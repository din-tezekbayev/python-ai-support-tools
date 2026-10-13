import json
import sqlite3
import unittest
from unittest.mock import patch

import httpx

from model_adapter import OpenAICompatibleProvider
from rag import KnowledgeBase
from workflow import GeneratedAnswer, SupportWorkflow


class FakeProvider:
    mode = 'test_stub'

    def __init__(self, callback):
        self.callback = callback

    def generate(self, question, evidence):
        return self.callback(question, evidence)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.connection = sqlite3.connect(':memory:')
        self.addCleanup(self.connection.close)
        self.kb = KnowledgeBase(self.connection)
        for tenant, text in [('clinic-a', 'Book an appointment online through the booking widget.'), ('shop-b', 'Book a private consultation by phone on Sunday.')]:
            self.kb.ingest(tenant_id=tenant, source_id='booking', title='Booking', text=text, version=1)

    def run_question(self, provider=None, question='booking appointment', tenant='clinic-a'):
        return SupportWorkflow(self.kb, provider).run(question, trusted_tenant_id=tenant)

    def test_graph_extractive_answer_is_cited_and_tenant_scoped(self):
        result = self.run_question()
        self.assertEqual(result.status, 'answered')
        self.assertEqual(result.answer, 'Book an appointment online through the booking widget.')
        self.assertEqual(result.citations[0].source_id, 'booking')
        self.assertEqual(result.steps, ['retrieve', 'generate', 'validate'])
        self.assertEqual(result.mode, 'offline_extractive')
        self.assertNotIn('Sunday', result.answer)

    def test_no_evidence_abstains_without_calling_provider(self):
        provider = FakeProvider(lambda *_: self.fail('Provider called without evidence'))
        result = self.run_question(provider, 'quasar obelisk')
        self.assertEqual(result.status, 'needs_human')
        self.assertEqual(result.reason, 'insufficient_evidence')
        self.assertEqual(result.citations, [])
        self.assertEqual(result.steps, ['retrieve', 'abstain'])

    def test_invalid_citation_indices_fail_closed(self):
        for indices in [[-1], [5], [0, 0]]:
            with self.subTest(indices=indices):
                result = self.run_question(FakeProvider(lambda *_: GeneratedAnswer(answer='Unverified answer', citation_indices=indices)))
                self.assertEqual(result.status, 'needs_human')
                self.assertEqual(result.reason, 'invalid_citations')
                self.assertNotIn('Unverified', result.answer)
                self.assertEqual(result.citations, [])

    def test_malformed_provider_response_fails_closed(self):
        for candidate in [{'answer': 'Hello', 'citation_indices': [True]}, {'answer': 'Hello', 'citation_indices': [0], 'tenant_id': 'shop-b'}]:
            result = self.run_question(FakeProvider(lambda *_: candidate))
            self.assertEqual(result.reason, 'provider_failure')
            self.assertEqual(result.status, 'needs_human')

    def test_provider_failure_does_not_expose_exception(self):
        def failure(*_):
            raise RuntimeError('secret-key-and-user-content')
        result = self.run_question(FakeProvider(failure))
        self.assertEqual(result.reason, 'provider_failure')
        self.assertNotIn('secret-key', json.dumps(result.to_dict()))

    def test_provider_can_abstain_on_lexically_related_but_insufficient_evidence(self):
        result = self.run_question(FakeProvider(lambda *_: GeneratedAnswer(answer='', citation_indices=[], abstain=True)))
        self.assertEqual(result.status, 'needs_human')
        self.assertEqual(result.reason, 'provider_abstained')
        self.assertEqual(result.citations, [])

    def test_ambient_langsmith_tracing_does_not_export_content(self):
        with patch.dict('os.environ', {'LANGSMITH_TRACING': 'true'}), patch('langsmith.run_trees.RunTree.post') as post:
            self.assertEqual(self.run_question().status, 'answered')
            post.assert_not_called()

    def test_source_revoked_during_generation_cannot_be_cited(self):
        def revoke(*_):
            self.kb.revoke('clinic-a', 'booking')
            return GeneratedAnswer(answer='Book online', citation_indices=[0])
        result = self.run_question(FakeProvider(revoke))
        self.assertEqual(result.reason, 'stale_or_unauthorized_evidence')
        self.assertEqual(result.citations, [])

    def test_question_and_tenant_require_trusted_nonempty_values(self):
        for question, tenant in [('', 'clinic-a'), ('a' * 2001, 'clinic-a'), ('booking', '')]:
            with self.assertRaises(ValueError):
                self.run_question(question=question, tenant=tenant)

    def test_separate_requests_do_not_share_history_or_trace(self):
        first = self.run_question()
        second = self.run_question(question='booking', tenant='shop-b')
        self.assertNotEqual(first.request_id, second.request_id)
        self.assertIn('Sunday', second.answer)
        self.assertEqual(len(second.steps), 3)


class AdapterTests(unittest.TestCase):
    def provider(self, handler):
        return OpenAICompatibleProvider(endpoint='https://provider.example/v1/chat/completions', model='synthetic-test', api_key='not-a-real-key', transport=httpx.MockTransport(handler))

    def evidence(self):
        connection = sqlite3.connect(':memory:')
        self.addCleanup(connection.close)
        kb = KnowledgeBase(connection)
        kb.ingest(tenant_id='a', source_id='policy', title='Booking', text='Book online.', version=1)
        return kb.search('a', 'booking')

    def test_adapter_sends_untrusted_reference_and_accepts_structured_response(self):
        def handler(request):
            payload = json.loads(request.content)
            self.assertEqual(payload['response_format'], {'type': 'json_object'})
            self.assertIn('untrusted_evidence', payload['messages'][1]['content'])
            self.assertNotIn('tools', payload)
            return httpx.Response(200, json={'choices': [{'message': {'content': '{"answer":"Book online.","citation_indices":[0]}'}}]})
        answer = self.provider(handler).generate('How to book?', self.evidence())
        self.assertEqual(answer.citation_indices, [0])

    def test_adapter_rejects_extra_fields_and_oversized_response(self):
        invalid = httpx.Response(200, json={'choices': [{'message': {'content': '{"answer":"Hi","citation_indices":[0],"action":"delete"}'}}]})
        with self.assertRaises(ValueError):
            self.provider(lambda _: invalid).generate('booking', self.evidence())
        with self.assertRaises(ValueError):
            self.provider(lambda _: httpx.Response(200, content=b'a' * 64001)).generate('booking', self.evidence())

    def test_endpoint_must_be_https_without_embedded_secrets(self):
        for endpoint in ['http://example.com', 'https://secret@example.com/v1', 'https://example.com?key=secret']:
            with self.assertRaises(ValueError):
                OpenAICompatibleProvider(endpoint=endpoint, model='test', api_key='test')


if __name__ == '__main__':
    unittest.main()
