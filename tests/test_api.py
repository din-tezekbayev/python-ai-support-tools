import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from api import create_app
from rag import KnowledgeBase
from workflow import GeneratedAnswer


class APITests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database = str(Path(self.directory.name) / 'knowledge.sqlite3')
        with sqlite3.connect(self.database) as connection:
            knowledge = KnowledgeBase(connection)
            knowledge.ingest(tenant_id='clinic-a', source_id='booking', title='Booking', text='Book clinic appointments online.', version=1)
            knowledge.ingest(tenant_id='shop-b', source_id='booking', title='Booking', text='Book shop appointments by phone.', version=1)
        self.client = TestClient(create_app(database_path=self.database, tokens={'secret-a': 'clinic-a', 'secret-b': 'shop-b'}))

    def ask(self, payload=None, token='secret-a', **kwargs):
        return self.client.post('/ask', json=payload or {'question': 'booking appointments'}, headers={'Authorization': f'Bearer {token}'}, **kwargs)

    def test_health_and_authenticated_question(self):
        self.assertEqual(self.client.get('/health').status_code, 200)
        response = self.ask()
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result['status'], 'answered')
        self.assertIn('clinic', result['answer'])
        self.assertEqual(result['steps'], ['retrieve', 'generate', 'validate'])

    def test_missing_invalid_and_header_only_auth_rejected(self):
        for headers in [{}, {'Authorization': 'Bearer unknown'}, {'X-Tenant-ID': 'clinic-a'}]:
            response = self.client.post('/ask', json={'question': 'booking'}, headers=headers)
            self.assertEqual(response.status_code, 401)

    def test_missing_auth_configuration_fails_closed(self):
        client = TestClient(create_app(database_path=self.database, tokens={}))
        response = client.post('/ask', json={'question': 'booking'}, headers={'Authorization': 'Bearer anything'})
        self.assertEqual(response.status_code, 503)

    def test_tenant_override_and_bad_schema_rejected(self):
        for payload in [{'question': 'booking', 'tenant_id': 'shop-b'}, {'question': 15}, {'question': ' '}, {'question': 'a' * 2001}, {'question': 'booking', 'intent': 'escalate'}]:
            with self.subTest(payload=str(payload)[:100]):
                self.assertEqual(self.ask(payload).status_code, 422)

    def test_server_token_mapping_prevents_cross_tenant_results(self):
        self.assertIn('clinic', self.ask().json()['answer'])
        self.assertNotIn('shop', self.ask().json()['answer'])
        self.assertIn('shop', self.ask(token='secret-b').json()['answer'])

    def test_oversized_body_rejected_before_json_validation(self):
        response = self.client.post('/ask', content=b' ' * 8193, headers={'Authorization': 'Bearer secret-a', 'Content-Type': 'application/json'})
        self.assertEqual(response.status_code, 413)

    def test_no_evidence_returns_handoff_and_empty_citations(self):
        response = self.ask({'question': 'quasar obelisk'})
        self.assertEqual(response.json()['status'], 'needs_human')
        self.assertEqual(response.json()['citations'], [])

    def test_request_logs_contain_metadata_not_user_content(self):
        with self.assertLogs('support_demo.requests', level='INFO') as logs:
            result = self.ask({'question': 'booking private-user-question-marker'}).json()
        record = logs.records[0]
        self.assertEqual(record.request_id, result['request_id'])
        self.assertNotIn('private-user-question-marker', json.dumps(record.__dict__, default=str))

    def test_invalid_provider_citations_and_errors_return_handoff(self):
        class Provider:
            mode = 'test_stub'
            fail = False

            def generate(self, question, evidence):
                if self.fail:
                    raise RuntimeError('private-provider-detail')
                return GeneratedAnswer(answer='unverified-provider-answer', citation_indices=[99])
        provider = Provider()
        client = TestClient(create_app(database_path=self.database, tokens={'a': 'clinic-a'}, provider=provider))
        for fails in [False, True]:
            provider.fail = fails
            response = client.post('/ask', json={'question': 'booking'}, headers={'Authorization': 'Bearer a'})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['status'], 'needs_human')
            self.assertEqual(response.json()['citations'], [])
            self.assertNotIn('private-provider-detail', response.text)
            self.assertNotIn('unverified-provider-answer', response.text)

    def test_malformed_environment_token_map_fails_closed(self):
        with patch.dict('os.environ', {'SUPPORT_API_TOKENS': '{bad json'}):
            client = TestClient(create_app(database_path=self.database))
        self.assertEqual(client.post('/ask', json={'question': 'booking'}).status_code, 503)


if __name__ == '__main__':
    unittest.main()
