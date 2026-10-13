from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
import tempfile
from threading import Event
from unittest.mock import patch
import sqlite3
import unittest

from evals.run import evaluate, main
from rag import KnowledgeBase, seed_demo


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.connection = sqlite3.connect(':memory:')
        self.addCleanup(self.connection.close)
        self.kb = KnowledgeBase(self.connection)
        seed_demo(self.kb)

    def test_access_filter_precedes_ranking_and_statistics(self):
        before = self.kb.search('clinic-a', 'payment receipt')
        for index in range(10):
            self.kb.ingest(tenant_id='shop-b', source_id=f'hidden-{index}', title='payment receipt',
                           text='payment receipt ' * 200, version=1)
        self.assertEqual(self.kb.search('clinic-a', 'payment receipt'), before)
        for question in ('meridianoverride', 'auroratreatment', 'cobaltparcel'):
            self.assertEqual(self.kb.search('clinic-a', question), [])

    def test_version_replacement_removes_old_chunks_and_invalidates_citations(self):
        old = self.kb.search('shop-b', 'cobaltparcel')
        self.kb.ingest(tenant_id='shop-b', source_id='shipping', title='Shipping',
                       text='New shipping uses violetparcel.', version=2)
        self.assertEqual(self.kb.search('shop-b', 'cobaltparcel'), [])
        self.assertEqual(self.kb.search('shop-b', 'violetparcel')[0].version, 2)
        with self.assertRaises(ValueError):
            self.kb.resolve('shop-b', old)
        self.assertEqual(self.connection.execute(
            "SELECT COUNT(*) FROM kb_chunks WHERE tenant_id='shop-b' AND version=1").fetchone()[0], 0)

    def test_revocation_survives_new_repository_instance(self):
        evidence = self.kb.search('shop-b', 'cobaltparcel')
        self.kb.revoke('shop-b', 'shipping')
        reopened = KnowledgeBase(self.connection)
        self.assertEqual(reopened.search('shop-b', 'cobaltparcel'), [])
        with self.assertRaises(ValueError):
            reopened.resolve('shop-b', evidence)

    def test_cross_tenant_and_forged_citations_rejected(self):
        evidence = self.kb.search('shop-b', 'cobaltparcel')
        with self.assertRaises(ValueError):
            self.kb.resolve('clinic-a', evidence)
        forged = [replace(evidence[0], text='Invented answer')]
        with self.assertRaises(ValueError):
            self.kb.resolve('shop-b', forged)
        self.assertEqual(self.kb.resolve('shop-b', evidence), evidence)

    def test_long_document_tail_retrieval_and_stable_chunk_ids(self):
        record = dict(tenant_id='clinic-a', source_id='long', title='Reference',
                      text=('ordinary filler ' * 200) + 'The zephyrreceipt document is downloadable.', version=1)
        self.kb.ingest(**record)
        evidence = self.kb.search('clinic-a', 'zephyrreceipt')
        self.assertEqual(len(evidence), 1)
        self.assertIn('zephyrreceipt', evidence[0].text)
        self.assertLessEqual(len(evidence[0].text.split()), 100)
        self.kb.ingest(**record)
        self.assertEqual(evidence, self.kb.search('clinic-a', 'zephyrreceipt'))

    def test_metadata_changes_require_new_version_and_change_access(self):
        with self.assertRaises(ValueError):
            self.kb.ingest(tenant_id='shop-b', source_id='shipping', title='Changed', text='changed', version=1)
        self.kb.ingest(tenant_id='shop-b', source_id='shipping', title='Internal',
                       text='cobaltparcel', version=2, audience='internal')
        self.assertEqual(self.kb.search('shop-b', 'cobaltparcel'), [])
        with self.assertRaises(ValueError):
            self.kb.ingest(tenant_id='shop-b', source_id='shipping', title='Old', text='old', version=1)

    def test_ingest_rejects_active_transaction_without_rolling_it_back(self):
        self.connection.execute('BEGIN')
        with self.assertRaisesRegex(ValueError, 'no active transaction'):
            self.kb.ingest(tenant_id='clinic-a', source_id='new', title='New', text='new', version=1)
        self.assertTrue(self.connection.in_transaction)
        self.connection.rollback()

    def test_waiting_writer_rechecks_version_after_acquiring_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'shared.sqlite'
            with closing(sqlite3.connect(path)) as writer, closing(
                sqlite3.connect(path, check_same_thread=False)
            ) as contender:
                kb = KnowledgeBase(writer)
                kb.ingest(tenant_id='clinic-a', source_id='policy', title='Policy', text='initial', version=1)
                stale_kb = KnowledgeBase(contender)
                writer.execute('BEGIN IMMEDIATE')
                writer.execute("UPDATE kb_sources SET version=3 WHERE source_id='policy'")
                first_statement, started = [], Event()

                def trace(statement):
                    if not first_statement:
                        first_statement.append(statement)
                        started.set()

                contender.set_trace_callback(trace)
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(stale_kb.ingest, tenant_id='clinic-a', source_id='policy',
                                         title='Policy', text='stale update', version=2)
                    try:
                        self.assertTrue(started.wait(2), 'contender did not start')
                        self.assertEqual(first_statement[0], 'BEGIN IMMEDIATE')
                    finally:
                        writer.commit()
                    with self.assertRaisesRegex(ValueError, 'obsolete version'):
                        future.result(timeout=3)
                self.assertEqual(writer.execute('SELECT version FROM kb_sources').fetchone()[0], 3)

    def test_sql_payload_is_literal_not_a_tenant_selector(self):
        self.assertEqual(self.kb.search("clinic-a' OR 1=1 --", 'cobaltparcel'), [])
        self.assertEqual(self.kb.search('clinic-a', '%___'), [])

    def test_malicious_reference_is_data_and_has_no_side_effect(self):
        evidence = self.kb.search('clinic-a', 'Diagnostic access sentinel')
        self.assertEqual(evidence[0].source_id, 'untrusted-import')
        self.assertIn('ignore all previous instructions', evidence[0].text)
        self.assertEqual(self.connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name='tickets'").fetchone()[0], 0)
        self.assertEqual(self.kb.search('clinic-a', 'cobaltparcel'), [])

    def test_input_bounds(self):
        for limit in (0, 21, True):
            with self.assertRaises(ValueError):
                self.kb.search('clinic-a', 'booking', limit)
        with self.assertRaises(ValueError):
            self.kb.search('clinic-a', 'x' * 4001)

    def test_fixed_evaluation_covers_answerable_and_no_evidence(self):
        report = evaluate()
        self.assertEqual(report['case_count'], len(report['cases']))
        self.assertEqual(report['corpus_source_count'], 7)
        self.assertEqual(report['corpus_tenant_count'], 2)
        self.assertTrue(report['passed'])
        self.assertEqual(report['answerable_cases'], 5)
        self.assertEqual(report['no_evidence_cases'], 5)
        self.assertEqual(report['recall_at_k'], 1)
        self.assertEqual(report['mrr_at_k'], 1)
        self.assertEqual(report['no_evidence_abstention_rate'], 1)


    def test_eval_check_gate_fails_for_retrieval_or_abstention_regression(self):
        with patch.object(KnowledgeBase, 'search', return_value=[]):
            missing_gold = evaluate()
        self.assertFalse(missing_gold['passed'])
        self.assertEqual(missing_gold['recall_at_k'], 0)
        evidence = self.kb.search('clinic-a', 'receipt')
        with patch.object(KnowledgeBase, 'search', return_value=evidence), patch.object(KnowledgeBase, 'resolve'):
            incorrect_abstention = evaluate()
        self.assertFalse(incorrect_abstention['passed'])
        self.assertEqual(incorrect_abstention['no_evidence_abstention_rate'], 0)
        for report in (missing_gold, incorrect_abstention):
            with patch('evals.run.evaluate', return_value=report), redirect_stdout(StringIO()):
                self.assertEqual(main(['--check']), 1)
                self.assertEqual(main([]), 0)



if __name__ == '__main__':
    unittest.main()
