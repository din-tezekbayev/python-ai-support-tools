"""Run: python -m evals.run. Small fixed fixtures, not production accuracy data."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3

from rag import KnowledgeBase, seed_demo


def evaluate(k: int = 3) -> dict:
    cases = json.loads((Path(__file__).parent / 'cases.json').read_text())
    corpus = json.loads((Path(__file__).parent.parent / 'fixtures' / 'knowledge.json').read_text())
    details, recalls, reciprocals, abstentions = [], [], [], []
    for case in cases:
        with closing(sqlite3.connect(':memory:')) as connection:
            kb = KnowledgeBase(connection)
            seed_demo(kb)
            if case.get('revoke'):
                kb.revoke(case['tenant'], case['revoke'])
            evidence = kb.search(case['tenant'], case['question'], limit=k)
            kb.resolve(case['tenant'], evidence)
            found = [item.source_id for item in evidence]
            gold = set(case['gold'])
            result = {'id': case['id'], 'expected_sources': sorted(gold), 'retrieved_sources': found}
            if gold:
                recall = len(gold.intersection(found)) / len(gold)
                reciprocal = next((1 / rank for rank, source in enumerate(found, 1) if source in gold), 0)
                recalls.append(recall)
                reciprocals.append(reciprocal)
                result.update(recall_at_k=recall, reciprocal_rank=reciprocal)
            else:
                abstentions.append(not found)
                result['correct_abstention'] = not found
            result['passed'] = gold.issubset(found) if gold else not found
            details.append(result)
    return {
        'method': 'lexical BM25, access filtering before ranking', 'k': k,
        'scope': 'Fixed synthetic retrieval cases; not generated-answer quality or production accuracy. Malicious text remains retrievable data; workflow tests verify its handling.',
        'corpus_source_count': len(corpus), 'corpus_tenant_count': len({row['tenant_id'] for row in corpus}),
        'case_count': len(cases), 'passed': all(case['passed'] for case in details),
        'answerable_cases': len(recalls), 'no_evidence_cases': len(abstentions),
        'recall_at_k': sum(recalls) / len(recalls),
        'mrr_at_k': sum(reciprocals) / len(reciprocals),
        'no_evidence_abstention_rate': sum(abstentions) / len(abstentions),
        'cases': details,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Exit nonzero if a gold source is missing or a no-evidence case retrieves content')
    args = parser.parse_args(argv)
    report = evaluate()
    print(json.dumps(report, indent=2))
    return int(args.check and not report['passed'])


if __name__ == '__main__':
    raise SystemExit(main())
