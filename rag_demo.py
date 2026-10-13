"""Run the deterministic RAG workflow or seed a synthetic API database."""
from contextlib import closing
import argparse
import json
import sqlite3

from rag import KnowledgeBase, seed_demo
from workflow import SupportWorkflow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', default=':memory:')
    args = parser.parse_args()
    with closing(sqlite3.connect(args.database)) as connection:
        knowledge = KnowledgeBase(connection)
        seed_demo(knowledge)
        for tenant, question in [('clinic-a', 'How do I book an appointment?'), ('shop-b', 'How long does shipping take?'), ('clinic-a', 'quantum rocket insurance')]:
            result = SupportWorkflow(knowledge).run(question, trusted_tenant_id=tenant)
            print(json.dumps(result.to_dict(), indent=2))


if __name__ == '__main__':
    main()
