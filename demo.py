"""Synthetic scenario: no credentials or network access required."""
import json
import sqlite3
from support_tools import Session, SupportTools, ToolError, initialize

def main() -> None:
    with sqlite3.connect(':memory:') as conn:
        initialize(conn)
        conn.executemany('INSERT INTO documents VALUES (?, ?, ?, ?)', [
            ('clinic-a', 'booking', 'Booking policy', 'Bookings can be changed 24 hours in advance.'),
            ('shop-b', 'booking', 'Private shop policy', 'A different tenant has different rules.'),
        ])
        conn.commit()
        tools = SupportTools(conn)
        session = Session('clinic-a', 'escalate')  # Authenticated application context.
        print(json.dumps(tools.execute(session, 'search_knowledge', '{"query":"booking"}', request_id='read-1'), indent=2))
        ticket = '{"summary":"Customer needs help changing a booking"}'
        first = tools.execute(session, 'create_ticket', ticket, request_id='op-42')
        retry = tools.execute(session, 'create_ticket', ticket, request_id='op-42')
        print(json.dumps({'first': first, 'retry': retry, 'same_ticket': first == retry}, indent=2))
        try:
            tools.execute(session, 'search_knowledge', '{"query":"policy","tenant_id":"shop-b"}', request_id='read-2')
        except ToolError as exc:
            print(f'Rejected model-supplied tenant override: {exc}')

if __name__ == '__main__':
    main()
