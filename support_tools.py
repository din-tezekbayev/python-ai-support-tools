"""Synthetic portfolio demo, October 2026. No production/client source code.

A deterministic boundary for model-proposed support tool calls. Authentication
and intent authorization come from application code, not model output.
"""
from __future__ import annotations
import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Literal

class ToolError(ValueError):
    """Expected tool failure, safe to expose to an orchestrator."""

@dataclass(frozen=True)
class Session:
    """Construct after authentication, never from model JSON."""
    tenant_id: str
    intent: Literal['answer_question', 'escalate']

SCHEMA = '''
CREATE TABLE IF NOT EXISTS documents (
 tenant_id TEXT NOT NULL, document_id TEXT NOT NULL, title TEXT NOT NULL,
 body TEXT NOT NULL, PRIMARY KEY (tenant_id, document_id)
);
CREATE TABLE IF NOT EXISTS tickets (
 id INTEGER PRIMARY KEY, tenant_id TEXT NOT NULL, request_id TEXT NOT NULL,
 summary TEXT NOT NULL, UNIQUE (tenant_id, request_id)
);
'''

def initialize(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)

def _text(value: Any, field: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ToolError(f'{field} must be non-empty text of at most {maximum} characters')
    return value.strip()

def _arguments(raw: str, expected: set[str]) -> dict[str, Any]:
    if len(raw) > 4096:
        raise ToolError('Tool arguments exceed size limit')
    def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ToolError(f'Duplicate argument: {key}')
            result[key] = value
        return result
    try:
        parsed = json.loads(raw, object_pairs_hook=unique_keys)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ToolError('Tool arguments must be valid JSON') from exc
    if not isinstance(parsed, dict) or set(parsed) != expected:
        raise ToolError('Missing or unexpected tool arguments')
    return parsed

class SupportTools:
    ALLOWED = {
        'answer_question': frozenset({'search_knowledge'}),
        'escalate': frozenset({'search_knowledge', 'create_ticket'}),
    }
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def execute(self, session: Session, tool: str, raw_arguments: str,
                *, request_id: str) -> dict[str, Any]:
        """Run a proposed call. App-generated request_id is reused on retries."""
        _text(session.tenant_id, 'tenant_id', 100)
        if tool not in self.ALLOWED.get(session.intent, frozenset()):
            raise ToolError('Tool is not permitted for this intent')
        if tool == 'search_knowledge':
            args = _arguments(raw_arguments, {'query'})
            return self._search(session.tenant_id, _text(args['query'], 'query', 200))
        args = _arguments(raw_arguments, {'summary'})
        return self._ticket(session.tenant_id, _text(request_id, 'request_id', 100),
                            _text(args['summary'], 'summary', 1000))

    def _search(self, tenant_id: str, query: str) -> dict[str, Any]:
        # Literal substring retrieval, deliberately not an embedding benchmark.
        literal = query.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
        rows = self.connection.execute(
            "SELECT document_id, title, body FROM documents "
            "WHERE tenant_id = ? AND (title LIKE ? ESCAPE '\\' "
            "OR body LIKE ? ESCAPE '\\') ORDER BY document_id LIMIT 5",
            (tenant_id, f'%{literal}%', f'%{literal}%'),
        ).fetchall()
        return {'kind': 'untrusted_reference_material', 'sources': [
            {'id': row[0], 'title': row[1], 'text': row[2]} for row in rows
        ]}

    def _ticket(self, tenant_id: str, request_id: str, summary: str) -> dict[str, Any]:
        # Transaction + uniqueness constraint persists across application instances.
        with self.connection:
            self.connection.execute(
                'INSERT INTO tickets (tenant_id, request_id, summary) VALUES (?, ?, ?) '
                'ON CONFLICT (tenant_id, request_id) DO NOTHING',
                (tenant_id, request_id, summary),
            )
            row = self.connection.execute(
                'SELECT id, summary FROM tickets WHERE tenant_id = ? AND request_id = ?',
                (tenant_id, request_id),
            ).fetchone()
            if row is None:
                raise RuntimeError('Ticket insert did not produce a record')
            if row[1] != summary:
                raise ToolError('request_id was already used with different arguments')
        return {'ticket_id': row[0], 'summary': row[1]}
