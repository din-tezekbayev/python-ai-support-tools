"""Small, offline lexical RAG index. Fixtures are synthetic, never client data.

BM25 ranks only authorized customer chunks. SQLite is the source of truth;
retrieved text remains untrusted data and must never grant tool permissions.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3


@dataclass(frozen=True)
class Evidence:
    source_id: str
    version: int
    chunk_id: str
    title: str
    text: str
    score: float


# Deliberately small English tokenizer: no embeddings or semantic-search claim.
_STOP = frozenset('a an and are as at be by can do does for from how i in is it my of on or our the to was what when where which with you your'.split())

def _tokens(text: str) -> list[str]:
    return [word for word in re.findall(r"[a-z0-9]+", text.lower()) if word not in _STOP]


def _chunks(text: str, size: int = 100, overlap: int = 20) -> list[str]:
    """Word windows with bounded overlap; stable for identical source content."""
    words = text.split()
    result = []
    for start in range(0, len(words), size - overlap):
        result.append(' '.join(words[start:start + size]))
        if start + size >= len(words):
            break
    return result


def _nonempty(value: str, field: str, maximum: int) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f'{field} must be non-empty text of at most {maximum} characters')


class KnowledgeBase:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self.connection.executescript('''
            CREATE TABLE IF NOT EXISTS kb_sources (
              tenant_id TEXT NOT NULL, source_id TEXT NOT NULL,
              title TEXT NOT NULL, text TEXT NOT NULL, version INTEGER NOT NULL,
              audience TEXT NOT NULL, approved INTEGER NOT NULL,
              revoked INTEGER NOT NULL DEFAULT 0,
              PRIMARY KEY (tenant_id, source_id)
            );
            CREATE TABLE IF NOT EXISTS kb_chunks (
              tenant_id TEXT NOT NULL, source_id TEXT NOT NULL,
              version INTEGER NOT NULL, chunk_id TEXT NOT NULL, text TEXT NOT NULL,
              PRIMARY KEY (tenant_id, source_id, chunk_id)
            );
        ''')

    def ingest(self, *, tenant_id: str, source_id: str, title: str, text: str,
               version: int, audience: str = 'customer', approved: bool = True) -> None:
        """Atomically replace a source using an idle, caller-owned connection.

        This method owns its transaction. An existing transaction is rejected
        without committing or rolling back the caller's work. BEGIN IMMEDIATE
        serializes writers before checking the current version.
        """
        if self.connection.in_transaction:
            raise ValueError('ingest requires a connection with no active transaction')
        for value, field, maximum in ((tenant_id, 'tenant_id', 100),
                                      (source_id, 'source_id', 100),
                                      (title, 'title', 300), (text, 'text', 100_000)):
            _nonempty(value, field, maximum)
        if type(version) is not int or version < 1:
            raise ValueError('version must be a positive integer')
        if audience not in {'customer', 'internal'} or type(approved) is not bool:
            raise ValueError('invalid audience or approval flag')
        with self.connection:
            self.connection.execute('BEGIN IMMEDIATE')
            existing = self.connection.execute(
                'SELECT title,text,version,audience,approved,revoked FROM kb_sources '
                'WHERE tenant_id=? AND source_id=?', (tenant_id, source_id)
            ).fetchone()
            if existing:
                if version < existing[2]:
                    raise ValueError('cannot ingest an obsolete version')
                if version == existing[2]:
                    if tuple(existing) == (title, text, version, audience, int(approved), 0):
                        return  # Idempotent exact replay.
                    raise ValueError('changed or revoked content requires a new version')
            self.connection.execute(
                'INSERT INTO kb_sources VALUES (?,?,?,?,?,?,?,0) '
                'ON CONFLICT(tenant_id,source_id) DO UPDATE SET '
                'title=excluded.title,text=excluded.text,version=excluded.version,'
                'audience=excluded.audience,approved=excluded.approved,revoked=0',
                (tenant_id, source_id, title, text, version, audience, int(approved)))
            self.connection.execute('DELETE FROM kb_chunks WHERE tenant_id=? AND source_id=?',
                                    (tenant_id, source_id))
            for index, chunk in enumerate(_chunks(text)):
                digest = hashlib.sha256(chunk.encode()).hexdigest()[:12]
                chunk_id = f'{source_id}:v{version}:{index}:{digest}'
                self.connection.execute('INSERT INTO kb_chunks VALUES (?,?,?,?,?)',
                                        (tenant_id, source_id, version, chunk_id, chunk))

    def revoke(self, tenant_id: str, source_id: str) -> None:
        with self.connection:
            self.connection.execute('UPDATE kb_sources SET revoked=1 WHERE tenant_id=? AND source_id=?',
                                    (tenant_id, source_id))

    def _authorized(self, tenant_id: str) -> list[tuple]:
        # Access filtering happens in SQL BEFORE tokenization, statistics or ranking.
        return self.connection.execute(
            'SELECT c.source_id,c.version,c.chunk_id,s.title,c.text '
            'FROM kb_chunks c JOIN kb_sources s '
            'ON c.tenant_id=s.tenant_id AND c.source_id=s.source_id AND c.version=s.version '
            "WHERE s.tenant_id=? AND s.audience='customer' AND s.approved=1 AND s.revoked=0 "
            'ORDER BY c.source_id,c.chunk_id', (tenant_id,)
        ).fetchall()

    def search(self, tenant_id: str, question: str, limit: int = 3) -> list[Evidence]:
        _nonempty(tenant_id, 'tenant_id', 100)
        _nonempty(question, 'question', 4000)
        if type(limit) is not int or not 1 <= limit <= 20:
            raise ValueError('limit must be between 1 and 20')
        query = set(_tokens(question))
        rows = self._authorized(tenant_id)
        if not rows or not query:
            return []
        counts = [Counter(_tokens(row[3] + ' ' + row[4])) for row in rows]
        lengths = [sum(count.values()) for count in counts]
        average = sum(lengths) / len(lengths) or 1
        document_frequency = Counter(token for count in counts for token in count)
        ranked = []
        for row, count, length in zip(rows, counts, lengths):
            score = 0.0
            for token in sorted(query):
                frequency = count[token]
                if frequency:
                    inverse = math.log(1 + (len(rows) - document_frequency[token] + 0.5) /
                                       (document_frequency[token] + 0.5))
                    score += inverse * frequency * 2.2 / (frequency + 1.2 * (0.25 + 0.75 * length / average))
            if score > 0:
                ranked.append(Evidence(*row, score=score))
        return sorted(ranked, key=lambda item: (-item.score, item.source_id, item.chunk_id))[:limit]

    def resolve(self, tenant_id: str, citations: list[Evidence]) -> list[Evidence]:
        """Revalidate request evidence immediately before returning a response.

        Exact text/title matching also rejects caller-created citations that reuse a
        legitimate ID with invented text. This checks authorization and freshness,
        not whether a generated answer is logically supported by its citations.
        """
        current = {(row[0], row[1], row[2]): row for row in self._authorized(tenant_id)}
        for item in citations:
            row = current.get((item.source_id, item.version, item.chunk_id))
            if row is None or tuple(row) != (item.source_id, item.version, item.chunk_id, item.title, item.text):
                raise ValueError('citation is stale, unauthorized or does not match current evidence')
        return list(citations)


def seed_demo(kb: KnowledgeBase) -> None:
    """Load only the public, invented fixture set shipped with this repository."""
    records = json.loads((Path(__file__).parent / 'fixtures' / 'knowledge.json').read_text())
    for record in records:
        kb.ingest(**record)
