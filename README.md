# Tenant-safe tools for an AI support assistant

A small Python demonstration of the application boundary around model-proposed tool calls. Synthetic knowledge-base content and local SQLite, with no API keys or network calls.

**New portfolio demonstration, October 2026.** Prepared with AI assistance. This is not historical production code or unaided handwritten code. It contains no Ping, CoFounder, healthcare-client source code or customer data.

## Run

Python 3.10+; standard library only.

```bash
python3 demo.py
python3 -m unittest discover -s tests -v
```

The demo retrieves one tenant's booking policy, creates a ticket, repeats the operation without a duplicate, and rejects a model-supplied tenant override.

## Design

- `Session` comes from trusted authentication and application intent selection, outside model output.
- Tools are allowed per intent; unknown tools and intents fail closed.
- JSON arguments have strict fields, bounded text and duplicate-key rejection.
- Retrieval binds the trusted tenant and uses SQL parameters. LIKE wildcard input is escaped.
- Retrieved documents are untrusted reference material, never interpreted as tool calls.
- A transaction and tenant-scoped unique operation key prevent duplicate tickets on retries. The same key with different arguments produces an error.
- Tests exercise cross-tenant access, permissions, malicious input, malformed JSON, retry persistence and payload conflicts.

## Scope

This demonstrates a tool boundary, not a complete solution to prompt injection. A model adapter must preserve the distinction between instructions and retrieved text, constrain the orchestration loop and use independently authenticated context. Search is deterministic substring retrieval, not embeddings or a RAG quality benchmark. Ticket creation is local; no helpdesk or customer system is contacted. Production work would add service authentication, resource limits, observability, data retention, provider failure handling and workload-specific evaluations.

For an interview, explain why tenant ID and operation key must not be chosen by the model, how database uniqueness makes retries reliable, and what this boundary does and does not prevent.
