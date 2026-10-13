# Engineering decisions

## One support workflow

This demonstration targets policy lookup: booking, cancellation and receipts for an invented clinic, plus an unrelated shop to exercise tenancy. The user receives evidence and a clear failure outcome. Ingestion is local/operator-controlled. A handoff is a suggested next step, not a completed action.

The LangGraph state carries question, retrieved evidence and answer candidates. Authenticated tenant context is captured outside model-editable state. Each request gets a fresh graph; this version does not need conversation storage or a checkpoint database. Original tool-call and idempotent-ticket examples remain separately reviewable.

## Retrieval before model choice

Word-window chunking preserves a bounded passage and stable locator. SQLite records source version, approval, audience and revocation. SQL filters the authorized corpus before BM25 statistics and ranking. Final citation resolution checks the current source and exact excerpt, including changes made while a provider is answering.

BM25 is intentionally a measurable lexical baseline. A semantic retriever would need an identified embedding model, dimensionality/version migration, the same authorization filter, deletion semantics and a comparison on representative questions. A vector store by itself would not demonstrate better answers. The current full-corpus in-memory ranking is not a scale design.

## Failure semantics

No retrieval result, model abstention/failure, invalid citation or stale evidence leads to `needs_human`. The live provider can abstain even when retrieval returns text. Offline mode returns an excerpt on lexical overlap and cannot decide semantic sufficiency. Citation provenance alone does not prove entailment or resolve contradictory policies.

A retrieved instruction cannot grant access or call a tool. It may still influence a live model or appear verbatim in extractive output. Safety of generated language requires separate adversarial and task-quality evaluation. Tests assert authorization and side-effect boundaries, not universal injection resistance.

## API and operation

Server-configured bearer credentials map to tenants; no request-supplied tenant override is allowed. This illustrates an authentication boundary without claiming an enterprise identity implementation. The API bounds request size, schema and model response size. Model/network errors are reduced to a safe failure reason; logs omit raw questions, evidence and credentials.

Request IDs, graph steps and elapsed time are lightweight diagnostic metadata. There is no fabricated dollar cost metric; pricing depends on a selected provider and measured usage. No external trace backend or production monitoring is claimed. The optional provider uses five-second I/O timeouts and checks a twenty-second response budget between received chunks. This is a cooperative bound, not a hard wall-clock cancellation of every network operation.

## Before a real customer deployment

Choose an actual workflow, corpus and permissions model with the customer. Establish approved sources and ownership; evaluate no-answer, stale, contradictory and adversarial cases alongside useful questions. Compare retrieval variants on the same set before changing infrastructure. Add OIDC/identity lifecycle, rate limits, retention, encrypted storage, audited ingestion, provider privacy controls and workload-appropriate concurrency. Measure answer support, latency, token use and escalation outcomes rather than projecting benefits from synthetic scores.

Human-approved action execution, MCP, streaming, hybrid retrieval and durable workflows are reasonable next features when a real integration requires them. This small project does not claim that these are already implemented.
