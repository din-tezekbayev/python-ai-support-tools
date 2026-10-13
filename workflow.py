"""Bounded, per-request LangGraph RAG workflow over trusted tenant context.

The offline mode returns an exact retrieved excerpt. Citation validation proves
source provenance and current authorization, not semantic truth of live output.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Annotated, Protocol, TypedDict
from uuid import uuid4
import operator

from langgraph.graph import END, START, StateGraph
from langsmith import tracing_context
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator

from rag import Evidence, KnowledgeBase


class GeneratedAnswer(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    answer: StrictStr = Field(max_length=4000)
    citation_indices: list[StrictInt] = Field(max_length=3)
    abstain: StrictBool = False

    @model_validator(mode='after')
    def coherent_answer(self):
        if self.abstain:
            if self.answer or self.citation_indices:
                raise ValueError('Abstention must have empty answer and citations')
        elif not self.answer.strip() or not self.citation_indices:
            raise ValueError('An answer requires content and citations')
        return self


class AnswerProvider(Protocol):
    mode: str

    def generate(self, question: str, evidence: list[Evidence]) -> GeneratedAnswer: ...


class ExtractiveProvider:
    """Deterministic smoke-test baseline; does not call or imitate an LLM."""
    mode = 'offline_extractive'

    def generate(self, question: str, evidence: list[Evidence]) -> GeneratedAnswer:
        return GeneratedAnswer(answer=evidence[0].text[:4000], citation_indices=[0])


class RAGState(TypedDict, total=False):
    question: str
    evidence: list[Evidence]
    candidate: GeneratedAnswer
    answer: str
    citations: list[Evidence]
    reason: str
    status: str
    steps: Annotated[list[str], operator.add]


@dataclass(frozen=True)
class AnswerResult:
    request_id: str
    status: str
    answer: str
    citations: list[Evidence]
    mode: str
    reason: str | None
    latency_ms: float
    steps: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


class SupportWorkflow:
    def __init__(self, knowledge: KnowledgeBase, provider: AnswerProvider | None = None):
        self.knowledge = knowledge
        self.provider = provider or ExtractiveProvider()

    def run(self, question: str, *, trusted_tenant_id: str) -> AnswerResult:
        if not isinstance(question, str) or not question.strip() or len(question) > 2000:
            raise ValueError('question must contain 1–2000 characters')
        if not isinstance(trusted_tenant_id, str) or not trusted_tenant_id.strip():
            raise ValueError('An authenticated tenant is required')
        # The tenant is captured by application code, never accepted as graph/model state.
        started = perf_counter()

        def retrieve(state: RAGState) -> dict:
            evidence = self.knowledge.search(trusted_tenant_id, state['question'], limit=3)
            return {'evidence': evidence, 'steps': ['retrieve']}

        def generate(state: RAGState) -> dict:
            try:
                candidate = self.provider.generate(state['question'], state['evidence'])
                # Re-validate even injected/custom providers; never trust a provider instance.
                candidate = GeneratedAnswer.model_validate(
                    candidate.model_dump() if isinstance(candidate, GeneratedAnswer) else candidate
                )
                return {'candidate': candidate, 'steps': ['generate']}
            except Exception:
                # Deliberately omit provider exceptions: they can contain prompts or keys.
                return {'reason': 'provider_failure', 'steps': ['generate']}

        def validate(state: RAGState) -> dict:
            if state.get('reason'):
                return {'steps': ['validate']}
            candidate = state['candidate']
            if candidate.abstain:
                return {'reason': 'provider_abstained', 'steps': ['validate']}
            indices = candidate.citation_indices
            if len(set(indices)) != len(indices) or any(
                index < 0 or index >= len(state['evidence']) for index in indices
            ) or not candidate.answer.strip():
                return {'reason': 'invalid_citations', 'steps': ['validate']}
            try:
                citations = self.knowledge.resolve(
                    trusted_tenant_id, [state['evidence'][index] for index in indices]
                )
            except ValueError:
                return {'reason': 'stale_or_unauthorized_evidence', 'steps': ['validate']}
            return {
                'answer': candidate.answer, 'citations': citations,
                'status': 'answered', 'steps': ['validate'],
            }

        def abstain(state: RAGState) -> dict:
            return {
                'status': 'needs_human', 'answer': 'I cannot answer from the available knowledge. Please contact support.',
                'citations': [], 'reason': state.get('reason', 'insufficient_evidence'),
                'steps': ['abstain'],
            }

        graph = StateGraph(RAGState)
        graph.add_node('retrieve', retrieve)
        graph.add_node('generate', generate)
        graph.add_node('validate', validate)
        graph.add_node('abstain', abstain)
        graph.add_edge(START, 'retrieve')
        graph.add_conditional_edges('retrieve', lambda state: 'generate' if state['evidence'] else 'abstain')
        graph.add_edge('generate', 'validate')
        graph.add_conditional_edges('validate', lambda state: 'abstain' if state.get('reason') else END)
        graph.add_edge('abstain', END)
        # No checkpointer, chat history, retries, model tools, or unbounded agent loop.
        # Never export prompts/evidence through ambient LangSmith tracing variables.
        with tracing_context(enabled=False):
            state = graph.compile().invoke(
                {'question': question.strip(), 'steps': []},
                {'recursion_limit': 8, 'callbacks': []},
            )
        return AnswerResult(
            request_id=str(uuid4()), status=state['status'], answer=state['answer'],
            citations=state['citations'], mode=self.provider.mode, reason=state.get('reason'),
            latency_ms=round((perf_counter() - started) * 1000, 2), steps=state['steps'],
        )
