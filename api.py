"""Minimal authenticated FastAPI boundary. Tokens map to tenants server-side."""

from contextlib import closing
import hmac
import json
import logging
import os
import sqlite3
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

from model_adapter import provider_from_env
from rag import Evidence, KnowledgeBase
from workflow import AnswerProvider, SupportWorkflow


class AskResponse(BaseModel):
    request_id: str
    status: str
    answer: str
    citations: list[Evidence]
    mode: str
    reason: str | None
    latency_ms: float
    steps: list[str]

logger = logging.getLogger('support_demo.requests')


class AskRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    question: StrictStr = Field(min_length=1, max_length=2000)

    @field_validator('question')
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError('question cannot be blank')
        return value.strip()


class BodyLimitMiddleware:
    """Bound the unauthenticated POST body before FastAPI JSON parsing."""
    def __init__(self, app, maximum: int = 8192):
        self.app, self.maximum = app, maximum

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope['path'] != '/ask' or scope['method'] != 'POST':
            return await self.app(scope, receive, send)
        body = bytearray()
        while True:
            message = await receive()
            if message['type'] == 'http.disconnect':
                return
            body.extend(message.get('body', b''))
            if len(body) > self.maximum:
                return await JSONResponse({'detail': 'Request body too large'}, status_code=413)(scope, receive, send)
            if not message.get('more_body', False):
                break
        sent = False

        async def bounded_receive():
            nonlocal sent
            if not sent:
                sent = True
                return {'type': 'http.request', 'body': bytes(body), 'more_body': False}
            return await receive()

        return await self.app(scope, bounded_receive, send)


def _tokens_from_env() -> dict[str, str]:
    try:
        tokens = json.loads(os.environ.get('SUPPORT_API_TOKENS', '{}'))
    except (ValueError, TypeError):
        return {}
    if not isinstance(tokens, dict) or not all(
        isinstance(key, str) and key.strip() and isinstance(value, str) and value.strip()
        for key, value in tokens.items()
    ):
        return {}
    return tokens


def create_app(*, database_path: str | None = None, tokens: dict[str, str] | None = None,
               provider: AnswerProvider | None = None) -> FastAPI:
    app = FastAPI(title='Tenant-scoped Support RAG Demo', version='0.2.0')
    app.add_middleware(BodyLimitMiddleware)
    configured_tokens = dict(_tokens_from_env() if tokens is None else tokens)
    database = database_path or os.environ.get('SUPPORT_DB_PATH', 'support-demo.sqlite3')
    answer_provider = provider if provider is not None else provider_from_env()
    security = HTTPBearer(auto_error=False)

    def authenticate(credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)]) -> str:
        if not configured_tokens:
            raise HTTPException(503, 'API authentication is not configured')
        if credentials is not None and credentials.scheme.lower() == 'bearer':
            for token, tenant in configured_tokens.items():
                if hmac.compare_digest(credentials.credentials.encode(), token.encode()):
                    return tenant
        raise HTTPException(401, 'Invalid bearer token', headers={'WWW-Authenticate': 'Bearer'})

    @app.get('/health')
    def health():
        return {'status': 'ok', 'authentication_configured': bool(configured_tokens)}

    @app.post('/ask', response_model=AskResponse)
    def ask(body: AskRequest, tenant: Annotated[str, Depends(authenticate)]):
        # One connection/workflow per request; no cross-user conversation state.
        with closing(sqlite3.connect(database, timeout=5.0)) as connection:
            result = SupportWorkflow(KnowledgeBase(connection), answer_provider).run(
                body.question, trusted_tenant_id=tenant,
            )
        logger.info('support_request', extra={
            'request_id': result.request_id, 'answer_status': result.status,
            'latency_ms': result.latency_ms, 'node_steps': result.steps,
        })
        return result.to_dict()

    return app


app = create_app()
