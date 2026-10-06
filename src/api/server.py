"""AgentCore Platform v1.0"""

# Standalone HTTP entry point for the agent.
# Entry points are adapters only — no business logic here.
# On the platform, the gateway calls agent.invoke() directly instead.

import json
import os
import secrets
from typing import Any, Dict, Optional, cast
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from framework.secrets.context import bound_secrets
from shared.secrets import factory as secrets_factory
from src.graph.graph import TelecomPlanServiceQAAgent, load_runtime_config

app = FastAPI(title="Agent")

# The declared runtime parameters (max_retry, timeout_s, retrieval tuning) are
# read here and handed to the graph. Constructing the agent with no config
# would leave every declared value dead and silently run on framework
# defaults.
agent = TelecomPlanServiceQAAgent(config=load_runtime_config())
agent.compile()
# The namespace and agent name must match the manifest values.
agent.provision_secrets(secrets_factory(namespace="tel", agent_name="TelecomPlanServiceQAAgent"))

# Upper bound on the serialized caller-data object. The pipeline bounds every
# field individually; this bounds the request as a whole so a single call
# cannot arrive with megabytes of catalog text attached.
MAX_INPUT_CONTEXT_BYTES = 256 * 1024


class InvokeRequest(BaseModel):
    input: str
    session_id: str = ""
    # Optional caller data — see src/caller_contract.py for the accepted shape.
    input_context: Optional[Dict[str, Any]] = None


@app.post("/invoke")
async def invoke(req: InvokeRequest, request: Request) -> Dict[str, Any]:
    trust = getattr(request.state, "trust_level", TrustLevel.ANONYMOUS)
    # Standalone caller auth: when INVOKE_AUTH_TOKEN is set on the server
    # environment, a caller that no upstream middleware vouched for (still
    # ANONYMOUS) must present it as a bearer token and then runs at
    # VERIFIED_EXTERNAL. Trust established by middleware is never demoted.
    # This adapter is the entry-point auth boundary — a deployment-level caller
    # credential rather than an agent secret, so the per-invocation secret
    # provider does not apply: no invocation context exists before auth.
    expected = os.environ.get("INVOKE_AUTH_TOKEN")
    if expected and trust is TrustLevel.ANONYMOUS:
        supplied = request.headers.get("authorization", "")
        # Compare bytes: compare_digest raises TypeError on non-ASCII str input
        # (headers decode as latin-1), which would 500 instead of the generic 401.
        if not secrets.compare_digest(supplied.encode(), f"Bearer {expected}".encode()):
            # Generic body on purpose — do not leak whether the token was
            # absent, malformed, or wrong.
            raise HTTPException(status_code=401, detail="Token is invalid or expired.")
        trust = TrustLevel.VERIFIED_EXTERNAL

    input_context = req.input_context or {}
    if input_context and len(json.dumps(input_context).encode("utf-8")) > MAX_INPUT_CONTEXT_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"input_context exceeds {MAX_INPUT_CONTEXT_BYTES} bytes.",
        )

    with bound_secrets(agent._secrets_provider):
        ctx = InvocationContext(
            session_id=req.session_id or str(uuid4()),
            caller_trust_level=trust,
            caller_id=getattr(request.state, "caller_id", ""),
        )
        return cast(Dict[str, Any], agent.invoke(req.input, ctx=ctx, input_context=input_context))


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "agent": "TelecomPlanServiceQAAgent"}
