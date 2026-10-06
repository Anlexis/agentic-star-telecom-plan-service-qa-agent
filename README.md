# Telecom Plan & Service Q&A Agent

AI agent for answering telecom plan and service questions, built with Agentic Star.

> **Category**: Cat 2 (domain-specific retrieval workflow)
> **Industry**: Telecommunications
> **Template ID**: TEL-C2-001

## Overview

A question-answering agent for telecom plans and services. A customer-service
representative, a sales assistant or a self-service portal asks a natural-language
question — about plan pricing, data allowances, contract terms, roaming, coverage or a
service procedure — and the agent answers from a plan and service catalog, quoting the
passages it used and citing each one.

The calling system supplies the catalog slice to answer from, so the agent reads the
operator's own live plan data rather than a fixed copy; a small sample catalog is
bundled so the template runs and its tests pass out of the box. Answer assembly is
deterministic — keyword retrieval and rule-based composition, no model call — which
makes every answer reproducible and traceable to a cited passage.

Every answer carries a standing informational disclaimer, and the output boundary
enforces that plus two more guarantees: every citation marker resolves to a listed
source, and no subscriber identifier or credential-shaped string is ever published.
The agent is read-only: it never looks up an account, changes a plan or calls a
billing system.

This is an agent template built with the **AGENTIC STAR** development platform and the
**AgentCore Framework**. It is intended to be taken as a starting point: fork it, adapt it to
your own data and policies, and run it inside your own AGENTIC STAR deployment.

## Requirements

**This template does not run standalone.** It requires:

| Requirement | Notes |
|---|---|
| **AGENTIC STAR platform** | The agent connects to the platform at start-up. Without it, start-up fails immediately (see *Behaviour without the platform* below). Deployment guides and API documentation: [AGENTIC STAR Developers](https://developers.fd.agenticstar.tm.softbank.jp/) |
| **AgentCore Framework** (`agenticstar-agentcore`) | Installed from PyPI as a dependency. |
| Python | >= 3.11 |

```bash
pip install -e .
```

### Behaviour without the platform

The framework is designed to run **only** on AGENTIC STAR. There is no fallback or degraded
mode. If the platform is unreachable or the SDK version does not match, the agent fails at
graph compile / start-up preflight rather than starting in a partially working state. This is
intentional — a half-running agent is worse than one that refuses to start.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests run without a platform connection. Running the agent itself does not.

## Project Structure

```
src/          agent implementation (nodes, services, schemas)
tests/        unit, integration and boundary tests
config/       agent configuration
docs/         design and test documentation
```

See `docs/02_design.md` for the architecture, the caller-data contract and the output
invariant, and `docs/03_test_spec.md` for the test plan.

## Customising

1. Adjust `config/config.yaml` — retrieval depth, relevance floor, retry and timeout.
2. Replace `config/kb/telecom_kb.json` with your own catalog, or pass a catalog slice
   per request in `input_context.catalog_documents` (see `docs/02_design.md`).
3. Review the node implementations under `src/nodes/`; `src/caller_contract.py` holds
   the accepted caller-data contract and `src/output_schema.py` the published answer
   shape that the output boundary enforces.
4. Re-run the test suite.

## License

MIT — see [LICENSE](LICENSE).

## Status of this repository

This template is published **as is**, by its individual author, under the MIT license. It carries
**no warranty and no support commitment**, and no organisation stands behind its behaviour or
fitness for any purpose. Issues and pull requests may or may not receive a response; that is at
the sole discretion of the repository owner.
