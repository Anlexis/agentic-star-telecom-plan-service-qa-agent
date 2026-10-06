# Answer Synthesis Prompt — TEL-C2-001

> **The bundled agent does not use this prompt at runtime.** `GenerateAnswerNode`
> assembles its answer deterministically from `ranked_documents`; no node reads
> this file. It records the contract a model-backed builder would work to, so
> that swap changes only the inside of `GenerateAnswerNode.execute()`.

## Contract for a model-backed answer node

- **Input:** the same `ranked_documents` JSON (id / title / category / source /
  score / excerpt) the bundled node reads.
- **Output:** the same state contract — `grounded_answer` (a string carrying
  numbered `[n]` citation markers) and `citations` (a JSON list of
  `{ref, id, title, source}`).
- **Grounding rule:** every factual statement must be traceable to one of the
  supplied passages via an `[n]` marker. A pricing, coverage or contract-term
  claim that is not in the passages must not be asserted — the agent reads the
  catalog, it does not extrapolate from it. The output boundary independently
  refuses any document whose markers do not resolve to a listed source.
- **No-coverage rule:** when no passage supports the question, say so and
  recommend rephrasing or contacting customer support — never answer plan
  pricing, coverage or contract terms from the model's own knowledge.
- **Tone:** neutral and catalog-accurate, with no account-specific guidance.
  The standing informational disclaimer is attached downstream by
  `OutputFormatNode`.
- **Caller text:** do not echo the caller's question into the answer body. The
  body is assembled from catalog passages only, and the output boundary treats
  a verbatim echo of the request as caller-controlled text reaching the
  external surface.

## Prompt template

```
You answer telecom plan and service questions strictly from the catalog
passages provided below.

Passages (each with a reference number):
{ranked_documents}

Rules:
1. Use ONLY the passages above. If they do not answer the question, say the
   catalog has insufficient coverage and stop.
2. Mark every factual statement with the [n] reference of its passage.
3. Do not state or imply account-specific pricing, balances, or promotions
   that are not present in the passages.
4. Keep the answer under 300 words.
```

## Configuration coupling

The `llm` block in `config/config.yaml` (`temperature`, `max_tokens`) is already
forwarded to the inner graph by `PlanServiceQAGraphNode._parent_config()` under
`config["configurable"]["llm"]`; a model-backed node reads it from there.
