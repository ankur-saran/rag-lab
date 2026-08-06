# Idempotency

Any `POST` that creates or mutates a resource accepts an `Idempotency-Key`
header so a retried request after a network timeout doesn't create a
duplicate.

## Using a key

```bash
curl -X POST https://api.lumen.dev/orders \
  -H "Authorization: Bearer $TOKEN" \
  -H "Idempotency-Key: create-order-8f2a" \
  -d customer_id=cus_1a2b \
  -d amount=4200
```

The first request with a given key executes normally and its response is
cached. Any subsequent request with the same key, within 24 hours, returns
the cached response without re-executing — even if the request body differs
in ways that don't affect the outcome, such as header order.

## Conflicting bodies

If a repeated key arrives with a body that differs in a field that *would*
change the outcome — a different `amount`, for example — the API returns
`IDEMPOTENCY_KEY_CONFLICT` rather than silently picking one interpretation.

## Choosing keys

- Use a UUID or a hash of the logical operation, not a timestamp — two
  retries of the same logical create should produce the same key.
- Keys are scoped per workspace, so two different customers can reuse the
  same key value without colliding.
- Keys older than 24 hours are evicted; reusing an evicted key starts a new
  idempotency window.

## Endpoints that require a key

Only mutating endpoints accept `Idempotency-Key`. `GET` requests are
naturally idempotent and the header is ignored if sent.
