# Sandbox Environment

The sandbox is a fully isolated copy of the API for integration testing —
separate data, separate rate limits, no real charges.

## Sandbox keys

Sandbox keys are prefixed `sk_test_` instead of `sk_live_`. Requests to
`https://api.lumen.dev` are routed to sandbox or production automatically
based on which prefix the key has; there is no separate sandbox hostname.

```python
client = Lumen(api_key="sk_test_51a2b3c")
```

## Triggering specific outcomes

Sandbox orders behave based on magic values in the request, so you can test
failure handling deterministically:

- `amount=100000` — always fails authorization with `card_declined`
- `amount=100001` — succeeds authorization, fails on capture
- Any other amount — succeeds normally

```bash
curl -X POST https://api.lumen.dev/orders \
  -H "Authorization: Bearer $TOKEN" \
  -d customer_id=cus_test_1 \
  -d amount=100000
```

## Data isolation

Sandbox and production data never intersect — a customer id created in
sandbox is meaningless in production and vice versa. Webhooks fire
separately per environment; register a sandbox endpoint explicitly if you
want to test webhook handling before going live.

## Resetting sandbox data

`POST /sandbox/reset` deletes all orders, customers, and webhook
subscriptions in the sandbox for the calling workspace. This endpoint does
not exist in production and returns 404 there.
