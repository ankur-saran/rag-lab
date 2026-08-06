# Webhooks

Webhooks push order and customer events to a URL you control, so you don't
have to poll.

## Registering an endpoint

```bash
curl -X POST https://api.lumen.dev/webhooks \
  -H "Authorization: Bearer $TOKEN" \
  -d url=https://example.com/hooks/lumen \
  -d events=order.created,order.refunded
```

An endpoint can subscribe to any subset of:

- `order.created`
- `order.updated`
- `order.refunded`
- `customer.created`
- `customer.deleted`

## Verifying signatures

Every delivery includes a `Lumen-Signature` header — an HMAC-SHA256 of the
raw request body, keyed with your webhook signing secret. Verify it before
trusting the payload.

```python
import hmac, hashlib

def verify(payload: bytes, signature: str, secret: str) -> bool:
    expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)
```

## Retries

A delivery that doesn't get a `2xx` response within 5 seconds is retried
with exponential backoff: 1 minute, 5 minutes, 30 minutes, then hourly for
24 hours. After 24 hours of failed deliveries, the endpoint is disabled and
an email is sent to the workspace owner.

## Replaying events

Past events for an endpoint can be replayed from the dashboard for up to 30
days after original delivery. Replays are marked with a `replayed: true`
field in the payload so consumers can distinguish them from live events.
