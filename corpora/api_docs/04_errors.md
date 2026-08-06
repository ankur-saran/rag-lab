# Errors

Every error response is JSON with a stable shape, regardless of status code.

```json
{
  "error": {
    "code": "RESOURCE_NOT_FOUND",
    "message": "No order with id ord_9f2a1b found in this workspace.",
    "request_id": "req_7c3d9e0a"
  }
}
```

Always log `request_id` — support uses it to trace a request through
internal services and cannot help without it.

## Status code ranges

- `4xx` — the request was malformed, unauthorized, or referenced something
  that doesn't exist. Retrying without changing the request will not help,
  except for `429`.
- `5xx` — something failed on our side. Safe to retry with backoff.

## Common error codes

- `INVALID_GRANT` — token refresh used a rotated or expired refresh token
- `RESOURCE_NOT_FOUND` — the id in the path does not exist in this workspace
- `VALIDATION_FAILED` — one or more fields in the request body were invalid;
  see `error.details` for a field-by-field breakdown
- `CURSOR_EXPIRED` — a pagination cursor older than 24 hours was used
- `IDEMPOTENCY_KEY_CONFLICT` — the same idempotency key was reused with a
  different request body

## Validation error detail

`VALIDATION_FAILED` errors include a `details` array:

```json
{
  "error": {
    "code": "VALIDATION_FAILED",
    "message": "2 fields failed validation.",
    "details": [
      {"field": "email", "reason": "not a valid email address"},
      {"field": "quantity", "reason": "must be a positive integer"}
    ]
  }
}
```
