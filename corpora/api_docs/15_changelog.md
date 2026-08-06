# Changelog

Notable changes to the API, newest first. Only breaking changes trigger a
new dated version; everything else applies immediately to all versions.

## 2024-06-01

- Added `metadata` field to customers (up to 20 keys)
- Added the search endpoint (`GET /search`)
- **Breaking:** `orders.status` value `pending_review` renamed to
  `authorized` to match the rest of the lifecycle terminology

```diff
- if order["status"] == "pending_review":
+ if order["status"] == "authorized":
```

## 2024-02-15

- Added batch operations endpoint
- Added `Idempotency-Key` support to all mutating endpoints
- Sandbox magic-value amounts documented publicly for the first time

## 2023-11-01

- Added webhook replay from the dashboard
- Increased default read rate limit from 300 to 600 requests per minute

## 2023-08-20

- **Breaking:** removed the deprecated `/v1/charges` endpoint in favor of
  `/orders`; the alias had returned a deprecation header since 2023-02-01
- Added file uploads endpoint

## 2023-02-01

- Initial public release of `/orders`, `/customers`, and OAuth
  client-credentials authentication
