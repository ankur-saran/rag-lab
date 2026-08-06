# Batch Operations

The batch endpoint executes up to 100 sub-requests in a single call, useful
for bulk imports where per-request overhead would otherwise dominate.

## Request shape

```json
{
  "requests": [
    {"method": "POST", "path": "/orders", "body": {"customer_id": "cus_1", "amount": 500}},
    {"method": "POST", "path": "/orders", "body": {"customer_id": "cus_2", "amount": 1200}}
  ]
}
```

## Response shape

Each sub-request gets its own result at the matching index, including its
own status code — a batch call itself returns `200` even if individual
items failed:

```json
{
  "results": [
    {"status": 201, "body": {"id": "ord_a1"}},
    {"status": 422, "body": {"error": {"code": "VALIDATION_FAILED"}}}
  ]
}
```

## Ordering and atomicity

Sub-requests execute in array order but are **not** transactional — a
failure partway through does not roll back earlier successful items.
Design batches so each item is independently retryable, and pass an
`Idempotency-Key` per item for safe retries of a partially-failed batch.

## Rate limit accounting

A batch call counts once against the batch rate limit regardless of how
many sub-requests it contains, but each sub-request still counts against
its own endpoint family's limit for quota-tracking purposes shown in the
dashboard.

## Size limits

Batches are capped at 100 items and 5 MB of total request body. Larger
imports should be chunked client-side into multiple batch calls.
