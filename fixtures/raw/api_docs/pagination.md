# Pagination

The API returns large collections in pages. Every list endpoint accepts the same
pagination parameters and returns the same envelope, so a client written against
one endpoint works against all of them.

## Cursor parameters

Pass `limit` to control page size and `cursor` to request a specific page. The
default limit is 25 and the maximum is 100. Requests above the maximum are
clamped rather than rejected.

```python
resp = client.orders.list(limit=50)
while resp.next_cursor:
    process(resp.items)
    resp = client.orders.list(limit=50, cursor=resp.next_cursor)
```

## Response envelope

Every paginated response carries `items`, `next_cursor`, and `has_more`. When
`has_more` is false, `next_cursor` is null and further requests return an empty
page rather than an error.

## Stability guarantees

Cursors are opaque and encode a snapshot of the sort order at the time the first
page was requested. Records inserted after that moment do not appear in the
remainder of the walk. Cursors expire after 24 hours; using an expired cursor
returns HTTP 410 with error code `CURSOR_EXPIRED`.
