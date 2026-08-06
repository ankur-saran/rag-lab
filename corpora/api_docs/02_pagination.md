# Pagination

The API returns large collections in pages. Every list endpoint accepts the
same pagination parameters and returns the same envelope, so a client written
against one endpoint works against all of them.

## Cursor parameters

- `limit` — page size, default 25, maximum 100. Requests above the maximum
  are clamped rather than rejected.
- `cursor` — opaque token identifying the page to fetch. Omit it to fetch the
  first page.

```python
resp = client.orders.list(limit=50)
while resp.next_cursor:
    process(resp.items)
    resp = client.orders.list(limit=50, cursor=resp.next_cursor)
```

## Response envelope

Every paginated response carries:

- `items` — the page of records
- `next_cursor` — cursor for the next page, or `null` if this is the last page
- `has_more` — boolean mirror of whether `next_cursor` is set

When `has_more` is false, further requests with a stale cursor return an
empty page rather than an error.

## Stability guarantees

Cursors are opaque and encode a snapshot of the sort order at the time the
first page was requested. Records inserted after that moment do not appear
in the remainder of the walk. Cursors expire after 24 hours; using an
expired cursor returns HTTP 410 with error code `CURSOR_EXPIRED`.

## Sorting

Pass `sort` to change the field pagination walks over — `created_at` is the
default. Reversing `sort` mid-walk invalidates the cursor, since the two
sort orders don't share a stable position.
