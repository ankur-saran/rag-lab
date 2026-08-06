# Search Endpoint

`GET /search` runs a single query across orders and customers using a small
query language, rather than requiring separate filtered list calls.

## Query syntax

```
GET /search?q=status:captured AND amount:>5000
```

Supported operators:

- `field:value` — exact match
- `field:>value` / `field:<value` — numeric or date comparison
- `AND` / `OR` — combine clauses; `AND` binds tighter than `OR`
- `"quoted text"` — free-text match against `name` and `email` fields

## Searchable fields

- Orders: `status`, `amount`, `currency`, `created_at`
- Customers: `email`, `name`, `created_at`

Cross-resource queries aren't supported in a single call — search orders and
customers separately and join client-side if you need both.

## Example

```python
results = client.search.query(
    'status:captured AND created_at:>2024-01-01 AND "jordan"'
)
for r in results.items:
    print(r.resource_type, r.id)
```

## Result ranking

Free-text clauses are ranked by relevance; exact-match clauses are not
ranked and simply filter. When a query mixes both, exact-match clauses apply
first and relevance ranking is computed only within the filtered set.

## Limits

Search indexes update within 30 seconds of a write, so a record created
moments ago may not yet appear. Queries longer than 200 characters return
`VALIDATION_FAILED`.
