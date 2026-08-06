# Orders Resource

An order represents a single purchase attempt and moves through a fixed set
of states from creation to settlement.

## Creating an order

```python
order = client.orders.create(
    customer_id="cus_1a2b3c",
    line_items=[{"sku": "widget-blue", "quantity": 2}],
    currency="usd",
)
```

## Fields

- `id` — `ord_` prefixed identifier
- `status` — one of `pending`, `authorized`, `captured`, `refunded`, `failed`
- `customer_id` — the owning customer
- `line_items` — array of `{sku, quantity, unit_amount}`
- `amount` — total in the smallest currency unit (cents for USD)
- `created_at` / `updated_at` — ISO 8601 timestamps

## State transitions

`pending` orders move to `authorized` once payment authorization succeeds,
then to `captured` when funds are actually collected. Only `captured` orders
can be `refunded`, partially or in full. An order that fails authorization
moves directly to `failed` and cannot be resumed — create a new order
instead.

```bash
curl -X POST https://api.lumen.dev/orders/ord_1a2b3c/refund \
  -H "Authorization: Bearer $TOKEN" \
  -d amount=1000
```

## Listing and filtering

`GET /orders` supports `status`, `customer_id`, and `created_after` /
`created_before` filters, combinable in a single request. Combining
`status=refunded` with a date range is the common query for reconciliation.
