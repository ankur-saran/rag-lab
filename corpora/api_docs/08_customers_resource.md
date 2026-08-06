# Customers Resource

A customer is the identity orders and payment methods attach to.

## Creating a customer

```python
customer = client.customers.create(
    email="jordan@example.com",
    name="Jordan Rivera",
    metadata={"external_id": "crm-9081"},
)
```

## Fields

- `id` — `cus_` prefixed identifier
- `email` — must be unique within a workspace
- `name` — display name, optional
- `metadata` — arbitrary key-value pairs, up to 20 keys, values under 500
  characters each
- `default_payment_method` — id of the payment method used when none is
  specified on an order

## Merging duplicate customers

When the same person is accidentally created twice, `POST
/customers/{id}/merge` reassigns all orders, payment methods, and webhook
history from a source customer onto a target customer, then deletes the
source. This operation cannot be undone.

```bash
curl -X POST https://api.lumen.dev/customers/cus_source/merge \
  -H "Authorization: Bearer $TOKEN" \
  -d target_id=cus_target
```

## Deleting a customer

Deleting a customer does not delete their historical orders — it detaches
`customer_id` from them and sets it to `null`, preserving records for
accounting purposes. To fully anonymize a customer for a data-deletion
request, also strip `email`, `name`, and `metadata` before deleting.
