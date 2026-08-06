# SDKs and Clients

Official SDKs handle retries, pagination iteration, and idempotency key
generation, so hand-rolled HTTP calls should be a last resort.

## Installing

```bash
pip install lumen-sdk
```

```python
from lumen import Lumen

client = Lumen(api_key="sk_live_...")
order = client.orders.create(customer_id="cus_1", amount=500)
```

## Retry behavior

The SDK retries `5xx` responses and connection errors up to 3 times with
exponential backoff, and automatically attaches a stable idempotency key
derived from the request body to any retried mutating call, so a caller
doesn't need to manage that manually. `4xx` responses are never retried.

## Pagination iteration

List methods return an iterator that walks cursors transparently:

```python
for order in client.orders.list(status="captured"):
    handle(order)
```

Iterating the whole result set makes one HTTP request per page under the
hood — the loop body does not need to know pagination exists.

## Available languages

Official SDKs exist for Python, Node.js, Ruby, and Go. Community-maintained
clients exist for PHP and Java but are not covered by the support SLA.

## Configuring timeouts

```python
client = Lumen(api_key="sk_live_...", timeout=10.0, max_retries=5)
```

Setting `timeout` too low on batch calls is a common source of spurious
retries, since a large batch can legitimately take several seconds.
