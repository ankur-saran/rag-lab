# Rate Limiting

Requests are limited per workspace, not per token, so rotating tokens does
not raise your ceiling.

## Default limits

- Read endpoints (`GET`): 600 requests per minute
- Write endpoints (`POST`/`PATCH`/`DELETE`): 120 requests per minute
- Batch endpoints: 10 requests per minute, each counted once regardless of
  batch size

## Reading limit headers

Every response carries three headers describing the caller's current state:

```
X-RateLimit-Limit: 600
X-RateLimit-Remaining: 583
X-RateLimit-Reset: 1712345678
```

`X-RateLimit-Reset` is a Unix timestamp for when the window rolls over, not a
duration.

## Handling 429s

When the limit is exceeded, the API returns HTTP 429 with a `Retry-After`
header giving the number of seconds to wait.

```python
import time

resp = client.orders.list()
if resp.status_code == 429:
    time.sleep(int(resp.headers["Retry-After"]))
    resp = client.orders.list()
```

## Requesting a higher limit

Workspaces on the Growth plan or above can request a limit increase from
the dashboard. Increases apply per-endpoint-family, not globally — raising
the write limit does not raise the read limit.
