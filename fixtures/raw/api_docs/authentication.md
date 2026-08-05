# Authentication

All requests must carry a bearer token. Tokens are issued per workspace and
inherit the permissions of the service account that created them.

## Obtaining a token

Exchange a client ID and secret at the token endpoint. The response contains an
access token valid for one hour and a refresh token valid for thirty days.

```bash
curl -X POST https://api.example.com/oauth/token \
  -d grant_type=client_credentials \
  -d client_id=$CLIENT_ID \
  -d client_secret=$CLIENT_SECRET
```

## Refreshing

Call the same endpoint with `grant_type=refresh_token`. Refresh rotates the
token: the previous refresh token is invalidated immediately, so a client that
retries a failed refresh with the old token receives `INVALID_GRANT`.

## Rate limits

Authentication endpoints are limited to 10 requests per minute per client ID.
Exceeding the limit returns HTTP 429 with a `Retry-After` header.
