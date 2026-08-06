# Authentication

Every request to the Lumen API must carry a bearer token in the `Authorization`
header. Tokens are issued per workspace and inherit the permissions of the
service account that created them.

## Obtaining a token

Exchange a client ID and secret at the token endpoint. The response contains
an access token valid for one hour and a refresh token valid for thirty days.

```bash
curl -X POST https://api.lumen.dev/oauth/token \
  -d grant_type=client_credentials \
  -d client_id=$CLIENT_ID \
  -d client_secret=$CLIENT_SECRET
```

The response body:

- `access_token` — opaque bearer token, valid for one hour
- `refresh_token` — opaque token, valid for thirty days
- `expires_in` — seconds until `access_token` expires
- `token_type` — always `Bearer`

## Refreshing a token

Call the same endpoint with `grant_type=refresh_token`. Refresh rotates the
token: the previous refresh token is invalidated immediately, so a client
that retries a failed refresh with the old token receives `INVALID_GRANT`.

```python
resp = client.post("/oauth/token", data={
    "grant_type": "refresh_token",
    "refresh_token": stored_refresh_token,
})
```

## Scopes

Tokens carry one or more scopes that gate which endpoints they can call:

- `orders:read` / `orders:write`
- `customers:read` / `customers:write`
- `webhooks:manage`
- `admin` — implies every other scope

## Rate limits on auth endpoints

Authentication endpoints are limited to 10 requests per minute per client ID.
Exceeding the limit returns HTTP 429 with a `Retry-After` header.
