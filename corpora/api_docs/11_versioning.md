# Versioning

The API is versioned by date, not by a monotonic integer, so a version name
also tells you roughly when its behavior was frozen.

## Selecting a version

Pass the version explicitly with the `Lumen-Version` header. Omitting it
uses the account's default version, which is set the first time a key is
issued and never changes automatically.

```bash
curl https://api.lumen.dev/orders \
  -H "Authorization: Bearer $TOKEN" \
  -H "Lumen-Version: 2024-06-01"
```

## What counts as a breaking change

- Removing a field or endpoint
- Changing a field's type or meaning
- Adding a new required request field
- Changing default behavior for an existing parameter

Adding a new optional field, a new endpoint, or a new enum value is **not**
considered breaking and ships without a version bump — clients should
tolerate unknown fields and unknown enum values gracefully.

```python
status = order.get("status", "unknown")
if status not in KNOWN_STATUSES:
    status = "unknown"  # forward-compatible with future enum values
```

## Deprecation window

A version is supported for at least 18 months after a newer version ships.
Deprecated versions return a `Lumen-Deprecation-Warning` header for the
last 90 days of support so you can detect stragglers before the cutoff.

## Checking your current version

`GET /account` includes `api_version_default`, the version applied when no
`Lumen-Version` header is sent.
