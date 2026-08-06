# File Uploads

Files — receipts, dispute evidence, customer-uploaded documents — are
uploaded separately from the resource they attach to, then referenced by id.

## Uploading a file

```bash
curl -X POST https://api.lumen.dev/files \
  -H "Authorization: Bearer $TOKEN" \
  -F purpose=dispute_evidence \
  -F file=@receipt.pdf
```

The response includes a `file_id` (`file_` prefixed) to reference elsewhere,
for example when submitting dispute evidence:

```python
client.disputes.submit_evidence(
    dispute_id="dp_1a2b",
    receipt_file_id=file.id,
)
```

## Supported formats and limits

- Formats: PDF, PNG, JPEG
- Maximum size: 8 MB per file
- Files are retained for 2 years, then automatically deleted

## Purpose values

The `purpose` field determines which downstream flows can reference the
file and affects retention:

- `dispute_evidence`
- `identity_document`
- `customer_upload`

A file uploaded with the wrong `purpose` cannot be reassigned — upload it
again with the correct value.

## Downloading a file

`GET /files/{id}/content` streams the raw bytes with the original
`Content-Type`. This endpoint accepts the same bearer token as everything
else; there is no separate signed-URL mechanism.
