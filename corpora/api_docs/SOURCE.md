# SOURCE

Synthesized for rag-lab. Not scraped or copied from any real company's docs —
entirely authored for this project as reference documentation for a fictional
product, "Lumen API."

**Stresses:** Fenced code blocks (curl/Python/JSON) and consistent markdown
heading structure, to differentiate structure-aware chunking (which can keep a
code sample intact) from fixed-size splitting (which can bisect one).

**Scale:** 15 documents. `doc_id = sha1(corpus + relative_path)[:12]`, so this
corpus can grow later without breaking any downstream artifact.
