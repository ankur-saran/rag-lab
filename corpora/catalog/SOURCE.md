# SOURCE

Synthesized for rag-lab. Not scraped or copied from any real product listing
— entirely authored for this project as short, multilingual product
description cards for a fictional general-merchandise catalog.

**Stresses:** Embedding model choice. Bodies are deliberately short
(150-400 characters) and span six languages (English, Spanish, French,
German, Japanese, Russian), so a multilingual embedder and an English-only
embedder should visibly diverge on this corpus specifically. Two English
entries (`desk_lamp`, `water_bottle`) deliberately omit an H1 heading, to
exercise the loader's filename-based title fallback.

**Scale:** 16 documents. `doc_id = sha1(corpus + relative_path)[:12]`, so this
corpus can grow later without breaking any downstream artifact.
