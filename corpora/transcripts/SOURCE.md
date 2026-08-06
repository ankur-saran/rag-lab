# SOURCE

Synthesized for rag-lab. Not scraped or copied from any real call or meeting
— entirely authored for this project as earnings-call and internal-meeting
style transcripts for a fictional company, "Solstice Analytics, Inc."

**Stresses:** Semantic chunking (Phase 7). Deliberately has **no markdown
headings anywhere** — plain speaker-turn text with several genuine topic
shifts per document, so a topic-boundary detector has real signal to find and
a structure-aware chunker has nothing to key off of. If this ever changes,
Phase 7's semantic-chunker demo has nothing to show.

**Scale:** 13 documents. `doc_id = sha1(corpus + relative_path)[:12]`, so this
corpus can grow later without breaking any downstream artifact.
