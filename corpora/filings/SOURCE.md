# SOURCE

Synthesized for rag-lab. Not scraped or copied from any real company's SEC
filing — entirely authored for this project as 10-K-style excerpts for a
fictional company, "Meridian Robotics, Inc." (fictional ticker: MRDN, fiscal
year 2023). All financial figures are invented and not derived from any real
filer.

**Stresses:** Financial tables in GFM pipe-table syntax, to demonstrate
table-summary indexing (Phase 7) against strategies that would otherwise
split a table's header row from its body.

**Scale:** 14 documents, 11 containing at least one table and 3 narrative-only
for realism. `doc_id = sha1(corpus + relative_path)[:12]`, so this corpus can
grow later without breaking any downstream artifact.
