"""Corpus pipeline: any data source to a validated, versioned, access-controlled index.

    spec (config/corpora/*.toml) -> source -> access policy -> document gates
      -> chunks -> chunk gates -> new build collection -> readback gates
      -> build record (status in Qdrant) -> alias promotion / rollback

Benchmarks are corpora like any other; see docs/adr/0007-corpus-pipeline.md.
"""
