# Consolidation notes

This folder consolidates the two uploaded project archives:

- `atmresolve-ai.zip`: the existing repository skeleton, ML evaluation helper, citation helper, and initial Streamlit page files.
- `ATMResolve_AI_Starter(1).zip`: the more complete implementation for SQLite persistence, synthetic transaction/ATM records, dataset download, local Ollama configuration, Chroma RAG ingestion/retrieval, LangGraph human review, FastAPI routes, and Streamlit workflow.

## Merge choices

- The more implemented starter code replaces stub implementations at matching paths.
- `src/ml/evaluate.py` and `src/rag/citations.py` from the existing repository were retained and adapted to the consolidated code.
- The existing `app/pages/` navigation was retained and its placeholder pages were made functional.
- The old unused `src/tools/atm_events.py` placeholder and minimal dataclass model were removed because the consolidated SQLite repository and `src/tools/atm_status.py` now provide the actual implementation.
- Existing ML tests were updated for the consolidated train/test-split training API.
- Python bytecode and pytest caches were excluded. No locally generated model, dataset, database, or vector-index artifacts are included. Generate these locally using the README commands.

## Important limits

The bundle includes application code, tests and synthetic examples; it does not contain BANKING77 CSV files, downloaded Ollama models, a trained classifier artifact, or the Chroma index. Follow `README.md` to generate them. The full integration requires installing `requirements.txt`, downloading the models, and running the end-to-end smoke test locally.
