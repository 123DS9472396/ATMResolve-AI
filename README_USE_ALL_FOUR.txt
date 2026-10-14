ATMResolve AI — Four-resource integration overlay

This is an overlay package, not a replacement project. Merge its folders into the existing folder containing api/, app/, src/, scripts/, data/, and requirements.txt.

Files added:
- scripts/prepare_atm_charge_review.py: creates an Excel-friendly queue for human review; it does not auto-label examples.
- src/ml/train_indian_complaint_router.py: trains a separate broad 12-label Indian-style grievance model; does not overwrite the current classifier.
- data/documents/RBI_FAILED_ATM_FAQ.md: source-linked paraphrased policy summary for RAG.
- docs/USE_ALL_FOUR_RESOURCES.md: exact workflow, commands, constraints and IndicBankBench setup.

After merging and activating .venv, start with:
  python scripts/prepare_atm_charge_review.py
  python -m src.ml.train_indian_complaint_router

Then, when Ollama and nomic-embed-text are available:
  python -m src.rag.ingest

Follow docs/USE_ALL_FOUR_RESOURCES.md for the separate IndicBankBench clone and subsequent fine-grained ATM model. The active app classifier is not switched automatically.
