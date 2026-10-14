ATMResolve AI — ML upgrade overlay

Merge this ZIP into your existing project folder (the folder containing api/, app/, src/, scripts/, data/, requirements.txt). It intentionally adds scripts/modules and updates src/ml/predict.py while leaving the existing classifier artifact in place.

Read docs/INDIA_ML_PLAYBOOK.md first. Recommended first commands in the activated .venv:

  python scripts/download_indian_bank_complaints.py
  python scripts/inspect_indian_bank_complaints.py
  python -m src.ml.benchmark_banking77

The first two commands download and audit the supplementary 6.96k-row Indian-style dataset. The benchmark command compares word-only TF-IDF with combined word+character TF-IDF using a validation split from BANKING77's training data, calibrates the selected model using only training data, evaluates once on BANKING77's official test split, and saves a new artifact. Your existing intent_classifier.joblib is not overwritten.

For actual India-specific ATM classification, follow docs/INDIA_ML_PLAYBOOK.md to create data/atm_intents_india.csv with manually reviewed labels and explicit train/validation/test splits, then run:

  python -m src.ml.train_atm_india

Do not expect a truly India-specific model merely by downloading an India-style dataset. The public supplementary dataset uses 12 broad complaint labels and its provenance is limited; it does not supply a clean set of fine-grained ATM dispute labels. Never relabel rows to categories unsupported by their text.
