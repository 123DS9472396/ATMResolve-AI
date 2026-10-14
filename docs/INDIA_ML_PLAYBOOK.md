# ATMResolve AI — India-focused ML plan

## What each dataset can and cannot do

### 1. BANKING77 (already downloaded and trained)

- 13,083 English online-banking utterances, 77 intent labels, official train/test split.
- Good general banking-intent baseline, but it is not a dataset of Indian ATM disputes.
- We can benchmark a word-only TF-IDF model against word + character TF-IDF and calibrate the selected model without using the official test split to choose the model.
- Source: <https://huggingface.co/datasets/PolyAI/banking77>

### 2. KunalEsM/bank_complaint_classifier (supplementary)

- Hugging Face dataset viewer reports about 6.96k rows and 12 labels, with rupee-denominated complaint text and some ATM-related charge examples.
- Do not assume these are authentic, anonymized bank customer messages. The visible examples appear templated, and the dataset card has limited provenance detail.
- The labels are broad complaint/grievance classes. Examples labelled as “charges deducted” can mention ATM fees, annual fees, SMS charges, and unrelated charges. Therefore, do not relabel every row as `atm_fee_dispute`, and do not turn that class into `cash_not_dispensed_debited`.
- Use it to audit Indian-style language and learn its actual label distribution. Use for a separate broad complaint-classification task only if its taxonomy fits your product.
- Source: <https://huggingface.co/datasets/KunalEsM/bank_complaint_classifier>

### 3. Fine-grained ATM intent dataset (the important missing piece)

The application needs a separately labelled dataset for the specific ATM complaint intents it must resolve. Create `data/atm_intents_india.csv` with columns:

- `text`: one complaint per row; remove account numbers, phone numbers, names, card numbers and other personal data.
- `label`: one label from the controlled taxonomy below.
- `split`: `train`, `validation`, or `test`.
- `source`: `human_labeled`, `synthetic`, or a permitted public-source identifier.

Keep near-duplicate paraphrases in the same split. Best practice is to collect the held-out test examples separately, rather than generating near-identical train/test rows. Do not publish private complaints or use customer data without permission and appropriate de-identification.

Recommended initial taxonomy:

| Label | Meaning / annotation rule |
|---|---|
| `cash_not_dispensed_debited` | Account debit is stated; ATM dispensed no cash. |
| `partial_cash_dispensed` | ATM dispensed less cash than requested. |
| `duplicate_atm_debit` | Same ATM withdrawal appears charged more than once. |
| `reversal_not_received` | Complaint is specifically that reversal/refund for a failed ATM transaction has not arrived. |
| `unauthorized_atm_withdrawal` | Customer says they did not make/authorize the ATM withdrawal. |
| `card_retained` | ATM retained/swallowed the card. |
| `declined_no_debit` | Withdrawal was declined and customer states no debit occurred. |
| `atm_fee_dispute` | ATM/card maintenance or withdrawal fee is disputed; not a cash-dispense failure. |
| `card_pin_or_atm_access_issue` | Card/PIN/technical access problem prevents withdrawal. |
| `other_or_uncertain` | Ambiguous, out-of-scope, or insufficient information; ask for clarification. |

If a complaint contains multiple issues, define and apply a consistent primary-intent rule. Do not mix transaction-outcome labels with RBI grievance-category labels.

## How many examples?

For a portfolio prototype, start with 50–100 reviewed training examples per label, plus at least 15–25 separate validation examples and 25–50 separate test examples per label if feasible. More varied, correctly labelled examples are more valuable than hundreds of near-duplicate templates. Record results separately for:

- Indian English (e.g. “amount was debited but cash did not come out”)
- Hinglish / Roman Hindi (e.g. “ATM se cash nahi nikla lekin paise kat gaye”)
- Hindi/Devanagari, Marathi and other supported languages, if you intend to claim support for them
- Short typo-heavy messages and longer narratives

These are targets, not a claim that the bundled project already has this many labelled examples. The blank template intentionally contains no invented training rows.

## Algorithms

### Baseline: word TF-IDF + Logistic Regression

Fast, explainable enough for a demo, CPU-friendly, and emits probability estimates. The current project measured 88.80% accuracy and 88.85% macro F1 on BANKING77's official test split. Those are general English banking metrics, not India-specific ATM metrics.

### Candidate improvement: word + character TF-IDF + Logistic Regression

Word n-grams capture phrases; character n-grams can help with misspellings, spelling variation and some transliteration patterns. This is worth benchmarking on the same split. Do not assume it will win—use the validation set and report the measured result.

### Confidence calibration

Raw `predict_proba` values are not automatically well-calibrated. The provided benchmark selects features using a validation split from BANKING77 training data, then fits sigmoid calibration using training data, and only then reports the official test metrics. Calibration on BANKING77 does not imply reliable confidence on Indian-English/Hinglish messages. The India-specific trainer only enables calibration when it has enough labelled examples per class; even then, real-world calibration must be assessed.

### Recommended model for a CPU-only local demo

Start with the word + character TF-IDF + Logistic Regression approach. Do not fine-tune a large transformer first. Only compare multilingual sentence embeddings or a multilingual transformer after building a representative Indian-language evaluation set.

## Commands

From the root folder `atmresolve-ai` in the activated virtual environment:

1. Download and audit supplementary Indian-style complaints:

   ```powershell
   python scripts/download_indian_bank_complaints.py
   python scripts/inspect_indian_bank_complaints.py
   ```

2. Benchmark BANKING77. This leaves the existing `intent_classifier.joblib` untouched and creates a separate calibrated artifact:

   ```powershell
   python -m src.ml.benchmark_banking77
   ```

   Outputs are under `artifacts/ml/benchmark_v2_*.json` and `benchmark_v2_confusion_matrix.csv`; model is `artifacts/ml/intent_classifier_v2_calibrated.joblib`.

3. Build the fine-grained ATM dataset. Copy `data/atm_intents_india_template.csv` to `data/atm_intents_india.csv`; add only human-reviewed, correctly labelled examples. Keep the explicit split column and use no exact duplicate text across splits. The trainer refuses cross-split text leakage and requires each class to appear in all three splits.

4. Train the India-specific ATM classifier:

   ```powershell
   python -m src.ml.train_atm_india
   ```

   This saves `artifacts/ml/atm_india_classifier.joblib` and independent validation/test metrics. Do not run this until the custom dataset is sufficiently populated and labels are reviewed.

5. Smoke-test a trained custom model without changing your application's default model:

   ```powershell
   python -c "from src.ml.predict import predict_intent; print(predict_intent('ATM se cash nahi nikla lekin account se paise kat gaye', 'artifacts/ml/atm_india_classifier.joblib'))"
   ```

6. Only when the India-specific model has been evaluated, set this line in `.env` to make the app use it:

   ```ini
   ML_MODEL_PATH=artifacts/ml/atm_india_classifier.joblib
   ```

   Restart FastAPI/Streamlit after changing `.env`.

## How to judge the model

Track accuracy and macro F1, but inspect per-class precision/recall, the confusion matrix, and a hand-reviewed error set. Evaluate thresholds on validation data. If maximum class probability or the margin between the top two classes is below a threshold chosen on validation data, route to clarification/review. Never make refunds or regulatory eligibility decisions from classifier confidence; use deterministic transaction evidence, current policy sources, and human approval.

Report all datasets distinctly:

1. BANKING77 official test performance — external general English baseline.
2. Indian-style complaint dataset results, if you train a broad category classifier — broad grievance taxonomy with limited provenance.
3. India-focused ATM held-out test performance — only meaningful if examples are separately labelled and representative of target customers.

Never combine these scores into one headline accuracy.
