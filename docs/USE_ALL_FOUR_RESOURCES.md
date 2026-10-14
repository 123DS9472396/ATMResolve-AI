# ATMResolve AI: how to use all four resources correctly

This plan uses all four resources in the application, but **does not merge them into one training CSV**, because they serve different purposes.

## Resource roles

1. **PolyAI/BANKING77** — general English banking intent-classification baseline. The current improved word+character TF-IDF model scored 90.58% accuracy and 90.54% macro F1 on BANKING77's official test split. These metrics are not Indian ATM customer metrics.
2. **KunalEsM/bank_complaint_classifier** — a supplementary broad Indian-style bank grievance dataset. The supplied labels contain 12 broad complaint categories, including an ATM AMC-charge category. Train a separate broad complaint router with `python -m src.ml.train_indian_complaint_router`; do not make it replace the BANKING77 model or the future fine-grained ATM classifier. The text appears heavily templated, so treat its held-out scores as source-distribution results only.
3. **NPCI/IndicBankBench** — a separate agent-safety/reliability benchmark, not supervised intent-training data. It has 799 synthetic multi-turn cases and evaluates tool use, grounding, clarification/refusal, privacy and workflow reliability. Download and run its official harness separately. Results require a configured OpenAI-compatible candidate endpoint with tool-calling support and a separate judge endpoint; a full 799-case, multi-pass run can be slow on CPU.
4. **RBI failed-ATM FAQ** — a policy source for RAG, not an ML label dataset. A paraphrased, source-linked summary is provided at `data/documents/RBI_FAILED_ATM_FAQ.md`. Verify current official wording before operational or customer-specific use.

## Step A — create a review queue from the 792 ATM-labelled complaint rows

In the existing ATMResolve project virtual environment:

```powershell
python scripts/prepare_atm_charge_review.py
```

Outputs:

- `data/atm_charge_review_queue.csv`
- `data/atm_charge_review_summary.json`

Open the queue CSV in Excel. Suggested subtypes and keyword flags are **review hints only**. Rows are not labelled as training-ready automatically. Mark `review_status` and `approved_training_label` only after checking the actual complaint text. Pay attention to card renewal/replacement charges, withdrawal fees, generic debits, and failed-transaction wording: these are not interchangeable.

## Step B — train a separate broad grievance router from the supplementary dataset

```powershell
python -m src.ml.train_indian_complaint_router
```

Outputs:

- `artifacts/ml/indian_complaint_router.joblib`
- `artifacts/ml/indian_complaint_router_metrics.json`
- `artifacts/ml/indian_complaint_router_classification_report.json`
- `artifacts/ml/indian_complaint_router_confusion_matrix.csv`

The script excludes normalized exact-text groups with conflicting labels, de-duplicates exact text, tunes `C` on a validation split and evaluates on a separate test split. Near-duplicate template leakage may remain. Do not change `.env` to make this broad 12-label model the active classifier; the classes differ from BANKING77 and from the planned fine-grained ATM label set.

## Step C — add the RBI source to local RAG

The project ingests `.md` and `.txt` files from `data/documents`. Ensure Ollama is running and the configured embedding model (default `nomic-embed-text`) is available, then run:

```powershell
python -m src.rag.ingest
```

This should index the RBI summary with source metadata alongside the existing demo documents. After indexing, use Policy Search in the app and verify that the RBI source URL is displayed. The file is a paraphrase, not an official verbatim publication.

## Step D — download and run IndicBankBench separately

Use a separate clone and virtual environment so its benchmark dependencies do not alter this project's environment. In a new PowerShell window:

```powershell
cd $HOME\Desktop
git clone https://github.com/npci/IndicBankBench.git
cd .\IndicBankBench
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
Copy-Item .env.example .env
hf download NPCI/IndicBankBench --repo-type dataset --local-dir .\data
$env:INDICBANKBENCH_DATA = "$PWD\data"
python -m harness.cli run --help
```

Read `HOW_TO_RUN.md` in that repo and configure both `CANDIDATE_*` and `JUDGE_*` settings in its `.env`. Both endpoints must be OpenAI-compatible; the candidate must support tool calls. Confirm endpoint model IDs and tool calling before running. Start with a small domain/case subset using the official `--cases` option and one pass; only run the full benchmark after the endpoint works. On this machine, a small local model on CPU can be slow and a weak judge can make scores hard to interpret. Do not report a full IndicBankBench score until the official harness has actually completed.

## Step E — later build the fine-grained Indian ATM classifier

Only after reviewing examples, create `data/atm_intents_india.csv` using the existing `data/atm_intents_india_template.csv` and `src/ml/train_atm_india.py`. The India ATM dataset must contain correctly labelled examples for categories such as `cash_not_dispensed_debited`, `partial_cash_dispensed`, `duplicate_atm_debit`, `reversal_not_received`, `unauthorized_atm_withdrawal`, `card_retained`, `atm_fee_dispute`, `declined_no_debit`, `card_pin_or_atm_access_issue`, and `other_or_uncertain`. Keep independent, human-labelled test examples, ideally from actual consented/appropriately de-identified complaint data if available. Synthetic examples can bootstrap a demo but cannot establish real-customer accuracy.

## Recommended architecture

- BANKING77 model: baseline / English generic banking intent benchmark.
- Indian complaint router: separate, optional broad grievance experiment.
- Fine-grained ATM classifier: dedicated intent prediction after reviewed local examples exist.
- RBI FAQ: cited retrieval context for the agent.
- IndicBankBench: end-to-end mocked tool-use and safety evaluation.

Never use a classifier probability alone to authorize a refund or other financial action. Keep deterministic evidence checks and human approval.
