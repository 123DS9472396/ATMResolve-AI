---
title: Synthetic Demo Bank ATM Complaint Procedure
source_url: local://SYNTHETIC_demo_bank_manual.md
source_type: synthetic_demo_policy
effective_date: 2026-10-13
---
# Synthetic Demo Bank ATM Complaint Procedure

> IMPORTANT: This is fictional demonstration content written for ATMResolve AI. It is not a real bank policy and must not be represented as official banking policy.

## Evidence required for a cash-not-dispensed complaint

For a complaint where a customer reports a debit but no cash was received, an investigator should check the transaction reference, transaction amount, timestamp, debit status, ATM dispense event and reversal status. In this prototype all of these records are synthetic.

## Handling missing or contradictory evidence

If the transaction reference does not exist, do not infer that a debit occurred. Mark the case as needing more information and send it for human review.

If ATM event records are missing, contradictory, or show both successful cash dispensing and a dispense failure for the same transaction, do not automatically resolve the case. Mark the evidence as incomplete or conflicting and send it for human review.

## Human approval

Every proposed resolution in this demo requires a human reviewer. The prototype must not initiate real refunds, change a real customer balance, contact a real customer or access a banking system.

## Recommendation format

A case recommendation should state: (1) verified transaction facts, (2) retrieved policy evidence, (3) missing or conflicting information, (4) a proposed next step, (5) source citations, and (6) whether human review is required.
