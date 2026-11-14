# Nimbu Mart AI Tools Acceptable Use Policy

Document ID: NM-AI-POL · Version 2026.2 · Effective 1 June 2026 · Owner: Head of Information Security with the DPO
Applies to: every employee and every AI tool (assistants, copilots, summarisers, translators, agents) used for work.

## 1. Principles

### AI-1.1 Work data stays under contract
Employees may use only AI tools that Information Security has reviewed and Procurement has recorded as approved for
the relevant data tier (see SEC-1.2). Personal accounts on consumer AI tools must never receive T2 or T3 data.

### AI-1.2 Humans decide
AI tool output that affects a customer (refunds, account actions, communications) must be reviewed by a person
before it is acted upon.

## 2. Training on our data

### AI-2.1 No training on Nimbu Mart data
A vendor may **not use Nimbu Mart data to train or improve its models**. Acceptable evidence is a published policy
or contract term stating that customer content is not used for training. Where a vendor trains by default but offers
an opt-out, the tool may be approved only on a plan where training is **contractually disabled**, and the review must
record which plan and how the opt-out is enforced. "You can email us to opt out" is not sufficient for T3 data.

### AI-2.2 Prompts are data
Prompts, uploaded files and generated outputs are Nimbu Mart data and carry the tier of the most sensitive content
in them.

## 3. Sub-processors and model providers

### AI-3.4 Model providers are sub-processors
When an AI tool sends our content to a third-party model provider, that provider is a sub-processor (DP-4.1) and
AI-2.1 applies to it as well. The review must identify every model provider in the vendor's sub-processor list and
confirm the vendor's contract with that provider forbids training on customer content.

## 4. Prohibited inputs

### AI-4.1 Never enter into any AI tool
Payment card numbers, government identity numbers, passwords, API keys, and unpublished financial results.

## 5. Approval tiers

### AI-5.1 Who approves
| Highest data tier | Reviewer | Approver |
|---|---|---|
| T1 Public | Team lead | Team lead |
| T2 Internal | Information Security | Head of Information Security |
| T3 Customer data | Information Security + DPO | Head of Information Security **and** DPO |

An automated assessment (including the vendor risk assistant) is an input to the review. It does not approve a tool.
