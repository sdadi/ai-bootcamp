# Nimbu Mart Information Security Policy (2026)

Document ID: NM-SEC-POL · Version 2026.1 · Effective 1 February 2026 · Owner: Head of Information Security
Applies to: all Nimbu Mart employees, contractors and third-party tools that store or process Nimbu Mart data.

This policy replaces the 2025 edition. The main change is in SEC-4.2: a SOC 2 **Type I** report is no longer
accepted on its own for tools that handle customer data.

## 1. Purpose and scope

### SEC-1.1 Purpose
This policy sets the minimum security requirements for systems and third-party vendors that handle Nimbu Mart
information. "Third-party vendor" includes any SaaS or AI tool, whether paid or free, used by an employee for work.

### SEC-1.2 Data tiers
Nimbu Mart classifies data into three tiers. Vendor requirements depend on the highest tier a tool will touch.

| Tier | Name | Examples |
|---|---|---|
| T1 | Public | Published catalogue pages, press releases, public help articles |
| T2 | Internal | Internal wikis, roadmaps, non-personal analytics, source code |
| T3 | Customer data | Customer names, addresses, phone numbers, order history, support tickets, payment references |

A tool that reads support tickets or chats is a **T3 (customer data)** tool, even if "only summaries" are produced.

## 4. Vendor security assurance

### SEC-4.1 Security review before use
No tool may store or process T2 or T3 data before the Information Security team has completed a vendor security
review and Procurement has recorded the outcome (see the Vendor Onboarding Checklist, PROC-1.1).

### SEC-4.2 Independent security assurance
A vendor that stores or processes **T3 (customer) data** must hold at least one of the following, current and
independently issued:

- a **SOC 2 Type II** report covering the service in question, dated within the last **15 months**, or
- an **ISO/IEC 27001** certificate covering the service in question, valid on the date of review.

A marketing statement such as "SOC 2 compliant" is **not** evidence. The reviewer must see the report, the
certificate, or a trust-centre entry that links to it. A **SOC 2 Type I** report, or a certification that is
"in progress", does not satisfy this clause for T3 tools. For T2 tools, Type I or a completed security questionnaire
is acceptable with Information Security approval.

### SEC-4.3 Security questionnaire
Where SEC-4.2 cannot be verified from public material, the vendor must complete the Nimbu Mart security
questionnaire (template NM-SEC-Q). Unanswered items are treated as "unknown", not as "pass".

## 5. Protecting data

### SEC-5.1 Encryption
Vendors must encrypt Nimbu Mart data in transit (TLS 1.2 or higher) and at rest (AES-256 or equivalent).

### SEC-5.2 Access by vendor staff
Vendor staff access to Nimbu Mart data must be limited to support cases we open, and must be logged.

## 7. Identity and access

### SEC-7.1 Single sign-on
Any tool that handles **T3 (customer) data**, or is used by **more than 20** employees, must support single sign-on
(SAML 2.0 or OIDC) with Nimbu Mart's identity provider, so that leavers lose access on their last day. If SSO is
offered only on a higher-priced plan, the review must state the plan required and Procurement must confirm the
budget before the tool is approved.

### SEC-7.2 Role-based access
Tools must support at least two roles (administrator and member) so that configuration changes are restricted.

## 9. Incidents

### SEC-9.1 Reporting
Employees must report any suspected security incident involving a vendor to security@nimbumart.example within
24 hours.

### SEC-9.3 Vendor incident history
A vendor with a **major security incident in the last 24 months** (unauthorised access to customer data, credential
exposure, or data shown to the wrong customer) may be approved only if the vendor has published a post-incident
report (post-mortem) describing the root cause and the fixes. A major incident **without** a published post-mortem
is disqualifying for T3 use until the vendor provides one. Minor incidents and availability outages are noted but
are not disqualifying.

### SEC-9.4 Rumours are not incidents
Forum posts, social media and unnamed sources are not evidence of an incident. Reviewers record them as
"unverified" and may ask the vendor directly.

## 12. Review

### SEC-12.1 Annual re-review
Every approved vendor is re-reviewed at least every 12 months, or sooner if the vendor reports an incident, changes
ownership, or changes where it stores data.
