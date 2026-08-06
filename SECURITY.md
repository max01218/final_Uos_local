# Security and Privacy

## Scope

This repository is a research prototype for mental-health-support conversations. It is not a medical device and must not be used as a substitute for professional diagnosis, treatment, emergency services, or crisis care.

## Secrets

- Never commit `.env`, API keys, access tokens, passwords, private model endpoints, or service-account credentials.
- Copy `.env.example` to `.env` and fill in values only on your local machine or deployment platform.
- If a secret is ever committed, deleting the file is not sufficient: revoke or rotate the credential, then remove it from Git history where appropriate.

## Personal and health data

- Do not commit real chat histories, session databases, identifiable user data, or unredacted clinical material.
- Use synthetic fixtures for demos and tests.
- Define retention, access control, encryption, deletion, and consent requirements before any real-user evaluation.

## Model safety

- Treat prompt filters and crisis classifiers as fallible safeguards, not guarantees.
- Test false positives, false negatives, prompt injection, unsafe advice, and out-of-distribution inputs.
- Keep a clear escalation path for crisis situations and communicate system limitations to users.

## Reporting

If you discover an exposed credential or privacy issue, do not publish the sensitive value in a public issue. Revoke the credential first, preserve only the minimum evidence needed, and contact the repository owner privately.
