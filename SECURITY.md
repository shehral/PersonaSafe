# Security Policy

## Reporting a Vulnerability

If you believe you have found a security issue, please email **ali.shehral@gmail.com** with:
- A description of the issue and potential impact
- Steps to reproduce
- Any relevant logs or proof of concept (do not include secrets)

We will investigate promptly and work with you on remediation. Please do not open public issues for security-sensitive reports.

## Supported Versions

This project is under active development. We aim to address security issues on the `main` branch and the latest tagged release.

## Handling Secrets

- Never commit API keys, tokens, or credentials
- Use local `.env` (not committed) or environment variables
- `.env.example` demonstrates expected variables without secrets
