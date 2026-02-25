# Security Policy

## Supported Versions

Security fixes are currently provided for the latest `main` branch and the latest release tag.

## Reporting a Vulnerability

Do not open public issues for security vulnerabilities.

Use the private reporting channel:

- Preferred: [GitHub Private Vulnerability Reporting](https://github.com/Laelaps43/mochiagent/security/advisories/new)
- Fallback: contact maintainer directly at [@Laelaps43](https://github.com/Laelaps43)

Please include:

- Clear description of the issue
- Reproduction steps or PoC
- Affected versions/files
- Impact assessment
- Suggested mitigation (if available)

## Response SLA

- Acknowledgement: within 48 hours
- Initial triage: within 7 calendar days
- Status updates: at least every 7 calendar days until resolved

## Scope

Priority areas include:

- Tool execution boundaries (`exec`, workspace restrictions, command guard)
- Data leakage and secret exposure paths
- Authentication/authorization gaps (if introduced by integrations)
- Dependency vulnerabilities
