# Security Policy

## Supported code

Security fixes target the latest commit on `master` and the latest published release. Older tags may not receive backports.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Use the repository's **Security** tab and submit a private vulnerability report. Include:

- the affected commit or version;
- the affected component;
- reproduction steps or a minimal proof of concept;
- expected and observed behavior;
- potential impact;
- any suggested mitigation.

Remove personal images, documents, access tokens, device names, absolute user paths, and other sensitive data before submitting evidence.

Maintainers will acknowledge reports and coordinate validation, remediation, and disclosure on a best-effort basis. Please allow time for a fix before publishing details.

## Security boundaries

Smart Stack processes untrusted filenames, images, documents, metadata, and local HTTP requests. Security-sensitive changes include:

- upload validation and path handling;
- document parsing and archive expansion;
- localhost and Tailscale exposure;
- subprocess execution;
- SQLite and LanceDB persistence;
- application bundle launchers;
- dependency and workflow changes;
- logging of user content.

The project is local-first, but local execution does not make untrusted input safe. Contributions must preserve path containment, resource limits, content validation, and least-privilege workflow permissions.
