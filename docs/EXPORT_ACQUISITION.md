# Authenticated Bet Tracker export acquisition

This adapter downloads the original export bytes and calls the existing immutable
importer. Manual CSV imports, reconciliation policies, research defaults, and
notebooks are unchanged. Scheduling and evaluation are not part of this change.

## What is confirmed, and what is still unknown

| Question | Current evidence |
| --- | --- |
| Export location | The user supplied OddsJam's Bet Tracker export endpoint; the host/path are fixed in code. |
| Programmatic use | The user's October 22, 2026 support screenshot confirms that the endpoint can export programmatically. |
| Browser request method and parameters | User observed GET for the direct download on October 22, 2026. The separate UI action uses POST to export-csv-email, returns 202, and emails CSV parts. Direct-download parameter requirements remain unverified. |
| Authentication mechanism | User observed a Cookie header in browser requests. Cookie replay from Python has not succeeded; required cookie/header set remains unverified. Commercial odds API credentials are not assumed. |
| Authenticated live download | User confirmed the direct GET downloads in signed-in Chrome, including a repeat check. Python returned HTTP 403. No successful programmatic acquisition is established. |
| Coverage | The old February/March CSVs contain historical and pending records, but that does not establish the scope of a future HTTP request. |
| Source identifiers and timezone | Existing CSVs lack standalone bet IDs and exhibit unexplained timestamp shifts. Fresh-download characteristics remain unverified. |
| Settlement revisions | Need observations from distinct exports and reliable chronology; one HTTP success cannot establish revision completeness. |
| Unattended operation/session renewal | Not demonstrated. No automatic login, cookie extraction, credential refresh, or browser automation is implemented. |

The current implementation is tested with synthetic HTTP responses. A successful
`fetch` is evidence of accepted CSV bytes using the supplied request, not proof
of complete account coverage, correct account selection, executable bets, or a
renewable unattended session.

### Current live validation status

- Direct GET export succeeds in signed-in Chrome.
- Python acquisition receives HTTP 403 with an explicit Cloudflare
  browser-challenge marker.
- Programmatic acquisition and unattended authentication remain blocked.
- Technical guidance from OddsJam Support is the next step; sending the
  follow-up has not yet been confirmed.
- Weekly scheduling remains disabled.
- Manual CSV ingestion remains available.

## Local setup, step by step

Run from the repository root on your acquisition branch. No new dependencies are
needed; the adapter uses Python's standard HTTPS client with default certificate
verification. It does not read browser stores or environment proxy settings.

1. Create an unconfigured private template (this makes no network request):

   ```bash
   poetry run python scripts/acquire_bet_tracker.py init-profile
   ```

   This creates `.local/oddsjam/request.json` with mode `0600` in a private directory.
   `.local/` and HAR files are Git-ignored. Existing profiles are never overwritten.
   Do not force-add these files to Git or share them in chat/screenshots. A blank
   template may already have been created during implementation; skip this step
   if it exists.

2. In your browser, sign in to OddsJam and open Bet Tracker. Open Developer Tools:
   on Chrome for macOS, press **Option + Command + I**, then select **Network**.
   Clear the request list, select the broadest intended history with pending and
   settled bets included where controls exist, and click the CSV export button.
   Filter the request list for `export-csv` and select that request.

3. Inspect the request locally. Under **Headers**, locate **Request Method** and
   **Request Headers**. Under **Payload**, check whether query parameters or form
   fields are present. Do not copy the entire request as cURL or export a HAR into
   the repository. Do not send header values to the coding agent.

4. Open `.local/oddsjam/request.json` in your local editor and replace its blank
   fields with the observed request. This is a structural example with deliberately
   unusable placeholders:

   ```json
   {
     "method": "GET",
     "headers": {
       "Cookie": "REPLACE_LOCALLY_WITH_OBSERVED_VALUE"
     },
     "query": {}
   }
   ```

   Use GET only if you observed GET. Supply `Authorization` only if observed;
   include both authentication headers only when needed. Supported optional CSRF
   headers are `X-CSRFToken`, `X-CSRF-Token`, and `X-XSRF-TOKEN`. Keep values private.
   Use only required authentication/session cookies rather than unrelated browser
   cookies where the required set is known. Do not remove a required CSRF header.
   `query` accepts observed string keys and values; leave it empty when none were
   present. Never put a full URL or headers in the public YAML configuration.

   POST with optional string-valued `form` fields is supported when actually
   observed. POST is tried only once. JSON bodies, additional custom headers,
   redirect-based export flows, and other hosts are intentionally unsupported;
   establish the actual request contract before extending support. A required
   unsupported header/body is a reason to adapt the implementation, not to guess.

5. Save and validate the local configuration without contacting OddsJam:

   ```bash
   poetry run python scripts/acquire_bet_tracker.py check-profile
   ```

   The loader requires an owner-owned regular file with no group/other permissions
   and rejects symlinks. If needed, restore its permissions locally:

   ```bash
   chmod 600 .local/oddsjam/request.json
   ```

6. Leave `configs/data_sources.yaml` scope/timezone/export time as unknown unless
   independently confirmed. Choose a non-secret namespace unique to this account.
   Do not reuse a namespace for a different account or paper/real account source.
   Public HTTP limits live in `configs/export_acquisition.yaml`.

7. Fetch once and, optionally, compare with a browser download from the same period:

   ```bash
   poetry run python scripts/acquire_bet_tracker.py fetch
   ```

   ```bash
   poetry run python scripts/acquire_bet_tracker.py fetch --compare-csv data/raw/YOUR_BROWSER_EXPORT.csv
   ```

   The comparison reports byte equality, column order/names equality, and counts.
   A changing account can legitimately produce different exports minutes apart.
   Matching row counts or identical bytes do not independently prove full history.
   Comparison failures prevent import. Original browser files remain unchanged.

All global options precede the subcommand. To use alternate locations:

```bash
poetry run python scripts/acquire_bet_tracker.py --profile .local/oddsjam/request.json --config configs/export_acquisition.yaml --source-config configs/data_sources.yaml check-profile
```

Credentials cannot be passed directly on the command line. Local profile paths
may be supplied; credential values belong only in the private file. Do not commit
alternate private storage locations either. Restrict profile and store access to
trusted local users and keep them out of cloud-synced/shared folders as appropriate.

## Success and failure behavior

The adapter uses only the fixed OddsJam export origin and path. Private query and
form values exist only in memory during the request; they are never persisted in
receipts or exception messages. Redirects are not followed, so authentication is
never forwarded to a redirect target. Server filenames, cookies, response bodies,
remote error messages, and response headers are not copied to logs.

Before calling PR 1's importer, it requires:

- HTTP 200 (not 204, 206, a login redirect, or an error status).
- A supported CSV/text/binary content type and identity content encoding.
- Valid, unambiguous Content-Length or a complete chunked transfer.
- A bounded, complete response with a valid Bet Tracker CSV schema.
- Successful optional browser-file comparison parsing.

Downloads without verifiable HTTP framing fail closed. A CSV cut off within its
HTTP framing or with an incomplete chunk is rejected. A server can still send a
properly framed but semantically partial dataset: neither CSV validity nor HTTP
framing can prove that all bets were exported. Verify source coverage separately.

GET retries only explicit transient statuses (408, 429, 500, 502, 503, 504), socket
timeouts, and incomplete transfers, for at most four attempts. Defaults are three
attempts, a 30-second socket timeout/body-read deadline, and 128 MiB. Retry waits
are bounded at ten seconds. A Retry-After longer than ten seconds or a non-numeric
Retry-After fails with `retry_later` rather than ignoring the server's instruction.
POST is not retried. DNS/TLS/other transport errors fail with a redacted category.
Socket timeouts do not constitute a hard process-wide deadline for OS DNS lookup;
a later scheduler should also impose an overall job deadline.

Malformed responses, oversized downloads, expired authentication, and incomplete
transfers publish no successful import. Failure receipts record safe error codes. Rejected non-authentication HTTP
responses include their numeric status, such as `http_response_rejected_status_406`;
remote response text and headers remain private.
No receipt or command output contains credential values, request URLs, or raw HTTP
messages. A local I/O failure after the importer publishes can leave a valid import
without a completed acquisition receipt; inspect the private store and retry
idempotently instead of deleting history.

## Provenance, freshness, and reconciliation

Acquisition receipts are atomically published under the ignored store:

```text
data/raw/ingestion/acquisitions/<acquisition-id>.json
```

They record acquisition start/download/finish times, request method, transport
limits, byte hash/size, safe observed schema/status counts, optional browser
comparison, import linkage, and outcome. They never include headers, parameter
names/values, raw captures, credential fingerprints, server filenames, or URLs.
The request method is user-supplied profile configuration, not automatically
verified against a captured browser request.

- `imported`: a new content-addressed source was accepted.
- `already_imported`: existing bytes were reused; this is not evidence of new bets.
- `failed`: no successful acquisition is reported; see the safe error category.

Rows, pending/open counts, explicit-offset counts, and recognized identifier/update
column names are observed facts. Coverage, timezone verification, and settlement
revision completeness remain explicitly unverified in the receipt. Declared source
metadata is retained separately in PR 1's import manifest; HTTP success does not
upgrade assumptions into verified metadata.

Acquisition/download time is **not** asserted to be server export time, bet time,
or settlement time. `source.exported_at` remains unknown unless confirmed. As a
result, PR 1 may quarantine changed payloads without confirmed export chronology,
even after a successful live fetch. Acquisition receipts preserve observation
chronology for a future reviewed extension; this PR does not silently change
reconciliation precedence or the research timezone gate.

Prefer verified full-history exports for the eventual weekly pipeline because
newly created bets alone cannot reveal all revisions to older bets. Do not mark
scope `full_history` until request controls and returned data have been checked.
Old unexplained five/six-hour timestamp shifts remain unresolved.

## Credential expiry and unattended operation

There is no documented/tested renewal mechanism available in the supplied evidence.
The adapter does not log in, refresh tokens, or bypass MFA, expired sessions,
CAPTCHA, or an access denial. On authentication failure, renew the session using
the normal browser login and replace only the necessary values in the private
profile. Check with OddsJam Support for a supported long-lived or renewable
account export authentication mechanism before claiming weekly unattended access.
Do not assume a commercial odds-feed API key authorizes account history exports.

Before scheduling, demonstrate successful downloads across the intended weekly
interval and credential expiry/renewal behavior. Require source coverage,
chronology, and research-readiness checks as separate gates. A failed/stale export
must produce a status report rather than reevaluating old data as if fresh.

## Validation and review groups

Synthetic tests use in-memory HTTP responses, including real standard-library
HTTP response parsing, with no network calls, real credentials, or private history.
They cover request/profile validation, fixed origin, redaction, redirects/auth
failures, CSV schema, length/chunk truncation, size/time limits, GET retries,
POST no-retry, idempotency, receipt integrity, and browser comparison.

Suggested logical commits (none created automatically):

1. `feat: define secure Bet Tracker acquisition configuration`
   Configuration module, public YAML, ignore rules, and configuration tests.
2. `feat: acquire Bet Tracker exports through immutable ingestion`
   Download/receipt module, CLI, and happy-path integration tests.
3. `test: cover export failures retries and credential redaction`
   Remaining synthetic HTTP and CLI cases, alongside any tested safeguards.
4. `docs: explain export authentication and live validation limits`
   This guide and README links, with actual live evidence added only after a test.

Keep the tests for each capability with its implementation when preparing commits;
there is no requirement to create a separate tests-only commit. A squash merge is
optional after reviewing the individual changes.

## Investigating browser success with Python HTTP 403

The user confirmed browser downloads still succeed while the adapter returns 403.
This does not prove credentials expired or identify which layer rejected access.
The adapter reports only safe classifications, never response bodies or headers:

- `browser_challenge_status_403` (or another numeric status) means an explicit
  `cf-mitigated: challenge` marker was observed.
- `access_rejected_status_403_html_origin_unverified` means HTML accompanied 403;
  this alone does not identify Cloudflare or prove a browser challenge.
- `authentication_rejected_status_403` means neither of those markers was detected;
  it still does not prove a bad cookie or an application-origin rejection.

The challenge marker is documented by
[Cloudflare](https://developers.cloudflare.com/cloudflare-challenges/challenge-types/challenge-pages/detect-response/).
Detection does not solve or bypass a challenge. Do not repeatedly copy cookies or
add browser impersonation to evade a denial. If a supported direct request cannot
be established, ask OddsJam Support for required authentication/request headers,
renewal, and whether non-browser weekly exports require an access exception.
Keep this PR in draft until live acquisition is validated, or explicitly scope
review to a tested adapter with live integration still blocked.
