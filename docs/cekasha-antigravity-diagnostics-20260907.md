# Antigravity HTTP diagnostics

The Python bridge transport previously discarded every HTTP error body, so a
Google account-verification refusal and an unrelated permission refusal both
became `bridge HTTP 403`. Both JSON and streaming request setup now preserve
the bridge's public reason code and correlation ID, without forwarding raw
Google text, credentials, account identifiers, or chat content.

## Wire boundary

The bridge returns `error.code` from a finite allowlist and an optional UUID in
`error.diagnostic_id` and/or `x-hermes-antigravity-request-id`. Python constructs
its own fixed explanation. It ignores `message`, `type`, unknown fields and
unknown codes. Conflicting valid body/header IDs discard both diagnostics.
Malformed or old bridge responses retain the original HTTP status. A valid
header ID can still correlate a response whose body cannot be parsed.

Diagnostic reads are capped at 8 KiB and one second, further bounded by the
caller's remaining timeout. They use bounded single-read socket operations,
not an unbounded response drain. Bodies require an exact, bounded
Content-Length; chunked bodies and unsupported response wrappers are not
read. They retain status/header correlation only. The paired bridge frames
its small diagnostic JSON with Content-Length.

`agent.antigravity_bridge_transport` writes `antigravity_http_error` plus JSON
containing `status`, `code`, and `diagnostic_id` through the existing logger.
Use that ID to correlate the final core failure with the plugin's private
per-attempt diagnostics. A generic HTTP 403 without a specific reason is not
evidence that an API key is invalid. Terminal guidance directs Antigravity
users to connected Google account settings rather than API-key setup.

Verification-required rows remain valid whether the account is automatically
disabled or explicitly re-enabled for a retry. Other enabled/status consistency
checks, aggregate checks and current-account checks remain enforced. Desktop
retains the optional finite account status and shows the verification reason
ahead of the generic enabled/disabled label. Merely toggling enable does not
hide the reason; a refreshed successful backend status does. Older snapshots
without status keep their existing enabled/disabled presentation.

No account migration, credentials, context handling, card lifecycle, fallback
configuration, runtime installation, packaging, or restart is changed here.
