# Shared Antigravity provider

Antigravity is an installation-scoped provider. All agents and profiles resolve
the managed bridge from `get_default_hermes_root()`, including custom installations
using `<root>/profiles/<name>`. Profile-local configuration, memory and all other
providers retain their existing scopes.

Each managed Python client starts its own lightweight bridge proxy with an
ephemeral token. Only the child environment receives the installation root and
the internal shared-service launch field and exact Python interpreter path for
the POSIX kernel-lock helper; neither `os.environ` nor the active
profile context is changed. Auth and catalog probes that explicitly pass the
managed executable follow the same route as inference clients. Custom explicit
commands retain the private contract. Managed launches require the
`antigravity-openai-shared-v1` handshake, failing closed with an update-required
error when an old bridge is installed.

The bridge repository owns the shared owner lifecycle and account state. RPC
adapters retain profile-scoped client lifetimes, but those clients see the same
installation account pool. Closing a client does not stop the shared owner.

Desktop Accounts settings render Google Antigravity in the regular provider list.
Its row expands inline to connect accounts, edit priorities, enable/disable and
remove accounts. Closing either disclosure or refreshing connection status
preserves mounted account controls and an active OAuth flow. Account
emails are displayed; opaque IDs remain mutation keys. Missing legacy emails use
a numbered account label instead of a random identifier. The settings text makes
the installation-wide scope explicit in every supported locale.

## Promotion gate

This is a paired core/bridge change. Main integration, installation, app packaging
and restart require separate authorization. Before promotion, verify that old
private bridge clients are drained and inspect account counts in profile-local
stores. The root/default pool is retained in place; profile-only accounts are not
silently imported or discarded. If any exist, stop and agree a reconnection plan.
Never copy account files or secrets into source repositories.

No chat histories, compaction rules, summaries, context limits, model prompts or
other providers are changed by this work.
