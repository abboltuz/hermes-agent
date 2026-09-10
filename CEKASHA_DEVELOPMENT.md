# Cekasha Hermes Development

This private repository is the authoritative source for Cekasha's Hermes
customization and integration work. It is based on the upstream
[`NousResearch/hermes-agent`](https://github.com/NousResearch/hermes-agent)
project and preserves upstream history and attribution.

## Repository roles

- This repository owns Hermes core, Desktop, TUI, gateway, provider framework,
  and shared runtime changes.
- Standalone plugin development lives in the private
  [`abboltuz/hermes-plugins`](https://github.com/abboltuz/hermes-plugins)
  monorepository.
- Installed Hermes checkouts, profiles, credentials, session databases, and
  runtime state are not development sources and must never be committed here.

## Branch model

- `main` is the integration branch for the verified Cekasha build.
- Use short-lived branches with descriptive prefixes such as `fix/`, `feat/`,
  `refactor/`, or `docs/`.
- Push the first coherent commit and open a draft pull request early.
- Review and test an exact candidate commit before integration.
- Remove clean local worktrees and local branches after the corresponding pull
  request is merged or closed and the remote state is verified.

Historical local development from the initial GitHub migration is preserved
under `archive/` refs. Those refs are for recovery and audit, not new work.

## Upstream synchronization

The upstream project remains the source for general Hermes releases. Sync
upstream deliberately, review the resulting diff, and keep Cekasha-specific
behavior on the private integration branch. Never overwrite local
customizations with an unreviewed upstream update.

## Quality and safety

Repository-specific development, testing, review, release, and promotion rules
are defined in [`AGENTS.md`](AGENTS.md) and
[`HERMES_BRANCH_RULES.md`](HERMES_BRANCH_RULES.md). Secrets, profiles, OAuth
material, session exports, generated caches, installed runtime state, and
packaged credentials do not belong in Git.

