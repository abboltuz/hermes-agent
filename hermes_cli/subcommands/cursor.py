"""``hermes cursor`` account and SDK bridge commands."""

from __future__ import annotations

from typing import Callable


def build_cursor_parser(subparsers, *, cmd_cursor: Callable) -> None:
    parser = subparsers.add_parser(
        "cursor",
        help="Manage the Cursor subscription account and SDK bridge",
    )
    actions = parser.add_subparsers(dest="cursor_action")
    login = actions.add_parser("login", help="Sign in to Cursor and store a profile-scoped credential")
    login.add_argument("--no-browser", action="store_true", help="Print the login URL without opening a browser")
    login.add_argument("--api-key-name", default="", help="Name for the expiring Cursor user API key")
    login.add_argument("--api-key-ttl-days", type=int, default=90, help="Credential lifetime in days (default: 90)")
    logout = actions.add_parser("logout", aliases=["clear"], help="Remove the stored Cursor credential")
    actions.add_parser("status", help="Show whether a usable Cursor credential and bridge are available")
    install = actions.add_parser("install", help="Download and verify the pinned official SDK bridge")
    install.add_argument("--version", default="", help="Pinned bridge release version")
    parser.set_defaults(func=cmd_cursor)


def cmd_cursor(args) -> None:
    from agent.cursor_bridge_transport import download_bridge, resolve_bridge_command
    from agent.cursor_sdk_auth import (
        clear_sdk_credentials,
        login,
        read_sdk_credentials,
        resolve_cursor_api_key,
    )

    action = getattr(args, "cursor_action", None) or "status"
    if action == "login":
        ttl_days = int(getattr(args, "api_key_ttl_days", 90))
        if ttl_days <= 0 or ttl_days > 365:
            raise SystemExit("--api-key-ttl-days must be between 1 and 365")
        result = login(
            on_login_url=lambda url: print("Open this Cursor login URL in your browser:\n" + url),
            open_browser=not getattr(args, "no_browser", False),
            api_key_name=getattr(args, "api_key_name", ""),
            api_key_ttl_ms=ttl_days * 24 * 60 * 60 * 1000,
        )
        print("Cursor login complete; credential stored securely for this Hermes profile.")
        if result.get("email"):
            print(f"Account: {result['email']}")
        return
    if action in {"logout", "clear"}:
        removed = clear_sdk_credentials()
        _, remaining_source = resolve_cursor_api_key()
        if removed:
            print("Cursor credential cleared.")
        if remaining_source == "env":
            raise SystemExit(
                "CURSOR_API_KEY remains configured in this profile's .env or process environment."
            )
        if not removed:
            print("No Cursor credential was stored.")
        return
    if action == "install":
        path = download_bridge(getattr(args, "version", ""))
        print(f"Cursor SDK bridge installed at {path}")
        return
    credential = read_sdk_credentials()
    api_key, source = resolve_cursor_api_key()
    print("Cursor account: configured" if api_key else "Cursor account: not configured")
    if api_key:
        print(f"Credential source: {source}")
    if credential and credential.get("email"):
        print(f"Account: {credential['email']}")
    print("Cursor SDK bridge: available" if resolve_bridge_command() else "Cursor SDK bridge: not installed")
