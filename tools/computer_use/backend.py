"""Abstract backend interface for computer use.

Any implementation (cua-driver over MCP, pyautogui, noop, future Linux/Windows)
must return the shape described below. All methods synchronous; async is
handled inside the backend implementation if needed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


class ComputerUseCapabilityError(RuntimeError):
    """A requested optional backend capability is unavailable.

    Optional hooks use a structured code so the model gets a fail-closed,
    actionable response instead of a generic backend exception.  Keeping the
    hooks non-abstract preserves compatibility with external backends that
    predate the capability.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.operation: Optional[str] = None
        self.next_step: Optional[str] = None


@dataclass
class UIElement:
    """One interactable element on the current screen."""

    index: int                       # 1-based SOM index
    role: str                        # AX role (AXButton, AXTextField, ...)
    label: str = ""                  # AXTitle / AXDescription / AXValue snippet
    bounds: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h (logical px)
    app: str = ""                    # owning bundle ID or app name
    pid: int = 0                     # owning process PID
    window_id: int = 0               # SkyLight / CG window ID
    attributes: Dict[str, Any] = field(default_factory=dict)
    # Opaque per-snapshot element handle from cua-driver
    # (trycua/cua#1961 — Surface 6 of NousResearch/hermes-agent#47072).
    # When set, downstream calls can pass it alongside `index` for
    # explicit stale-detection: a stale token returns an error from
    # cua-driver rather than silently re-resolving to a different
    # element. None for pre-#1961 drivers that didn't carry the field.
    element_token: Optional[str] = None

    def center(self) -> Tuple[int, int]:
        x, y, w, h = self.bounds
        return x + w // 2, y + h // 2


@dataclass
class CaptureResult:
    """Result of a screen capture call.

    At least one of png_b64 / elements is populated depending on capture mode:
      * mode="vision" → png_b64 only
      * mode="ax"     → elements only
      * mode="som"    → both (default): a plain PNG plus a separately
                         numbered `elements` list. The PNG is not annotated.
    """

    mode: str
    width: int                      # screenshot width (logical px, pre-Anthropic-scale)
    height: int
    png_b64: Optional[str] = None
    elements: List[UIElement] = field(default_factory=list)
    # Optional: the target app/window the elements were captured for.
    app: str = ""
    window_title: str = ""
    # Raw bytes we sent to Anthropic, for token estimation.
    png_bytes_len: int = 0
    # Explicit MIME type for `png_b64` when the backend supplied it
    # (cua-driver-rs emits `mimeType` on every image part as of
    # trycua/cua#1961 — Surface 7 of NousResearch/hermes-agent#47072).
    # When None, downstream consumers fall back to base64-prefix
    # sniffing for back-compat with older drivers.
    image_mime_type: Optional[str] = None
    # Optional guidance appended to the human-readable summary — used by
    # capture lanes that intentionally return no elements (e.g. full-screen
    # composited grabs) to tell the model how to reach an interactive lane.
    note: str = ""
    # Exact native identity resolved for this capture. Non-window backends
    # leave these unset.
    pid: Optional[int] = None
    window_id: Optional[int] = None
    # A failed target resolution carries a machine-readable code and safe
    # candidate metadata without arming a sticky mutation target.
    error: Optional[str] = None
    available_windows: List[Dict[str, Any]] = field(default_factory=list)
    # Full count reported by cua-driver before any source-side max_elements
    # truncation. Kept last to preserve the positional constructor ABI.
    # None means the older driver did not publish a count.
    total_elements: Optional[int] = None


@dataclass
class ActionResult:
    """Result of any action (click / type / scroll / drag / key / wait).

    Beyond the transport-level ``ok`` flag, this carries cua-driver's
    structured action verdict so the model can follow the documented
    verify → escalate ladder (NousResearch/hermes-agent#67052). ``ok`` stays
    tool/transport success only — it is NOT the semantic verdict. Read
    ``effect`` / ``escalation`` to decide the next rung. All structured
    fields are optional and additive: an older driver that omits
    ``structuredContent`` leaves them ``None`` and behavior is unchanged.
    """

    ok: bool
    action: str
    message: str = ""                # human-readable summary
    # Optional trailing screenshot — set when the caller asked for a
    # post-action capture or the backend always returns one.
    capture: Optional[CaptureResult] = None
    # Arbitrary extra fields for debugging / telemetry.
    meta: Dict[str, Any] = field(default_factory=dict)
    # ── cua-driver structured verdict (additive; None on old drivers) ──
    # AX read-back verification: True = driver read the effect back,
    # False = ran but unconfirmed, None = tool doesn't carry the field.
    verified: Optional[bool] = None
    # Confidence signal: "confirmed" | "unverifiable" | "suspected_noop".
    effect: Optional[str] = None
    # Machine-readable next-rung hint. Current drivers use
    # {"target": "px"|"foreground", "reason_code": str}; the legacy
    # {"recommended": ..., "reason": ...} shape remains accepted.
    escalation: Optional[Dict[str, Any]] = None
    # Delivery rung that actually ran (e.g. "ax", "x11_pixel", "cgevent_fg").
    path: Optional[str] = None
    # Current cua-driver vocabulary. ``path`` remains as a legacy alias.
    route: Optional[str] = None
    delivery: Optional[Dict[str, Any]] = None
    evidence: Optional[Dict[str, Any]] = None
    # True when an AX walk found no actionable elements (act by px instead).
    degraded: Optional[bool] = None
    # The delivery_mode the caller requested for this action, echoed back.
    delivery_mode: Optional[str] = None
    # A structured refusal code (e.g. "background_unavailable",
    # "foreground_unsupported", "desktop_scope_disabled") when present.
    code: Optional[str] = None


class ComputerUseBackend(ABC):
    """Lifecycle: `start()` before first use, `stop()` at shutdown."""

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the backend can be used on this host right now.

        Used by check_fn gating and by the post-setup wizard.
        """

    # ── Capture ─────────────────────────────────────────────────────
    @abstractmethod
    def capture(
        self,
        mode: str = "som",
        app: Optional[str] = None,
        pid: Optional[int] = None,
        window_id: Optional[int] = None,
        max_elements: Optional[int] = None,
    ) -> CaptureResult: ...

    # ── Pointer actions ─────────────────────────────────────────────
    @abstractmethod
    def click(
        self,
        *,
        element: Optional[int] = None,
        x: Optional[int] = None,
        y: Optional[int] = None,
        button: str = "left",           # left | right | middle
        click_count: int = 1,
        modifiers: Optional[List[str]] = None,
        delivery_mode: Optional[str] = None,   # background (default) | foreground
        bring_to_front: bool = False,
    ) -> ActionResult: ...

    @abstractmethod
    def drag(
        self,
        *,
        from_element: Optional[int] = None,
        to_element: Optional[int] = None,
        from_xy: Optional[Tuple[int, int]] = None,
        to_xy: Optional[Tuple[int, int]] = None,
        button: str = "left",
        modifiers: Optional[List[str]] = None,
        delivery_mode: Optional[str] = None,
        bring_to_front: bool = False,
    ) -> ActionResult: ...

    @abstractmethod
    def scroll(
        self,
        *,
        direction: str,                 # up | down | left | right
        amount: int = 3,                # wheel ticks
        element: Optional[int] = None,
        x: Optional[int] = None,
        y: Optional[int] = None,
        modifiers: Optional[List[str]] = None,
        delivery_mode: Optional[str] = None,
        bring_to_front: bool = False,
    ) -> ActionResult: ...

    # ── Keyboard ────────────────────────────────────────────────────
    @abstractmethod
    def type_text(self, text: str, *, element: Optional[int] = None,
                  x: Optional[int] = None, y: Optional[int] = None,
                  delivery_mode: Optional[str] = None,
                  bring_to_front: bool = False) -> ActionResult: ...

    @abstractmethod
    def key(self, keys: str, *, element: Optional[int] = None,
            x: Optional[int] = None, y: Optional[int] = None,
            delivery_mode: Optional[str] = None,
            bring_to_front: bool = False) -> ActionResult:
        """Send a key combo, e.g. 'cmd+s', 'ctrl+alt+t', 'return'."""

    # ── Introspection ───────────────────────────────────────────────
    @abstractmethod
    def list_apps(self) -> List[Dict[str, Any]]:
        """Return running apps with bundle IDs, PIDs, window counts."""

    def list_windows(self) -> List[Dict[str, Any]]:
        """Return visible native windows with PID and window identifiers.

        Optional compatibility hook: backends that predate window discovery
        remain instantiable and simply report no windows.
        """
        return []

    def list_windows_exact(
        self,
        *,
        pid: Optional[int] = None,
        on_screen_only: bool = True,
    ) -> Dict[str, Any]:
        """Return an exact window inventory when the backend supports it.

        The compatibility fallback is intentionally limited to the historical
        visible-window behavior.  It is labelled as non-exact and refuses an
        off-screen request rather than pretending a legacy backend can see
        windows on other Spaces/desktops.
        """
        if not on_screen_only:
            raise ComputerUseCapabilityError(
                "exact_window_listing_unsupported",
                "This computer-use backend cannot enumerate off-screen windows.",
            )
        windows = self.list_windows()
        if pid is not None:
            windows = [window for window in windows if window.get("pid") == pid]
        return {
            "windows": windows,
            "count": len(windows),
            "on_screen_only": True,
            "exact_metadata": False,
            "compatibility_fallback": True,
        }

    def verify_state(
        self,
        *,
        pid: int,
        window_id: int,
        expect: List[Dict[str, Any]],
        timeout_ms: Optional[int] = None,
        stable_samples: Optional[int] = None,
        include_screenshot: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Evaluate bounded predicates against one exact native window."""
        raise ComputerUseCapabilityError(
            "verify_state_unsupported",
            "This computer-use backend does not support deterministic state verification.",
        )

    def launch_app(
        self,
        *,
        bundle_id: Optional[str] = None,
        name: Optional[str] = None,
        creates_new_application_instance: bool = False,
    ) -> Dict[str, Any]:
        """Launch an app without selecting or foregrounding its windows."""
        raise ComputerUseCapabilityError(
            "launch_app_unsupported",
            "This computer-use backend does not support application launch.",
        )

    def set_window_frame(
        self,
        *,
        pid: int,
        window_id: int,
        frame: Dict[str, Any],
    ) -> ActionResult:
        """Set exact desktop window geometry without changing focus."""
        raise ComputerUseCapabilityError(
            "set_window_frame_unsupported",
            "This computer-use backend does not support window geometry changes.",
        )

    @abstractmethod
    def focus_app(self, app: str, raise_window: bool = False) -> ActionResult:
        """Route input to `app` (by name or bundle ID). Default: focus without raise."""

    # ── Native-value mutation ────────────────────────────────────────
    @abstractmethod
    def set_value(self, value: str, element: Optional[int] = None) -> ActionResult:
        """Set a native value on an element (e.g. AXPopUpButton selection).

        `element` is the 1-based SOM index returned by a prior capture call.
        """

    # ── Timing ──────────────────────────────────────────────────────
    def wait(self, seconds: float) -> ActionResult:
        """Default implementation: time.sleep."""
        import time
        time.sleep(max(0.0, min(seconds, 30.0)))
        return ActionResult(ok=True, action="wait", message=f"waited {seconds:.2f}s")
