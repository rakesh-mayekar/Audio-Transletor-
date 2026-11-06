"""Overlay window behaviour per OS, expressed through Qt window flags.

Capabilities below are DOCUMENTED Qt/OS behaviour, not measured on real Windows/Linux machines
(only the Qt-free logic is unit-tested). `apply()` reports what it could not do rather than
pretending - e.g. Wayland compositors may ignore always-on-top and absolute positioning.
"""

from __future__ import annotations

import os
from typing import Any

from core.config.models import OverlayConfig
from oslayer.base.overlay_provider import OverlayApplyResult, OverlayCapabilities, OverlayProvider
from oslayer.selector import OSName, detect_os


def is_wayland(environ: dict[str, str] | None = None) -> bool:
    env = environ if environ is not None else dict(os.environ)
    return env.get("XDG_SESSION_TYPE", "").lower() == "wayland" or bool(env.get("WAYLAND_DISPLAY"))


class QtOverlayProvider(OverlayProvider):
    def __init__(self, os_name: OSName | None = None, wayland: bool | None = None) -> None:
        self.os = os_name or detect_os()
        self.wayland = is_wayland() if wayland is None else wayland

    def capabilities(self) -> OverlayCapabilities:
        if self.os is OSName.LINUX and self.wayland:
            return OverlayCapabilities(
                transparent=True,
                always_on_top=False,
                click_through=True,
                positionable=False,
                over_fullscreen_apps=False,
                notes=(
                    "Wayland compositors decide stacking and placement; the overlay may sit "
                    "behind other windows and cannot be placed at absolute coordinates.",
                    "For a reliable overlay, log in to an X11 session.",
                ),
            )
        if self.os is OSName.MACOS:
            return OverlayCapabilities(
                True,
                True,
                True,
                True,
                False,
                (
                    "Above normal windows; not above native full-screen Spaces (needs NSWindow "
                    "collection behaviour, not implemented).",
                ),
            )
        if self.os is OSName.WINDOWS:
            return OverlayCapabilities(
                True,
                True,
                True,
                True,
                False,
                ("Above windowed and borderless apps; exclusive-fullscreen games may hide it.",),
            )
        return OverlayCapabilities(
            True,
            True,
            True,
            True,
            False,
            ("X11: depends on the window manager honouring the always-on-top hint.",),
        )

    def apply(self, window: Any, config: OverlayConfig) -> OverlayApplyResult:
        from PySide6.QtCore import Qt  # lazy: this module imports without Qt installed

        caps = self.capabilities()
        applied: set[str] = set()
        unsupported: set[str] = set()
        msgs: list[str] = list(caps.notes)
        flags = Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool
        if config.always_on_top:
            flags |= Qt.WindowType.WindowStaysOnTopHint
            (applied if caps.always_on_top else unsupported).add("always_on_top")
        if config.click_through:
            flags |= Qt.WindowType.WindowTransparentForInput
            (applied if caps.click_through else unsupported).add("click_through")
        flags |= Qt.WindowType.WindowDoesNotAcceptFocus
        window.setWindowFlags(flags)
        window.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        window.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        window.setWindowOpacity(config.opacity)
        applied.update({"transparent", "opacity"})
        if config.position is not None and not caps.positionable:
            unsupported.add("position")
        else:
            applied.add("position")
        return OverlayApplyResult(frozenset(applied), frozenset(unsupported), tuple(msgs))
