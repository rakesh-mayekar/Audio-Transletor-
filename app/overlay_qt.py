"""PySide6 caption overlay window. Thin: all timing/history logic lives in CaptionPresenter.

STATUS: written but NOT executed in this build (PySide6 was not installed when it was written).
Qt objects may only be touched on the GUI thread, so bus events are marshalled with a Signal.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, QPoint, QTimer, Signal
from PySide6.QtGui import QFont, QGuiApplication, QMouseEvent
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from app.captions import CaptionPresenter
from core.config.models import OverlayConfig
from core.events import EventBus
from core.events.types import CaptionValidated
from oslayer.overlay import QtOverlayProvider


class _Bridge(QObject):
    caption = Signal(object)


class OverlayWindow(QWidget):
    def __init__(
        self, cfg: OverlayConfig, bus: EventBus, provider: QtOverlayProvider | None = None
    ):
        super().__init__()
        self.cfg = cfg
        self.presenter = CaptionPresenter(cfg, bus)
        self.provider = provider or QtOverlayProvider()
        self.apply_result = self.provider.apply(self, cfg)
        self._label = QLabel("")
        self._label.setWordWrap(True)
        self._label.setFont(QFont("", cfg.font_size_pt, QFont.Weight.DemiBold))
        self._label.setStyleSheet(
            "color: white; background: rgba(0,0,0,170); padding: 10px 18px; border-radius: 10px;"
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._label)
        self.setMinimumWidth(480)
        self.setMaximumWidth(1100)
        self._drag: QPoint | None = None
        self._bridge = _Bridge()
        self._bridge.caption.connect(self._on_caption)
        self._sub = bus.subscribe("overlay", CaptionValidated, handler=self._bridge.caption.emit)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(100)
        self.presenter.on_change = self._refresh
        self._refresh()
        self._place()

    def _on_caption(self, e: CaptionValidated) -> None:
        self.presenter.show(e)

    def _tick(self) -> None:
        self.presenter.tick()

    def _refresh(self) -> None:
        text = self.presenter.text()
        self._label.setText(text)
        self.setVisible(self.cfg.visible and bool(text))
        self.adjustSize()
        if self.isVisible():
            self._place()

    def _place(self) -> None:
        if self.cfg.position is not None:
            self.move(*self.cfg.position)
            return
        screen = QGuiApplication.primaryScreen().availableGeometry()
        self.move(screen.center().x() - self.width() // 2, screen.bottom() - self.height() - 60)

    # drag to reposition (ignored when click-through is on, since input passes through)
    def mousePressEvent(self, ev: QMouseEvent) -> None:
        self._drag = ev.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:
        if self._drag is not None:
            self.move(ev.globalPosition().toPoint() - self._drag)

    def mouseReleaseEvent(self, ev: QMouseEvent) -> None:
        self._drag = None

    def closeEvent(self, ev: object) -> None:  # type: ignore[override]
        self._timer.stop()
        self._sub.close()
        super().closeEvent(ev)  # type: ignore[arg-type]
