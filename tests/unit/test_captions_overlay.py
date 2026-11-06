import py_compile
from pathlib import Path

from app.captions import CaptionPresenter, display_seconds
from core.config.models import OverlayConfig
from core.events import EventBus
from core.events.types import CaptionDisplayed, CaptionExpired, CaptionValidated
from oslayer.overlay import QtOverlayProvider
from oslayer.selector import OSName


class Clock:
    t = 100.0

    def __call__(self):
        return self.t


def cap(i, text="Hello there", s=0.0, e=1000.0):
    return CaptionValidated(caption_id=f"c{i}", text=text, start_ms=s, end_ms=e)


def test_show_expire_and_events():
    clk, bus, got = Clock(), EventBus(), []
    bus.subscribe("t", handler=got.append)
    p = CaptionPresenter(OverlayConfig(caption_timeout_s=6.0), bus, clk)
    assert p.show(cap(1))
    assert p.text() == "Hello there"
    clk.t += 5.9
    assert not p.tick() and p.text()
    clk.t += 0.2
    assert p.tick() and p.text() == ""
    bus.wait_idle()
    kinds = [type(e) for e in got]
    assert kinds == [CaptionDisplayed, CaptionExpired]
    bus.close()


def test_two_visible_max_oldest_pushed_out():
    p = CaptionPresenter(OverlayConfig(), clock=Clock())
    for i in range(3):
        p.show(cap(i, f"line {i}", e=1000.0 * (i + 1)))
    assert [c.text for c in p.visible()] == ["line 1", "line 2"]


def test_out_of_order_caption_is_not_shown():
    p = CaptionPresenter(OverlayConfig(), clock=Clock())
    assert p.show(cap(1, e=5000.0))
    assert not p.show(cap(2, "older", e=3000.0))
    assert p.skipped_out_of_order == 1 and p.text() == "Hello there"


def test_long_captions_stay_longer():
    assert display_seconds("one two", 6.0) == 6.0
    assert display_seconds(" ".join(["w"] * 30), 6.0) == 10.0


def test_history_respects_setting():
    p = CaptionPresenter(OverlayConfig(history_size=2), clock=Clock())
    for i in range(4):
        p.show(cap(i, f"t{i}", e=1000.0 * (i + 1)))
    assert [c.text for c in p.history()] == ["t2", "t3"]
    off = CaptionPresenter(OverlayConfig(), clock=Clock(), keep_history=False)
    off.show(cap(1))
    assert off.history() == []


def test_wayland_limits_are_reported_honestly():
    caps = QtOverlayProvider(OSName.LINUX, wayland=True).capabilities()
    assert not caps.always_on_top and not caps.positionable and caps.notes
    assert QtOverlayProvider(OSName.LINUX, wayland=False).capabilities().always_on_top
    assert QtOverlayProvider(OSName.WINDOWS, wayland=False).capabilities().notes


def test_qt_widget_module_is_valid_python():
    # PySide6 is optional/not installed: at least guarantee the file compiles.
    py_compile.compile(str(Path(__file__).parents[2] / "app" / "overlay_qt.py"), doraise=True)
