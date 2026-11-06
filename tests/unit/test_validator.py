import pytest

from core.config.models import ValidationConfig
from core.types import CaptionState, ModelDiagnostics, TranslationResult
from core.validation import CaptionValidator, strip_seam_overlap


def res(text, s=0.0, e=2000.0, nsp=0.05, lp=-0.3, cr=1.4, lang_p=None):
    return TranslationResult(
        "ja", text, s, e, "small", diagnostics=ModelDiagnostics(nsp, lp, cr, lang_p)
    )


def test_good_caption_accepted():
    v = CaptionValidator().validate(res("The weather is nice today."))
    assert v.accepted and v.text == "The weather is nice today."


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"nsp": 0.8}, "no_speech"),
        ({"lp": -1.4}, "low_logprob"),
        ({"cr": 3.0}, "repetitive"),
        ({"lang_p": 0.3}, "low_language_confidence"),
    ],
)
def test_diagnostic_rejections(kwargs, reason):
    v = CaptionValidator().validate(res("Hello there my friend", **kwargs))
    assert not v.accepted and reason in v.reasons


def test_hallucination_on_silence_rejected_but_real_thank_you_kept():
    assert not CaptionValidator().validate(res("Thank you for watching!", nsp=0.8)).accepted
    # clear speech saying thank you: strong acoustics -> accepted
    assert CaptionValidator().validate(res("Thank you.", nsp=0.02, lp=-0.2)).accepted


def test_empty_and_punctuation_only():
    v = CaptionValidator()
    assert v.validate(res("   ")).reasons == ("empty",)
    assert v.validate(res("...")).reasons == ("empty",)


def test_repetition_loop():
    assert "repetition_loop" in CaptionValidator().validate(res("no no no no no no")).reasons
    t = "I will go. " * 6
    assert "repetition_loop" in CaptionValidator().validate(res(t)).reasons
    assert CaptionValidator().validate(res("no no, I said no")).accepted


def test_missing_diagnostics_are_not_treated_as_failures():
    r = TranslationResult("ja", "Hello friends", 0, 1000, "small")
    assert CaptionValidator().validate(r).accepted


def test_duplicate_overlapping_caption_rejected_but_new_one_kept():
    v = CaptionValidator()
    assert v.validate(res("I am going to the garden.", 0, 3000)).accepted
    assert "duplicate" in v.validate(res("I'm going to the garden", 2500, 5000)).reasons
    assert v.validate(res("Then we eat dinner.", 5200, 7000)).accepted


def test_seam_overlap_stripped():
    assert strip_seam_overlap(
        "we went to the market today", "to the market today and bought rice"
    ) == ("and bought rice")
    assert strip_seam_overlap("hello there", "there is a cat") == "there is a cat"  # 1 word: keep
    v = CaptionValidator()
    assert v.validate(res("we went to the market today", 0, 4000)).accepted
    out = v.validate(res("to the market today and bought rice", 3700, 7000), seam=True)
    assert out.accepted and out.text == "and bought rice"


def test_staleness():
    v = CaptionValidator(ValidationConfig(max_staleness_ms=5000))
    assert "stale" in v.validate(res("Late caption", 0, 1000), now_audio_ms=9000).reasons
    assert v.validate(res("Fresh caption", 0, 1000), now_audio_ms=4000).accepted


def test_state_enum_untouched():
    assert CaptionState.FINAL.value == "final"
