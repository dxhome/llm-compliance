from __future__ import annotations

from mpid.crossmodal import check_crossmodal


def test_clean_text_not_suspicious():
    result = check_crossmodal({"text": "What is the capital of France?", "image": None})
    assert result.label == "clean"
    assert not result.suspicious


def test_figstep_metadata_marks_indirect():
    result = check_crossmodal(
        {
            "text": "The image shows a list numbered 1, 2, and 3, but the items are empty.",
            "image": "runs/_datasets/raw/jailbreakv_28k/JailBreakV_28K/figstep/foo.png",
            "source": "jailbreakv_28k",
            "metadata": {"format": "figstep"},
        }
    )
    assert result.label == "indirect"
    assert result.suspicious
    assert "figstep_metadata_or_path" in result.reasons


def test_figstep_text_pattern_marks_indirect():
    result = check_crossmodal(
        {
            "text": "The image shows a list numbered 1, 2, and 3, but the items are empty.",
            "image": None,
            "source": "jailbreakv_28k",
            "metadata": {},
        }
    )
    assert result.label == "indirect"
    assert "figstep_text_pattern" in result.reasons
