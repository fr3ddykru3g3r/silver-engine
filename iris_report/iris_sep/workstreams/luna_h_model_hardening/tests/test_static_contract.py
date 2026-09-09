from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_colab_runtime_test_is_present_and_no_dataset_paths_are_embedded() -> None:
    text = (ROOT / "colab_runtime_test.py").read_text()
    assert "torch" in text
    assert "synthetic" in text.lower()
    forbidden = ("locked_test", "data/", "/datasets/", "parquet")
    assert not any(token in text for token in forbidden)


def test_public_api_mentions_all_missing_abstention_and_primary_default() -> None:
    text = (ROOT / "luna_model.py").read_text()
    for token in ("all_missing", "abstain", "feature_masks", "event_conditional", "censored"):
        assert token in text

