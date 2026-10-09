import pytest

from intent_classifier import IntentClassifier


@pytest.fixture()
def classifier(tmp_path):
    """Force the keyword fallback by pointing at a directory with no model."""
    return IntentClassifier(model_path=tmp_path / "missing")


def test_fallback_backend_when_model_missing(classifier):
    assert classifier.backend == "keyword" and not classifier.is_transformer


def test_normal_query(classifier):
    result = classifier.classify("My payment failed and I was charged")
    assert result.intent == "payment_issue"
    assert 0.0 <= result.confidence <= 1.0


def test_unrecognised_query_is_low_confidence(classifier):
    result = classifier.classify("zzzz qqqq")
    assert result.intent == "general_inquiry" and result.confidence < 0.5


@pytest.mark.parametrize("bad", ["", "   ", None, 123])
def test_empty_or_invalid_query_raises(classifier, bad):
    with pytest.raises(ValueError):
        classifier.classify(bad)


def test_default_classifier_returns_valid_result():
    result = IntentClassifier().classify("Where is my order?")
    assert result.intent and 0.0 <= result.confidence <= 1.0
