from unittest.mock import patch, MagicMock
from app.context_inference import infer_context
from app.config import settings


def _make_mock_client(content: str) -> MagicMock:
    """Build a mock Groq client whose chat.completions.create returns content."""
    mock_message = MagicMock()
    mock_message.content = content

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


def test_infer_context_returns_summary():
    """Stripped summary is returned when the API succeeds."""
    raw = "  A team standup about the Apollo project.  "
    mock_client = _make_mock_client(raw)

    with patch("app.context_inference._get_client", return_value=mock_client):
        result = infer_context("some rough words")

    assert result == "A team standup about the Apollo project."


def test_infer_context_sends_rough_text():
    """The rough text appears in the user message and the correct model is used."""
    mock_client = _make_mock_client("Some description.")

    with patch("app.context_inference._get_client", return_value=mock_client):
        infer_context("quarterly earnings call transcript excerpt")

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == settings.inference_model

    messages = call_kwargs["messages"]
    user_messages = [m for m in messages if m["role"] == "user"]
    assert len(user_messages) == 1
    assert "quarterly earnings call transcript excerpt" in user_messages[0]["content"]


def test_infer_context_returns_empty_on_error():
    """Returns "" when the Groq client raises any exception."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("API down")

    with patch("app.context_inference._get_client", return_value=mock_client):
        result = infer_context("some text")

    assert result == ""


def test_infer_context_empty_input_skips_api():
    """Returns "" immediately for empty/whitespace input without calling the API."""
    mock_client = MagicMock()

    with patch("app.context_inference._get_client", return_value=mock_client) as mock_getter:
        assert infer_context("") == ""
        assert infer_context("   ") == ""

    mock_getter.assert_not_called()
    mock_client.chat.completions.create.assert_not_called()
