import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import litellm
import pytest

from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel


def test_authentication_error_enhanced_message():
    """Test that AuthenticationError gets enhanced with config set instruction."""
    model = LitellmTextbasedModel(model_name="gpt-4")

    # Create a mock exception that behaves like AuthenticationError
    original_error = Mock(spec=litellm.exceptions.AuthenticationError)
    original_error.message = "Invalid API key"

    with patch("litellm.completion") as mock_completion:
        # Make completion raise the mock error
        def side_effect(*args, **kwargs):
            raise litellm.exceptions.AuthenticationError("Invalid API key", llm_provider="openai", model="gpt-4")

        mock_completion.side_effect = side_effect

        with pytest.raises(litellm.exceptions.AuthenticationError) as exc_info:
            model._query([{"role": "user", "content": "test"}])

        # Check that the error message was enhanced
        assert "You can permanently set your API key with `mini-extra config set KEY VALUE`." in str(exc_info.value)


def test_model_registry_loading():
    """Test that custom model registry is loaded and registered when provided."""
    model_costs = {
        "my-custom-model": {
            "max_tokens": 4096,
            "input_cost_per_token": 0.0001,
            "output_cost_per_token": 0.0002,
            "litellm_provider": "openai",
            "mode": "chat",
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(model_costs, f)
        registry_path = f.name

    try:
        with patch("litellm.utils.register_model") as mock_register:
            _model = LitellmTextbasedModel(model_name="my-custom-model", litellm_model_registry=Path(registry_path))

            # Verify register_model was called with the correct data
            mock_register.assert_called_once_with(model_costs)
    except Exception as e:
        print(e)
        raise e
    finally:
        Path(registry_path).unlink()


def test_model_registry_none():
    """Test that no registry loading occurs when litellm_model_registry is None."""
    with patch("litellm.register_model") as mock_register:
        _model = LitellmTextbasedModel(model_name="gpt-4", litellm_model_registry=None)

        # Verify register_model was not called
        mock_register.assert_not_called()


def test_model_registry_not_provided():
    """Test that no registry loading occurs when litellm_model_registry is not provided."""
    with patch("litellm.register_model") as mock_register:
        _model = LitellmTextbasedModel(model_name="gpt-4o")

        # Verify register_model was not called
        mock_register.assert_not_called()


def test_litellm_model_cost_tracking_ignore_errors():
    """Test that models work with cost_tracking='ignore_errors'."""
    model = LitellmTextbasedModel(model_name="gpt-4o", cost_tracking="ignore_errors")

    initial_cost = GLOBAL_MODEL_STATS.cost

    with patch("litellm.completion") as mock_completion:
        mock_response = Mock()
        mock_message = Mock()
        mock_message.content = "```mswea_bash_command\necho test\n```"
        mock_message.model_dump.return_value = {
            "role": "assistant",
            "content": "```mswea_bash_command\necho test\n```",
        }
        mock_response.choices = [Mock(message=mock_message)]
        mock_response.model_dump.return_value = {"test": "response"}
        mock_completion.return_value = mock_response

        with patch("litellm.cost_calculator.completion_cost", side_effect=ValueError("Model not found")):
            messages = [{"role": "user", "content": "test"}]
            result = model.query(messages)

            assert result["content"] == "```mswea_bash_command\necho test\n```"
            assert result["extra"]["actions"] == [{"command": "echo test"}]
            assert GLOBAL_MODEL_STATS.cost == initial_cost


def test_litellm_model_cost_validation_zero_cost():
    """Test that zero cost raises error when cost tracking is enabled."""
    model = LitellmTextbasedModel(model_name="gpt-4o")

    with patch("litellm.completion") as mock_completion:
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        mock_response.model_dump.return_value = {"test": "response"}
        mock_completion.return_value = mock_response

        with patch("litellm.cost_calculator.completion_cost", return_value=0.0):
            messages = [{"role": "user", "content": "test"}]

            with pytest.raises(RuntimeError) as exc_info:
                model.query(messages)

            assert "Cost must be > 0.0, got 0.0" in str(exc_info.value)
            assert "MSWEA_COST_TRACKING='ignore_errors'" in str(exc_info.value)


# ── Unclosed-think fallback ────────────────────────────────────────────────

_ONE_ACTION = "Look at the file.\n\n```mswea_bash_command\nls /testbed\n```"


def _response(content, reasoning_content, finish_reason="stop"):
    message = {"role": "assistant", "content": content}
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    return litellm.ModelResponse(
        choices=[{"finish_reason": finish_reason, "index": 0, "message": message}],
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


def test_unclosed_think_fallback_takes_action_from_reasoning():
    model = LitellmTextbasedModel(model_name="gpt-4", cost_tracking="ignore_errors")
    assert model._parse_actions(_response(None, _ONE_ACTION)) == [{"command": "ls /testbed"}]
    assert model._parse_actions(_response("  \n", _ONE_ACTION)) == [{"command": "ls /testbed"}]


@pytest.mark.parametrize(
    "content,reasoning,finish_reason",
    [
        (None, _ONE_ACTION, "length"),  # cut off by the token limit
        (None, "just thinking, no command", "stop"),  # zero actions
        (None, _ONE_ACTION + "\n" + _ONE_ACTION, "stop"),  # several actions
        (None, None, "stop"),  # nothing at all
        ("no block here", _ONE_ACTION, "stop"),  # content present but malformed: content wins
    ],
)
def test_unclosed_think_fallback_not_applied(content, reasoning, finish_reason):
    from minisweagent.exceptions import FormatError

    model = LitellmTextbasedModel(model_name="gpt-4", cost_tracking="ignore_errors")
    with pytest.raises(FormatError):
        model._parse_actions(_response(content, reasoning, finish_reason))


def test_unclosed_think_fallback_disabled():
    from minisweagent.exceptions import FormatError

    model = LitellmTextbasedModel(model_name="gpt-4", cost_tracking="ignore_errors", unclosed_think_fallback=False)
    with pytest.raises(FormatError):
        model._parse_actions(_response(None, _ONE_ACTION))


def test_content_still_preferred_over_reasoning():
    model = LitellmTextbasedModel(model_name="gpt-4", cost_tracking="ignore_errors")
    resp = _response("```mswea_bash_command\npwd\n```", "draft:\n```mswea_bash_command\nls\n```")
    assert model._parse_actions(resp) == [{"command": "pwd"}]


def test_query_rewrites_message_on_fallback():
    model = LitellmTextbasedModel(model_name="gpt-4", cost_tracking="ignore_errors")
    with patch.object(model, "_query", return_value=_response(None, _ONE_ACTION)):
        message = model.query([{"role": "user", "content": "go"}])
    assert message["content"] == _ONE_ACTION
    assert "reasoning_content" not in message
    assert message["extra"]["action_source"] == "reasoning_content"
    assert message["extra"]["actions"] == [{"command": "ls /testbed"}]
    # the raw response is still on record
    assert message["extra"]["response"]["choices"][0]["message"]["reasoning_content"] == _ONE_ACTION
    assert message["extra"]["response"]["choices"][0]["message"]["content"] is None


def test_query_leaves_normal_message_alone():
    model = LitellmTextbasedModel(model_name="gpt-4", cost_tracking="ignore_errors")
    with patch.object(model, "_query", return_value=_response(_ONE_ACTION, "thinking...")):
        message = model.query([{"role": "user", "content": "go"}])
    assert message["content"] == _ONE_ACTION
    assert message["reasoning_content"] == "thinking..."
    assert "action_source" not in message["extra"]
