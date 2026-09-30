"""Large diagnostic output must not grow trajectory/event metadata unboundedly."""

import json
from pathlib import Path

import pytest
import yaml
from jinja2 import StrictUndefined, Template

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel
from minisweagent.models.utils.actions_text import format_observation_messages
from minisweagent.models.utils.actions_toolcall import format_toolcall_observation_messages
from minisweagent.models.utils.actions_toolcall_response import (
    format_toolcall_observation_messages as format_response_observations,
)
from minisweagent.models.utils.observation import RAW_OUTPUT_MAX_CHARS, observation_metadata


@pytest.fixture(params=["text", "toolcall", "response"])
def format_output(request):
    def render(output, template):
        kwargs = {"outputs": [output], "observation_template": template}
        if request.param == "text":
            return format_observation_messages(**kwargs)[0]
        kwargs["actions"] = [{"command": "test", "tool_call_id": "call_1"}]
        formatter = format_toolcall_observation_messages if request.param == "toolcall" else format_response_observations
        return formatter(**kwargs)[0]

    return render


def test_large_raw_metadata_is_bounded_but_model_input_is_identical(format_output):
    config_path = Path(__file__).resolve().parents[2] / "src/minisweagent/config/benchmarks/swebench_backticks.yaml"
    template = yaml.safe_load(config_path.read_text())["model"]["observation_template"]
    raw = "HEAD\n" + "ログ行\\\"\n" * 300_000 + "\nTAIL"
    output = {"output": raw, "returncode": -1, "exception_info": "timeout", "extra": {"exception_type": "TimeoutExpired"}}
    expected = Template(template, undefined=StrictUndefined).render(output=output)
    msg = format_output(output, template)
    assert msg.get("content", msg.get("output")) == expected
    assert len(msg["extra"]["raw_output"]) == RAW_OUTPUT_MAX_CHARS
    assert msg["extra"]["raw_output"].startswith("HEAD\n")
    assert msg["extra"]["raw_output"].endswith("\nTAIL")
    assert msg["extra"]["raw_output_original_chars"] == len(raw)
    assert msg["extra"]["raw_output_truncated"] is True
    assert msg["extra"]["exception_type"] == "TimeoutExpired"
    assert output["output"] == raw
    assert "raw_output_truncated" not in output["extra"]
    assert len(json.dumps(msg)) < 12 * RAW_OUTPUT_MAX_CHARS


@pytest.mark.parametrize("size", [0, 100, RAW_OUTPUT_MAX_CHARS])
def test_small_output_retains_original_schema(format_output, size):
    raw = "x" * size
    msg = format_output({"output": raw, "returncode": 0}, "{{ output.output }}")
    assert msg.get("content", msg.get("output")) == raw
    assert msg["extra"]["raw_output"] == raw
    assert "raw_output_truncated" not in msg["extra"]


def test_extra_raw_output_cannot_bypass_limit():
    original = {"output": "small", "extra": {"raw_output": "x" * (RAW_OUTPUT_MAX_CHARS + 1)}}
    assert len(observation_metadata(original)["raw_output"]) == RAW_OUTPUT_MAX_CHARS


def test_saved_trajectory_and_events_only_retain_bounded_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("MSWEA_EVENT_LOG_DIR", str(tmp_path))
    agent = DefaultAgent(
        DeterministicModel(outputs=[]), LocalEnvironment(), system_template="system", instance_template="{{task}}"
    )
    raw = "x" * 2_000_000
    msg = format_observation_messages(
        [{"output": raw, "returncode": 0}], observation_template="{{ output.output[:10] }}"
    )[0]
    agent.add_messages(msg)
    saved = agent.save(tmp_path / "trajectory.json")
    assert json.loads((tmp_path / "trajectory.json").read_text()) == saved
    event = json.loads((tmp_path / "events.jsonl").read_text())
    assert event["message"] == saved["messages"][0]
    assert event["message"]["content"] == raw[:10]
    assert (tmp_path / "trajectory.json").stat().st_size < 100_000
    assert (tmp_path / "events.jsonl").stat().st_size < 100_000
