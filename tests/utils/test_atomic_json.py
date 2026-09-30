"""Interrupted checkpoints must leave the last complete trajectory readable."""

import json
import os
import signal
import subprocess
import sys

import pytest

from minisweagent.utils.serialize import atomic_write_json


def test_atomic_json_replaces_complete_snapshot(tmp_path):
    path = tmp_path / "nested" / "trajectory.json"
    atomic_write_json(path, {"messages": ["old"]})
    payload = {"messages": ["新しい", {"extra": {"calls": 134}}]}
    atomic_write_json(path, payload)
    assert json.loads(path.read_text()) == payload
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize("failure", ["serialization", "replace"])
def test_failed_write_preserves_previous_snapshot(tmp_path, monkeypatch, failure):
    path = tmp_path / "trajectory.json"
    atomic_write_json(path, {"messages": ["previous"]})
    previous = path.read_bytes()
    if failure == "serialization":
        payload = {"messages": ["new", object()]}
        expected = TypeError
    else:
        payload = {"messages": ["new"]}
        expected = OSError

        def fail_replace(*args):
            raise OSError("simulated failed replace")

        monkeypatch.setattr("minisweagent.utils.serialize.os.replace", fail_replace)
    with pytest.raises(expected):
        atomic_write_json(path, payload)
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.skipif(os.name != "posix", reason="SIGKILL is a POSIX failure mode")
def test_sigkill_during_write_preserves_previous_snapshot(tmp_path):
    path = tmp_path / "trajectory.json"
    atomic_write_json(path, {"messages": ["previous"]})
    previous = path.read_bytes()
    child = r'''
import os, signal, sys
from pathlib import Path
from minisweagent.utils import serialize

def interrupted_dump(data, stream, **kwargs):
    stream.write('{"messages": ["incomplete')
    stream.flush()
    os.kill(os.getpid(), signal.SIGKILL)

serialize.json.dump = interrupted_dump
serialize.atomic_write_json(Path(sys.argv[1]), {"messages": ["new"]})
'''
    result = subprocess.run([sys.executable, "-c", child, str(path)], capture_output=True, timeout=10)
    assert result.returncode == -signal.SIGKILL, result.stderr.decode()
    assert path.read_bytes() == previous
    assert json.loads(path.read_text()) == {"messages": ["previous"]}
