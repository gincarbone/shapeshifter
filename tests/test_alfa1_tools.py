"""Tests for alfa1_tools.py — workspace-scoped file I/O, the path-traversal
guard, and shell command execution."""
from __future__ import annotations

import asyncio
import sys

import pytest

import alfa1_tools
from alfa1_tools import Alfa1Error


@pytest.fixture(autouse=True)
def _reset_workspace(tmp_path_factory, monkeypatch):
    alfa1_tools._workspace_root = None
    # Redirect the last-workspace pointer file to a throwaway location for
    # every test — it must never read/write the real one next to
    # alfa1_tools.py, or tests would pollute (and be polluted by) the
    # actual user's remembered workspace.
    monkeypatch.setattr(
        alfa1_tools, "_LAST_WORKSPACE_PATH",
        tmp_path_factory.mktemp("alfa1_last_ws") / ".alfa1_last_workspace.json",
    )
    yield
    alfa1_tools._workspace_root = None


def test_set_workspace_rejects_nonexistent_path(tmp_path):
    with pytest.raises(Alfa1Error):
        alfa1_tools.set_workspace(str(tmp_path / "does-not-exist"))


def test_set_workspace_rejects_file(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("hi")
    with pytest.raises(Alfa1Error):
        alfa1_tools.set_workspace(str(f))


def test_set_workspace_accepts_directory(tmp_path):
    result = alfa1_tools.set_workspace(str(tmp_path))
    assert result["root"] == str(tmp_path.resolve())
    assert alfa1_tools.get_workspace() == tmp_path.resolve()


def test_set_workspace_creates_session_dir_and_remembers_last_workspace(tmp_path):
    alfa1_tools.set_workspace(str(tmp_path))
    assert (tmp_path / ".alfa1").is_dir()
    assert alfa1_tools.get_last_workspace() == tmp_path.resolve()


def test_get_last_workspace_none_when_pointer_file_missing():
    assert alfa1_tools.get_last_workspace() is None


def test_get_last_workspace_none_when_remembered_dir_no_longer_exists(tmp_path):
    import shutil
    ghost = tmp_path / "ghost"
    ghost.mkdir()
    alfa1_tools.set_workspace(str(ghost))
    shutil.rmtree(ghost)  # set_workspace creates a .alfa1 subdir, so plain rmdir() would fail
    assert alfa1_tools.get_last_workspace() is None


def test_save_and_load_history_round_trip(tmp_path):
    alfa1_tools.set_workspace(str(tmp_path))
    conversation = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    alfa1_tools.save_history(conversation)
    assert alfa1_tools.load_history() == conversation


def test_load_history_none_when_never_saved(tmp_path):
    alfa1_tools.set_workspace(str(tmp_path))
    assert alfa1_tools.load_history() is None


def test_load_history_none_without_workspace():
    assert alfa1_tools.load_history() is None


def test_save_history_no_op_without_workspace():
    alfa1_tools.save_history([{"role": "user", "content": "hi"}])  # must not raise


@pytest.mark.parametrize("bad_path", [
    "../etc/passwd",
    "../../secret.txt",
    "/etc/passwd",
    "C:\\Windows\\System32",
    "..\\..\\secret",
])
def test_safe_join_rejects_traversal(tmp_path, bad_path):
    alfa1_tools.set_workspace(str(tmp_path))
    with pytest.raises(Alfa1Error):
        alfa1_tools._safe_join(bad_path)


def test_safe_join_accepts_nested_valid_path(tmp_path):
    alfa1_tools.set_workspace(str(tmp_path))
    (tmp_path / "sub").mkdir()
    result = alfa1_tools._safe_join("sub/file.txt")
    assert result == (tmp_path / "sub" / "file.txt").resolve()


def test_safe_join_requires_workspace():
    with pytest.raises(Alfa1Error):
        alfa1_tools._safe_join("anything.txt")


def test_write_read_delete_roundtrip(tmp_path):
    alfa1_tools.set_workspace(str(tmp_path))
    w = alfa1_tools.write_file("a/b.txt", "hello world")
    assert w["bytes_written"] == len("hello world")
    assert (tmp_path / "a" / "b.txt").read_text() == "hello world"

    r = alfa1_tools.read_file("a/b.txt")
    assert r["content"] == "hello world"
    assert r["binary"] is False

    d = alfa1_tools.delete_file("a/b.txt")
    assert d["deleted"] is True
    assert not (tmp_path / "a" / "b.txt").exists()


def test_read_file_detects_binary(tmp_path):
    alfa1_tools.set_workspace(str(tmp_path))
    (tmp_path / "bin.dat").write_bytes(b"\x00\x01\x02\x03")
    r = alfa1_tools.read_file("bin.dat")
    assert r["binary"] is True
    assert r["content"] is None


def test_delete_file_refuses_workspace_root(tmp_path):
    alfa1_tools.set_workspace(str(tmp_path))
    with pytest.raises(Alfa1Error):
        alfa1_tools.delete_file(".")


def test_list_tree_skips_noise_dirs_and_finds_files(tmp_path):
    alfa1_tools.set_workspace(str(tmp_path))
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("x")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print(1)")

    entries = alfa1_tools.list_tree(".")
    paths = {e["path"] for e in entries}
    assert "src" in paths
    assert "src/main.py" in paths
    assert not any(p.startswith(".git") for p in paths)


def test_search_files_finds_plain_text_across_files(tmp_path):
    alfa1_tools.set_workspace(str(tmp_path))
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.py").write_text("def bar():\n    return foo()\n", encoding="utf-8")

    results = alfa1_tools.search_files("foo")
    paths = {(r["path"], r["line"]) for r in results}
    assert ("a.py", 1) in paths
    assert ("sub/b.py", 2) in paths


def test_search_files_is_case_insensitive_by_default(tmp_path):
    alfa1_tools.set_workspace(str(tmp_path))
    (tmp_path / "a.txt").write_text("Hello World\n", encoding="utf-8")
    results = alfa1_tools.search_files("hello world")
    assert len(results) == 1


def test_search_files_falls_back_to_literal_on_invalid_regex(tmp_path):
    alfa1_tools.set_workspace(str(tmp_path))
    (tmp_path / "a.txt").write_text("a(b test\n", encoding="utf-8")
    # "a(b" is invalid regex syntax (unbalanced paren) — must not raise,
    # should fall back to a literal substring match.
    results = alfa1_tools.search_files("a(b")
    assert len(results) == 1


def test_search_files_skips_binary_and_noise_dirs(tmp_path):
    alfa1_tools.set_workspace(str(tmp_path))
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("needle", encoding="utf-8")
    (tmp_path / "bin.dat").write_bytes(b"\x00needle\x00")
    (tmp_path / "real.txt").write_text("needle here\n", encoding="utf-8")

    results = alfa1_tools.search_files("needle")
    assert {r["path"] for r in results} == {"real.txt"}


def test_search_files_respects_max_results(tmp_path):
    alfa1_tools.set_workspace(str(tmp_path))
    (tmp_path / "a.txt").write_text("\n".join("needle" for _ in range(10)), encoding="utf-8")
    results = alfa1_tools.search_files("needle", max_results=3)
    assert len(results) == 3


def test_run_command_captures_exit_code_and_stdout(tmp_path):
    alfa1_tools.set_workspace(str(tmp_path))
    result = asyncio.run(alfa1_tools.run_command(f'"{sys.executable}" -c "print(1)"'))
    assert result["exit_code"] == 0
    assert "1" in result["stdout"]
    assert result["timed_out"] is False


def test_run_command_times_out(tmp_path):
    alfa1_tools.set_workspace(str(tmp_path))
    result = asyncio.run(alfa1_tools.run_command(
        f'"{sys.executable}" -c "import time; time.sleep(5)"', timeout_s=1,
    ))
    assert result["timed_out"] is True
    assert result["exit_code"] is None


def test_run_command_truncates_large_output(tmp_path):
    alfa1_tools.set_workspace(str(tmp_path))
    result = asyncio.run(alfa1_tools.run_command(
        f'"{sys.executable}" -c "print(\'x\' * 1000)"', max_output_bytes=10,
    ))
    assert result["truncated"] is True
    assert len(result["stdout"]) == 10
