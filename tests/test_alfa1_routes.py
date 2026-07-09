"""HTTP-level tests for alfa1_routes.py using fastapi.testclient.TestClient.

This is a deliberate deviation from the rest of the test suite's
plain-function-call convention (see test_wrapper_pipeline.py): these routes
are thin FastAPI wiring around alfa1_tools, so testing them at the HTTP
layer is the pragmatic choice; the underlying logic is already covered by
test_alfa1_tools.py and test_alfa1_agent.py.
"""
from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import alfa1_routes
import alfa1_tools


@pytest.fixture(autouse=True)
def _reset_state(tmp_path_factory, monkeypatch):
    alfa1_tools._workspace_root = None
    alfa1_routes._conversation.clear()
    alfa1_routes._turn_status = "idle"
    alfa1_routes._current_task = None
    alfa1_routes._pending_queue.clear()
    # Redirect the last-workspace pointer so these tests never read/write
    # the real user's remembered workspace (see test_alfa1_tools.py's
    # equivalent fixture for the same reasoning).
    monkeypatch.setattr(
        alfa1_tools, "_LAST_WORKSPACE_PATH",
        tmp_path_factory.mktemp("alfa1_last_ws") / ".alfa1_last_workspace.json",
    )
    yield
    alfa1_tools._workspace_root = None
    alfa1_routes._conversation.clear()
    alfa1_routes._turn_status = "idle"
    alfa1_routes._current_task = None
    alfa1_routes._pending_queue.clear()


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(alfa1_routes.router, prefix="/alfa1")
    return TestClient(app)


def test_get_workspace_initially_none(client):
    r = client.get("/alfa1/workspace")
    assert r.status_code == 200
    assert r.json() == {"root": None}


def test_set_workspace_round_trip(client, tmp_path):
    r = client.post("/alfa1/workspace", json={"path": str(tmp_path)})
    assert r.status_code == 200
    assert r.json()["root"] == str(tmp_path.resolve())

    r2 = client.get("/alfa1/workspace")
    assert r2.json()["root"] == str(tmp_path.resolve())


def test_set_workspace_rejects_malformed_json_body(client):
    r = client.post(
        "/alfa1/workspace",
        content=b"{not valid json",
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 400


def test_chat_rejects_malformed_json_body(client, tmp_path):
    client.post("/alfa1/workspace", json={"path": str(tmp_path)})
    r = client.post(
        "/alfa1/chat",
        content=b"{not valid json",
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 400


def test_get_workspace_auto_restores_last_used_folder(client, tmp_path):
    """Regression coverage for the session-persistence feature: a fresh
    process (workspace_root back to None, as after a real server restart)
    should transparently pick back up the last folder used, instead of
    forcing the user to re-pick it — this is exactly what a page load's
    checkWorkspace() -> GET /alfa1/workspace relies on."""
    alfa1_tools.set_workspace(str(tmp_path))  # simulates the *previous* process run
    alfa1_tools._workspace_root = None        # simulates the restart itself

    r = client.get("/alfa1/workspace")
    assert r.json()["root"] == str(tmp_path.resolve())


def test_get_workspace_restores_persisted_conversation_history(client, tmp_path):
    alfa1_tools.set_workspace(str(tmp_path))
    alfa1_tools.save_history([{"role": "user", "content": "earlier turn"}])
    alfa1_tools._workspace_root = None
    alfa1_routes._conversation.clear()

    client.get("/alfa1/workspace")
    assert alfa1_routes._conversation == [{"role": "user", "content": "earlier turn"}]


def test_set_workspace_route_loads_that_folders_own_history(client, tmp_path):
    alfa1_tools.set_workspace(str(tmp_path))
    alfa1_tools.save_history([{"role": "user", "content": "hi from before"}])
    alfa1_tools._workspace_root = None
    alfa1_routes._conversation.clear()

    client.post("/alfa1/workspace", json={"path": str(tmp_path)})
    assert alfa1_routes._conversation == [{"role": "user", "content": "hi from before"}]


def test_set_workspace_rejects_bad_path(client, tmp_path):
    r = client.post("/alfa1/workspace", json={"path": str(tmp_path / "nope")})
    assert r.status_code == 400


def test_file_routes_reject_traversal(client, tmp_path):
    client.post("/alfa1/workspace", json={"path": str(tmp_path)})

    r = client.get("/alfa1/files/content", params={"path": "../../etc/passwd"})
    assert r.status_code == 400

    r = client.put("/alfa1/files/content", json={"path": "../escape.txt", "content": "x"})
    assert r.status_code == 400

    r = client.delete("/alfa1/files/content", params={"path": "../escape.txt"})
    assert r.status_code == 400


def test_file_write_read_delete_via_routes(client, tmp_path):
    client.post("/alfa1/workspace", json={"path": str(tmp_path)})

    r = client.put("/alfa1/files/content", json={"path": "note.txt", "content": "hello"})
    assert r.status_code == 200

    r = client.get("/alfa1/files/content", params={"path": "note.txt"})
    assert r.json()["content"] == "hello"

    r = client.delete("/alfa1/files/content", params={"path": "note.txt"})
    assert r.status_code == 200
    assert not (tmp_path / "note.txt").exists()


def test_chat_rejects_without_workspace(client):
    r = client.post("/alfa1/chat", json={"message": "hi"})
    assert r.status_code == 400


def test_chat_accepts_and_appends_user_message(client, tmp_path, monkeypatch):
    # Stub out the agent loop so this test never makes a real network call —
    # only the route's own bookkeeping (appending the user message, setting
    # status) is under test here; the loop itself is covered by
    # test_alfa1_agent.py.
    async def fake_run_agent_turn(conversation, model=None, on_event=None):
        return conversation
    monkeypatch.setattr(alfa1_routes, "run_agent_turn", fake_run_agent_turn)

    client.post("/alfa1/workspace", json={"path": str(tmp_path)})
    r = client.post("/alfa1/chat", json={"message": "hello agent"})
    assert r.status_code == 200
    assert r.json() == {"accepted": True, "queued": False}
    assert alfa1_routes._conversation[-1] == {"role": "user", "content": "hello agent"}


def test_chat_queues_instead_of_rejecting_when_a_turn_is_in_progress(client, tmp_path):
    client.post("/alfa1/workspace", json={"path": str(tmp_path)})
    alfa1_routes._turn_status = "working"

    r = client.post("/alfa1/chat", json={"message": "hi"})
    assert r.status_code == 200
    assert r.json() == {"accepted": True, "queued": True, "position": 1}
    assert alfa1_routes._pending_queue == [{"message": "hi", "model": None}]
    # the queued message must NOT be appended to the live conversation yet —
    # it's only added when its turn actually starts
    assert alfa1_routes._conversation == []

    r2 = client.post("/alfa1/chat", json={"message": "second"})
    assert r2.json()["position"] == 2
    assert len(alfa1_routes._pending_queue) == 2


def test_drain_queue_starts_the_next_message_once_idle(monkeypatch, client, tmp_path):
    client.post("/alfa1/workspace", json={"path": str(tmp_path)})

    async def fake_run_agent_turn(conversation, model=None, on_event=None):
        return conversation
    monkeypatch.setattr(alfa1_routes, "run_agent_turn", fake_run_agent_turn)

    alfa1_routes._pending_queue.append({"message": "queued message", "model": None})
    asyncio.run(alfa1_routes._drain_queue())

    assert alfa1_routes._pending_queue == []
    assert alfa1_routes._conversation[-1] == {"role": "user", "content": "queued message"}


def test_drain_queue_is_a_no_op_when_a_turn_is_already_working(client, tmp_path):
    client.post("/alfa1/workspace", json={"path": str(tmp_path)})
    alfa1_routes._pending_queue.append({"message": "queued message", "model": None})
    alfa1_routes._turn_status = "working"

    asyncio.run(alfa1_routes._drain_queue())

    assert alfa1_routes._pending_queue == [{"message": "queued message", "model": None}]
    assert alfa1_routes._conversation == []


def test_reset_clears_the_pending_queue_too(client, tmp_path):
    client.post("/alfa1/workspace", json={"path": str(tmp_path)})
    alfa1_routes._pending_queue.append({"message": "stale", "model": None})

    client.post("/alfa1/reset")

    assert alfa1_routes._pending_queue == []


def test_reset_clears_conversation_and_persists_empty_history(client, tmp_path):
    client.post("/alfa1/workspace", json={"path": str(tmp_path)})
    alfa1_routes._conversation.append({"role": "user", "content": "hi"})

    r = client.post("/alfa1/reset")
    assert r.status_code == 200
    assert alfa1_routes._conversation == []
    assert (tmp_path / ".alfa1" / "history.json").read_text(encoding="utf-8") == "[]"


def test_clear_all_sessions_deletes_the_history_file(client, tmp_path):
    client.post("/alfa1/workspace", json={"path": str(tmp_path)})
    alfa1_tools.save_history([{"role": "user", "content": "hi"}])
    assert (tmp_path / ".alfa1" / "history.json").exists()

    r = client.delete("/alfa1/history")
    assert r.status_code == 200
    assert r.json() == {"ok": True, "deleted": True}
    assert not (tmp_path / ".alfa1" / "history.json").exists()
    assert alfa1_routes._conversation == []


def test_clear_all_sessions_without_a_prior_history_file_is_still_ok(client, tmp_path):
    client.post("/alfa1/workspace", json={"path": str(tmp_path)})
    r = client.delete("/alfa1/history")
    assert r.status_code == 200
    assert r.json() == {"ok": True, "deleted": False}


def test_cancel_with_no_running_turn_is_a_no_op(client):
    r = client.post("/alfa1/cancel")
    assert r.status_code == 200
    assert r.json() == {"cancelled": False}
    assert alfa1_routes._turn_status == "idle"


def test_cancel_current_task_cancels_a_running_task():
    """Regression test: a run_command the agent invokes can block forever
    (e.g. a script waiting on stdin) — /alfa1/cancel must be able to force
    the turn back to idle without waiting for it to finish on its own.
    Exercised directly against the asyncio primitives (asyncio.run) rather
    than through TestClient/HTTP: TestClient does not guarantee an
    asyncio.create_task scheduled during one .post() call keeps running (or
    is even the same event loop) across a second .post() call, which made an
    HTTP-level version of this test flaky in a way that had nothing to do
    with the cancellation logic itself."""
    import asyncio

    async def scenario():
        async def hang():
            await asyncio.sleep(3600)

        task = asyncio.create_task(hang())
        alfa1_routes._current_task = task
        await asyncio.sleep(0)  # let the task actually start running
        assert not task.done()

        cancelled = await alfa1_routes._cancel_current_task()
        assert cancelled is True
        assert task.cancelled()
        assert alfa1_routes._current_task is None

    asyncio.run(scenario())
