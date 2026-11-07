#!/usr/bin/env python3
"""README's "What holds it together", tested against the real TUI binary.

A stub server plays Ollama and scripts the model's turns, so these results don't
depend on what a model decides to do. Needs a built TUI (`make test` builds it).
"""

import json
import subprocess
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from driver import TUI_BIN, Session, run_end


def tool(name, **args):
    return {"tool_calls": [{"function": {"name": name, "arguments": args}}]}


class Stub(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, body: dict):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.reply({"models": [{"name": "stub:1"}], "version": "0.0.0"})

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path != "/api/chat":
            return self.reply({"capabilities": ["completion", "tools"]})
        server = self.server
        server.requests.append(request)
        turn = server.script.pop(0) if server.script else {"content": "Done."}
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Connection", "close")
        self.end_headers()
        if turn.get("hang"):
            # Keep generating until the client hangs up, as Ollama would.
            self.wfile.write(json.dumps({"message": {"content": "thinking "}, "done": False}).encode() + b"\n")
            self.wfile.flush()
            self.connection.settimeout(60)
            server.disconnected.set() if self.connection.recv(1) == b"" else None
            return
        line = {"message": {"content": turn.get("content", ""), **{k: v for k, v in turn.items() if k == "tool_calls"}}, "done": True}
        self.wfile.write(json.dumps(line).encode() + b"\n")


@unittest.skipUnless(TUI_BIN.is_file(), "TUI isn't built")
class Harness(unittest.TestCase):
    def start(self, script, *flags):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
        server.daemon_threads = True
        server.script, server.requests, server.disconnected = list(script), [], threading.Event()
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.ws, self.home = self.root / "ws", self.root / "home"
        self.ws.mkdir()
        self.home.mkdir()
        (self.ws / "notes.txt").write_text("hello\n")
        self.server = server
        self.session = Session(
            self.ws, self.home,
            ["--template", "notes-to-spreadsheet", "--task", "go", *flags],
            env={"AMADEUS_OLLAMA_URL": f"http://127.0.0.1:{server.server_address[1]}", "AMADEUS_MAIN_MODEL": "stub:1"},
        )
        self.addCleanup(self.session.close)
        return self.session

    def wait_kind(self, kind, n=1, timeout=15):
        found = self.session.wait_for(lambda ev: len([e for e in ev if e.get("kind") == kind]) >= n, timeout)
        self.assertTrue(found, f"no {kind} #{n} in the run log: {self.session.events()}")

    def write_pending(self, n=1):
        seen = self.session.wait_for(
            lambda ev: sum(e.get("kind") == "tool_call" and e.get("name") == "write_file" for e in ev) >= n, 15
        )
        self.assertTrue(seen, f"the write never reached the gate: {self.session.events()}")
        # Give the approval card time to arrive; the file must still not exist.
        for _ in range(5):
            self.session.pump()

    def test_reads_run_without_asking(self):
        self.start([tool("read_file", path="notes.txt")])
        self.wait_kind("run_end")
        kinds = [e["kind"] for e in self.session.events()]
        self.assertNotIn("approval", kinds)
        self.assertIn("hello", [e for e in self.session.events() if e["kind"] == "tool_result"][0]["output"])

    def test_write_waits_for_approval_and_y_applies_it(self):
        s = self.start([tool("write_file", path="out.csv", content="a,b\n")])
        self.write_pending()
        time.sleep(1)
        self.assertFalse((self.ws / "out.csv").exists(), "written before approval")
        s.send("y")
        self.wait_kind("run_end")
        self.assertEqual((self.ws / "out.csv").read_text(), "a,b\n")

    def test_decline_sends_the_reason_and_writes_nothing(self):
        s = self.start([tool("write_file", path="out.csv", content="a,b\n")])
        self.write_pending()
        s.send("n")
        s.pump(0.3)
        s.send("leave vendor empty\r")
        self.wait_kind("run_end")
        self.assertFalse((self.ws / "out.csv").exists())
        sent = json.dumps(self.server.requests[-1]["messages"])
        self.assertIn("leave vendor empty", sent)

    def test_exact_repeat_of_a_declined_call_is_refused_without_asking(self):
        call = tool("write_file", path="out.csv", content="a,b\n")
        s = self.start([call, call])
        self.write_pending()
        s.send("n")
        s.pump(0.3)
        s.send("no\r")
        self.wait_kind("run_end")
        approvals = [e for e in s.events() if e["kind"] == "approval"]
        self.assertEqual(len(approvals), 1, "asked twice about the same call")
        self.assertFalse((self.ws / "out.csv").exists())

    def test_auto_applies_to_one_task_only(self):
        call = tool("write_file", path="first.csv", content="1\n")
        s = self.start([call], "--auto")
        self.wait_kind("run_end")
        self.assertTrue((self.ws / "first.csv").exists())
        self.server.script.append(tool("write_file", path="second.csv", content="2\n"))
        time.sleep(0.5)
        s.send("again\r")
        seen = s.wait_for(lambda ev: any(e.get("headline") == "write_file second.csv" for e in ev), 15)
        self.assertTrue(seen, "the second task never proposed its write")
        for _ in range(10):
            s.pump()
        self.assertFalse((self.ws / "second.csv").exists(), "--auto leaked into the next task")

    def test_file_tools_refuse_to_leave_the_workspace(self):
        s = self.start(
            [
                tool("write_file", path="../outside.txt", content="x"),
                tool("write_file", path="/tmp/amadeus-eval-outside.txt", content="x"),
                tool("read_file", path="../home/secret.txt"),
            ],
            "--auto",
        )
        (self.home / "secret.txt").write_text("TOPSECRET")
        self.wait_kind("run_end")
        results = [e for e in s.events() if e["kind"] == "tool_result"]
        self.assertTrue(results)
        self.assertFalse(any(e.get("ok") for e in results), results)
        self.assertNotIn("TOPSECRET", json.dumps(s.events()))
        self.assertFalse((self.root / "outside.txt").exists())
        self.assertFalse(Path("/tmp/amadeus-eval-outside.txt").exists())

    def test_ctrl_c_stops_generation_on_the_server(self):
        s = self.start([{"hang": True}])
        self.wait_kind("run_start")
        for _ in range(100):
            s.pump()
            if self.server.requests:
                break
        s.pump(0.5)
        s.send("\x03")
        self.wait_kind("run_end", timeout=10)
        self.assertEqual(run_end(s.events())["status"], "cancelled")
        self.assertTrue(self.server.disconnected.wait(5), "the server never saw the disconnect")

    def test_every_step_is_logged_and_replay_reads_it(self):
        s = self.start([tool("read_file", path="notes.txt"), {"content": "All done."}])
        self.wait_kind("run_end")
        kinds = [e["kind"] for e in s.events()]
        for kind in ["run_start", "model_turn", "tool_call", "tool_result", "run_end"]:
            self.assertIn(kind, kinds)
        out = subprocess.run([TUI_BIN, "replay", str(s.log_path())], capture_output=True, text=True, timeout=10)
        self.assertEqual(out.returncode, 0)
        self.assertIn("read_file notes.txt", out.stdout)
        self.assertIn("END     done", out.stdout)


if __name__ == "__main__":
    # forkpty warns because the stub server has threads; the child execs straight away.
    unittest.main(warnings="ignore")
