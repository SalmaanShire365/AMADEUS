"""Drive the real amadeus-tui through a pty, the way a person would.

The TUI has no headless mode, and a headless path would measure something that
doesn't ship. Each run gets a throwaway $HOME, so run logs stay out of the
user's state dir and user template overrides can't change the result.
"""

import fcntl
import json
import os
import pty
import select
import signal
import struct
import termios
import time
from pathlib import Path

AMADEUS = Path(__file__).resolve().parent.parent
TUI_BIN = AMADEUS / "tui" / "target" / "release" / "amadeus-tui"


class Session:
    """One TUI process. Call pump() often so its output never fills the pty."""

    def __init__(self, workspace: Path, home: Path, args: list[str], env: dict | None = None):
        self.home = home
        full_env = {k: v for k, v in os.environ.items() if not k.startswith("XDG_")}
        full_env.update(
            HOME=str(home),
            AMADEUS_HOME=str(AMADEUS),
            AMADEUS_MOUSE="0",
            TERM="xterm-256color",
            **(env or {}),
        )
        self.pid, self.fd = pty.fork()
        if self.pid == 0:
            os.chdir(workspace)
            os.execve(TUI_BIN, [str(TUI_BIN), *args], full_env)
        fcntl.ioctl(self.fd, termios.TIOCSWINSZ, struct.pack("HHHH", 50, 160, 0, 0))

    def pump(self, seconds: float = 0.1) -> None:
        ready, _, _ = select.select([self.fd], [], [], seconds)
        if ready:
            try:
                os.read(self.fd, 65536)
            except OSError:
                pass

    def send(self, text: str) -> None:
        os.write(self.fd, text.encode())

    def events(self) -> list[dict]:
        logs = sorted(self.home.glob(".local/state/amadeus/projects/*/runs/*.jsonl"))
        if not logs:
            return []
        out = []
        for line in logs[-1].read_text().splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # the writer is mid-line
        return out

    def log_path(self) -> Path | None:
        logs = sorted(self.home.glob(".local/state/amadeus/projects/*/runs/*.jsonl"))
        return logs[-1] if logs else None

    def wait_for(self, predicate, timeout: float):
        """Pump until predicate(events) is truthy; return its value, or None on timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.pump()
            found = predicate(self.events())
            if found:
                return found
        return None

    def close(self) -> None:
        """Quit the TUI, then make sure the process is gone."""
        try:
            self.send("\x03")  # cancel a run still in progress
            for _ in range(10):
                self.pump()
            self.send("/quit\r")
            for _ in range(20):
                self.pump()
                if os.waitpid(self.pid, os.WNOHANG)[0]:
                    os.close(self.fd)
                    return
        except OSError:
            pass
        try:
            os.kill(self.pid, signal.SIGKILL)
            os.waitpid(self.pid, 0)
        except (ProcessLookupError, ChildProcessError):
            pass
        os.close(self.fd)


def run_end(events: list[dict]) -> dict | None:
    return next((e for e in events if e.get("kind") == "run_end"), None)


def run_task(workspace: Path, home: Path, template: str, task: str, timeout: float) -> dict:
    """Run one task with --auto. Returns status, answer, events and elapsed seconds."""
    started = time.monotonic()
    session = Session(workspace, home, ["--template", template, "--task", task, "--auto"])
    try:
        end = session.wait_for(run_end, timeout)
        events = session.events()
        log = session.log_path()
    finally:
        session.close()
    return {
        "status": end["status"] if end else "timeout",
        "answer": end.get("answer", "") if end else "",
        "events": events,
        "log": log,
        "seconds": time.monotonic() - started,
    }
