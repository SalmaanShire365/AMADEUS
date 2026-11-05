//! Child processes that can be cancelled.
//!
//! Every child gets its own process group, stdin closed, and both pipes read
//! on their own threads. Cancel or timeout kills the whole group, so a
//! `sh -c` wrapper can't leave a grandchild holding the pipes open, and a
//! command that waits on stdin gets EOF instead of hanging the UI forever.

use std::io::{self, BufRead, BufReader, Read};
use std::os::unix::process::CommandExt;
use std::process::{Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

/// Keep at most this much of each stream in memory.
const CAP: usize = 1 << 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stream {
    Out,
    Err,
}

#[derive(Debug)]
pub enum Exit {
    Status(ExitStatus),
    Cancelled,
    TimedOut,
}

#[derive(Debug)]
pub struct Output {
    pub exit: Exit,
    pub stdout: String,
    pub stderr: String,
}

impl Output {
    pub fn success(&self) -> bool {
        matches!(&self.exit, Exit::Status(s) if s.success())
    }

    pub fn describe(&self) -> String {
        match &self.exit {
            Exit::Status(s) => match s.code() {
                Some(c) => format!("exit code {c}"),
                None => s.to_string(),
            },
            Exit::Cancelled => "cancelled".into(),
            Exit::TimedOut => "timed out".into(),
        }
    }
}

fn pump<R: Read + Send + 'static>(r: R, which: Stream, tx: mpsc::Sender<(Stream, String)>) {
    thread::spawn(move || {
        let mut reader = BufReader::new(r);
        let mut buf = Vec::new();
        loop {
            buf.clear();
            match reader.read_until(b'\n', &mut buf) {
                Ok(0) | Err(_) => return,
                Ok(_) => {
                    let line = String::from_utf8_lossy(&buf)
                        .trim_end_matches(['\n', '\r'])
                        .to_string();
                    if tx.send((which, line)).is_err() {
                        return;
                    }
                }
            }
        }
    });
}

fn kill_group(pid: u32) {
    // SAFETY: plain syscall; a negative pid addresses the process group we
    // created with process_group(0), whose id equals the child's pid.
    unsafe {
        libc::kill(-(pid as i32), libc::SIGKILL);
    }
}

/// Run `cmd` to completion, calling `on_line` for each line as it arrives.
pub fn run(
    mut cmd: Command,
    cancel: &AtomicBool,
    timeout: Option<Duration>,
    mut on_line: impl FnMut(Stream, &str),
) -> io::Result<Output> {
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0);
    let mut child = cmd.spawn()?;
    let pid = child.id();

    let (tx, rx) = mpsc::channel();
    if let Some(o) = child.stdout.take() {
        pump(o, Stream::Out, tx.clone());
    }
    if let Some(e) = child.stderr.take() {
        pump(e, Stream::Err, tx.clone());
    }
    drop(tx);

    let started = Instant::now();
    let (mut stdout, mut stderr) = (String::new(), String::new());
    let mut keep = |which: Stream, line: &str, on: &mut dyn FnMut(Stream, &str)| {
        on(which, line);
        let buf = match which {
            Stream::Out => &mut stdout,
            Stream::Err => &mut stderr,
        };
        if buf.len() < CAP {
            buf.push_str(line);
            buf.push('\n');
        }
    };

    let mut forced: Option<Exit> = None;
    let status = loop {
        match rx.recv_timeout(Duration::from_millis(50)) {
            Ok((w, l)) => keep(w, &l, &mut on_line),
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                // Both pipes closed; the child is exiting or already gone.
                break child.wait()?;
            }
        }
        if forced.is_none() {
            if cancel.load(Ordering::Relaxed) {
                forced = Some(Exit::Cancelled);
            } else if timeout.is_some_and(|t| started.elapsed() > t) {
                forced = Some(Exit::TimedOut);
            }
            if forced.is_some() {
                kill_group(pid);
            }
        }
        if let Some(s) = child.try_wait()? {
            // Exited, but a detached grandchild may still hold a pipe. Drain
            // briefly, then stop waiting for it.
            let deadline = Instant::now() + Duration::from_millis(500);
            while let Ok((w, l)) =
                rx.recv_timeout(deadline.saturating_duration_since(Instant::now()))
            {
                keep(w, &l, &mut on_line);
            }
            break s;
        }
    };

    Ok(Output {
        exit: forced.unwrap_or(Exit::Status(status)),
        stdout,
        stderr,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sh(script: &str) -> Command {
        let mut c = Command::new("sh");
        c.arg("-c").arg(script);
        c
    }

    #[test]
    fn captures_both_streams_and_status() {
        let cancel = AtomicBool::new(false);
        let mut seen = Vec::new();
        let out = run(
            sh("echo hi; echo oops >&2; exit 3"),
            &cancel,
            None,
            |w, l| seen.push((w, l.to_string())),
        )
        .unwrap();
        assert_eq!(out.stdout, "hi\n");
        assert_eq!(out.stderr, "oops\n");
        assert!(!out.success());
        assert_eq!(out.describe(), "exit code 3");
        assert!(seen.contains(&(Stream::Out, "hi".into())));
    }

    #[test]
    fn stdin_is_closed_so_readers_do_not_hang() {
        let cancel = AtomicBool::new(false);
        let out = run(
            sh("cat; echo after"),
            &cancel,
            Some(Duration::from_secs(5)),
            |_, _| {},
        )
        .unwrap();
        assert!(out.success());
        assert_eq!(out.stdout, "after\n");
    }

    #[test]
    fn cancel_kills_the_whole_group_promptly() {
        let cancel = std::sync::Arc::new(AtomicBool::new(false));
        let c2 = cancel.clone();
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(200));
            c2.store(true, Ordering::Relaxed);
        });
        let t = Instant::now();
        // The grandchild `sleep` inherits the pipes; killing only `sh` would
        // leave the reader blocked for 30s.
        let out = run(sh("sleep 30 & sleep 30; wait"), &cancel, None, |_, _| {}).unwrap();
        assert!(matches!(out.exit, Exit::Cancelled));
        assert!(
            t.elapsed() < Duration::from_secs(5),
            "took {:?}",
            t.elapsed()
        );
    }

    #[test]
    fn timeout_is_reported() {
        let cancel = AtomicBool::new(false);
        let out = run(
            sh("sleep 30"),
            &cancel,
            Some(Duration::from_millis(200)),
            |_, _| {},
        )
        .unwrap();
        assert!(matches!(out.exit, Exit::TimedOut));
        assert_eq!(out.describe(), "timed out");
    }
}
