//! A running agent's stdin, shared by its host's tasks.

use std::sync::Arc;

use tokio::io::AsyncWriteExt;
use tokio::process::ChildStdin;

/// Lines written in order; closed once (the agent then exits).
#[derive(Clone)]
pub(crate) struct Stdin(Arc<tokio::sync::Mutex<Option<ChildStdin>>>);

impl Stdin {
    pub(crate) fn new(stdin: ChildStdin) -> Self {
        Self(Arc::new(tokio::sync::Mutex::new(Some(stdin))))
    }

    /// Write `text`; a pipe that fails is dropped (the agent has gone).
    pub(crate) async fn write(&self, text: &str) {
        let _ = self.try_write(text).await;
    }

    /// Write `text`, returning `true` when it was flushed to an open pipe and
    /// `false` if the pipe was already closed or failed (dropping it).
    pub(crate) async fn try_write(&self, text: &str) -> bool {
        let mut stdin = self.0.lock().await;
        if let Some(pipe) = stdin.as_mut() {
            if pipe.write_all(text.as_bytes()).await.is_ok() && pipe.flush().await.is_ok() {
                return true;
            }
            *stdin = None;
        }
        false
    }

    pub(crate) async fn close(&self) {
        self.0.lock().await.take();
    }
}
