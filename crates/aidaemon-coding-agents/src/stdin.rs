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
        let mut stdin = self.0.lock().await;
        if let Some(pipe) = stdin.as_mut()
            && (pipe.write_all(text.as_bytes()).await.is_err() || pipe.flush().await.is_err())
        {
            *stdin = None;
        }
    }

    pub(crate) async fn close(&self) {
        self.0.lock().await.take();
    }
}
