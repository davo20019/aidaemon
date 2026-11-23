//! A relayed session's state: its cipher, replay buffers and shell.

use super::*;

pub(crate) struct CryptoSession {
    pub(crate) session_id: String,
    pub(crate) cipher: Aes256Gcm,
    pub(crate) send_counter: u64,
    pub(crate) recv_counter: u64,
    pub(crate) agent: Option<String>,
    pub(crate) agent_args: Vec<String>,
    pub(crate) bootstrapped_agent: bool,
}

pub(crate) struct ReplayFrame {
    pub(crate) seq: u64,
    pub(crate) data: String,
}

pub(crate) struct ReviewStreamFrame {
    pub(crate) seq: u64,
    pub(crate) request_id: String,
    pub(crate) stream: String,
    pub(crate) data: String,
}

pub(crate) struct ActiveSession {
    pub(crate) crypto: CryptoSession,
    pub(crate) shell: ShellProcess,
    pub(crate) cwd: PathBuf,
    pub(crate) replay: VecDeque<ReplayFrame>,
    pub(crate) replay_bytes: usize,
    pub(crate) next_stdout_seq: u64,
    pub(crate) review_stream_replay: VecDeque<ReviewStreamFrame>,
    pub(crate) review_stream_replay_bytes: usize,
    pub(crate) next_review_stream_seq: u64,
    pub(crate) pending_uploads: HashMap<String, PendingUpload>,
    pub(crate) review_job: Option<ReviewJob>,
    pub(crate) last_review_progress: Option<Value>,
    pub(crate) last_review_result: Option<Value>,
}
