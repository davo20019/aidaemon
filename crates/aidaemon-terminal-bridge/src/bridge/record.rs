//! Recording session output for replay and fanning it out.

use crate::*;

impl TerminalBridge {
    pub(crate) fn record_stdout_chunks(
        &mut self,
        session_id: &str,
        data: &str,
    ) -> Vec<(u64, String)> {
        let Some(active) = self.sessions.get_mut(session_id) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for chunk in split_utf8_chunks(data, MAX_ENCRYPTED_PAYLOAD_BYTES) {
            let seq = active.next_stdout_seq;
            active.next_stdout_seq = active.next_stdout_seq.saturating_add(1);
            active.replay_bytes = active.replay_bytes.saturating_add(chunk.len());
            active.replay.push_back(ReplayFrame {
                seq,
                data: chunk.clone(),
            });

            while active.replay.len() > REPLAY_MAX_FRAMES || active.replay_bytes > REPLAY_MAX_BYTES
            {
                if let Some(old) = active.replay.pop_front() {
                    active.replay_bytes = active.replay_bytes.saturating_sub(old.data.len());
                } else {
                    break;
                }
            }
            out.push((seq, chunk));
        }
        out
    }

    pub(crate) fn record_review_stream_chunks(
        &mut self,
        session_id: &str,
        request_id: &str,
        stream: &str,
        data: &str,
    ) -> Vec<(u64, String, String, String)> {
        let Some(active) = self.sessions.get_mut(session_id) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for chunk in split_utf8_chunks(data, REVIEW_STREAM_CHUNK_MAX_BYTES) {
            let seq = active.next_review_stream_seq;
            active.next_review_stream_seq = active.next_review_stream_seq.saturating_add(1);
            active.review_stream_replay_bytes = active
                .review_stream_replay_bytes
                .saturating_add(chunk.len());
            active.review_stream_replay.push_back(ReviewStreamFrame {
                seq,
                request_id: request_id.to_string(),
                stream: stream.to_string(),
                data: chunk.clone(),
            });

            while active.review_stream_replay.len() > REVIEW_STREAM_REPLAY_MAX_FRAMES
                || active.review_stream_replay_bytes > REVIEW_STREAM_REPLAY_MAX_BYTES
            {
                if let Some(old) = active.review_stream_replay.pop_front() {
                    active.review_stream_replay_bytes = active
                        .review_stream_replay_bytes
                        .saturating_sub(old.data.len());
                } else {
                    break;
                }
            }

            out.push((seq, request_id.to_string(), stream.to_string(), chunk));
        }
        out
    }

    pub(crate) fn record_review_progress_payload(
        &mut self,
        session_id: &str,
        request_id: &str,
        stage: &str,
        message: &str,
    ) -> Value {
        let payload = json!({
            "kind":"review_progress",
            "request_id": request_id,
            "stage": stage,
            "message": truncate_with_note(message, 600),
        });
        if let Some(active) = self.sessions.get_mut(session_id) {
            active.last_review_progress = Some(payload.clone());
        }
        payload
    }

    pub(crate) fn record_review_error_payload(
        &mut self,
        session_id: &str,
        request_id: &str,
        code: &str,
        message: &str,
    ) -> Value {
        let payload = json!({
            "kind":"review_error",
            "request_id": request_id,
            "code": code,
            "message": truncate_with_note(message, 1000),
        });
        if let Some(active) = self.sessions.get_mut(session_id) {
            active.last_review_progress = Some(payload.clone());
            if active
                .review_job
                .as_ref()
                .map(|job| job.request_id == request_id)
                .unwrap_or(false)
            {
                active.review_job = None;
            }
        }
        payload
    }

    pub(crate) fn record_review_result_payload(
        &mut self,
        session_id: &str,
        request_id: &str,
        payload: Value,
    ) -> Value {
        let mut normalized = payload;
        if normalized.get("kind").and_then(|v| v.as_str()) != Some("review_result") {
            normalized["kind"] = Value::String("review_result".to_string());
        }
        normalized["request_id"] = Value::String(request_id.to_string());
        if let Some(active) = self.sessions.get_mut(session_id) {
            active.last_review_result = Some(normalized.clone());
            active.last_review_progress = None;
            if active
                .review_job
                .as_ref()
                .map(|job| job.request_id == request_id)
                .unwrap_or(false)
            {
                active.review_job = None;
            }
        }
        normalized
    }

    pub(crate) fn drain_shell_events(&mut self, max_events: usize) {
        let mut drained = 0usize;
        loop {
            if drained >= max_events {
                break;
            }
            match self.shell_events_rx.try_recv() {
                Ok(event) => {
                    if let Err(err) = self.handle_shell_event(event, None) {
                        warn!(error = %err, "Failed to process buffered PTY event");
                    }
                    drained += 1;
                }
                Err(tokio::sync::mpsc::error::TryRecvError::Empty) => break,
                Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => break,
            }
        }
        if drained > 0 {
            debug!(
                drained,
                "Drained buffered PTY output while transport was unavailable"
            );
        }
    }

    pub(crate) async fn drain_local_events(&mut self, max_events: usize) {
        let mut drained = 0usize;
        while drained < max_events {
            match self.local_events_rx.try_recv() {
                Ok(event) => {
                    if let Err(err) = self.handle_local_attach_event(event).await {
                        warn!(error = %err, "Local attach event handling failed");
                    }
                    drained = drained.saturating_add(1);
                }
                Err(tokio::sync::mpsc::error::TryRecvError::Empty) => break,
                Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => break,
            }
        }
    }

    pub(crate) fn broadcast_local_payload(&mut self, session_id: &str, payload: Value) {
        let mut stale = Vec::new();
        for (client_id, client) in &self.local_clients {
            if client.session_id != session_id {
                continue;
            }
            if client.tx.send(payload.clone()).is_err() {
                stale.push(*client_id);
            }
        }
        for client_id in stale {
            self.local_clients.remove(&client_id);
        }
    }
}
