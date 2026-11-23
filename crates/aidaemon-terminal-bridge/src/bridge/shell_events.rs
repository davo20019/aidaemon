//! Turning shell events into relay frames.

use crate::*;

impl TerminalBridge {
    pub(crate) fn handle_shell_event(
        &mut self,
        event: ShellEvent,
        outbound_queue: Option<&mut OutboundQueue>,
    ) -> anyhow::Result<()> {
        match event {
            ShellEvent::Output { session_id, data } => {
                let frames = self.record_stdout_chunks(&session_id, &data);
                for (_, chunk) in &frames {
                    self.broadcast_local_payload(
                        &session_id,
                        json!({"type":"stdout","data": chunk}),
                    );
                }
                if let Some(queue) = outbound_queue {
                    let mut dropped = 0usize;
                    for (seq, chunk) in frames {
                        if queue
                            .enqueue(
                                OutboundPriority::Low,
                                QueuedOutboundPayload {
                                    session_id: session_id.clone(),
                                    payload: json!({"kind":"stdout","seq":seq,"data":chunk}),
                                },
                            )
                            .is_some()
                        {
                            dropped = dropped.saturating_add(1);
                        }
                    }
                    if dropped > 0 {
                        warn!(
                            session_id = %session_id,
                            dropped,
                            "Terminal bridge low-priority outbound queue saturated; dropped stale stdout frames"
                        );
                    }
                }
            }
            ShellEvent::ReviewProgress {
                session_id,
                request_id,
                stage,
                message,
            } => {
                let payload =
                    self.record_review_progress_payload(&session_id, &request_id, &stage, &message);
                if let Some(queue) = outbound_queue
                    && queue
                        .enqueue(
                            OutboundPriority::High,
                            QueuedOutboundPayload {
                                session_id: session_id.clone(),
                                payload,
                            },
                        )
                        .is_some()
                {
                    warn!(
                        session_id = %session_id,
                        "Terminal bridge high-priority outbound queue saturated; dropped oldest progress frame"
                    );
                }
            }
            ShellEvent::ReviewResult {
                session_id,
                request_id,
                payload,
            } => {
                let payload = self.record_review_result_payload(&session_id, &request_id, payload);
                if let Some(queue) = outbound_queue
                    && queue
                        .enqueue(
                            OutboundPriority::High,
                            QueuedOutboundPayload {
                                session_id: session_id.clone(),
                                payload,
                            },
                        )
                        .is_some()
                {
                    warn!(
                        session_id = %session_id,
                        "Terminal bridge high-priority outbound queue saturated; dropped oldest result frame"
                    );
                }
            }
            ShellEvent::ReviewError {
                session_id,
                request_id,
                code,
                message,
            } => {
                let payload =
                    self.record_review_error_payload(&session_id, &request_id, &code, &message);
                if let Some(queue) = outbound_queue
                    && queue
                        .enqueue(
                            OutboundPriority::High,
                            QueuedOutboundPayload {
                                session_id: session_id.clone(),
                                payload,
                            },
                        )
                        .is_some()
                {
                    warn!(
                        session_id = %session_id,
                        "Terminal bridge high-priority outbound queue saturated; dropped oldest error frame"
                    );
                }
            }
            ShellEvent::ReviewStream {
                session_id,
                request_id,
                stream,
                data,
            } => {
                let frames =
                    self.record_review_stream_chunks(&session_id, &request_id, &stream, &data);
                if let Some(queue) = outbound_queue {
                    let mut dropped = 0usize;
                    for (seq, request_id, stream, data) in frames {
                        if queue
                            .enqueue(
                                OutboundPriority::Low,
                                QueuedOutboundPayload {
                                    session_id: session_id.clone(),
                                    payload: json!({
                                        "kind":"review_stream",
                                        "seq": seq,
                                        "request_id": request_id,
                                        "stream": stream,
                                        "data": data,
                                    }),
                                },
                            )
                            .is_some()
                        {
                            dropped = dropped.saturating_add(1);
                        }
                    }
                    if dropped > 0 {
                        warn!(
                            session_id = %session_id,
                            dropped,
                            "Terminal bridge low-priority outbound queue saturated; dropped stale review stream frames"
                        );
                    }
                }
            }
        }
        Ok(())
    }
}
