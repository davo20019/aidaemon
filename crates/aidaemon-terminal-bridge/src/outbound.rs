//! The two-lane outbound queue for relay frames.

use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OutboundPriority {
    High,
    Low,
}

pub(crate) struct QueuedOutboundPayload {
    pub(crate) session_id: String,
    pub(crate) payload: Value,
}

#[derive(Default)]
pub(crate) struct OutboundQueue {
    // High-priority control/result frames are drained before bulk stdout/review stream frames.
    pub(crate) high: VecDeque<QueuedOutboundPayload>,
    pub(crate) low: VecDeque<QueuedOutboundPayload>,
}

impl OutboundQueue {
    pub(crate) fn is_empty(&self) -> bool {
        self.high.is_empty() && self.low.is_empty()
    }

    pub(crate) fn enqueue(
        &mut self,
        priority: OutboundPriority,
        entry: QueuedOutboundPayload,
    ) -> Option<QueuedOutboundPayload> {
        match priority {
            OutboundPriority::High => {
                let dropped = if self.high.len() >= OUTBOUND_HIGH_QUEUE_CAP {
                    self.high.pop_front()
                } else {
                    None
                };
                self.high.push_back(entry);
                dropped
            }
            OutboundPriority::Low => {
                let dropped = if self.low.len() >= OUTBOUND_LOW_QUEUE_CAP {
                    self.low.pop_front()
                } else {
                    None
                };
                self.low.push_back(entry);
                dropped
            }
        }
    }

    pub(crate) fn pop_next(&mut self) -> Option<QueuedOutboundPayload> {
        self.high.pop_front().or_else(|| self.low.pop_front())
    }
}
