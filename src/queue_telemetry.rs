use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use chrono::{TimeZone, Utc};
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueuePressure {
    Normal,
    Warning,
    Overload,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PressureObservation {
    pub pressure: QueuePressure,
    pub entered_warning: bool,
    pub entered_overload: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct QueueSnapshot {
    pub name: String,
    pub capacity: usize,
    pub warn_depth: usize,
    pub overload_depth: usize,
    pub current_depth: usize,
    pub high_watermark: usize,
    pub received_total: u64,
    pub completed_total: u64,
    pub dropped_total: u64,
    pub failed_total: u64,
    pub overload_events_total: u64,
    pub overload_active: bool,
    pub last_overload_backlog: usize,
    pub last_overload_at: Option<String>,
    pub saturation: f64,
}

#[derive(Debug)]
struct QueueCounters {
    name: &'static str,
    capacity: usize,
    warn_depth: usize,
    overload_depth: usize,
    current_depth: AtomicUsize,
    high_watermark: AtomicUsize,
    received_total: AtomicU64,
    completed_total: AtomicU64,
    dropped_total: AtomicU64,
    failed_total: AtomicU64,
    overload_events_total: AtomicU64,
    last_overload_backlog: AtomicUsize,
    last_overload_epoch_secs: AtomicU64,
    warning_active: AtomicBool,
    overload_active: AtomicBool,
}

impl QueueCounters {
    fn new(name: &'static str, capacity: usize, warning_ratio: f32, overload_ratio: f32) -> Self {
        let safe_capacity = capacity.max(1);
        let warning_ratio = warning_ratio.clamp(0.0, 1.0);
        let overload_ratio = overload_ratio.clamp(warning_ratio, 1.0);

        let warn_depth = depth_from_ratio(safe_capacity, warning_ratio).max(1);
        let mut overload_depth = depth_from_ratio(safe_capacity, overload_ratio).max(warn_depth);
        if safe_capacity > 1 && overload_depth <= warn_depth {
            overload_depth = (warn_depth + 1).min(safe_capacity);
        }

        Self {
            name,
            capacity: safe_capacity,
            warn_depth,
            overload_depth,
            current_depth: AtomicUsize::new(0),
            high_watermark: AtomicUsize::new(0),
            received_total: AtomicU64::new(0),
            completed_total: AtomicU64::new(0),
            dropped_total: AtomicU64::new(0),
            failed_total: AtomicU64::new(0),
            overload_events_total: AtomicU64::new(0),
            last_overload_backlog: AtomicUsize::new(0),
            last_overload_epoch_secs: AtomicU64::new(0),
            warning_active: AtomicBool::new(false),
            overload_active: AtomicBool::new(false),
        }
    }

    fn observe_depth(&self, depth: usize) -> PressureObservation {
        self.current_depth.store(depth, Ordering::Relaxed);
        self.update_high_watermark(depth);

        if depth >= self.overload_depth {
            let entered_overload = !self.overload_active.swap(true, Ordering::Relaxed);
            self.warning_active.store(true, Ordering::Relaxed);
            if entered_overload {
                self.overload_events_total.fetch_add(1, Ordering::Relaxed);
                self.last_overload_backlog.store(depth, Ordering::Relaxed);
                self.last_overload_epoch_secs
                    .store(now_epoch_secs(), Ordering::Relaxed);
            }
            return PressureObservation {
                pressure: QueuePressure::Overload,
                entered_warning: false,
                entered_overload,
            };
        }

        self.overload_active.store(false, Ordering::Relaxed);

        if depth >= self.warn_depth {
            let entered_warning = !self.warning_active.swap(true, Ordering::Relaxed);
            return PressureObservation {
                pressure: QueuePressure::Warning,
                entered_warning,
                entered_overload: false,
            };
        }

        self.warning_active.store(false, Ordering::Relaxed);
        PressureObservation {
            pressure: QueuePressure::Normal,
            entered_warning: false,
            entered_overload: false,
        }
    }

    fn mark_received(&self) {
        self.received_total.fetch_add(1, Ordering::Relaxed);
    }

    fn mark_completed(&self) {
        self.completed_total.fetch_add(1, Ordering::Relaxed);
    }

    fn mark_dropped(&self, count: u64) {
        self.dropped_total.fetch_add(count, Ordering::Relaxed);
    }

    fn mark_failed(&self) {
        self.failed_total.fetch_add(1, Ordering::Relaxed);
    }

    fn snapshot(&self) -> QueueSnapshot {
        let current_depth = self.current_depth.load(Ordering::Relaxed);
        let saturation = if self.capacity == 0 {
            0.0
        } else {
            (current_depth as f64) / (self.capacity as f64)
        };
        let last_overload_epoch = self.last_overload_epoch_secs.load(Ordering::Relaxed);
        QueueSnapshot {
            name: self.name.to_string(),
            capacity: self.capacity,
            warn_depth: self.warn_depth,
            overload_depth: self.overload_depth,
            current_depth,
            high_watermark: self.high_watermark.load(Ordering::Relaxed),
            received_total: self.received_total.load(Ordering::Relaxed),
            completed_total: self.completed_total.load(Ordering::Relaxed),
            dropped_total: self.dropped_total.load(Ordering::Relaxed),
            failed_total: self.failed_total.load(Ordering::Relaxed),
            overload_events_total: self.overload_events_total.load(Ordering::Relaxed),
            overload_active: self.overload_active.load(Ordering::Relaxed),
            last_overload_backlog: self.last_overload_backlog.load(Ordering::Relaxed),
            last_overload_at: if last_overload_epoch == 0 {
                None
            } else {
                Utc.timestamp_opt(last_overload_epoch as i64, 0)
                    .single()
                    .map(|dt| dt.to_rfc3339())
            },
            saturation,
        }
    }

    fn update_high_watermark(&self, depth: usize) {
        let mut current = self.high_watermark.load(Ordering::Relaxed);
        while depth > current {
            match self.high_watermark.compare_exchange_weak(
                current,
                depth,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(observed) => current = observed,
            }
        }
    }
}

#[derive(Debug)]
pub struct QueueTelemetry {
    approval: QueueCounters,
    media: QueueCounters,
    trigger_events: QueueCounters,
}

impl QueueTelemetry {
    #[allow(dead_code)]
    pub fn new(approval_capacity: usize, media_capacity: usize, trigger_capacity: usize) -> Self {
        Self::new_with_policy(
            approval_capacity,
            media_capacity,
            trigger_capacity,
            0.75,
            0.90,
        )
    }

    pub fn new_with_policy(
        approval_capacity: usize,
        media_capacity: usize,
        trigger_capacity: usize,
        warning_ratio: f32,
        overload_ratio: f32,
    ) -> Self {
        Self {
            approval: QueueCounters::new(
                "approval",
                approval_capacity,
                warning_ratio,
                overload_ratio,
            ),
            media: QueueCounters::new("media", media_capacity, warning_ratio, overload_ratio),
            trigger_events: QueueCounters::new(
                "trigger_events",
                trigger_capacity,
                warning_ratio,
                overload_ratio,
            ),
        }
    }

    pub fn observe_approval_depth(&self, depth: usize) -> PressureObservation {
        self.approval.observe_depth(depth)
    }

    pub fn observe_media_depth(&self, depth: usize) -> PressureObservation {
        self.media.observe_depth(depth)
    }

    pub fn observe_trigger_depth(&self, depth: usize) -> PressureObservation {
        self.trigger_events.observe_depth(depth)
    }

    pub fn mark_approval_received(&self) {
        self.approval.mark_received();
    }

    pub fn mark_approval_completed(&self) {
        self.approval.mark_completed();
    }

    pub fn mark_approval_dropped(&self, count: u64) {
        self.approval.mark_dropped(count);
    }

    pub fn mark_approval_failed(&self) {
        self.approval.mark_failed();
    }

    pub fn mark_media_received(&self) {
        self.media.mark_received();
    }

    pub fn mark_media_completed(&self) {
        self.media.mark_completed();
    }

    pub fn mark_media_dropped(&self) {
        self.media.mark_dropped(1);
    }

    pub fn mark_media_failed(&self) {
        self.media.mark_failed();
    }

    pub fn mark_trigger_received(&self) {
        self.trigger_events.mark_received();
    }

    pub fn mark_trigger_completed(&self) {
        self.trigger_events.mark_completed();
    }

    pub fn mark_trigger_dropped(&self, count: u64) {
        self.trigger_events.mark_dropped(count);
    }

    pub fn mark_trigger_failed(&self) {
        self.trigger_events.mark_failed();
    }

    pub fn snapshots(&self) -> Vec<QueueSnapshot> {
        vec![
            self.approval.snapshot(),
            self.media.snapshot(),
            self.trigger_events.snapshot(),
        ]
    }
}

fn now_epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn depth_from_ratio(capacity: usize, ratio: f32) -> usize {
    ((capacity as f64) * (ratio as f64)).ceil() as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queue_tracks_high_watermark_and_transitions() {
        let telemetry = QueueTelemetry::new(10, 10, 10);
        let warn = telemetry.observe_media_depth(8);
        assert_eq!(warn.pressure, QueuePressure::Warning);
        assert!(warn.entered_warning);

        let overload = telemetry.observe_media_depth(10);
        assert_eq!(overload.pressure, QueuePressure::Overload);
        assert!(overload.entered_overload);

        let snap = telemetry
            .snapshots()
            .into_iter()
            .find(|s| s.name == "media")
            .expect("media snapshot");
        assert_eq!(snap.high_watermark, 10);
        assert!(snap.overload_events_total >= 1);
    }

    #[test]
    fn trigger_drop_counter_accumulates() {
        let telemetry = QueueTelemetry::new(16, 16, 64);
        telemetry.mark_trigger_dropped(4);
        telemetry.mark_trigger_dropped(2);

        let snap = telemetry
            .snapshots()
            .into_iter()
            .find(|s| s.name == "trigger_events")
            .expect("trigger snapshot");
        assert_eq!(snap.dropped_total, 6);
    }
}
