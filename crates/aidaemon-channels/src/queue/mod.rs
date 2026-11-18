//! Bounded queues between the daemon's producers and the channel hub:
//! their settings ([`QueuePolicyConfig`]), pressure telemetry and overload
//! policy. Internal to aidaemon, no stability promise.

pub mod policy;
pub mod telemetry;

use serde::Deserialize;

fn default_queue_capacity_approval() -> usize {
    16
}

fn default_queue_capacity_media() -> usize {
    16
}

fn default_queue_capacity_trigger_events() -> usize {
    64
}

fn default_queue_warning_ratio() -> f32 {
    0.75
}

fn default_queue_overload_ratio() -> f32 {
    0.90
}

fn default_queue_adaptive_shedding() -> bool {
    true
}

fn default_queue_fair_trigger_sessions() -> bool {
    true
}

fn default_queue_fair_window_secs() -> u64 {
    60
}

fn default_queue_fair_max_events_per_session() -> u32 {
    4
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
pub struct QueueLanePolicyConfig {
    #[serde(default = "default_queue_adaptive_shedding")]
    pub adaptive_shedding: bool,
    #[serde(default = "default_queue_fair_trigger_sessions")]
    pub fair_sessions: bool,
    #[serde(default = "default_queue_fair_window_secs")]
    pub fair_session_window_secs: u64,
    #[serde(default = "default_queue_fair_max_events_per_session")]
    pub fair_max_events_per_session: u32,
}

impl Default for QueueLanePolicyConfig {
    fn default() -> Self {
        Self {
            adaptive_shedding: default_queue_adaptive_shedding(),
            fair_sessions: default_queue_fair_trigger_sessions(),
            fair_session_window_secs: default_queue_fair_window_secs(),
            fair_max_events_per_session: default_queue_fair_max_events_per_session(),
        }
    }
}

impl QueueLanePolicyConfig {
    pub fn normalized(&self) -> Self {
        let mut lane = self.clone();
        lane.fair_session_window_secs = lane.fair_session_window_secs.max(1);
        lane.fair_max_events_per_session = lane.fair_max_events_per_session.max(1);
        lane
    }
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct QueueLanePoliciesConfig {
    #[serde(default)]
    pub approval: QueueLanePolicyConfig,
    #[serde(default)]
    pub media: QueueLanePolicyConfig,
    #[serde(default)]
    pub trigger: QueueLanePolicyConfig,
}

impl QueueLanePoliciesConfig {
    pub fn normalized(&self) -> Self {
        Self {
            approval: self.approval.normalized(),
            media: self.media.normalized(),
            trigger: self.trigger.normalized(),
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct QueuePolicyConfig {
    #[serde(default = "default_queue_capacity_approval")]
    pub approval_capacity: usize,
    #[serde(default = "default_queue_capacity_media")]
    pub media_capacity: usize,
    #[serde(default = "default_queue_capacity_trigger_events")]
    pub trigger_event_capacity: usize,
    #[serde(default = "default_queue_warning_ratio")]
    pub warning_ratio: f32,
    #[serde(default = "default_queue_overload_ratio")]
    pub overload_ratio: f32,
    #[serde(default)]
    pub lanes: QueueLanePoliciesConfig,

    // Legacy shared overload knobs retained for backward compatibility.
    #[serde(default)]
    pub adaptive_shedding: Option<bool>,
    #[serde(default)]
    pub fair_trigger_sessions: Option<bool>,
    #[serde(default)]
    pub fair_trigger_session_window_secs: Option<u64>,
    #[serde(default)]
    pub fair_trigger_max_events_per_session: Option<u32>,
}

impl Default for QueuePolicyConfig {
    fn default() -> Self {
        Self {
            approval_capacity: default_queue_capacity_approval(),
            media_capacity: default_queue_capacity_media(),
            trigger_event_capacity: default_queue_capacity_trigger_events(),
            warning_ratio: default_queue_warning_ratio(),
            overload_ratio: default_queue_overload_ratio(),
            lanes: QueueLanePoliciesConfig::default(),
            adaptive_shedding: None,
            fair_trigger_sessions: None,
            fair_trigger_session_window_secs: None,
            fair_trigger_max_events_per_session: None,
        }
    }
}

impl QueuePolicyConfig {
    pub fn normalized(&self) -> Self {
        let mut policy = self.clone();
        policy.approval_capacity = policy.approval_capacity.max(1);
        policy.media_capacity = policy.media_capacity.max(1);
        policy.trigger_event_capacity = policy.trigger_event_capacity.max(1);
        policy.warning_ratio = if policy.warning_ratio.is_finite() {
            policy.warning_ratio.clamp(0.0, 1.0)
        } else {
            default_queue_warning_ratio()
        };
        policy.overload_ratio = if policy.overload_ratio.is_finite() {
            policy.overload_ratio.clamp(policy.warning_ratio, 1.0)
        } else {
            default_queue_overload_ratio().clamp(policy.warning_ratio, 1.0)
        };

        let default_lane = QueueLanePolicyConfig::default().normalized();
        let legacy_lane = QueueLanePolicyConfig {
            adaptive_shedding: policy
                .adaptive_shedding
                .unwrap_or(default_queue_adaptive_shedding()),
            fair_sessions: policy
                .fair_trigger_sessions
                .unwrap_or(default_queue_fair_trigger_sessions()),
            fair_session_window_secs: policy
                .fair_trigger_session_window_secs
                .unwrap_or(default_queue_fair_window_secs()),
            fair_max_events_per_session: policy
                .fair_trigger_max_events_per_session
                .unwrap_or(default_queue_fair_max_events_per_session()),
        }
        .normalized();

        policy.lanes = policy.lanes.normalized();
        if policy.lanes.approval == default_lane {
            policy.lanes.approval = legacy_lane.clone();
        }
        if policy.lanes.media == default_lane {
            policy.lanes.media = legacy_lane.clone();
        }
        if policy.lanes.trigger == default_lane {
            policy.lanes.trigger = legacy_lane;
        }

        policy
    }
}
