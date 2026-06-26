use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::sync::{mpsc, Mutex};

use crate::channels::attachments::save_tool_observation_image;
use crate::config::{ComputerUseConfig, ProviderKind, VisionConfig};
use crate::events::{Event, EventStore, EventType};
use crate::tools::ApprovalBroker;
use crate::traits::{
    Tool, ToolCallMetadata, ToolCallOutcome, ToolCallSemantics, ToolCapabilities, ToolRole,
};
use crate::types::{MediaKind, MediaMessage, StatusUpdate};

mod approvals;
pub mod cache;
pub mod capability;
#[allow(clippy::too_many_arguments)]
pub mod harness;
#[cfg(all(target_os = "macos", feature = "computer_use-macos"))]
pub mod macos;
mod mock;
pub mod pin_registry;
mod policy;
mod telemetry;
pub mod types;

#[cfg(test)]
mod tests;

use approvals::ApprovalState;
use cache::SnapshotCache;
use capability::pick_capable_model;
use harness::{ComputerHarness, HarnessRequestContext};
use pin_registry::ComputerUsePinRegistry;
use policy::{classify_target, is_prohibited_bundle, ActionClass, ComputerActionKind};
use telemetry::{ActionLog, ActionRecord, ElementTarget, MutationBudget, SessionTelemetry};
use types::{format_condensed_refresh, format_full_tree, AppSnapshot, IndexedElement};

const TOOL_NAME: &str = "computer_use";

#[derive(Default)]
struct PendingActionMeta {
    mutation_budget: Option<MutationBudget>,
    element_target: ElementTarget,
    click_method: Option<String>,
}

pub struct ComputerUseTool {
    config: ComputerUseConfig,
    vision: VisionConfig,
    inbox_dir: PathBuf,
    harness: Arc<dyn ComputerHarness>,
    cache: Arc<Mutex<SnapshotCache>>,
    approval: ApprovalBroker,
    approval_state: ApprovalState,
    pins: ComputerUsePinRegistry,
    media_tx: mpsc::Sender<MediaMessage>,
    session_telemetry: SessionTelemetry,
    pending_meta: tokio::sync::Mutex<PendingActionMeta>,
    /// Per-task signature of the last element-targeted mutation, to flag an
    /// immediate exact repeat (e.g. clicking a Like/toggle twice) that could
    /// undo the first action. Cleared by any other action (observation included).
    last_mutation_sig: tokio::sync::Mutex<std::collections::HashMap<String, String>>,
    /// Optional events store: when present, every action is also persisted as a
    /// structured event so the full trajectory (incl. click_method/outcome/
    /// timing) is auditable from the DB, not only from stdout logs.
    events: Option<Arc<EventStore>>,
}

const DUPLICATE_MUTATION_CAUTION: &str = "\n[NOTE] This repeats your previous action on the same \
target with no get_app_state in between. If the first one already worked, doing it again may UNDO \
it (e.g. toggle a Like off). Call get_app_state to check the current state before repeating.";

const NO_VISIBLE_CHANGE_NOTICE: &str = "\n[VERIFY] The accessibility state did not change after \
this click — it may NOT have taken effect (e.g. the click hit the wrong sub-element, the control \
needs a hover/second step, or the page had not updated). Do NOT assume success: re-read with \
get_app_state (or a screenshot) and confirm the intended change before reporting done.";

impl ComputerUseTool {
    pub fn new(
        config: ComputerUseConfig,
        vision: VisionConfig,
        inbox_dir: PathBuf,
        approval: ApprovalBroker,
        media_tx: mpsc::Sender<MediaMessage>,
        events: Option<Arc<EventStore>>,
    ) -> Self {
        #[cfg(all(not(test), target_os = "macos", feature = "computer_use-macos"))]
        let harness: Arc<dyn ComputerHarness> = Arc::new(macos::MacOsHarness::new(config.clone()));
        #[cfg(any(test, not(all(target_os = "macos", feature = "computer_use-macos"))))]
        let harness: Arc<dyn ComputerHarness> = Arc::new(mock::MockHarness::new(config.clone()));

        Self {
            config,
            vision,
            inbox_dir,
            harness,
            cache: Arc::new(Mutex::new(SnapshotCache::default())),
            approval,
            approval_state: ApprovalState::new(),
            pins: ComputerUsePinRegistry::shared(),
            media_tx,
            session_telemetry: SessionTelemetry::default(),
            pending_meta: tokio::sync::Mutex::new(PendingActionMeta::default()),
            last_mutation_sig: tokio::sync::Mutex::new(std::collections::HashMap::new()),
            events,
        }
    }

    /// Persist one action as a structured DecisionPoint event (decision_type
    /// "computer_use_action") so the full trajectory — including click_method,
    /// outcome and timing that stdout-only `log_action` carries — is queryable
    /// from the events store. No-op when no events store is wired in (tests).
    #[allow(clippy::too_many_arguments)]
    async fn record_action_event(
        &self,
        session_id: &str,
        task_id: &str,
        action: &str,
        app: &str,
        success: bool,
        error: Option<&str>,
        click_method: Option<&str>,
        duration_ms: u64,
        is_mutation: bool,
        target: &ElementTarget,
    ) {
        let Some(events) = &self.events else {
            return;
        };
        if session_id.is_empty() {
            return;
        }
        let data = json!({
            "decision_type": "computer_use_action",
            "name": TOOL_NAME,
            "task_id": task_id,
            "action": action,
            "app": app,
            "outcome": if success { "ok" } else { "error" },
            "error": error,
            "click_method": click_method,
            "duration_ms": duration_ms,
            "is_mutation": is_mutation,
            "target": {
                "index": target.index,
                "title": target.title,
                "role": target.role,
            },
        });
        if let Err(e) = events
            .append(Event::new(session_id, EventType::DecisionPoint, data))
            .await
        {
            tracing::warn!(error = %e, "failed to persist computer_use action event");
        }
    }

    /// Flag (don't block) an exact repeat of an element-targeted mutation with no
    /// observation in between — the pattern that can silently undo a toggle. Any
    /// other action (or an observation) resets the streak for the task.
    async fn duplicate_mutation_caution(
        &self,
        task_id: &str,
        is_mutation: bool,
        element_target: &ElementTarget,
        args: &Value,
    ) -> bool {
        let mut last = self.last_mutation_sig.lock().await;
        // Only element-targeted mutations are toggle-prone; everything else
        // (observations, press_key, page scrolls) resets the streak.
        if !is_mutation || element_target.index.is_none() {
            last.remove(task_id);
            return false;
        }
        let action = args.get("action").and_then(|v| v.as_str()).unwrap_or("");
        let value = args.get("value").and_then(|v| v.as_str()).unwrap_or("");
        let text = args.get("text").and_then(|v| v.as_str()).unwrap_or("");
        let sig = format!(
            "{action}|{:?}|{}|{value}|{text}",
            element_target.index,
            element_target.title.as_deref().unwrap_or("")
        );
        let repeat = last.get(task_id) == Some(&sig);
        last.insert(task_id.to_string(), sig);
        repeat
    }

    async fn clear_pending_meta(&self) {
        *self.pending_meta.lock().await = PendingActionMeta::default();
    }

    async fn set_element_target(&self, element: Option<&IndexedElement>, index: Option<u32>) {
        let mut meta = self.pending_meta.lock().await;
        meta.element_target = element_target_from(element, index);
    }

    async fn set_click_method(&self, method: &'static str) {
        self.pending_meta.lock().await.click_method = Some(method.to_string());
    }

    async fn take_pending_meta(&self) -> PendingActionMeta {
        std::mem::take(&mut *self.pending_meta.lock().await)
    }

    async fn resolve_element_target(
        &self,
        args: &Value,
        ctx: &HarnessRequestContext,
        bundle_id: &str,
    ) -> ElementTarget {
        let generation = match args.get("snapshot_generation").and_then(|v| v.as_u64()) {
            Some(g) => g,
            None => return ElementTarget::default(),
        };
        let index = optional_u32(args, "element_index");
        let cache = self.cache.lock().await;
        let key = self.snapshot_key(bundle_id, ctx);
        match index {
            Some(index) => cache
                .element_by_index(&key, generation, index)
                .ok()
                .map(|el| element_target_from(Some(el), Some(index)))
                .unwrap_or_else(|| element_target_from(None, Some(index))),
            None => ElementTarget::default(),
        }
    }

    fn parse_provider_kind(args: &Value) -> ProviderKind {
        args.get("_provider_kind")
            .and_then(|v| v.as_str())
            .and_then(|raw| match raw {
                "OpenaiCompatible" => Some(ProviderKind::OpenaiCompatible),
                "Anthropic" => Some(ProviderKind::Anthropic),
                "GoogleGenai" => Some(ProviderKind::GoogleGenai),
                "XaiNative" => Some(ProviderKind::XaiNative),
                _ => None,
            })
            .unwrap_or(ProviderKind::OpenaiCompatible)
    }

    fn parse_model_chain(args: &Value, current_model: &str) -> Vec<String> {
        if let Some(chain) = args.get("_model_chain").and_then(|v| v.as_array()) {
            let models: Vec<String> = chain
                .iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect();
            if !models.is_empty() {
                return models;
            }
        }
        vec![current_model.to_string()]
    }

    async fn ensure_model_pin(
        &self,
        args: &Value,
        ctx: &HarnessRequestContext,
    ) -> Result<(), String> {
        if self.pins.get(&ctx.task_id).await.is_some() {
            return Ok(());
        }
        let current_model = args.get("_model").and_then(|v| v.as_str()).unwrap_or("");
        let chain = Self::parse_model_chain(args, current_model);
        let provider_kind = Self::parse_provider_kind(args);
        let capable = pick_capable_model(&chain, &self.vision, provider_kind)?;
        self.pins.pin(ctx.task_id.clone(), capable).await;
        Ok(())
    }

    async fn ensure_action_approvals(
        &self,
        ctx: &HarnessRequestContext,
        action: ComputerActionKind,
        bundle_id: Option<&str>,
        app_name: Option<&str>,
        action_class: ActionClass,
        summary: Option<&str>,
    ) -> Result<(), String> {
        let observation = matches!(
            action,
            ComputerActionKind::GetAppState
                | ComputerActionKind::ListApps
                | ComputerActionKind::Screenshot
        );
        // One combined per-app approval (inspect + control) — the user is asked
        // once per app, not once per scope or per session.
        if let (Some(bundle_id), Some(app_name)) = (bundle_id, app_name) {
            if is_prohibited_bundle(bundle_id) {
                return Err(format!(
                    "App '{app_name}' ({bundle_id}) is blocked by policy"
                ));
            }
            self.approval_state
                .ensure_app(
                    &self.approval,
                    &self.config,
                    &ctx.session_id,
                    &ctx.task_id,
                    bundle_id,
                    app_name,
                )
                .await?;
        }

        if action_class == ActionClass::Consequential {
            let label = summary.unwrap_or("consequential desktop action");
            self.approval_state
                .ensure_consequential(&self.approval, &ctx.session_id, &ctx.task_id, label)
                .await?;
        }

        if !observation {
            let budget = self
                .approval_state
                .record_mutating_action(&ctx.task_id, &self.config)
                .await?;
            self.pending_meta.lock().await.mutation_budget = Some(budget);
        }
        Ok(())
    }

    fn parse_context(args: &Value) -> Result<HarnessRequestContext, String> {
        let session_id = args
            .get("_session_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if session_id.is_empty() {
            return Err("computer_use actions require a session id".to_string());
        }
        let task_id = args
            .get("_task_id")
            .and_then(|v| v.as_str())
            .unwrap_or("default")
            .to_string();
        Ok(HarnessRequestContext {
            task_id,
            session_id,
        })
    }

    async fn ensure_session_ready(
        &self,
        ctx: &HarnessRequestContext,
        args: &Value,
        _action: ComputerActionKind,
    ) -> Result<(), String> {
        self.harness.check_permissions()?;
        if !self.vision.enabled {
            return Err(
                "Vision is disabled in config — computer_use requires vision-capable models"
                    .to_string(),
            );
        }
        self.ensure_model_pin(args, ctx).await?;
        Ok(())
    }

    async fn build_outcome(
        &self,
        text: String,
        snapshot: Option<&AppSnapshot>,
        session_id: &str,
    ) -> Result<ToolCallOutcome, String> {
        let mut metadata = ToolCallMetadata::default();
        if let Some(snapshot) = snapshot {
            if !snapshot.png.is_empty() {
                let attachment = save_tool_observation_image(
                    &self.inbox_dir,
                    &snapshot.png,
                    "screenshot.png",
                    "image/png",
                    TOOL_NAME,
                )
                .map_err(|e| format!("Screenshot captured but failed to save: {e}"))?;
                metadata.attachments.push(attachment);

                if self.config.mirror_screenshots_to_channel {
                    let _ = self
                        .media_tx
                        .send(MediaMessage {
                            session_id: session_id.to_string(),
                            kind: MediaKind::Photo {
                                data: snapshot.png.clone(),
                            },
                            caption: format!("Screenshot of {}", snapshot.app_name),
                            result_tx: None,
                        })
                        .await;
                }
            }
        }
        Ok(ToolCallOutcome {
            output: text,
            metadata,
        })
    }

    async fn dispatch(&self, args: &Value) -> Result<ToolCallOutcome, String> {
        let action_raw = args
            .get("action")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Missing required parameter: action".to_string())?;
        let action = ComputerActionKind::parse(action_raw)?;

        if action == ComputerActionKind::ListApps {
            self.harness.check_permissions()?;
            let apps = self.harness.list_apps().await?;
            let mut lines = String::from("Running apps:\n");
            for app in apps {
                lines.push_str(&format!(
                    "- {} ({}) pid={}\n",
                    app.name, app.bundle_id, app.pid
                ));
            }
            return Ok(ToolCallOutcome::from_output(lines));
        }

        let ctx = Self::parse_context(args)?;
        self.ensure_session_ready(&ctx, args, action).await?;

        match action {
            ComputerActionKind::GetAppState => {
                let app = required_app(args)?;
                let resolved = self.resolve_app(&app).await?;
                self.ensure_action_approvals(
                    &ctx,
                    action,
                    Some(&resolved.bundle_id),
                    Some(&resolved.name),
                    ActionClass::Observation,
                    None,
                )
                .await?;
                let mut cache = self.cache.lock().await;
                let snapshot = self.harness.get_app_state(&app, &ctx, &mut cache).await?;
                let text = format_full_tree(&snapshot);
                self.build_outcome(text, Some(&snapshot), &ctx.session_id)
                    .await
            }
            ComputerActionKind::Screenshot => {
                // A screenshot is a pure observation: capture the image only and
                // deliver it to the chat (and attach it for the model). It must
                // NOT advance the element generation, and it reports the current
                // cached snapshot_generation so the model keeps using the right
                // value — silently bumping it here used to make every
                // post-screenshot mutation fail as stale.
                let app = required_app(args)?;
                let resolved = self.resolve_app(&app).await?;
                self.ensure_action_approvals(
                    &ctx,
                    action,
                    Some(&resolved.bundle_id),
                    Some(&resolved.name),
                    ActionClass::Observation,
                    None,
                )
                .await?;
                let snapshot = self.harness.capture_screenshot(&app).await?;
                if !snapshot.png.is_empty() {
                    let _ = self
                        .media_tx
                        .send(MediaMessage {
                            session_id: ctx.session_id.clone(),
                            kind: MediaKind::Photo {
                                data: snapshot.png.clone(),
                            },
                            caption: format!("Screenshot of {}", snapshot.app_name),
                            result_tx: None,
                        })
                        .await;
                }
                let generation_hint = {
                    let cache = self.cache.lock().await;
                    let key = self.snapshot_key(&resolved.bundle_id, &ctx);
                    match cache.current_generation(&key) {
                        Some(g) => format!(
                            " Current snapshot_generation={g} — reuse it for your next click/type \
                             (a screenshot does not change it); call get_app_state only if the UI \
                             has changed."
                        ),
                        None => " No snapshot captured yet — call get_app_state to get element \
                                 indices before any click/type."
                            .to_string(),
                    }
                };
                let text = format!(
                    "Screenshot of {} ({}) captured and sent to the chat.{}",
                    snapshot.app_name, snapshot.bundle_id, generation_hint
                );
                self.build_outcome(text, Some(&snapshot), &ctx.session_id)
                    .await
            }
            ComputerActionKind::ActivateApp => {
                let app = required_app(args)?;
                // Optional: activation has no element target, and it is often
                // the first action on an app — before any get_app_state.
                let generation = args.get("snapshot_generation").and_then(|v| v.as_u64());
                // "Activate this app" naturally means "open it" when it isn't
                // running yet — fall back to launching so the model doesn't dead
                // end on a not-running error.
                let resolved = match self.resolve_app(&app).await {
                    Ok(found) => found,
                    Err(_) => self.harness.launch_app(&app).await?,
                };
                self.ensure_action_approvals(
                    &ctx,
                    action,
                    Some(&resolved.bundle_id),
                    Some(&resolved.name),
                    ActionClass::LocalMutation,
                    None,
                )
                .await?;
                let mut cache = self.cache.lock().await;
                let snapshot = self
                    .harness
                    .activate_app(&app, generation, &ctx, &mut cache)
                    .await?;
                let text = format_condensed_refresh(&snapshot, None);
                self.build_outcome(text, Some(&snapshot), &ctx.session_id)
                    .await
            }
            ComputerActionKind::Click => {
                let app = required_app(args)?;
                let generation = required_generation(args)?;
                let x = optional_f64(args, "x");
                let y = optional_f64(args, "y");
                let resolved = self.resolve_app(&app).await?;
                let bundle_id = resolved.bundle_id.clone();
                let mut cache = self.cache.lock().await;
                let key = self.snapshot_key(&bundle_id, &ctx);
                let element_index = resolve_target_index(args, &cache, &key, generation)?;
                let mut action_class = ActionClass::LocalMutation;
                let mut summary = None;
                // Capture the targeted element's identity so we can verify the
                // click actually changed *it* (a whole-snapshot diff is useless on
                // a live page where unrelated content churns every frame).
                let mut before_target: Option<(u32, String, String)> = None;
                if let Some(index) = element_index {
                    let element = cache.element_by_index(&key, generation, index)?.clone();
                    before_target = Some((index, element.role.clone(), element.title.clone()));
                    self.set_element_target(Some(&element), Some(index)).await;
                    action_class = classify_target(action, Some(&element), None);
                    if action_class == ActionClass::Prohibited {
                        return Err("Target element is prohibited".to_string());
                    }
                    if action_class == ActionClass::Consequential {
                        summary = Some(format!("Click '{}'", element.title));
                    }
                }
                self.ensure_action_approvals(
                    &ctx,
                    action,
                    Some(&bundle_id),
                    Some(&resolved.name),
                    action_class,
                    summary.as_deref(),
                )
                .await?;
                let (snapshot, focus, click_method) = self
                    .harness
                    .click(&app, generation, element_index, x, y, &ctx, &mut cache)
                    .await?;
                self.set_click_method(click_method).await;
                drop(cache);
                // Element-specific verification: if the targeted element is still
                // present unchanged (same role + title) after the click, the click
                // had no effect on it — flag it instead of implying success.
                let target_unchanged = before_target.as_ref().is_some_and(|(i, role, title)| {
                    snapshot
                        .elements
                        .iter()
                        .any(|e| e.index == *i && &e.role == role && &e.title == title)
                });
                let mut text = format_condensed_refresh(&snapshot, focus);
                if target_unchanged {
                    text.push_str(NO_VISIBLE_CHANGE_NOTICE);
                }
                self.build_outcome(text, Some(&snapshot), &ctx.session_id)
                    .await
            }
            ComputerActionKind::TypeText => {
                let app = required_app(args)?;
                let generation = required_generation(args)?;
                let text = required_str(args, "text")?;
                let resolved = self.resolve_app(&app).await?;
                let class = classify_target(action, None, Some(&text));
                if class == ActionClass::Prohibited {
                    return Err("Typed content is prohibited".to_string());
                }
                self.ensure_action_approvals(
                    &ctx,
                    action,
                    Some(&resolved.bundle_id),
                    Some(&resolved.name),
                    class,
                    Some(&format!("Type text into {}", resolved.name)),
                )
                .await?;
                let mut cache = self.cache.lock().await;
                let key = self.snapshot_key(&resolved.bundle_id, &ctx);
                let element_index = resolve_target_index(args, &cache, &key, generation)?;
                let snapshot = self
                    .harness
                    .type_text(&app, generation, element_index, &text, &ctx, &mut cache)
                    .await?;
                let body = format_condensed_refresh(&snapshot, None);
                self.build_outcome(body, Some(&snapshot), &ctx.session_id)
                    .await
            }
            ComputerActionKind::PressKey => {
                let app = required_app(args)?;
                let generation = required_generation(args)?;
                let key = required_str(args, "key")?;
                let resolved = self.resolve_app(&app).await?;
                self.ensure_action_approvals(
                    &ctx,
                    action,
                    Some(&resolved.bundle_id),
                    Some(&resolved.name),
                    ActionClass::LocalMutation,
                    Some(&format!("Press key {key} in {}", resolved.name)),
                )
                .await?;
                let mut cache = self.cache.lock().await;
                let snapshot = self
                    .harness
                    .press_key(&app, generation, &key, &ctx, &mut cache)
                    .await?;
                let body = format_condensed_refresh(&snapshot, None);
                self.build_outcome(body, Some(&snapshot), &ctx.session_id)
                    .await
            }
            ComputerActionKind::Scroll => {
                let app = required_app(args)?;
                let generation = required_generation(args)?;
                let direction = required_str(args, "direction")?;
                let pages = args.get("pages").and_then(|v| v.as_f64()).unwrap_or(1.0);
                let resolved = self.resolve_app(&app).await?;
                // Element target is optional: with no element, scroll the focused
                // window/page — the common "scroll the feed" case.
                let element = {
                    let cache = self.cache.lock().await;
                    let key = self.snapshot_key(&resolved.bundle_id, &ctx);
                    match resolve_target_index(args, &cache, &key, generation)? {
                        Some(index) => Some((
                            index,
                            cache.element_by_index(&key, generation, index)?.clone(),
                        )),
                        None => None,
                    }
                };
                let element_index = element.as_ref().map(|(i, _)| *i);
                if let Some((index, el)) = &element {
                    self.set_element_target(Some(el), Some(*index)).await;
                }
                self.ensure_action_approvals(
                    &ctx,
                    action,
                    Some(&resolved.bundle_id),
                    Some(&resolved.name),
                    ActionClass::LocalMutation,
                    None,
                )
                .await?;
                let mut cache = self.cache.lock().await;
                let (snapshot, focus) = self
                    .harness
                    .scroll(
                        &app,
                        generation,
                        element_index,
                        &direction,
                        pages,
                        &ctx,
                        &mut cache,
                    )
                    .await?;
                let body = format_condensed_refresh(&snapshot, focus);
                self.build_outcome(body, Some(&snapshot), &ctx.session_id)
                    .await
            }
            ComputerActionKind::SetValue => {
                let app = required_app(args)?;
                let generation = required_generation(args)?;
                let value = required_str(args, "value")?;
                let resolved = self.resolve_app(&app).await?;
                let bundle_id = resolved.bundle_id.clone();
                let mut cache = self.cache.lock().await;
                let key = self.snapshot_key(&bundle_id, &ctx);
                let element_index = resolve_target_index(args, &cache, &key, generation)?
                    .ok_or_else(|| {
                        "set_value requires element_index or element_title/element_role".to_string()
                    })?;
                let element = cache
                    .element_by_index(&key, generation, element_index)?
                    .clone();
                self.set_element_target(Some(&element), Some(element_index))
                    .await;
                let class = classify_target(action, Some(&element), Some(&value));
                if class == ActionClass::Prohibited {
                    return Err("Target element or value is prohibited".to_string());
                }
                self.ensure_action_approvals(
                    &ctx,
                    action,
                    Some(&bundle_id),
                    Some(&resolved.name),
                    class,
                    Some(&format!("Set value on '{}'", element.title)),
                )
                .await?;
                let (snapshot, focus) = self
                    .harness
                    .set_value(&app, generation, element_index, &value, &ctx, &mut cache)
                    .await?;
                let body = format_condensed_refresh(&snapshot, Some(focus));
                self.build_outcome(body, Some(&snapshot), &ctx.session_id)
                    .await
            }
            ComputerActionKind::LaunchApp => {
                let app = required_app(args)?;
                // Launch first (this only starts the process — no screenshot),
                // then run the per-app approval before any get_app_state captures
                // the window, preserving "approve before we look at it".
                let resolved = self.harness.launch_app(&app).await?;
                self.ensure_action_approvals(
                    &ctx,
                    action,
                    Some(&resolved.bundle_id),
                    Some(&resolved.name),
                    ActionClass::LocalMutation,
                    None,
                )
                .await?;
                let mut cache = self.cache.lock().await;
                let snapshot = self
                    .harness
                    .get_app_state(&resolved.name, &ctx, &mut cache)
                    .await?;
                let text = format_full_tree(&snapshot);
                self.build_outcome(text, Some(&snapshot), &ctx.session_id)
                    .await
            }
            ComputerActionKind::ListApps => {
                unreachable!("list_apps handled before match");
            }
        }
    }

    async fn resolve_app(&self, app: &str) -> Result<types::AppInfo, String> {
        let apps = self.harness.list_apps().await?;
        let needle = app.trim();
        if let Some(found) = apps.iter().find(|a| {
            a.bundle_id.eq_ignore_ascii_case(needle) || a.name.eq_ignore_ascii_case(needle)
        }) {
            return Ok(found.clone());
        }
        if let Some(found) = apps.iter().find(|a| {
            a.name
                .to_ascii_lowercase()
                .contains(&needle.to_ascii_lowercase())
        }) {
            return Ok(found.clone());
        }
        Err(policy::no_running_app_message(app))
    }

    async fn resolve_bundle_id(&self, app: &str) -> Result<String, String> {
        Ok(self.resolve_app(app).await?.bundle_id)
    }

    fn snapshot_key(&self, bundle_id: &str, ctx: &HarnessRequestContext) -> cache::SnapshotKey {
        cache::SnapshotKey {
            task_id: ctx.task_id.clone(),
            session_id: ctx.session_id.clone(),
            bundle_id: bundle_id.to_string(),
        }
    }
}

fn required_str(args: &Value, key: &str) -> Result<String, String> {
    args.get(key)
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .ok_or_else(|| format!("Missing required parameter: {key}"))
}

// Small models recover from validation errors far more reliably when the
// message states the literal next step instead of just naming the gap.
fn required_app(args: &Value) -> Result<String, String> {
    args.get("app")
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .ok_or_else(|| {
            "Missing required parameter: app — repeat the same call with app set to the \
             application you are controlling (use the name from your last get_app_state or \
             list_apps result)"
                .to_string()
        })
}

fn required_generation(args: &Value) -> Result<u64, String> {
    args.get("snapshot_generation")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| {
            "Missing required parameter: snapshot_generation — call get_app_state for this app \
             and copy the snapshot_generation value from its result into this call"
                .to_string()
        })
}

fn required_u64(args: &Value, key: &str) -> Result<u64, String> {
    args.get(key)
        .and_then(|v| v.as_u64())
        .ok_or_else(|| format!("Missing required parameter: {key}"))
}

fn required_u32(args: &Value, key: &str) -> Result<u32, String> {
    args.get(key)
        .and_then(|v| v.as_u64())
        .and_then(|v| u32::try_from(v).ok())
        .ok_or_else(|| format!("Missing required parameter: {key}"))
}

fn element_target_from(element: Option<&IndexedElement>, index: Option<u32>) -> ElementTarget {
    match element {
        Some(el) => ElementTarget {
            index: Some(el.index),
            title: if el.title.is_empty() {
                None
            } else {
                Some(el.title.clone())
            },
            role: if el.role.is_empty() {
                None
            } else {
                Some(el.role.clone())
            },
        },
        None => ElementTarget {
            index,
            ..Default::default()
        },
    }
}

fn optional_u32(args: &Value, key: &str) -> Option<u32> {
    args.get(key)
        .and_then(|v| v.as_u64())
        .and_then(|v| u32::try_from(v).ok())
}

/// Resolve the target element index for an action. An explicit `element_index`
/// wins; otherwise a descriptor (`element_title` and/or `element_role`, with an
/// optional 1-based `occurrence`) is resolved against the current snapshot, so
/// the model can target a control by its stable label rather than a positional
/// index that renumbers across re-renders. Returns `Ok(None)` when neither an
/// index nor a descriptor was supplied (e.g. a coordinate click).
fn resolve_target_index(
    args: &Value,
    snapshot_cache: &SnapshotCache,
    key: &cache::SnapshotKey,
    generation: u64,
) -> Result<Option<u32>, String> {
    if let Some(index) = optional_u32(args, "element_index") {
        return Ok(Some(index));
    }
    let role = args.get("element_role").and_then(|v| v.as_str());
    let title = args.get("element_title").and_then(|v| v.as_str());
    if role.is_none() && title.is_none() {
        return Ok(None);
    }
    let occurrence = args
        .get("occurrence")
        .and_then(|v| v.as_u64())
        .map(|n| n.max(1) as usize)
        .unwrap_or(1);
    snapshot_cache
        .resolve_descriptor(key, generation, role, title, occurrence)
        .map(Some)
}

fn optional_f64(args: &Value, key: &str) -> Option<f64> {
    args.get(key).and_then(|v| v.as_f64())
}

#[async_trait]
impl Tool for ComputerUseTool {
    fn name(&self) -> &str {
        TOOL_NAME
    }

    fn description(&self) -> &str {
        "Inspect and control native macOS applications via accessibility trees and screenshots. \
         Only apps that are already running can be controlled — if the target app is not listed by \
         list_apps, call launch_app to start it first. Call get_app_state before mutating actions; \
         copy the exact snapshot_generation from the most recent result into every mutation (do not \
         increment or guess it). To click or type into a control, prefer targeting it by \
         element_title (and element_role) rather than element_index — titles are stable while \
         indices renumber on every re-render; use occurrence for repeated labels. type_text focuses \
         the element you target before typing, so always pass element_title/element_index when \
         typing into a specific field (e.g. an address bar). After your final mutating action, call \
         get_app_state and confirm the visible state matches the goal before reporting success."
    }

    fn schema(&self) -> Value {
        json!({
            "name": TOOL_NAME,
            "description": self.description(),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "list_apps",
                            "launch_app",
                            "get_app_state",
                            "screenshot",
                            "activate_app",
                            "click",
                            "type_text",
                            "press_key",
                            "scroll",
                            "set_value"
                        ],
                        "description": "The desktop action to perform. computer_use can only control apps that are already running; if the target app is not in list_apps, use launch_app to start it first."
                    },
                    "app": {
                        "type": "string",
                        "description": "Application name or bundle id. Required for every action except list_apps. If the app is installed but not running, call launch_app with this name first."
                    },
                    "snapshot_generation": {
                        "type": "integer",
                        "description": "Generation from the latest get_app_state for this app. Required for mutating actions; optional for activate_app (activation may be your first action on an app)."
                    },
                    "element_index": {
                        "type": "integer",
                        "description": "Indexed element from the accessibility tree. WARNING: indices renumber whenever the screen changes (page load, scroll, new element). Prefer targeting by element_title/element_role, which are stable. Used by click, type_text, scroll, set_value."
                    },
                    "element_title": {
                        "type": "string",
                        "description": "Target an element by its visible title/label (case-insensitive substring), resolved against the latest snapshot. More robust than element_index because it survives re-renders. E.g. 'Address and search bar', 'Like', 'Send'. Combine with occurrence to disambiguate repeats."
                    },
                    "element_role": {
                        "type": "string",
                        "description": "Target an element by accessibility role (case-insensitive substring), e.g. 'button', 'textfield'. Use with or instead of element_title to disambiguate."
                    },
                    "occurrence": {
                        "type": "integer",
                        "description": "1-based: which match to use when element_title/element_role match several elements (e.g. occurrence 1 = first 'Like' button in a feed). Default 1."
                    },
                    "x": { "type": "number", "description": "Coordinate click x (global points)" },
                    "y": { "type": "number", "description": "Coordinate click y (global points)" },
                    "text": { "type": "string", "description": "Text to type. For type_text, also pass element_title (or element_index) to focus that field first; otherwise the text goes to whatever currently has keyboard focus." },
                    "key": { "type": "string", "description": "Key combo such as Return or Command+s" },
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down", "left", "right"],
                        "description": "Scroll direction. For scroll, element_index/element_title are OPTIONAL — omit them to scroll the whole page/feed; provide one only to scroll a specific pane."
                    },
                    "pages": { "type": "number", "description": "Scroll amount in pages (default 1)" },
                    "value": { "type": "string", "description": "Value for set_value" }
                },
                "required": ["action"],
                "additionalProperties": false
            }
        })
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let outcome = self.call_with_status_outcome(arguments, None).await?;
        Ok(outcome.output)
    }

    async fn call_with_status_outcome(
        &self,
        arguments: &str,
        _status_tx: Option<mpsc::Sender<StatusUpdate>>,
    ) -> anyhow::Result<ToolCallOutcome> {
        let args: Value = serde_json::from_str(arguments)?;
        self.clear_pending_meta().await;
        let started = std::time::Instant::now();
        let result = self.dispatch(&args).await;
        let pending = self.take_pending_meta().await;
        let duration_ms = started.elapsed().as_millis() as u64;

        let action = args.get("action").and_then(|v| v.as_str()).unwrap_or("?");
        let app = args.get("app").and_then(|v| v.as_str()).unwrap_or("");
        let generation = args.get("snapshot_generation").and_then(|v| v.as_u64());
        let task_id = args
            .get("_task_id")
            .and_then(|v| v.as_str())
            .unwrap_or("default");
        let session_id = args
            .get("_session_id")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let is_mutation = !matches!(action, "list_apps" | "get_app_state" | "screenshot");

        let mut element_target = pending.element_target;
        if element_target.index.is_none() {
            if let Some(index) = optional_u32(&args, "element_index") {
                element_target.index = Some(index);
            }
        }
        if element_target.title.is_none() && !app.is_empty() {
            if let Ok(ctx) = Self::parse_context(&args) {
                if let Ok(bundle_id) = self.resolve_bundle_id(app).await {
                    let resolved = self.resolve_element_target(&args, &ctx, &bundle_id).await;
                    if resolved.title.is_some() || resolved.role.is_some() {
                        element_target = resolved;
                    }
                }
            }
        }

        let click_method = pending.click_method.as_deref();
        let mut budget = pending.mutation_budget;
        if is_mutation && budget.is_none() {
            let used = self.approval_state.mutations_used(task_id).await;
            budget = Some(ApprovalState::mutation_budget(&self.config, used));
        }

        match result {
            Ok(outcome) => {
                let screenshot_bytes: usize = outcome
                    .metadata
                    .attachments
                    .iter()
                    .map(|a| a.size_bytes as usize)
                    .sum();
                let screenshot_path = outcome
                    .metadata
                    .attachments
                    .first()
                    .map(|a| a.local_path.as_str());
                let truncated = outcome.output.contains("TRUNCATED");
                telemetry::log_action(&ActionLog {
                    task_id,
                    action,
                    app,
                    generation,
                    target: Some(&element_target),
                    click_method,
                    duration_ms,
                    success: true,
                    error: None,
                    screenshot_bytes,
                    screenshot_path,
                    truncated,
                    budget,
                    is_mutation,
                });
                self.session_telemetry
                    .record_action(&ActionRecord {
                        task_id,
                        action,
                        app,
                        is_mutation,
                        success: true,
                        budget,
                        target: Some(&element_target),
                    })
                    .await;
                self.record_action_event(
                    session_id,
                    task_id,
                    action,
                    app,
                    true,
                    None,
                    click_method,
                    duration_ms,
                    is_mutation,
                    &element_target,
                )
                .await;
                let mut outcome = outcome;
                if self
                    .duplicate_mutation_caution(task_id, is_mutation, &element_target, &args)
                    .await
                {
                    outcome.output.push_str(DUPLICATE_MUTATION_CAUTION);
                }
                Ok(outcome)
            }
            Err(err) => {
                telemetry::log_action(&ActionLog {
                    task_id,
                    action,
                    app,
                    generation,
                    target: Some(&element_target),
                    click_method,
                    duration_ms,
                    success: false,
                    error: Some(&err),
                    screenshot_bytes: 0,
                    screenshot_path: None,
                    truncated: false,
                    budget,
                    is_mutation,
                });
                self.session_telemetry
                    .record_action(&ActionRecord {
                        task_id,
                        action,
                        app,
                        is_mutation,
                        success: false,
                        budget,
                        target: Some(&element_target),
                    })
                    .await;
                self.record_action_event(
                    session_id,
                    task_id,
                    action,
                    app,
                    false,
                    Some(&err),
                    click_method,
                    duration_ms,
                    is_mutation,
                    &element_target,
                )
                .await;
                // A failed action didn't take effect — reset so a later retry
                // isn't wrongly flagged as a no-op repeat.
                self.last_mutation_sig.lock().await.remove(task_id);
                Ok(ToolCallOutcome::from_output(format!("Error: {err}")))
            }
        }
    }

    fn call_semantics(&self, arguments: &str) -> ToolCallSemantics {
        let Ok(args) = serde_json::from_str::<Value>(arguments) else {
            return ToolCallSemantics::default();
        };
        let action = args
            .get("action")
            .and_then(|v| v.as_str())
            .and_then(|a| ComputerActionKind::parse(a).ok());
        let observation = matches!(
            action,
            Some(
                ComputerActionKind::ListApps
                    | ComputerActionKind::GetAppState
                    | ComputerActionKind::Screenshot
            )
        );
        if observation {
            ToolCallSemantics::observation()
        } else {
            ToolCallSemantics::mutation()
        }
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: false,
            external_side_effect: true,
            needs_approval: true,
            idempotent: false,
            // Not high_impact_write: the policy tool filter strips high-impact
            // tools on low-risk turns, which would hide computer_use from the
            // model entirely (it then improvises with browser). Risk is gated by
            // the tool's own approval ladder (session → per-app → consequential),
            // matching how `browser` declares its capabilities.
            high_impact_write: false,
        }
    }

    fn tool_role(&self) -> ToolRole {
        ToolRole::Action
    }

    async fn on_task_end(&self, task_id: &str, session_id: &str) -> anyhow::Result<()> {
        if let Some(summary) = self.session_telemetry.finish_task(task_id).await {
            telemetry::log_session_end(task_id, session_id, &summary);
        }
        self.cache.lock().await.clear_task(task_id);
        self.pins.clear_task(task_id).await;
        self.approval_state.clear_task(task_id).await;
        self.last_mutation_sig.lock().await.remove(task_id);
        Ok(())
    }

    fn is_available(&self) -> bool {
        self.config.enabled
    }
}

#[cfg(test)]
pub async fn test_tool(config: ComputerUseConfig, inbox: PathBuf) -> ComputerUseTool {
    use crate::config::FilesConfig;
    use crate::tools::ApprovalBroker;

    let mut files = FilesConfig::default();
    files.vision_enabled = true;
    let (media_tx, _media_rx) = mpsc::channel(1);
    let (approval_tx, _approval_rx) = mpsc::channel(1);
    let tool = ComputerUseTool::new(
        config,
        VisionConfig::from_files(&files),
        inbox,
        ApprovalBroker::new(approval_tx),
        media_tx,
        None,
    );
    tool.approval_state
        .approve_all_for_test("telegram:1", "com.apple.calculator")
        .await;
    tool
}
