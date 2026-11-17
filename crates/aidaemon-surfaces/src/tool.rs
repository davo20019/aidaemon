//! `hand_over` (feature `tool`): the agent asks the owner to do a step themselves on a live
//! surface (sign in, enter a one-time code, solve a challenge) and waits.
//!
//! The ask is an ordinary owner card through the agent kernel, never a wait
//! inside the tool: `owner_approval_request` gets the surface ready (a
//! browser moves onto a profile that keeps sign-ins), marks the handoff on
//! the surface registry and returns a take-over card. The run is interrupted
//! there, so it survives a restart like any card. Done (allow once) runs the
//! call, which returns where the surface is now, as typed facts; Not now
//! (deny) binds the run, so the agent does not ask again in it.
//!
//! Done is not proof: the result is an observation (site, title, whether a
//! password field is still there) for the model to judge.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::{SurfaceId, SurfaceRegistry};
use anyhow::Context;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use aidaemon_tools::OutcomeTool as Tool;
use aidaemon_tools::owner_answers::OwnerAnsweredCalls;
use aidaemon_tools::semantics::{
    ApprovalCall, ApprovalKind, ApprovalResponse, OwnerApprovalNeed, ToolCallContext,
    ToolCallMetadata, ToolCallOutcome, ToolCallSemantics, ToolCapabilities, ToolOutcomeStatus,
    ToolSemantics, TrustRoots, UserRole,
};

/// The one effect every handoff declares: a "Not now" binds the run.
const HANDOFF_EFFECT: &str = "hand_over";
/// What a granted handoff is kept under until its call runs.
const SUBJECT: &str = "hand_over";
/// The longest `what` a card shows.
const MAX_WHAT_CHARS: usize = 120;

/// Where the owner's app opens a surface's live view, when the daemon has an
/// address a phone can reach.
pub type LiveViewLink = Arc<dyn Fn(&SurfaceId) -> Option<String> + Send + Sync>;

#[derive(Debug, Deserialize)]
struct Args {
    #[serde(default)]
    surface: Option<String>,
    what: String,
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

pub struct HandOverTool {
    registry: SurfaceRegistry,
    link: LiveViewLink,
    owner_answers: OwnerAnsweredCalls,
    /// The surface a card is waiting on, per session.
    asking: Mutex<HashMap<String, SurfaceId>>,
    /// The surface the owner said Done on, per session, until its call runs.
    done: Mutex<HashMap<String, SurfaceId>>,
    /// The site another handoff already asks about, per session whose call
    /// was not put to the owner for it.
    already_asked: Mutex<HashMap<String, String>>,
}

impl HandOverTool {
    pub fn new(registry: SurfaceRegistry, link: LiveViewLink) -> Self {
        Self {
            registry,
            link,
            owner_answers: OwnerAnsweredCalls::default(),
            asking: Mutex::new(HashMap::new()),
            done: Mutex::new(HashMap::new()),
            already_asked: Mutex::new(HashMap::new()),
        }
    }

    fn what_of(args: &Args) -> String {
        let what: String = args
            .what
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .chars()
            .take(MAX_WHAT_CHARS)
            .collect();
        if what.is_empty() {
            "do a step yourself".into()
        } else {
            what
        }
    }
}

#[async_trait]
impl Tool for HandOverTool {
    fn name(&self) -> &str {
        "hand_over"
    }

    fn description(&self) -> &str {
        "Ask the owner to do a step themselves on a screen you are working on, then wait."
    }

    fn schema(&self) -> Value {
        json!({
            "name": "hand_over",
            "description": "Ask the owner to do something themselves on a browser tab you are working on, such as signing in, entering a verification code or solving a challenge, then wait until they say they are done. Use it whenever a step needs the owner's password, a one-time code or their personal judgement; never ask for a password or a code in chat. The owner does it from the live view in their app or at the computer, and you never see what they type. Returns where the tab is afterwards: check it before going on.",
            "parameters": {
                "type": "object",
                "properties": {
                    "surface": {
                        "type": "string",
                        "description": "The browser tab_id to hand over, as the browser tool returned it. Omit for the tab you are on."
                    },
                    "what": {
                        "type": "string",
                        "description": "What the owner should do, in a few plain words, e.g. 'Sign in to your Acme account'."
                    }
                },
                "required": ["what"],
                "additionalProperties": false
            }
        })
    }

    /// The card is the handoff: Done or Not now, this once.
    fn approval_card_kind(&self, _request: &aidaemon_core::OwnerApprovalRequest) -> ApprovalKind {
        ApprovalKind::Handoff
    }

    async fn record_owner_answer(
        &self,
        _call: &ApprovalCall<'_>,
        facts: &ToolCallContext,
        response: Option<ApprovalResponse>,
    ) {
        let Some(session) = facts.session() else {
            return;
        };
        let Some(surface) = lock(&self.asking).remove(session) else {
            return;
        };
        // Answered or not, the surface is the agent's again.
        self.registry.end_handoff(&surface);
        if matches!(
            response,
            Some(
                ApprovalResponse::AllowOnce
                    | ApprovalResponse::AllowSession
                    | ApprovalResponse::AllowAlways
            )
        ) {
            lock(&self.done).insert(session.to_string(), surface);
            self.owner_answers.approve(session, SUBJECT);
        }
    }

    async fn execute(
        &self,
        arguments: &str,
        facts: &ToolCallContext,
    ) -> anyhow::Result<ToolCallOutcome> {
        let args: Args =
            serde_json::from_str(arguments).context("hand_over arguments must be valid JSON")?;
        anyhow::ensure!(
            facts.user_role == Some(UserRole::Owner),
            "Only the owner's own conversations can hand a screen to the owner"
        );
        let session = facts
            .session()
            .context("hand_over needs a conversation")?
            .to_string();
        if let Some(site) = lock(&self.already_asked).remove(&session) {
            anyhow::bail!(
                "The owner is already asked to do a step on {site} for another task. Wait for that to finish, then look at the page again before asking."
            );
        }
        if !self.owner_answers.take(&session, SUBJECT) {
            let found = self
                .registry
                .find_for_agent(&session, args.surface.as_deref())
                .await;
            anyhow::ensure!(
                found.is_some(),
                "No browser tab of this conversation matches. Open the page with the browser tool first, then hand it over."
            );
            anyhow::bail!(
                "Handing a screen to the owner needs their answer through the agent kernel"
            );
        }
        let surface = lock(&self.done).remove(&session);
        let observation = match &surface {
            Some(id) => self.registry.observe(id).await.ok(),
            None => None,
        };
        let output = match observation {
            Some(seen) => json!({
                "owner_answer": "done",
                "tab_id": seen.agent_ref,
                "site": seen.site,
                "title": seen.title,
                "password_field_present": seen.password_field_present,
                "interpretation": if seen.password_field_present {
                    "The owner pressed Done, but a password field is still on the page: the step may not be finished. Look at the page before going on; if it still needs the owner, ask once more."
                } else {
                    "The owner pressed Done. Look at the page to confirm the step worked before going on."
                },
            }),
            None => json!({
                "owner_answer": "done",
                "interpretation": "The owner pressed Done, but the tab is gone. Open the page again and check whether the step worked.",
            }),
        };
        Ok(ToolCallOutcome {
            output: serde_json::to_string_pretty(&output)?,
            metadata: ToolCallMetadata {
                outcome_status: Some(ToolOutcomeStatus::Succeeded),
                semantics: self.call_semantics(arguments),
                ..ToolCallMetadata::default()
            },
            finish: None,
        })
    }
}

#[async_trait]
impl ToolSemantics for HandOverTool {
    /// The owner's answer comes back to the tool (`record_owner_answer`),
    /// never as a kernel session grant: each handoff is asked anew.
    fn dispatch_facts(&self) -> aidaemon_tools::semantics::ToolDispatchFacts {
        aidaemon_tools::semantics::ToolDispatchFacts {
            keeps_own_grants: true,
            ..Default::default()
        }
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: false,
            external_side_effect: false,
            needs_approval: true,
            idempotent: false,
            high_impact_write: false,
        }
    }

    /// Every handoff is the owner's to answer. A call nobody could hand over
    /// (no tab, not the owner's chat) shows no card; execution says why.
    async fn owner_approval_request(
        &self,
        call: &ApprovalCall<'_>,
        facts: &ToolCallContext,
    ) -> OwnerApprovalNeed {
        if call.refused || facts.user_role != Some(UserRole::Owner) {
            return OwnerApprovalNeed::NotNeeded;
        }
        let Some(session) = facts.session() else {
            return OwnerApprovalNeed::NotNeeded;
        };
        let Ok(args) = serde_json::from_value::<Args>(call.arguments.clone()) else {
            return OwnerApprovalNeed::NotNeeded;
        };
        let Some(surface) = self
            .registry
            .find_for_agent(session, args.surface.as_deref())
            .await
        else {
            return OwnerApprovalNeed::NotNeeded;
        };
        // One ask per site at a time: a later run (an automation's next
        // occurrence) does not stack a second card on the owner.
        if let Some(site) = surface.site.clone()
            && self.registry.list().iter().any(|other| {
                other.id != surface.id
                    && other.handoff_pending
                    && other.site.as_ref() == Some(&site)
            })
        {
            lock(&self.already_asked).insert(session.to_string(), site);
            return OwnerApprovalNeed::NotNeeded;
        }
        let prep = match self.registry.prepare_handoff(&surface.id).await {
            Ok(prep) => prep,
            Err(error) => {
                tracing::warn!(%error, "Could not get a screen ready to hand over");
                return OwnerApprovalNeed::NotNeeded;
            }
        };
        if self.registry.begin_handoff(&prep.id).is_err() {
            return OwnerApprovalNeed::NotNeeded;
        }
        let site = self
            .registry
            .get(&prep.id)
            .and_then(|info| info.site)
            .or(surface.site);
        lock(&self.asking).insert(session.to_string(), prep.id.clone());
        let what = Self::what_of(&args);
        let summary = match site.as_deref() {
            Some(site) => format!("{what} on {site}, then press Done."),
            None => format!("{what}, then press Done."),
        };
        OwnerApprovalNeed::Ask(aidaemon_core::OwnerApprovalRequest {
            effects: vec![aidaemon_core::EffectKey::named(
                aidaemon_core::EffectClass::Call,
                HANDOFF_EFFECT,
            )],
            summary,
            subject: format!("{SUBJECT} {}", prep.id),
            risk: aidaemon_core::ApprovalRisk::Low,
            warnings: Vec::new(),
            card: Some(Box::new(aidaemon_core::ApprovalCard::new(
                aidaemon_core::CardAction::TakeOver {
                    surface_id: prep.id.to_string(),
                    what,
                    site,
                    keeps_sign_in: prep.keeps_sign_in,
                    open_url: (self.link)(&prep.id),
                },
            ))),
        })
    }

    fn call_effects(&self, _arguments: &str) -> aidaemon_core::CallEffects {
        aidaemon_core::CallEffects {
            keys: vec![aidaemon_core::EffectKey::named(
                aidaemon_core::EffectClass::Call,
                HANDOFF_EFFECT,
            )],
            complete: true,
            understood: true,
        }
    }

    /// What comes back names a page: someone else wrote its title.
    fn output_trust(
        &self,
        _arguments: &str,
        _roots: &TrustRoots<'_>,
    ) -> aidaemon_core::SourceTrust {
        aidaemon_core::SourceTrust::Untrusted
    }

    fn call_semantics(&self, _arguments: &str) -> ToolCallSemantics {
        ToolCallSemantics::observation()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::MockSurface;
    use crate::{SurfaceObservation, SurfacesConfig};

    fn owner_facts(session: &str) -> ToolCallContext {
        let mut facts = aidaemon_tools::testing::call_facts(session);
        facts.user_role = Some(UserRole::Owner);
        facts
    }

    fn setup() -> (HandOverTool, MockSurface, SurfaceRegistry) {
        let registry = SurfaceRegistry::new(SurfacesConfig::default());
        let mock = MockSurface::new();
        mock.set_notifier(registry.notifier());
        registry.add_backend(Arc::new(mock.clone()));
        let tool = HandOverTool::new(
            registry.clone(),
            Arc::new(|id: &SurfaceId| {
                Some(format!("https://daemon.example.invalid/#surface={id}"))
            }),
        );
        (tool, mock, registry)
    }

    fn args() -> Value {
        json!({"what": "Sign in to your Acme account"})
    }

    #[tokio::test]
    async fn the_card_names_the_step_and_marks_the_surface_waiting() {
        let (tool, mock, registry) = setup();
        let id = mock.add("app:s1", "tab-1");
        let args = args();
        let need = tool
            .owner_approval_request(&ApprovalCall::as_sent(&args), &owner_facts("app:s1"))
            .await;
        let OwnerApprovalNeed::Ask(request) = need else {
            panic!("a handoff always asks");
        };
        assert_eq!(
            request.summary,
            "Sign in to your Acme account on example.test, then press Done."
        );
        assert!(matches!(
            tool.approval_card_kind(&request),
            ApprovalKind::Handoff
        ));
        let card = request.card.unwrap();
        assert!(matches!(
            &card.action,
            aidaemon_core::CardAction::TakeOver { surface_id, keeps_sign_in: Some(true), open_url: Some(url), .. }
                if *surface_id == id.to_string() && url.ends_with(&id.to_string())
        ));
        assert!(registry.get(&id).unwrap().handoff_pending);
    }

    #[tokio::test]
    async fn done_returns_where_the_tab_is_and_not_now_ends_the_handoff() {
        let (tool, mock, registry) = setup();
        let id = mock.add("app:s1", "tab-1");
        mock.set_observation(SurfaceObservation {
            site: Some("example.test".into()),
            title: "Account".into(),
            password_field_present: false,
            agent_ref: None,
        });
        let args = args();
        let call = ApprovalCall::as_sent(&args);
        let facts = owner_facts("app:s1");
        let _ = tool.owner_approval_request(&call, &facts).await;
        tool.record_owner_answer(&call, &facts, Some(ApprovalResponse::AllowOnce))
            .await;
        assert!(!registry.get(&id).unwrap().handoff_pending);
        let outcome = tool.execute(&args.to_string(), &facts).await.unwrap();
        let output: Value = serde_json::from_str(&outcome.output).unwrap();
        assert_eq!(output["owner_answer"], "done");
        assert_eq!(output["tab_id"], "tab-1");
        assert_eq!(output["password_field_present"], false);
        // The answer is spent: running again without a new card refuses.
        assert!(tool.execute(&args.to_string(), &facts).await.is_err());

        let _ = tool.owner_approval_request(&call, &facts).await;
        assert!(registry.get(&id).unwrap().handoff_pending);
        tool.record_owner_answer(&call, &facts, Some(ApprovalResponse::Deny))
            .await;
        assert!(!registry.get(&id).unwrap().handoff_pending);
        assert!(tool.execute(&args.to_string(), &facts).await.is_err());
    }

    #[tokio::test]
    async fn nothing_to_hand_over_or_not_the_owner_asks_nothing() {
        let (tool, _mock, _registry) = setup();
        let args = args();
        let call = ApprovalCall::as_sent(&args);
        assert!(matches!(
            tool.owner_approval_request(&call, &owner_facts("app:none"))
                .await,
            OwnerApprovalNeed::NotNeeded
        ));
        let error = tool
            .execute(&args.to_string(), &owner_facts("app:none"))
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("No browser tab"), "{error}");
        let mut guest = owner_facts("app:none");
        guest.user_role = Some(UserRole::Guest);
        assert!(matches!(
            tool.owner_approval_request(&call, &guest).await,
            OwnerApprovalNeed::NotNeeded
        ));
    }

    #[tokio::test]
    async fn a_second_ask_for_a_site_already_waiting_is_not_put_to_the_owner() {
        let (tool, mock, _registry) = setup();
        mock.add("app:first", "tab-1");
        mock.add("app:second", "tab-1");
        let args = args();
        let call = ApprovalCall::as_sent(&args);
        assert!(matches!(
            tool.owner_approval_request(&call, &owner_facts("app:first"))
                .await,
            OwnerApprovalNeed::Ask(_)
        ));
        assert!(matches!(
            tool.owner_approval_request(&call, &owner_facts("app:second"))
                .await,
            OwnerApprovalNeed::NotNeeded
        ));
        let error = tool
            .execute(&args.to_string(), &owner_facts("app:second"))
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("already asked"), "{error}");
    }

    #[test]
    fn a_decline_binds_every_later_handoff_in_the_run() {
        let (tool, _mock, _registry) = setup();
        let one = tool.call_effects(&json!({"what": "a"}).to_string());
        let two = tool.call_effects(&json!({"what": "b", "surface": "t"}).to_string());
        assert_eq!(one.keys, two.keys);
        assert!(one.complete);
    }
}
