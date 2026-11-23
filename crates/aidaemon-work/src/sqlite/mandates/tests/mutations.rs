//! Tests of mutation reservations, dispatch claims and the rolling quota.

use super::*;

#[tokio::test]
async fn policy_update_marks_unclaimed_reservation_never_dispatched_and_releases_rolling_quota() {
    let (store, _database) = test_store().await;
    let (goal, mandate) = controller("owner-session", 2);
    let run = claim_and_start_run(&store, &goal, &mandate).await;
    let act = MandateDecisionCycle::new(
        &mandate.id,
        &run.id,
        MandateDecisionOutcome::Act,
        "A bounded action was justified before the policy update",
        mandate.version,
    );
    let intention = Intention::new(
        &mandate.id,
        &act.id,
        &run.id,
        "Perform one bounded action",
        "It tests pre-dispatch invalidation",
    );
    store
        .record_mandate_decision(&act, Some(&intention), None)
        .await
        .unwrap();
    let fence = claim_mutation_fence(&store, &goal, &mandate, &run).await;
    let now = chrono::Utc::now();
    let unclaimed = reservation(&mandate, &act, &run, &fence, 1, 41, now);
    assert!(
        store
            .reserve_mandate_action_attempt(&unclaimed)
            .await
            .unwrap()
            .is_some()
    );

    let mut updated = store.get_mandate(&mandate.id).await.unwrap().unwrap();
    updated.version += 1;
    updated
        .constraints
        .push("Owner narrowed the policy".to_string());
    store.update_mandate(&updated).await.unwrap();

    let attempts = store
        .list_mandate_mutation_attempts_for_run(&run.id)
        .await
        .unwrap();
    assert_eq!(attempts.len(), 1);
    assert_eq!(
        attempts[0].status,
        MandateMutationAttemptStatus::NeverDispatched
    );
    assert!(attempts[0].completed_at.is_some());
    let quota = store
        .get_mandate_mutation_quota_state(
            &mandate.id,
            &(now + chrono::Duration::seconds(1)).to_rfc3339(),
        )
        .await
        .unwrap()
        .unwrap();
    assert_eq!(quota.reserved_in_rolling_24h, 0);
    assert_eq!(quota.remaining_in_rolling_24h, 24);
}

#[tokio::test]
async fn pause_marks_claimed_unresolved_reservation_ambiguous_and_run_reconciliation_required() {
    let (store, _database) = test_store().await;
    let (goal, mandate) = controller("owner-session", 1);
    let run = claim_and_start_run(&store, &goal, &mandate).await;
    let act = MandateDecisionCycle::new(
        &mandate.id,
        &run.id,
        MandateDecisionOutcome::Act,
        "One mutation was authorized before the pause",
        mandate.version,
    );
    let intention = Intention::new(
        &mandate.id,
        &act.id,
        &run.id,
        "Perform one bounded action",
        "It tests claimed-dispatch invalidation",
    );
    store
        .record_mandate_decision(&act, Some(&intention), None)
        .await
        .unwrap();
    let fence = claim_mutation_fence(&store, &goal, &mandate, &run).await;
    let now = chrono::Utc::now();
    let claimed = reservation(&mandate, &act, &run, &fence, 1, 42, now);
    assert!(
        store
            .reserve_mandate_action_attempt(&claimed)
            .await
            .unwrap()
            .is_some()
    );
    assert!(
        store
            .claim_mandate_mutation_dispatch(&dispatch_claim(
                &claimed,
                now + chrono::Duration::milliseconds(1),
            ))
            .await
            .unwrap()
    );

    assert!(
        store
            .transition_mandate_status(&mandate.id, MandateStatus::Active, MandateStatus::Paused,)
            .await
            .unwrap()
    );
    let attempts = store
        .list_mandate_mutation_attempts_for_run(&run.id)
        .await
        .unwrap();
    assert_eq!(attempts.len(), 1);
    assert_eq!(attempts[0].status, MandateMutationAttemptStatus::Ambiguous);
    assert!(attempts[0].completed_at.is_some());
    let closed_run = store
        .get_goal_runs(&goal.id)
        .await
        .unwrap()
        .into_iter()
        .find(|candidate| candidate.id == run.id)
        .unwrap();
    assert_eq!(closed_run.status, "cancelled");
    assert_eq!(
        closed_run.outcome_summary.as_deref(),
        Some("mandate_reconciliation_required:lifecycle_invalidated_after_dispatch_claim")
    );
    let notice_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM notification_queue
             WHERE id = ? AND notification_type = 'mandate_reconciliation_required'
               AND priority = 'critical' AND expires_at IS NULL",
    )
    .bind(format!("mandate-run-notice:{}", run.id))
    .fetch_one(&store.pool)
    .await
    .unwrap();
    assert_eq!(notice_count, 1);
    let suspended = store.get_mandate(&mandate.id).await.unwrap().unwrap();
    assert_eq!(suspended.status, MandateStatus::AwaitingInput);
    assert_eq!(
        suspended.suspension.as_ref().map(|value| value.kind),
        Some(MandateSuspensionKind::ReconciliationRequired)
    );
    assert!(
        store
            .resolve_mandate_suspension(
                &mandate.id,
                suspended.version,
                MandateSuspensionKind::ReconciliationRequired,
                None,
                Some(MandateReconciliationResolution::ConfirmedNoEffect),
                "I inspected the exact target and confirmed no effect occurred.",
                "owner-session",
            )
            .await
            .unwrap()
    );
    let resumed = store.get_mandate(&mandate.id).await.unwrap().unwrap();
    assert_eq!(resumed.status, MandateStatus::Active);
    assert!(resumed.suspension.is_none());
    let reconciliations: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM mandate_reconciliations
             WHERE mandate_id = ? AND resolution = 'confirmed_no_effect'",
    )
    .bind(&mandate.id)
    .fetch_one(&store.pool)
    .await
    .unwrap();
    assert_eq!(reconciliations, 1);
}

#[tokio::test]
async fn policy_update_with_claimed_unresolved_mutation_lands_awaiting_input() {
    let (store, _database) = test_store().await;
    let (goal, mandate) = controller("owner-session", 1);
    let run = claim_and_start_run(&store, &goal, &mandate).await;
    let act = MandateDecisionCycle::new(
        &mandate.id,
        &run.id,
        MandateDecisionOutcome::Act,
        "One action was authorized under the old policy",
        mandate.version,
    );
    let intention = Intention::new(
        &mandate.id,
        &act.id,
        &run.id,
        "Perform one bounded action",
        "It tests policy-race reconciliation",
    );
    store
        .record_mandate_decision(&act, Some(&intention), None)
        .await
        .unwrap();
    let fence = claim_mutation_fence(&store, &goal, &mandate, &run).await;
    let now = chrono::Utc::now();
    let claimed = reservation(&mandate, &act, &run, &fence, 1, 43, now);
    assert!(
        store
            .reserve_mandate_action_attempt(&claimed)
            .await
            .unwrap()
            .is_some()
    );
    assert!(
        store
            .claim_mandate_mutation_dispatch(&dispatch_claim(
                &claimed,
                now + chrono::Duration::milliseconds(1),
            ))
            .await
            .unwrap()
    );

    let mut updated = store.get_mandate(&mandate.id).await.unwrap().unwrap();
    updated.version += 1;
    updated
        .constraints
        .push("Owner narrowed the policy after dispatch".to_string());
    store.update_mandate(&updated).await.unwrap();

    let stored = store.get_mandate(&mandate.id).await.unwrap().unwrap();
    assert_eq!(stored.version, updated.version);
    assert_eq!(stored.status, MandateStatus::AwaitingInput);
    assert_eq!(
        store.get_goal(&goal.id).await.unwrap().unwrap().status,
        "paused"
    );
    assert_eq!(
        store
            .list_mandate_mutation_attempts_for_run(&run.id)
            .await
            .unwrap()[0]
            .status,
        MandateMutationAttemptStatus::Ambiguous
    );
    let notices: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM notification_queue WHERE id = ?")
        .bind(format!("mandate-run-notice:{}", run.id))
        .fetch_one(&store.pool)
        .await
        .unwrap();
    assert_eq!(notices, 1);
}

#[tokio::test]
async fn action_reservations_require_a_unique_live_proof_and_respect_the_cycle_cap() {
    let (store, _database) = test_store().await;
    let (goal, mandate) = controller("owner-session", 2);
    let run = claim_and_start_run(&store, &goal, &mandate).await;
    let act = MandateDecisionCycle::new(
        &mandate.id,
        &run.id,
        MandateDecisionOutcome::Act,
        "A sourced reply is timely and useful",
        mandate.version,
    );
    let intention = Intention::new(
        &mandate.id,
        &act.id,
        &run.id,
        "Post one sourced reply",
        "It contributes concrete information",
    );
    store
        .record_mandate_decision(&act, Some(&intention), None)
        .await
        .unwrap();
    assert_eq!(
        store.list_intentions(&mandate.id, 10).await.unwrap(),
        vec![intention]
    );
    let fence = claim_mutation_fence(&store, &goal, &mandate, &run).await;
    let now = chrono::Utc::now();
    let first = reservation(&mandate, &act, &run, &fence, 1, 10, now);

    assert!(
        store
            .reserve_mandate_action_attempt(&first)
            .await
            .unwrap()
            .is_some()
    );
    assert!(
        store
            .reserve_mandate_action_attempt(&first)
            .await
            .unwrap()
            .is_none()
    );
    assert!(
        store
            .reserve_mandate_action_attempt(&reservation(
                &mandate,
                &act,
                &run,
                &fence,
                2,
                11,
                now + chrono::Duration::seconds(901),
            ))
            .await
            .unwrap()
            .is_some()
    );
    assert!(
        store
            .reserve_mandate_action_attempt(&reservation(
                &mandate,
                &act,
                &run,
                &fence,
                3,
                12,
                now + chrono::Duration::seconds(1_802),
            ))
            .await
            .unwrap()
            .is_none()
    );
}

#[tokio::test]
async fn owner_policy_update_invalidates_an_existing_action_budget() {
    let (store, _database) = test_store().await;
    let (goal, mandate) = controller("owner-session", 3);
    let run = claim_and_start_run(&store, &goal, &mandate).await;
    let act = MandateDecisionCycle::new(
        &mandate.id,
        &run.id,
        MandateDecisionOutcome::Act,
        "One action currently fits the policy",
        mandate.version,
    );
    let intention = Intention::new(
        &mandate.id,
        &act.id,
        &run.id,
        "Perform the authorized action",
        "It advances the mandate",
    );
    store
        .record_mandate_decision(&act, Some(&intention), None)
        .await
        .unwrap();
    let fence = claim_mutation_fence(&store, &goal, &mandate, &run).await;
    let now = chrono::Utc::now();
    assert!(
        store
            .reserve_mandate_action_attempt(&reservation(&mandate, &act, &run, &fence, 1, 20, now,))
            .await
            .unwrap()
            .is_some()
    );

    let mut owner_update = store.get_mandate(&mandate.id).await.unwrap().unwrap();
    owner_update.version += 1;
    owner_update
        .constraints
        .push("New owner constraint".to_string());
    store.update_mandate(&owner_update).await.unwrap();

    assert!(
        store
            .reserve_mandate_action_attempt(&reservation(
                &mandate,
                &act,
                &run,
                &fence,
                2,
                21,
                now + chrono::Duration::seconds(901),
            ))
            .await
            .unwrap()
            .is_none()
    );
}

#[tokio::test]
async fn concurrent_reservations_have_one_last_slot_winner() {
    let (store, _database) = test_store().await;
    let (goal, mut mandate) = controller("owner-session", 1);
    mandate.authority.max_mutating_actions_per_rolling_24h = 1;
    let run = claim_and_start_run(&store, &goal, &mandate).await;
    let act = MandateDecisionCycle::new(
        &mandate.id,
        &run.id,
        MandateDecisionOutcome::Act,
        "One exact action is justified",
        mandate.version,
    );
    let intention = Intention::new(
        &mandate.id,
        &act.id,
        &run.id,
        "Perform exactly one action",
        "It tests the atomic quota fence",
    );
    store
        .record_mandate_decision(&act, Some(&intention), None)
        .await
        .unwrap();
    let fence = claim_mutation_fence(&store, &goal, &mandate, &run).await;
    let now = chrono::Utc::now();
    let left = reservation(&mandate, &act, &run, &fence, 1, 101, now);
    let right = reservation(&mandate, &act, &run, &fence, 1, 102, now);
    let (left, right) = tokio::join!(
        store.reserve_mandate_action_attempt(&left),
        store.reserve_mandate_action_attempt(&right),
    );
    let winners = [left.unwrap(), right.unwrap()]
        .into_iter()
        .filter(Option::is_some)
        .count();
    assert_eq!(winners, 1);
    let attempts = store
        .list_mandate_mutation_attempts_for_run(&run.id)
        .await
        .unwrap();
    assert_eq!(attempts.len(), 1);
}

async fn start_existing_claimed_run(
    store: &SqliteWorkStore,
    goal: &Goal,
    mandate: &Mandate,
) -> GoalRun {
    let claimed = store
        .claim_due_mandates(1, "test-heartbeat-next", 300)
        .await
        .unwrap();
    assert_eq!(claimed.len(), 1);
    assert_eq!(claimed[0].id, mandate.id);
    let root_task_id = uuid::Uuid::new_v4().to_string();
    let run = store
        .start_goal_run(&goal.id, "mandate", None, Some(&root_task_id))
        .await
        .unwrap();
    store
        .create_task(&review_root(goal, &run.id, &root_task_id))
        .await
        .unwrap();
    sqlx::query("UPDATE goal_runs SET status = 'running' WHERE id = ?")
        .bind(&run.id)
        .execute(&store.pool)
        .await
        .unwrap();
    run
}

#[tokio::test]
async fn cooldown_failure_and_rolling_expiry_are_cross_cycle_safety_state() {
    let (store, _database) = test_store().await;
    let (goal, mut mandate) = controller("owner-session", 2);
    mandate.authority.max_mutating_actions_per_rolling_24h = 2;
    let run = claim_and_start_run(&store, &goal, &mandate).await;
    let act = MandateDecisionCycle::new(
        &mandate.id,
        &run.id,
        MandateDecisionOutcome::Act,
        "Two spaced actions fit the envelope",
        mandate.version,
    );
    let intention = Intention::new(
        &mandate.id,
        &act.id,
        &run.id,
        "Perform two governed actions",
        "It tests rolling quota semantics",
    );
    store
        .record_mandate_decision(&act, Some(&intention), None)
        .await
        .unwrap();
    let fence = claim_mutation_fence(&store, &goal, &mandate, &run).await;
    let base = chrono::Utc::now();
    let first = reservation(&mandate, &act, &run, &fence, 1, 201, base);
    assert!(
        store
            .reserve_mandate_action_attempt(&first)
            .await
            .unwrap()
            .is_some()
    );
    assert!(
        store
            .claim_mandate_mutation_dispatch(&dispatch_claim(
                &first,
                base + chrono::Duration::milliseconds(1),
            ))
            .await
            .unwrap()
    );
    assert!(
        !store
            .claim_mandate_mutation_dispatch(&dispatch_claim(
                &first,
                base + chrono::Duration::milliseconds(2),
            ))
            .await
            .unwrap()
    );
    assert!(
        finish_attempt(
            &store,
            &first,
            MandateMutationAttemptStatus::Failed,
            Some(400),
            base + chrono::Duration::seconds(1)
        )
        .await
    );
    let too_soon = reservation(
        &mandate,
        &act,
        &run,
        &fence,
        2,
        202,
        base + chrono::Duration::seconds(899),
    );
    assert!(
        store
            .reserve_mandate_action_attempt(&too_soon)
            .await
            .unwrap()
            .is_none()
    );
    let cooldown = store
        .get_mandate_mutation_quota_state(
            &mandate.id,
            &(base + chrono::Duration::seconds(899)).to_rfc3339(),
        )
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        cooldown.block_reason,
        Some(MandateMutationQuotaBlockReason::Cooldown)
    );
    assert_eq!(cooldown.reserved_in_rolling_24h, 1);

    let second = reservation(
        &mandate,
        &act,
        &run,
        &fence,
        2,
        203,
        base + chrono::Duration::seconds(900),
    );
    assert!(
        store
            .reserve_mandate_action_attempt(&second)
            .await
            .unwrap()
            .is_some()
    );
    let full = store
        .get_mandate_mutation_quota_state(
            &mandate.id,
            &(base + chrono::Duration::seconds(901)).to_rfc3339(),
        )
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        full.block_reason,
        Some(MandateMutationQuotaBlockReason::Rolling24hExhausted)
    );
    assert_eq!(full.remaining_in_rolling_24h, 0);

    let observation = crate::traits::ToolCallSemantics::observation().with_target_hint(
        crate::traits::ToolTargetHintKind::Url,
        "https://api.x.com/2/tweets",
    );
    assert!(
        crate::authority::authorize_mandate_observation(
            &mandate,
            "http_request",
            r#"{"method":"GET","url":"https://api.x.com/2/tweets"}"#,
            &observation,
            &(base + chrono::Duration::seconds(901)),
        )
        .is_ok()
    );

    let expired = store
        .get_mandate_mutation_quota_state(
            &mandate.id,
            &(base + chrono::Duration::hours(24) + chrono::Duration::seconds(1)).to_rfc3339(),
        )
        .await
        .unwrap()
        .unwrap();
    assert_eq!(expired.reserved_in_rolling_24h, 1);
    assert_eq!(expired.remaining_in_rolling_24h, 1);
    assert!(expired.available_now);
}

#[tokio::test]
async fn rolling_quota_survives_policy_versions_pause_resume_and_new_cycles() {
    let (store, _database) = test_store().await;
    let (goal, mut mandate) = controller("owner-session", 1);
    mandate.authority.max_mutating_actions_per_rolling_24h = 1;
    let first_run = claim_and_start_run(&store, &goal, &mandate).await;
    let first_act = MandateDecisionCycle::new(
        &mandate.id,
        &first_run.id,
        MandateDecisionOutcome::Act,
        "Use the sole daily slot",
        mandate.version,
    );
    let first_intention = Intention::new(
        &mandate.id,
        &first_act.id,
        &first_run.id,
        "Use one daily action",
        "It tests cross-cycle accounting",
    );
    store
        .record_mandate_decision(&first_act, Some(&first_intention), None)
        .await
        .unwrap();
    let first_fence = claim_mutation_fence(&store, &goal, &mandate, &first_run).await;
    let base = chrono::Utc::now();
    let first = reservation(&mandate, &first_act, &first_run, &first_fence, 1, 301, base);
    assert!(
        store
            .reserve_mandate_action_attempt(&first)
            .await
            .unwrap()
            .is_some()
    );
    assert!(
        store
            .claim_mandate_mutation_dispatch(&dispatch_claim(
                &first,
                base + chrono::Duration::milliseconds(1),
            ))
            .await
            .unwrap()
    );
    assert!(
        finish_attempt(
            &store,
            &first,
            MandateMutationAttemptStatus::Failed,
            Some(400),
            base + chrono::Duration::seconds(1)
        )
        .await
    );
    assert!(
        store
            .transition_mandate_status(&mandate.id, MandateStatus::Active, MandateStatus::Paused,)
            .await
            .unwrap()
    );
    assert!(
        store
            .transition_mandate_status(&mandate.id, MandateStatus::Paused, MandateStatus::Active,)
            .await
            .unwrap()
    );
    let current = store.get_mandate(&mandate.id).await.unwrap().unwrap();
    assert!(current.version > mandate.version);
    let second_run = start_existing_claimed_run(&store, &goal, &current).await;
    let second_act = MandateDecisionCycle::new(
        &current.id,
        &second_run.id,
        MandateDecisionOutcome::Act,
        "A new cycle still sees the old reservation",
        current.version,
    );
    let second_intention = Intention::new(
        &current.id,
        &second_act.id,
        &second_run.id,
        "Try another action",
        "The daily ledger must reject it",
    );
    // The new cycle's ACT is refused when it is committed: the store
    // reads the same rolling ledger across policy versions.
    let refused = store
        .record_mandate_decision(&second_act, Some(&second_intention), None)
        .await
        .expect_err("ACT must be refused while the rolling quota is used up");
    assert!(
        refused
            .to_string()
            .starts_with("mandate_mutation_quota_unavailable: choose WAIT"),
        "{refused}"
    );
    let quota = store
        .get_mandate_mutation_quota_state(
            &current.id,
            &(base + chrono::Duration::seconds(901)).to_rfc3339(),
        )
        .await
        .unwrap()
        .unwrap();
    assert_eq!(quota.reserved_in_rolling_24h, 1);
    assert_eq!(
        quota.block_reason,
        Some(MandateMutationQuotaBlockReason::Rolling24hExhausted)
    );
}
