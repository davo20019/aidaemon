//! Old checks against new checks on one run of every fixture.
//!
//! Each fixture runs once through [`DaemonSubject`]. The old
//! `assert_expectations` and the crate's checks then judge that same run, on
//! the original expectations and on many mutated ones. Temporary: it exists
//! to justify deleting the old checks.

use aidaemon_eval::{check_trajectory, Expect, ScriptedArguments, ScriptedCall, ScriptedReply};
use aidaemon_eval::{Subject, ToolLaneExpect, Trajectory};

use super::fixture::{
    assert_expectations, fixtures_dir, load_fixtures_dir, ExpectBlock, HarnessEvalFixture,
    HarnessEvalRunResult, MockResponseSpec,
};
use super::runner::run_fixture;
use super::subject::{
    count_names, trajectory_of, DaemonExpect, DaemonScenario, DaemonSetup, DaemonSubject,
};

fn translate_expect(old: &ExpectBlock) -> Expect<DaemonExpect> {
    Expect {
        tool_dispatch_counts: old.tool_dispatch_counts.clone(),
        final_text_contains_exact: old.response_contains.clone(),
        max_turns: old.execution_llm_calls_max,
        proposed: ToolLaneExpect {
            tools_used: old.tools_used.clone(),
            tools_not_used: old.tools_not_used.clone(),
            tools_in_order: old.tools_in_order.clone(),
            tool_call_counts: old.tool_call_counts.clone(),
        },
        ext: Some(DaemonExpect {
            orchestration_route: old.orchestration_route.clone(),
            tools_required_predicted: old.tools_required_predicted,
            outcome: old.outcome.clone(),
            stop_reason: old.stop_reason.clone(),
            llm_calls_min: old.llm_calls_min,
            llm_calls_max: old.llm_calls_max,
            execution_llm_calls_min: old.execution_llm_calls_min,
            tool_calls_min: old.tool_calls_min,
            tool_calls_max: old.tool_calls_max,
            weighted_tokens_max: old.weighted_tokens_max,
            forbidden_mutation_attempts_max: old.forbidden_mutation_attempts_max,
            forbids_mutation: old.forbids_mutation,
            plan_steps_completed_min: old.plan_steps_completed_min,
            routing_accuracy_min: old.routing_accuracy_min,
            progress_yield_min: old.progress_yield_min,
            contract_fulfillment_min: old.contract_fulfillment_min,
            cost_efficiency_min: old.cost_efficiency_min,
            overall_min: old.overall_min,
            direct_return: old.direct_return,
            response_fallthrough: old.response_fallthrough,
            guard_fired: old.guard_fired.clone(),
            decision_types_seen: old.decision_types_seen.clone(),
        }),
        ..Expect::default()
    }
}

fn translate(old: &HarnessEvalFixture) -> DaemonScenario {
    let script: Vec<ScriptedReply> = old
        .mock_responses
        .iter()
        .map(|spec| match spec {
            MockResponseSpec::Text { text } => ScriptedReply {
                text: Some(text.clone()),
                tool_calls: None,
                usage: None,
            },
            MockResponseSpec::ToolCall { tool_call } => ScriptedReply {
                text: None,
                tool_calls: Some(vec![ScriptedCall {
                    id: None,
                    name: tool_call.name.clone(),
                    arguments: ScriptedArguments::Text(tool_call.arguments.clone()),
                }]),
                usage: None,
            },
        })
        .collect();
    DaemonScenario {
        name: old.name.clone(),
        description: old.description.clone(),
        input: old.user_text.clone(),
        instructions: None,
        tags: Vec::new(),
        script: (!script.is_empty()).then_some(script),
        limits: None,
        setup: Some(DaemonSetup {
            session_id: Some(old.session_id.clone()),
            user_role: old.user_role.clone(),
            orchestrator: old.orchestrator,
            routing_models: old.routing_models,
            session_mode: old.session_mode.clone(),
            seed: old.seed.clone(),
        }),
        expect: translate_expect(&old.expect),
        source: None,
    }
}

/// `true` when the new checks pass this run.
fn new_verdict(expect: &ExpectBlock, trajectory: &Trajectory<HarnessEvalRunResult>) -> bool {
    let expect = translate_expect(expect);
    let mut failures = check_trajectory(&expect, trajectory);
    failures.extend(DaemonSubject.check_ext(expect.ext.as_ref().unwrap(), trajectory));
    failures.is_empty()
}

const GHOST: &str = "__differential_ghost__";

struct Mutant {
    label: String,
    expect: ExpectBlock,
    must_pass: bool,
}

/// Flip the case of every ASCII letter.
fn flip_case(text: &str) -> String {
    text.chars()
        .map(|c| {
            if c.is_ascii_uppercase() {
                c.to_ascii_lowercase()
            } else {
                c.to_ascii_uppercase()
            }
        })
        .collect()
}

fn mutants(base: &ExpectBlock, r: &HarnessEvalRunResult) -> Vec<Mutant> {
    let eval = &r.harness_eval;
    let mut out = vec![
        Mutant {
            label: "original".into(),
            expect: base.clone(),
            must_pass: true,
        },
        Mutant {
            label: "empty".into(),
            expect: ExpectBlock::default(),
            must_pass: true,
        },
    ];
    let mut add = |label: &str, must_pass: bool, edit: &dyn Fn(&mut ExpectBlock)| {
        let mut expect = base.clone();
        edit(&mut expect);
        out.push(Mutant {
            label: label.to_string(),
            expect,
            must_pass,
        });
    };
    let ghost = || GHOST.to_string();
    let calls = r.task_end.tool_calls_count;
    let weighted = eval.cost.weighted_tokens;
    let attempts = eval.quality.contract.forbidden_mutation_attempts;
    let plan_steps = eval.progress.plan_steps_completed.unwrap_or(0);
    let proposed = count_names(&r.tool_names);
    let dispatched = count_names(&r.dispatched_tool_names);

    add("route wrong", false, &|e| {
        e.orchestration_route = Some(ghost())
    });
    add("route right", true, &|e| {
        e.orchestration_route = Some(eval.orchestration_route.clone())
    });
    add("predicted wrong", false, &|e| {
        e.tools_required_predicted = Some(!eval.routing.tools_required_predicted)
    });
    add("outcome wrong", false, &|e| e.outcome = Some(ghost()));
    add("outcome right", true, &|e| {
        e.outcome = Some(r.task_end.effective_outcome().as_str().to_string())
    });
    add("stop wrong", false, &|e| e.stop_reason = Some(ghost()));
    add("stop right", true, &|e| {
        e.stop_reason = Some(eval.quality.stop_reason.clone())
    });

    add("used ghost", false, &|e| e.tools_used.push(ghost()));
    add("not used ghost", true, &|e| e.tools_not_used.push(ghost()));
    add("order ghost", false, &|e| e.tools_in_order = vec![ghost()]);
    add("order actual", true, &|e| {
        e.tools_in_order = r.tool_names.clone()
    });
    add("count ghost", false, &|e| {
        e.tool_call_counts.insert(ghost(), 1);
    });
    add("dispatch ghost", false, &|e| {
        e.tool_dispatch_counts.insert(ghost(), 1);
    });
    for (tool, n) in &proposed {
        add(&format!("used {tool}"), true, &|e| {
            e.tools_used.push(tool.clone())
        });
        add(&format!("not used {tool}"), false, &|e| {
            e.tools_not_used.push(tool.clone())
        });
        add(&format!("count {tool} exact"), true, &|e| {
            e.tool_call_counts.insert(tool.clone(), *n);
        });
        add(&format!("count {tool} high"), false, &|e| {
            e.tool_call_counts.insert(tool.clone(), n + 1);
        });
    }
    for (tool, n) in &dispatched {
        add(&format!("dispatch {tool} exact"), true, &|e| {
            e.tool_dispatch_counts.insert(tool.clone(), *n);
        });
        add(&format!("dispatch {tool} high"), false, &|e| {
            e.tool_dispatch_counts.insert(tool.clone(), n + 1);
        });
    }
    let mut reversed = r.tool_names.clone();
    reversed.reverse();
    if reversed != r.tool_names {
        add("order reversed", false, &|e| {
            e.tools_in_order = reversed.clone()
        });
    }

    add("llm min high", false, &|e| {
        e.llm_calls_min = Some(r.llm_calls + 1)
    });
    add("llm bounds exact", true, &|e| {
        e.llm_calls_min = Some(r.llm_calls);
        e.llm_calls_max = Some(r.llm_calls);
    });
    add("exec min high", false, &|e| {
        e.execution_llm_calls_min = Some(r.execution_llm_calls + 1)
    });
    add("exec bounds exact", true, &|e| {
        e.execution_llm_calls_min = Some(r.execution_llm_calls);
        e.execution_llm_calls_max = Some(r.execution_llm_calls);
    });
    add("calls min high", false, &|e| {
        e.tool_calls_min = Some(calls + 1)
    });
    add("calls bounds exact", true, &|e| {
        e.tool_calls_min = Some(calls);
        e.tool_calls_max = Some(calls);
    });
    add("weighted exact", true, &|e| {
        e.weighted_tokens_max = Some(weighted)
    });
    add("attempts exact", true, &|e| {
        e.forbidden_mutation_attempts_max = Some(attempts)
    });
    if r.llm_calls > 0 {
        add("llm max low", false, &|e| {
            e.llm_calls_max = Some(r.llm_calls - 1)
        });
    }
    if r.execution_llm_calls > 0 {
        add("exec max low", false, &|e| {
            e.execution_llm_calls_max = Some(r.execution_llm_calls - 1)
        });
    }
    if calls > 0 {
        add("calls max low", false, &|e| {
            e.tool_calls_max = Some(calls - 1)
        });
    }
    if weighted > 0 {
        add("weighted low", false, &|e| {
            e.weighted_tokens_max = Some(weighted - 1)
        });
    }
    if attempts > 0 {
        add("attempts low", false, &|e| {
            e.forbidden_mutation_attempts_max = Some(attempts - 1)
        });
    }

    add("forbids wrong", false, &|e| {
        e.forbids_mutation = Some(!eval.quality.contract.forbids_mutation)
    });
    add("plan high", false, &|e| {
        e.plan_steps_completed_min = Some(plan_steps + 1)
    });
    add("plan exact", true, &|e| {
        e.plan_steps_completed_min = Some(plan_steps)
    });
    add("routing score high", false, &|e| {
        e.routing_accuracy_min = Some(eval.scores.routing_accuracy + 0.5)
    });
    add("progress score high", false, &|e| {
        e.progress_yield_min = Some(eval.scores.progress_yield + 0.5)
    });
    add("contract score high", false, &|e| {
        e.contract_fulfillment_min = Some(eval.scores.contract_fulfillment + 0.5)
    });
    add("cost score high", false, &|e| {
        e.cost_efficiency_min = Some(eval.scores.cost_efficiency + 0.5)
    });
    add("overall score high", false, &|e| {
        e.overall_min = Some(eval.scores.overall + 0.5)
    });
    add("scores exact", true, &|e| {
        e.routing_accuracy_min = Some(eval.scores.routing_accuracy);
        e.progress_yield_min = Some(eval.scores.progress_yield);
        e.contract_fulfillment_min = Some(eval.scores.contract_fulfillment);
        e.cost_efficiency_min = Some(eval.scores.cost_efficiency);
        e.overall_min = Some(eval.scores.overall);
    });
    add("direct wrong", false, &|e| {
        e.direct_return = Some(!eval.routing.direct_return_attempted)
    });
    add("fallthrough wrong", false, &|e| {
        e.response_fallthrough = Some(!eval.routing.response_fallthrough)
    });

    add("guard unknown", false, &|e| e.guard_fired.push(ghost()));
    for (guard, fires) in [
        (
            "RepetitiveCallDetection",
            eval.progress.repetition_guard_fires,
        ),
        ("Stall", eval.progress.stall_guard_fires),
        ("stall", eval.progress.stall_guard_fires),
    ] {
        add(&format!("guard {guard}"), fires > 0, &|e| {
            e.guard_fired.push(guard.to_string())
        });
    }
    add("decision ghost", false, &|e| {
        e.decision_types_seen.push(ghost())
    });
    if let Some(seen) = r.decision_types.first() {
        add("decision seen", true, &|e| {
            e.decision_types_seen.push(seen.clone())
        });
    }

    add("text ghost", false, &|e| e.response_contains.push(ghost()));
    let head: String = r.response_text.chars().take(24).collect();
    if !head.is_empty() {
        add("text exact", true, &|e| {
            e.response_contains.push(head.clone())
        });
    }
    let flipped = flip_case(&head);
    if flipped != head && !r.response_text.contains(&flipped) {
        add("text case flipped", false, &|e| {
            e.response_contains.push(flipped.clone())
        });
    }
    for needle in &base.response_contains {
        let flipped = flip_case(needle);
        if flipped != *needle && !r.response_text.contains(&flipped) {
            add("expected text case flipped", false, &|e| {
                e.response_contains = vec![flipped.clone()]
            });
        }
    }
    out
}

fn judge(
    fixture: &HarnessEvalFixture,
    trajectory: &Trajectory<HarnessEvalRunResult>,
    mutant: &Mutant,
) -> Option<String> {
    let mut mutated = fixture.clone();
    mutated.expect = mutant.expect.clone();
    let old = assert_expectations(&mutated, &trajectory.observed);
    let new = new_verdict(&mutant.expect, trajectory);
    (old.is_ok() != new || new != mutant.must_pass).then(|| {
        format!(
            "{} / {}: old {:?}, new {new}, must_pass {}",
            fixture.name, mutant.label, old, mutant.must_pass
        )
    })
}

/// Every fixture runs through the subject exactly as through the old runner,
/// and both sets of checks agree on every expectation, real and mutated.
#[tokio::test]
async fn old_and_new_checks_agree_on_every_fixture() {
    let fixtures = load_fixtures_dir(&fixtures_dir()).unwrap();
    assert!(fixtures.len() >= 17, "got {} fixtures", fixtures.len());
    let mut disagreements = Vec::new();
    let (mut mutant_count, mut failing_mutants) = (0usize, 0usize);
    for (path, fixture) in &fixtures {
        let scenario = translate(fixture);
        let trajectory = DaemonSubject
            .run(&scenario, 0)
            .await
            .unwrap_or_else(|err| panic!("{}: {err}", path.display()));

        let old_run = run_fixture(fixture).await.unwrap();
        let new_run = &trajectory.observed;
        assert_eq!(old_run.tool_names, new_run.tool_names, "{}", fixture.name);
        assert_eq!(
            old_run.dispatched_tool_names, new_run.dispatched_tool_names,
            "{}",
            fixture.name
        );
        assert_eq!(
            old_run.response_text, new_run.response_text,
            "{}",
            fixture.name
        );
        assert_eq!(old_run.llm_calls, new_run.llm_calls, "{}", fixture.name);
        assert_eq!(
            old_run.task_end.effective_outcome().as_str(),
            new_run.task_end.effective_outcome().as_str(),
            "{}",
            fixture.name
        );

        for mutant in mutants(&fixture.expect, new_run) {
            mutant_count += 1;
            failing_mutants += usize::from(!mutant.must_pass);
            disagreements.extend(judge(fixture, &trajectory, &mutant));
        }
    }
    assert!(
        disagreements.is_empty(),
        "{} of {mutant_count} mutants disagree:\n{}",
        disagreements.len(),
        disagreements.join("\n")
    );
    assert!(
        failing_mutants > 100,
        "only {failing_mutants} must-fail mutants"
    );
    eprintln!("differential: {mutant_count} mutants, {failing_mutants} must fail");
}

/// Proposals and receipts are separate lanes: a proposal with no receipt moves
/// `tool_call_counts` but not `tool_dispatch_counts`, in both implementations.
#[tokio::test]
async fn old_and_new_checks_read_the_same_lanes_when_proposals_and_receipts_differ() {
    let path = fixtures_dir().join("mock_tool_then_reply.yaml");
    let fixture = super::fixture::load_fixture_file(&path).unwrap();
    let mut result = run_fixture(&fixture).await.unwrap();
    let tool = result.dispatched_tool_names[0].clone();
    let dispatched = count_names(&result.dispatched_tool_names)[&tool];
    let proposed = count_names(&result.tool_names)[&tool];
    assert_eq!(proposed, dispatched, "control run has no suppression");
    result.tool_names.push(tool.clone());
    let trajectory = trajectory_of(result);

    let check = |must_pass: bool, label: &str, edit: &dyn Fn(&mut ExpectBlock)| {
        let mut expect = ExpectBlock::default();
        edit(&mut expect);
        let mutant = Mutant {
            label: label.into(),
            expect,
            must_pass,
        };
        assert_eq!(judge(&fixture, &trajectory, &mutant), None);
    };
    check(true, "proposal lane counts the extra", &|e| {
        e.tool_call_counts.insert(tool.clone(), proposed + 1);
    });
    check(false, "proposal lane is not the receipt count", &|e| {
        e.tool_call_counts.insert(tool.clone(), dispatched);
    });
    check(true, "dispatch lane ignores the extra", &|e| {
        e.tool_dispatch_counts.insert(tool.clone(), dispatched);
    });
    check(false, "dispatch lane is not the proposal count", &|e| {
        e.tool_dispatch_counts.insert(tool.clone(), proposed + 1);
    });
}

/// The files that replace the fixtures say exactly what the translation says.
#[test]
fn scenario_files_match_the_translation() {
    let dir = fixtures_dir().parent().unwrap().join("scenarios");
    let fixtures = load_fixtures_dir(&fixtures_dir()).unwrap();
    for (path, fixture) in &fixtures {
        let file = dir.join(path.file_name().unwrap());
        let mut parsed = DaemonScenario::parse_file(&file).unwrap();
        parsed.source = None;
        assert_eq!(parsed, translate(fixture), "{}", file.display());
    }
    assert_eq!(
        std::fs::read_dir(&dir).unwrap().count(),
        fixtures.len(),
        "scenario files without a fixture"
    );
}
