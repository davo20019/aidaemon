use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use aidaemon_core::{
    Agent, ModelRequest, ModelResponse, Provider, ProviderError, ReceiptStatus, Usage,
};
use aidaemon_eval::examples::{example_tools, ExampleFactory};
use aidaemon_eval::{
    compare, CallRecord, CaseResult, EvalError, ExpectedStatus, FactoryError, FnFactory, Report,
    Scenario, Subject, Suite, Trajectory,
};
use async_trait::async_trait;

fn scenario(yaml: &str) -> Scenario {
    Scenario::from_yaml_str(yaml, "inline.yaml").expect("valid scenario")
}

async fn run(yaml: &str) -> CaseResult {
    scenario(yaml).run(&ExampleFactory::new()).await
}

fn has_failure(result: &CaseResult, prefix: &str) -> bool {
    result.failures.iter().any(|f| f.starts_with(prefix))
}

fn scratch_dir(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("aidaemon-eval-{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

const ROUND_TRIP: &str = r#"
name: rt
input: weather?
script:
  - tool_calls:
      - { name: get_weather, arguments: { city: Springfield } }
      - { name: delete_file, arguments: { path: x.txt } }
  - text: It is Sunny in Springfield.
"#;

// ---------- parsing ----------

#[test]
fn parses_full_scenario() {
    let s = scenario(
        r#"
name: full
description: d
input: hi
instructions: be brief
tags: [a, b]
limits: { max_turns: 3, max_tokens: 1000 }
script:
  - text: hello
    usage: { input_tokens: 10, output_tokens: 5 }
  - tool_calls:
      - { id: c1, name: read_file, arguments: { path: a } }
      - { name: read_file }
expect:
  status: limit_reached
  final_text_contains: [x]
  final_text_not_contains: [y]
  tools_used: [read_file]
  tools_not_used: [delete_file]
  tools_in_order: [read_file]
  tool_call_counts: { read_file: 2 }
  receipt_statuses: { delete_file: [denied, invalid_arguments] }
  max_turns: 2
  max_billable_tokens: 500
  no_failed_receipts: true
"#,
    );
    assert_eq!(s.name, "full");
    assert_eq!(s.tags, vec!["a", "b"]);
    assert_eq!(s.expect.status, Some(ExpectedStatus::LimitReached));
    assert_eq!(
        s.expect.receipt_statuses["delete_file"],
        vec![ReceiptStatus::Denied, ReceiptStatus::InvalidArguments]
    );
    let responses = s.scripted_responses();
    assert_eq!(responses[0].usage.input_tokens, 10);
    assert_eq!(responses[1].tool_calls[0].id, "c1");
    assert_eq!(responses[1].tool_calls[1].id, "call_2_1");
    assert!(responses[1].tool_calls[1].arguments.is_object());
    assert_eq!(s.source.as_deref(), Some(Path::new("inline.yaml")));
}

#[test]
fn script_arguments_are_a_string_or_an_object() {
    let s = scenario(
        r#"
name: args
input: hi
script:
  - tool_calls:
      - { name: a, arguments: { path: x } }
      - { name: b, arguments: '{"path":"y"}' }
      - { name: c, arguments: 'not json' }
      - { name: d }
"#,
    );
    let calls = &s.scripted_responses()[0].tool_calls;
    assert_eq!(calls[0].arguments["path"], "x");
    assert_eq!(calls[1].arguments["path"], "y");
    assert_eq!(calls[2].arguments, serde_json::json!("not json"));
    assert_eq!(calls[3].arguments, serde_json::json!({}));

    let replies = s.script.unwrap();
    let raw = replies[0].tool_calls.as_ref().unwrap();
    assert_eq!(raw[0].arguments.to_json_string(), r#"{"path":"x"}"#);
    assert_eq!(raw[1].arguments.to_json_string(), r#"{"path":"y"}"#);
    assert_eq!(raw[2].arguments.to_json_string(), "not json");
}

#[test]
fn unknown_fields_are_rejected_everywhere() {
    let cases = [
        ("top level", "name: a\ninput: b\nexpekt: {}\n"),
        ("expect", "name: a\ninput: b\nexpect: { tools_usd: [x] }\n"),
        ("script reply", "name: a\ninput: b\nscript: [{ txt: hi }]\n"),
        (
            "tool call",
            "name: a\ninput: b\nscript: [{ tool_calls: [{ name: x, args: {} }] }]\n",
        ),
        ("limits", "name: a\ninput: b\nlimits: { turns: 3 }\n"),
        (
            "status value",
            "name: a\ninput: b\nexpect: { status: done }\n",
        ),
    ];
    for (what, yaml) in cases {
        let err = Scenario::from_yaml_str(yaml, "cases/typo.yaml").unwrap_err();
        let msg = err.to_string();
        assert!(matches!(err, EvalError::Parse { .. }), "{what}: {msg}");
        assert!(msg.contains("cases/typo.yaml"), "{what}: {msg}");
    }
}

#[test]
fn validation_errors_name_the_problem() {
    let err = Scenario::from_yaml_str("name: a\ninput: b\nscript: [{}]\n", "v.yaml").unwrap_err();
    assert!(matches!(err, EvalError::Invalid { .. }));
    assert!(err.to_string().contains("reply 1"), "{err}");
    let err = Scenario::from_yaml_str("name: a\ninput: b\nscript: []\n", "v.yaml").unwrap_err();
    assert!(err.to_string().contains("at least one reply"), "{err}");
    let err = Scenario::from_yaml_str("name: ' '\ninput: b\n", "v.yaml").unwrap_err();
    assert!(err.to_string().contains("name"), "{err}");
    let err = Scenario::from_yaml_str("name: a\ninput: b\nlimits: {max_turns: 0}\n", "v.yaml")
        .unwrap_err();
    assert!(err.to_string().contains("max_turns"), "{err}");
}

#[test]
fn from_dir_sorts_and_reports_paths() {
    let dir = scratch_dir("load");
    std::fs::write(dir.join("b.yml"), "name: second\ninput: x\n").unwrap();
    std::fs::write(dir.join("a.yaml"), "name: first\ninput: x\n").unwrap();
    std::fs::write(dir.join("notes.txt"), "ignored").unwrap();
    let suite = Suite::from_dir(&dir).unwrap();
    let names: Vec<&str> = suite.scenarios.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, ["first", "second"]);

    std::fs::write(
        dir.join("c.yaml"),
        "name: bad\ninput: x\nexpect: {nope: 1}\n",
    )
    .unwrap();
    let err = Suite::from_dir(&dir).unwrap_err().to_string();
    assert!(err.contains("c.yaml") && err.contains("nope"), "{err}");

    std::fs::write(dir.join("c.yaml"), "name: first\ninput: x\n").unwrap();
    let err = Suite::from_dir(&dir).unwrap_err();
    assert!(matches!(err, EvalError::DuplicateName { .. }), "{err}");
    assert!(err.to_string().contains("a.yaml"), "{err}");

    let empty = scratch_dir("empty");
    assert!(matches!(
        Suite::from_dir(&empty).unwrap_err(),
        EvalError::Empty { .. }
    ));
    let missing = empty.join("missing");
    let err = Suite::from_dir(&missing).unwrap_err();
    assert!(matches!(err, EvalError::Io { .. }));
    assert!(err.to_string().contains("missing"), "{err}");
}

// ---------- expectations ----------

#[tokio::test]
async fn all_expectations_pass_on_a_matching_run() {
    let yaml = format!(
        "{ROUND_TRIP}expect:
  status: completed
  final_text_contains: [SUNNY, springfield]
  final_text_not_contains: [rain]
  tools_used: [get_weather, delete_file]
  tools_not_used: [read_file]
  tools_in_order: [get_weather, delete_file]
  tool_call_counts: {{ get_weather: 1, delete_file: 1, read_file: 0 }}
  receipt_statuses: {{ get_weather: [succeeded], delete_file: [denied] }}
  max_turns: 2
  max_billable_tokens: 240
  no_failed_receipts: true
"
    );
    let result = run(&yaml).await;
    assert!(result.passed, "{:?}", result.failures);
    assert_eq!(result.status, "completed");
    assert_eq!(result.turns, 2);
    assert_eq!(result.billable_tokens, 240);
    assert_eq!(result.tools, ["get_weather", "delete_file"]);
}

#[tokio::test]
async fn every_failing_expectation_is_reported() {
    let yaml = format!(
        "{ROUND_TRIP}expect:
  status: stuck
  final_text_contains: [rain]
  final_text_not_contains: [sunny]
  tools_used: [read_file]
  tools_not_used: [get_weather]
  tools_in_order: [delete_file, get_weather]
  tool_call_counts: {{ get_weather: 2 }}
  receipt_statuses: {{ delete_file: [succeeded] }}
  max_turns: 1
  max_billable_tokens: 100
"
    );
    let result = run(&yaml).await;
    assert!(!result.passed);
    let prefixes = [
        "status: expected `stuck`, got `completed`",
        "final_text_contains: missing `rain`",
        "final_text_not_contains: found `sunny`",
        "tools_used: `read_file` was never called",
        "tools_not_used: `get_weather` was called 1 time(s)",
        "tools_in_order:",
        "tool_call_counts: `get_weather` expected 2 call(s), got 1",
        "receipt_statuses: `delete_file` expected [succeeded], got [denied]",
        "max_turns: used 2 turns, limit 1",
        "max_billable_tokens: used 240 billable tokens, limit 100",
    ];
    for p in prefixes {
        assert!(
            has_failure(&result, p),
            "missing `{p}` in {:#?}",
            result.failures
        );
    }
    assert_eq!(
        result.failures.len(),
        prefixes.len(),
        "{:#?}",
        result.failures
    );
}

#[tokio::test]
async fn no_failed_receipts_flags_real_failures_but_not_denials() {
    let result = run(r#"
name: bad_args
input: x
script:
  - tool_calls:
      - { name: read_file, arguments: {} }
      - { name: nonexistent }
      - { name: delete_file, arguments: { path: a } }
  - text: done
expect:
  no_failed_receipts: true
"#)
    .await;
    assert_eq!(result.failures.len(), 1, "{:?}", result.failures);
    let msg = &result.failures[0];
    assert!(msg.contains("read_file invalid_arguments"), "{msg}");
    assert!(msg.contains("nonexistent unknown_tool"), "{msg}");
    assert!(!msg.contains("delete_file"), "{msg}");
}

#[tokio::test]
async fn missing_final_text_is_explained() {
    let result = run(r#"
name: limit
input: x
limits: { max_turns: 1 }
script:
  - tool_calls: [{ name: read_file, arguments: { path: a } }]
expect:
  status: limit_reached
  final_text_contains: [anything]
"#)
    .await;
    assert_eq!(result.failures.len(), 1, "{:?}", result.failures);
    assert!(result.failures[0].contains("no final text"));
}

// ---------- script consumption ----------

#[tokio::test]
async fn script_running_out_fails() {
    let result = run(r#"
name: short
input: x
script:
  - tool_calls: [{ name: read_file, arguments: { path: a } }]
expect:
  status: provider_failed
"#)
    .await;
    assert_eq!(result.status, "provider_failed");
    assert!(!result.passed);
    assert!(
        has_failure(&result, "script ran out"),
        "{:?}",
        result.failures
    );
}

#[tokio::test]
async fn unconsumed_script_fails() {
    let result = run(r#"
name: long
input: x
script:
  - text: done early
  - text: never sent
expect:
  status: completed
"#)
    .await;
    assert!(!result.passed);
    assert_eq!(
        result.failures,
        ["script not fully consumed: 1 of 2 replies unused"]
    );
}

#[tokio::test]
async fn unscripted_without_live_provider_fails_clearly() {
    let result = run("name: live\ninput: x\n").await;
    assert_eq!(result.status, "error");
    assert!(result.failures[0].contains("no live provider"));
}

#[tokio::test]
async fn factory_errors_are_failures_and_configure_applies_limits() {
    let failing = FnFactory::new(|_s: &Scenario, _p: Arc<dyn Provider>| Err("no tools".into()));
    let result = scenario(ROUND_TRIP).run(&failing).await;
    assert_eq!(result.failures, ["agent factory failed: no tools"]);

    // A closure factory that applies the scenario's limits.
    let factory = FnFactory::new(|s: &Scenario, p: Arc<dyn Provider>| {
        let mut b = Agent::builder("fn")
            .allow_unguarded()
            .model("m")
            .provider(p);
        for t in example_tools() {
            b = b.tool_arc(t);
        }
        Ok(s.configure(b).build()?)
    });
    let result = scenario(
        r#"
name: capped
input: x
limits: { max_tokens: 100 }
script:
  - tool_calls: [{ name: read_file, arguments: { path: a } }]
expect: { status: limit_reached, max_turns: 1 }
"#,
    )
    .run(&factory)
    .await;
    assert!(result.passed, "{:?}", result.failures);
}

// ---------- N-of-M ----------

/// Answers correctly on every other call, whatever the request.
struct Flaky {
    calls: AtomicUsize,
}

#[async_trait]
impl Provider for Flaky {
    fn name(&self) -> &str {
        "flaky"
    }
    async fn complete(&self, _request: &ModelRequest) -> Result<ModelResponse, ProviderError> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        tokio::task::yield_now().await;
        let text = if n.is_multiple_of(2) {
            "The answer is 42."
        } else {
            "I am not sure."
        };
        Ok(ModelResponse {
            text: text.into(),
            tool_calls: Vec::new(),
            usage: Usage {
                input_tokens: 50,
                output_tokens: 10,
                cached_input_tokens: 20,
                cache_creation_tokens: 0,
            },
            ..Default::default()
        })
    }
}

#[tokio::test]
async fn n_of_m_with_flaky_provider() {
    let live: Arc<dyn Provider> = Arc::new(Flaky {
        calls: AtomicUsize::new(0),
    });
    let factory = ExampleFactory::with_live_provider(live);
    let suite = Suite::new(vec![
        scenario("name: flaky\ninput: answer?\nexpect: { final_text_contains: ['42'] }\n"),
        scenario("name: steady\ninput: x\nscript: [{ text: ok }]\nexpect: { status: completed }\n"),
    ])
    .parallelism(3);

    let report = suite.run(&factory, 10).await;
    assert_eq!(report.total_runs, 20);
    assert_eq!(report.total_passes, 15);
    assert!((report.pass_rate - 0.75).abs() < 1e-9);

    let flaky = report.scenario("flaky").unwrap();
    assert_eq!((flaky.runs, flaky.passes), (10, 5));
    assert!((flaky.pass_rate - 0.5).abs() < 1e-9);
    assert!((flaky.mean_turns - 1.0).abs() < 1e-9);
    assert!((flaky.mean_billable_tokens - 40.0).abs() < 1e-9);
    assert!(flaky.mean_duration_ms >= 0.0);
    assert_eq!(flaky.failures.len(), 1);
    assert_eq!(flaky.failures[0].count, 5);
    assert!(flaky.failures[0].message.contains("missing `42`"));

    let steady = report.scenario("steady").unwrap();
    assert_eq!(steady.passes, 10);
    assert!((steady.mean_billable_tokens - 120.0).abs() < 1e-9);
    assert!(!report.all_passed());

    let cases = suite.run_cases(&factory, 3).await;
    assert_eq!(cases.len(), 2);
    assert!(cases.iter().all(|runs| runs.len() == 3));
    assert!(cases[1].iter().all(|r| r.scenario == "steady"));
}

fn case(
    name: &str,
    passed: bool,
    failures: &[&str],
    turns: u32,
    tokens: u64,
    ms: f64,
) -> CaseResult {
    CaseResult {
        scenario: name.into(),
        passed,
        failures: failures.iter().map(|f| f.to_string()).collect(),
        status: "completed".into(),
        final_text: None,
        tools: Vec::new(),
        turns,
        billable_tokens: tokens,
        duration_ms: ms,
    }
}

fn sample_report() -> Report {
    Report::from_results(&[
        vec![
            case("alpha", true, &[], 1, 100, 10.0),
            case("alpha", false, &["b", "a"], 3, 300, 30.0),
            case("alpha", false, &["a"], 2, 200, 20.0),
            case("alpha", true, &[], 2, 200, 20.0),
        ],
        vec![case("beta|pipe", true, &[], 1, 50, 5.0)],
        vec![],
    ])
}

#[test]
fn aggregation_math() {
    let report = sample_report();
    assert_eq!(report.scenarios.len(), 2, "empty groups are skipped");
    assert_eq!((report.total_runs, report.total_passes), (5, 3));
    assert!((report.pass_rate - 0.6).abs() < 1e-9);
    let a = report.scenario("alpha").unwrap();
    assert!((a.pass_rate - 0.5).abs() < 1e-9);
    assert!((a.mean_turns - 2.0).abs() < 1e-9);
    assert!((a.mean_billable_tokens - 200.0).abs() < 1e-9);
    assert!((a.mean_duration_ms - 20.0).abs() < 1e-9);
    let grouped: Vec<(&str, usize)> = a
        .failures
        .iter()
        .map(|f| (f.message.as_str(), f.count))
        .collect();
    assert_eq!(grouped, [("a", 2), ("b", 1)]);
}

#[test]
fn markdown_and_json_shape() {
    let report = sample_report();
    let md = report.to_markdown();
    assert!(md.starts_with("# Eval report"), "{md}");
    assert!(md.contains("Overall pass rate: 60.0% (3/5)"), "{md}");
    assert!(md.contains("| Scenario | Pass rate |"), "{md}");
    assert!(
        md.contains("| alpha | 50.0% | 2/4 | 2.0 | 200 | 20.0 |"),
        "{md}"
    );
    assert!(md.contains("beta\\|pipe"), "{md}");
    assert!(
        md.contains("## Failures\n\n### alpha\n\n- 2x a\n- 1x b"),
        "{md}"
    );

    let json: serde_json::Value = serde_json::from_str(&report.to_json()).unwrap();
    assert_eq!(json["total_runs"], 5);
    assert_eq!(json["scenarios"][0]["name"], "alpha");
    assert_eq!(json["scenarios"][0]["failures"][0]["count"], 2);
    assert_eq!(Report::from_json(&report.to_json()).unwrap(), report);
    assert!(matches!(
        Report::from_json("{"),
        Err(EvalError::ReportJson(_))
    ));
}

#[test]
fn compare_shows_deltas() {
    let baseline = Report::from_results(&[
        vec![
            case("a", true, &[], 1, 1, 1.0),
            case("a", false, &["x"], 1, 1, 1.0),
        ],
        vec![case("b", true, &[], 1, 1, 1.0)],
        vec![case("gone", true, &[], 1, 1, 1.0)],
    ]);
    let candidate = Report::from_results(&[
        vec![
            case("a", true, &[], 1, 1, 1.0),
            case("a", true, &[], 1, 1, 1.0),
        ],
        vec![case("b", false, &["y"], 1, 1, 1.0)],
        vec![case("new", true, &[], 1, 1, 1.0)],
    ]);
    let cmp = compare(&baseline, &candidate);
    let rows: Vec<(&str, Option<f64>)> = cmp
        .rows
        .iter()
        .map(|r| (r.scenario.as_str(), r.delta))
        .collect();
    assert_eq!(
        rows,
        [
            ("a", Some(0.5)),
            ("b", Some(-1.0)),
            ("gone", None),
            ("new", None)
        ]
    );
    assert_eq!(cmp.rows[2].candidate_pass_rate, None);
    assert_eq!(cmp.rows[3].baseline_pass_rate, None);
    let regressions: Vec<&str> = cmp.regressions().map(|r| r.scenario.as_str()).collect();
    assert_eq!(regressions, ["b"]);
    assert!((cmp.delta - 0.0).abs() < 1e-9);

    let md = cmp.to_markdown();
    assert!(
        md.contains("| a | 50.0% (n=2) | 100.0% (n=2) | +50.0 pp |"),
        "{md}"
    );
    assert!(
        md.contains("| b | 100.0% (n=1) | 0.0% (n=1) | -100.0 pp |"),
        "{md}"
    );
    assert!(md.contains("| gone | 100.0% (n=1) | - | - |"), "{md}");
    assert!(md.contains("| new | - | 100.0% (n=1) | - |"), "{md}");
    let json: serde_json::Value = serde_json::from_str(&cmp.to_json()).unwrap();
    assert_eq!(json["rows"][1]["delta"], -1.0);
}

// ---------- shipped scenarios ----------

#[tokio::test]
async fn shipped_scenarios_all_pass() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/scenarios");
    let suite = Suite::from_dir(dir).unwrap();
    assert!(suite.scenarios.len() >= 5);
    let report = suite.run(&ExampleFactory::new(), 3).await;
    assert!(report.all_passed(), "{}", report.to_markdown());
    let statuses: Vec<String> = suite
        .run_cases(&ExampleFactory::new(), 1)
        .await
        .into_iter()
        .map(|runs| runs[0].status.clone())
        .collect();
    for s in ["completed", "stuck", "limit_reached"] {
        assert!(statuses.iter().any(|x| x == s), "{statuses:?}");
    }
}

/// A run a tool finished has its own status, never "unknown", and a
/// scenario can expect it.
#[test]
fn finished_runs_have_their_own_status() {
    let status = aidaemon_core::RunStatus::Finished {
        outcome: aidaemon_core::FinishOutcome::Partial,
        tool: "file_report".into(),
    };
    assert_eq!(aidaemon_eval::status_name(&status), "finished");
    let expected: ExpectedStatus = serde_yaml::from_str("finished").unwrap();
    assert_eq!(expected, ExpectedStatus::Finished);
    assert_eq!(expected.as_str(), aidaemon_eval::status_name(&status));
}

/// `completed` says the run stopped; `succeeded` says its objective was met.
#[test]
fn succeeded_is_checked_separately_from_completed() {
    let expect = |yaml: &str| -> aidaemon_eval::Expect { serde_yaml::from_str(yaml).unwrap() };
    let mut outcome = aidaemon_core::RunOutcome {
        status: aidaemon_core::RunStatus::Completed,
        final_text: Some("done".into()),
        state: aidaemon_core::RunState::new("a", "s", "go"),
    };
    let both = expect("{ status: completed, succeeded: true }");
    assert!(aidaemon_eval::check_expectations(&both, &outcome).is_empty());

    outcome.state.unmet = vec!["checklist: [-] email the report".into()];
    let failures = aidaemon_eval::check_expectations(&both, &outcome);
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(failures[0].starts_with("succeeded:") && failures[0].contains("email the report"));
    let stopped_unmet = expect("{ status: completed, succeeded: false }");
    assert!(aidaemon_eval::check_expectations(&stopped_unmet, &outcome).is_empty());
}

// ---------- the Subject seam ----------

/// A subject that is not a kernel agent: it just reports what it did.
struct Canned(Trajectory);

#[async_trait]
impl Subject for Canned {
    type Observed = ();

    async fn run(&self, _scenario: &Scenario, _run: usize) -> Result<Trajectory, FactoryError> {
        Ok(self.0.clone())
    }
}

fn canned_trajectory() -> Trajectory {
    Trajectory {
        status: "completed".into(),
        status_detail: String::new(),
        succeeded: true,
        unmet: Vec::new(),
        final_text: Some("Report for Acme Corp".into()),
        receipts: vec![CallRecord {
            tool: "read_file".into(),
            status: None,
            dispatched: true,
        }],
        turns: 2,
        billable_tokens: 30,
        run_failures: Vec::new(),
        observed: (),
    }
}

#[tokio::test]
async fn any_subject_is_checked_against_the_same_expectations() {
    let s = scenario(
        r#"
name: canned
input: hi
expect:
  status: completed
  final_text_contains: [acme]
  tools_used: [read_file]
  max_turns: 1
  no_failed_receipts: true
"#,
    );
    let result = s.run_on(&Canned(canned_trajectory()), 0).await;
    assert_eq!(result.status, "completed");
    assert_eq!(result.tools, vec!["read_file"]);
    assert_eq!(result.failures.len(), 1, "{:?}", result.failures);
    assert!(has_failure(&result, "max_turns"), "{:?}", result.failures);
}

#[tokio::test]
async fn run_failures_fail_the_case_and_subject_errors_are_reported() {
    let s = scenario("name: canned\ninput: hi\n");
    let mut trajectory = canned_trajectory();
    trajectory.run_failures.push("script left over".into());
    let result = s.run_on(&Canned(trajectory), 0).await;
    assert!(!result.passed);
    assert_eq!(result.failures, vec!["script left over"]);

    struct Broken;
    #[async_trait]
    impl Subject for Broken {
        type Observed = ();
        async fn run(&self, _: &Scenario, _: usize) -> Result<Trajectory, FactoryError> {
            Err("no database".into())
        }
    }
    let result = s.run_on(&Broken, 0).await;
    assert_eq!(result.status, "error");
    assert_eq!(result.failures, vec!["no database"]);
}
