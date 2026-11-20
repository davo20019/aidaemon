//! How fast the app's own work is on the real engine, headless: startup,
//! a long conversation, a long sidebar and a streamed reply. Ignored by
//! default; `cargo test --release --test perf -- --ignored --nocapture`.

use std::time::{Duration, Instant};

use aidaemon_client_core::Update;
use aidaemon_client_protocol::ServerFrame;
use nsh_app_aidaemon::{Offline, feed, sent};
use nsh_engine::{EngineConfig, Points, Root, Size};
use nsh_testing::Headless;

fn frame(json: &str) -> Update {
    Update::Frame(serde_json::from_str::<ServerFrame>(json).expect("a frame"))
}

fn wait_sent(kind: &str) -> Result<String, String> {
    for _ in 0..250 {
        if let Some(id) = sent(kind) {
            return Ok(id);
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    Err(format!("no {kind} was sent"))
}

/// CPU time this process has used, user and system.
fn cpu() -> Duration {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    let tv = |t: libc::timeval| Duration::new(t.tv_sec as u64, t.tv_usec as u32 * 1000);
    tv(usage.ru_utime) + tv(usage.ru_stime)
}

const SESSIONS: usize = match option_env!("PERF_FEW") {
    Some(_) => 10,
    None => 500,
};
const HIST: usize = match option_env!("PERF_HIST") {
    Some(_) => 6,
    None => 200,
};

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn session(i: usize) -> String {
    format!(
        r#"{{"id":"app:perf-{i}","channel":"app","title":"Synthetic conversation {i}","writable":true,"last_activity_ms":{}}}"#,
        1_000_000 - i
    )
}

fn history(n: usize) -> String {
    (0..n)
        .map(|i| {
            let role = if i % 2 == 0 { "user" } else { "assistant" };
            let text = if i % 2 == 0 {
                format!("Question {i}: how does the synthetic module handle case {i}?")
            } else {
                format!(
                    "Answer {i}. It **parses** the input, checks `limit_{i}`, then:\n\n- reads the file\n- writes the result\n\n```rust\nfn case_{i}() -> u32 {{ {i} }}\n```\n\nThat covers case {i}."
                )
            };
            serde_json::json!({"id": format!("m{i}"), "role": role, "text": text, "at_ms": 1000 + i as u64})
                .to_string()
        })
        .collect::<Vec<_>>()
        .join(",")
}

#[test]
#[ignore]
fn perf_report() -> Result<(), String> {
    let mut config = EngineConfig::new(Root::of::<Offline>());
    config.viewport = Size::new(Points(1280.0), Points(820.0));
    config.environment.os = nsh_core::platform::Os::MacOs;
    config.environment.idiom = nsh_core::platform::Idiom::Desktop;

    let t = Instant::now();
    let mut app = Headless::start(config).unwrap();
    app.wait_for("What can I help you with?")?;
    let startup = t.elapsed();

    // A long sidebar.
    let sessions: Vec<String> = (0..SESSIONS).map(session).collect();
    let t = Instant::now();
    feed([
        Update::Connected,
        frame(&format!(
            r#"{{"type":"ack","request_id":"r0","result":{{"type":"sessions","sessions":[{}]}}}}"#,
            sessions.join(",")
        )),
    ]);
    app.wait_for("Synthetic conversation 0")?;
    let sidebar = t.elapsed();

    // A long conversation: 200 messages of Markdown with code.
    app.press("Synthetic conversation 0 >> nth=0")?;
    let id = wait_sent("get_history")?;
    let t = Instant::now();
    feed([frame(&format!(
        r#"{{"type":"ack","request_id":"{id}","result":{{"type":"history","session_id":"app:perf-0","messages":[{}]}}}}"#,
        history(HIST)
    ))]);
    app.wait_for(&format!("That covers case {}.", HIST - 1))?;
    let open_long = t.elapsed();

    // Does a long conversation follow a reply that starts streaming?
    feed([frame(
        r#"{"type":"event","seq":1,"at_ms":5000,"session_id":"app:perf-0","event":{"type":"turn_started","turn_id":"t0","session_id":"app:perf-0","origin":"app"}}"#,
    )]);
    let follows = app.wait_for("Working…").is_ok();
    println!("PERF long_conversation_follows_live_row={follows}");

    // A streamed reply in another conversation: 400 small deltas at
    // 100 a second, as a fast model sends them.
    app.press("Synthetic conversation 1 >> nth=0")?;
    std::thread::sleep(Duration::from_millis(100));
    let id = wait_sent("get_history")?;
    feed([frame(&format!(
        r#"{{"type":"ack","request_id":"{id}","result":{{"type":"history","session_id":"app:perf-1","messages":[{}]}}}}"#,
        history(6)
    ))]);
    app.wait_for("That covers case 5.")?;
    feed([frame(
        r#"{"type":"event","seq":2,"at_ms":5000,"session_id":"app:perf-1","event":{"type":"turn_started","turn_id":"t1","session_id":"app:perf-1","origin":"app"}}"#,
    )]);
    app.wait_for("Working…")?;

    // Idle: what the engine costs with nothing happening.
    std::thread::sleep(Duration::from_millis(300));
    let (c, t) = (cpu(), Instant::now());
    std::thread::sleep(Duration::from_secs(2));
    let idle_pct = ms(cpu() - c) / ms(t.elapsed()) * 100.0;

    let (c0, t) = (cpu(), Instant::now());
    for i in 0..400 {
        let word = if i == 399 {
            "\\n\\nStream done.".to_owned()
        } else if i % 40 == 39 {
            "\\n\\n".to_owned()
        } else {
            format!("w{i} ")
        };
        feed([frame(&format!(
            r#"{{"type":"event","at_ms":{},"session_id":"app:perf-1","event":{{"type":"turn_delta","turn_id":"t1","text":"{word}"}}}}"#,
            5001 + i
        ))]);
        std::thread::sleep(Duration::from_millis(10));
    }
    app.wait_for("Stream done.")?;
    let stream = t.elapsed();
    let stream_cpu = cpu() - c0;

    println!("PERF startup_ms={:.1}", ms(startup));
    println!("PERF sidebar_{SESSIONS}_ms={:.1}", ms(sidebar));
    println!("PERF open_{HIST}_msgs_ms={:.1}", ms(open_long));
    println!("PERF idle_cpu_pct={idle_pct:.1}");
    println!(
        "PERF stream_400_deltas_wall_ms={:.0} cpu_ms={:.0} cpu_per_delta_ms={:.2} cpu_pct={:.0}",
        ms(stream),
        ms(stream_cpu),
        ms(stream_cpu) / 400.0,
        ms(stream_cpu) / ms(stream) * 100.0
    );
    Ok(())
}
