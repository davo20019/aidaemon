// Runs the shadow LLM intent classifier against a hand-curated corpus and
// reports agreement with the heuristic baseline plus latency.
//
// `#[ignore]`d by default — hits a real LLM API and costs money.
// Run with:
//
//   cargo test --lib intent_classifier_bench -- --ignored --nocapture
//
// Requires `PROVIDER_API_KEY` in env (loaded from `.env`). Uses OpenRouter
// as the base URL and `google/gemini-2.5-flash` as the classifier model
// unless overridden via `PROVIDER_BASE_URL` / `CLASSIFIER_MODEL`.

use std::time::Instant;

use crate::agent::llm_classifier::{classify_intent, LlmIntentClass};
use crate::providers::OpenAiCompatibleProvider;

struct CorpusCase {
    user_text: &'static str,
    heuristic_class: LlmIntentClass,
    note: &'static str,
}

fn corpus() -> Vec<CorpusCase> {
    use LlmIntentClass::*;
    vec![
        // --- Memory storage ---
        CorpusCase {
            user_text: "Remember my birthday is October 15",
            heuristic_class: MemoryStorage,
            note: "schedule-hijack regression — date must not trigger scheduling",
        },
        CorpusCase {
            user_text: "Remember these facts about me: I drink coffee black, I live in Miami",
            heuristic_class: MemoryStorage,
            note: "bulk fact storage",
        },
        CorpusCase {
            user_text: "Please save my preferences",
            heuristic_class: MemoryStorage,
            note: "bare verb",
        },
        CorpusCase {
            user_text: "Note that I work remotely",
            heuristic_class: MemoryStorage,
            note: "imperative note-taking",
        },
        CorpusCase {
            user_text: "Keep in mind I'm allergic to peanuts",
            heuristic_class: MemoryStorage,
            note: "indirect storage phrasing",
        },
        CorpusCase {
            user_text: "Update my work hours to 9am-5pm",
            heuristic_class: MemoryStorage,
            note: "update-style storage",
        },
        // --- Schedule one-shot ---
        CorpusCase {
            user_text: "Remind me at 5pm to call mom",
            heuristic_class: ScheduleOneShot,
            note: "canonical reminder",
        },
        CorpusCase {
            user_text: "Set a reminder for tomorrow morning",
            heuristic_class: ScheduleOneShot,
            note: "relative-time reminder",
        },
        CorpusCase {
            user_text: "Alert me in 30 minutes",
            heuristic_class: ScheduleOneShot,
            note: "duration-based one-shot",
        },
        CorpusCase {
            user_text: "Ping me when the deploy finishes",
            heuristic_class: ScheduleOneShot,
            note: "event-triggered one-shot",
        },
        // --- Schedule recurring ---
        CorpusCase {
            user_text: "Remind me every Monday at 9am to do standup prep",
            heuristic_class: ScheduleRecurring,
            note: "weekly recurring",
        },
        CorpusCase {
            user_text: "Every day at noon, check the production logs",
            heuristic_class: ScheduleRecurring,
            note: "daily recurring",
        },
        // --- Memory recall ---
        CorpusCase {
            user_text: "What do you know about me?",
            heuristic_class: MemoryRecall,
            note: "canonical recall",
        },
        CorpusCase {
            user_text: "Do I have any pets?",
            heuristic_class: MemoryRecall,
            note: "boolean recall",
        },
        CorpusCase {
            user_text: "What's my coffee preference?",
            heuristic_class: MemoryRecall,
            note: "specific fact recall",
        },
        CorpusCase {
            user_text: "Tell me about my daughter",
            heuristic_class: MemoryRecall,
            note: "entity-focused recall",
        },
        // --- Action ---
        CorpusCase {
            user_text: "Create a Python script that prints fibonacci numbers",
            heuristic_class: Action,
            note: "code generation action",
        },
        CorpusCase {
            user_text: "Search the web for the latest Rust release notes",
            heuristic_class: Action,
            note: "web search action",
        },
        CorpusCase {
            user_text: "Run cargo test and tell me what fails",
            heuristic_class: Action,
            note: "shell command action",
        },
        CorpusCase {
            user_text: "Write a haiku and save it to ~/poems.txt",
            heuristic_class: Action,
            note: "compound action (write+save)",
        },
        CorpusCase {
            user_text: "Deploy the changes to production",
            heuristic_class: Action,
            note: "deployment action",
        },
        // --- Knowledge question ---
        CorpusCase {
            user_text: "What is the speed of light?",
            heuristic_class: KnowledgeQuestion,
            note: "factual",
        },
        CorpusCase {
            user_text: "Explain how OAuth2 refresh tokens work",
            heuristic_class: KnowledgeQuestion,
            note: "technical explanation",
        },
        CorpusCase {
            user_text: "What's the difference between TCP and UDP?",
            heuristic_class: KnowledgeQuestion,
            note: "comparison question",
        },
        // --- Tricky / ambiguous ---
        CorpusCase {
            user_text: "What do you know about me? After that, create a Python script with my info.",
            heuristic_class: Action,
            note: "compound: heuristic upgrades to action because of trailing verb",
        },
        CorpusCase {
            user_text: "Do not stop until the deploy succeeds",
            heuristic_class: Action,
            note: "negated cancel — should NOT be cancel intent",
        },
        CorpusCase {
            user_text: "I drink my coffee black",
            heuristic_class: LlmIntentClass::Other,
            note: "implicit fact share, no imperative verb",
        },
    ]
}

#[tokio::test]
#[ignore = "hits a real LLM API; run with `cargo test -- --ignored --nocapture`"]
async fn intent_classifier_bench_run_corpus() {
    let _ = dotenvy::dotenv();

    let api_key = std::env::var("PROVIDER_API_KEY")
        .expect("PROVIDER_API_KEY must be set (in .env or shell) to run this bench");
    let base_url = std::env::var("PROVIDER_BASE_URL")
        .unwrap_or_else(|_| "https://openrouter.ai/api/v1".to_string());
    let model = std::env::var("CLASSIFIER_MODEL")
        .unwrap_or_else(|_| "google/gemini-2.5-flash".to_string());

    println!("\n=== Intent classifier corpus run ===");
    println!("base_url: {base_url}");
    println!("model:    {model}");

    let provider = OpenAiCompatibleProvider::new(&base_url, &api_key)
        .expect("failed to construct provider");

    let cases = corpus();
    let total = cases.len();
    let mut agree = 0usize;
    let mut disagree = 0usize;
    let mut unknown = 0usize;
    let mut total_latency_ms = 0u128;
    let mut disagreements: Vec<(String, LlmIntentClass, LlmIntentClass, &'static str)> = Vec::new();

    for (idx, case) in cases.iter().enumerate() {
        let start = Instant::now();
        let llm = classify_intent(&provider, &model, case.user_text, None).await;
        let elapsed_ms = start.elapsed().as_millis();
        total_latency_ms += elapsed_ms;

        let marker = if llm == LlmIntentClass::Unknown {
            unknown += 1;
            "??"
        } else if llm == case.heuristic_class {
            agree += 1;
            "OK"
        } else {
            disagree += 1;
            disagreements.push((case.user_text.to_string(), case.heuristic_class, llm, case.note));
            "!!"
        };

        println!(
            "[{:>2}/{:>2}] {marker} ({:>5}ms) heur={:<22} llm={:<22} {:?}",
            idx + 1,
            total,
            elapsed_ms,
            case.heuristic_class.as_label(),
            llm.as_label(),
            case.user_text,
        );
    }

    println!("\n=== Summary ===");
    println!("Total cases:    {total}");
    println!(
        "Agree:          {agree} ({:.1}%)",
        100.0 * agree as f64 / total as f64
    );
    println!(
        "Disagree:       {disagree} ({:.1}%)",
        100.0 * disagree as f64 / total as f64
    );
    println!(
        "LLM unknown:    {unknown} ({:.1}%)",
        100.0 * unknown as f64 / total as f64
    );
    println!(
        "Avg latency:    {} ms",
        if total > 0 {
            total_latency_ms / total as u128
        } else {
            0
        }
    );

    if !disagreements.is_empty() {
        println!("\n=== Disagreements (heuristic is not ground truth — review each) ===");
        for (text, heur, llm, note) in &disagreements {
            println!(
                "- heur={:<22} llm={:<22} :: {:?}",
                heur.as_label(),
                llm.as_label(),
                text
            );
            if !note.is_empty() {
                println!("  note: {note}");
            }
        }
    }
}
