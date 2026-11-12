//! A run streamed over Server-Sent Events from axum: `GET /run?q=...`
//! starts a run and streams its events (text deltas, tool calls, the end)
//! as they happen. The example serves one request from itself and exits;
//! with `--serve` it keeps serving on 127.0.0.1:3000 for curl:
//!
//! curl -N 'http://127.0.0.1:3000/run?q=Weather%20in%20Lisbon'
//!
//! Live with ANTHROPIC_API_KEY or OPENAI_API_KEY, otherwise scripted:
//! cargo run -p aidaemon-sdk --example sse_server --features testing
use std::convert::Infallible;

use aidaemon_sdk::prelude::*;
use aidaemon_sdk::testing::{call, text, tool_calls, ScriptedProvider, StreamingProvider};
use aidaemon_sdk::{CancellationToken, RunId};
use axum::extract::{Query, State};
use axum::response::sse::{Event as Sse, Sse as SseResponse};
use axum::routing::get;
use axum::Router;
use futures::Stream;
use serde::Deserialize;
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::{broadcast, mpsc};

/// Today's weather for a city.
#[tool(read_only)]
async fn forecast(city: String) -> String {
    format!("{city}: 21°C and sunny (synthetic data)")
}

#[derive(Clone)]
struct App {
    agent: Agent,
    events: broadcast::Sender<(RunId, Event)>,
}

#[derive(Deserialize)]
struct Ask {
    q: String,
}

async fn run(
    State(app): State<App>,
    Query(ask): Query<Ask>,
) -> SseResponse<impl Stream<Item = Result<Sse, Infallible>>> {
    // Make the state first so its run id is known before any event.
    let state = RunState::new(app.agent.id(), "http:synthetic-user-1", ask.q);
    let run_id = state.run_id;
    let mut events = app.events.subscribe();
    let (tx, rx) = mpsc::channel::<Sse>(256);
    tokio::spawn(async move {
        let outcome = app.agent.run_from(state, CancellationToken::new()).await;
        // Every event of the run was emitted before it returned.
        while let Ok((id, event)) = events.try_recv() {
            if id == run_id {
                let _ = tx.send(sse(&event)).await;
            }
        }
        let status = match outcome {
            Ok(outcome) => json!({"status": outcome.status, "text": outcome.final_text}),
            Err(err) => json!({"error": err.to_string()}),
        };
        let _ = tx
            .send(Sse::default().event("outcome").data(status.to_string()))
            .await;
    });
    let stream = futures::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|event| (Ok(event), rx))
    });
    SseResponse::new(stream)
}

fn sse(event: &Event) -> Sse {
    let body = serde_json::to_value(event).unwrap_or_default();
    let kind = body["type"].as_str().unwrap_or("event").to_string();
    Sse::default().event(kind).data(body.to_string())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (events, _) = broadcast::channel(4096);
    let sink = events.clone();
    let agent = aidaemon_sdk::preset("weather")
        .provider_from_env_or(StreamingProvider(ScriptedProvider::new([
            tool_calls([call("c1", "forecast", json!({"city": "Lisbon"}))]),
            text("Lisbon is 21°C and sunny today."),
        ])))
        .instructions("Answer weather questions in one sentence. Use the forecast tool.")
        .tool(Forecast)
        .configure(move |builder| {
            builder.event_sink(move |run: RunId, event: &Event| {
                let _ = sink.send((run, event.clone()));
            })
        })
        .build()
        .await?;
    let app = Router::new()
        .route("/run", get(run))
        .with_state(App { agent, events });

    let serve = std::env::args().any(|a| a == "--serve");
    let listener = tokio::net::TcpListener::bind(if serve {
        "127.0.0.1:3000"
    } else {
        "127.0.0.1:0"
    })
    .await?;
    let addr = listener.local_addr()?;
    if serve {
        println!("serving on http://{addr}/run?q=...");
        axum::serve(listener, app).await?;
        return Ok(());
    }
    tokio::spawn(async move { axum::serve(listener, app).await });

    // One request, as a browser's EventSource would make it.
    let mut socket = tokio::net::TcpStream::connect(addr).await?;
    let request = "GET /run?q=Weather%20in%20Lisbon%3F HTTP/1.1\r\nHost: localhost\r\n\
                   Accept: text/event-stream\r\nConnection: close\r\n\r\n";
    socket.write_all(request.as_bytes()).await?;
    let mut response = String::new();
    socket.read_to_string(&mut response).await?;
    for line in response.lines().filter(|l| l.starts_with("event:")) {
        println!("{line}");
    }
    let last = response
        .lines()
        .rfind(|l| l.starts_with("data:"))
        .unwrap_or_default();
    println!("{last}");
    Ok(())
}
