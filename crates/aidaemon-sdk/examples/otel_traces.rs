//! Traces exported with OpenTelemetry (the `otel` feature of
//! `aidaemon-policies`): one span per run segment, with a child per model
//! call and tool call, following the GenAI conventions. Set
//! OTEL_EXPORTER_OTLP_ENDPOINT (a collector, Langfuse, LangSmith...) to
//! export over OTLP/HTTP; without it the spans are kept in memory and
//! printed. By default spans hold no prompt or reply text.
//!
//! Live with ANTHROPIC_API_KEY or OPENAI_API_KEY, otherwise scripted:
//! cargo run -p aidaemon-sdk --example otel_traces --features "testing otel"
use aidaemon_sdk::policies::{otlp_provider, OtelSink};
use aidaemon_sdk::prelude::*;
use aidaemon_sdk::testing::{call, text, tool_calls, ScriptedProvider};
use opentelemetry_sdk::trace::{InMemorySpanExporter, SdkTracerProvider};
use serde_json::json;

/// Today's weather for a city.
#[tool(read_only)]
async fn forecast(city: String) -> String {
    format!("{city}: 21°C and sunny (synthetic data)")
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let memory = InMemorySpanExporter::default();
    let tracer = match std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT") {
        Ok(endpoint) => otlp_provider(&endpoint, Default::default(), "weather-agent")?,
        Err(_) => SdkTracerProvider::builder()
            .with_simple_exporter(memory.clone())
            .build(),
    };

    let sink = OtelSink::new(&tracer);
    let agent = aidaemon_sdk::preset("weather")
        .provider_from_env_or(ScriptedProvider::new([
            tool_calls([call("c1", "forecast", json!({"city": "Lisbon"}))]),
            text("Lisbon is 21°C and sunny today."),
        ]))
        .instructions("Answer weather questions in one sentence. Use the forecast tool.")
        .tool(Forecast)
        .configure(move |builder| builder.event_sink(sink))
        .build()
        .await?;
    let outcome = agent
        .run("cli:synthetic-user-1", "Weather in Lisbon?")
        .await?;
    println!("{}", outcome.final_text.unwrap_or_default());

    tracer.force_flush()?;
    for span in memory.get_finished_spans()? {
        println!(
            "span {:<28} {} attributes",
            span.name,
            span.attributes.len()
        );
    }
    tracer.shutdown()?;
    Ok(())
}
