//! The shortest working agent: one tool written with `#[tool]`, the preset
//! for everything else. Live with ANTHROPIC_API_KEY or OPENAI_API_KEY,
//! otherwise against a scripted model.
//!
//! cargo run -p aidaemon-sdk --example quickstart --features testing
use aidaemon_sdk::prelude::*;
use aidaemon_sdk::testing::{call, text, tool_calls, ScriptedProvider};

/// Today's weather for a city.
#[tool(read_only)]
async fn forecast(city: String) -> String {
    format!("{city}: 21°C and sunny (synthetic data)")
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let agent = aidaemon_sdk::preset("weather")
        .provider_from_env_or(scripted())
        .instructions("Answer weather questions in one sentence. Use the forecast tool.")
        .tool(Forecast)
        .build()
        .await?;
    let outcome = agent
        .run("cli:synthetic-user-1", "Weather in Lisbon?")
        .await?;
    println!("{}", outcome.final_text.unwrap_or_default());
    Ok(())
}

/// What the model says when no API key is set.
fn scripted() -> ScriptedProvider {
    let lookup = call("c1", "forecast", serde_json::json!({"city": "Lisbon"}));
    ScriptedProvider::new([tool_calls([lookup]), text("Lisbon is 21°C and sunny.")])
}
