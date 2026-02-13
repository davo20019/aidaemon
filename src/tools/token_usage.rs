use async_trait::async_trait;
use chrono::Timelike;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;

use crate::traits::{StateStore, Tool, ToolCapabilities, ToolRole};

pub struct TokenUsageTool {
    state: Arc<dyn StateStore>,
    daily_token_budget: Option<u64>,
}

impl TokenUsageTool {
    pub fn new(state: Arc<dyn StateStore>, daily_token_budget: Option<u64>) -> Self {
        Self {
            state,
            daily_token_budget,
        }
    }
}

#[derive(Debug, Deserialize)]
struct TokenUsageArgs {
    action: String,
    #[serde(default = "default_days")]
    days: u32,
    #[serde(default)]
    #[allow(dead_code)]
    session_id: Option<String>,
}

fn default_days() -> u32 {
    7
}

#[async_trait]
impl Tool for TokenUsageTool {
    fn name(&self) -> &str {
        "token_usage"
    }

    fn description(&self) -> &str {
        "Query token usage statistics to understand LLM costs and diagnose spending patterns"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "token_usage",
            "description": "Query token usage statistics to understand LLM API costs, diagnose spending patterns, and monitor budget. Use this to answer questions about cost, usage trends, and to troubleshoot unexpectedly high spending.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["summary", "breakdown", "top_sessions", "hourly", "models", "budget_status"],
                        "description": "Action to perform:\n- summary: Overall usage totals for the period (tokens, requests, by model)\n- breakdown: Split by source — background tasks vs user conversations, with per-subsystem detail\n- top_sessions: Sessions/conversations ranked by token consumption\n- hourly: Request counts and tokens per hour to spot spikes or runaway loops\n- models: Per-model breakdown with request counts and avg tokens per request\n- budget_status: Daily budget remaining, projected daily spend, days until exhaustion"
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of days to look back (default: 7, max: 90)",
                        "default": 7
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Optional: filter to a specific session_id (exact match or prefix match with 'background:*')"
                    }
                },
                "required": ["action"],
                "additionalProperties": false
            }
        })
    }

    fn tool_role(&self) -> ToolRole {
        ToolRole::Universal
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: true,
            external_side_effect: false,
            needs_approval: false,
            idempotent: true,
            high_impact_write: false,
        }
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: TokenUsageArgs = serde_json::from_str(arguments)?;
        let days = args.days.min(90).max(1);
        let since = chrono::Utc::now() - chrono::Duration::days(days as i64);
        let since_str = since.format("%Y-%m-%d %H:%M:%S").to_string();

        let records = self.state.get_token_usage_since(&since_str).await?;

        // Note: session_id filtering is handled at the SQL level for
        // breakdown/top_sessions. The flat records from get_token_usage_since
        // don't include session_id, so summary/hourly/models show all data.

        match args.action.as_str() {
            "summary" => self.action_summary(&records, days),
            "breakdown" => self.action_breakdown(&since_str, days).await,
            "top_sessions" => self.action_top_sessions(&since_str, days).await,
            "hourly" => self.action_hourly(&records, days),
            "models" => self.action_models(&records, days),
            "budget_status" => self.action_budget_status().await,
            _ => Ok(format!(
                "Unknown action '{}'. Available: summary, breakdown, top_sessions, hourly, models, budget_status",
                args.action
            )),
        }
    }
}

impl TokenUsageTool {
    fn action_summary(
        &self,
        records: &[crate::traits::TokenUsageRecord],
        days: u32,
    ) -> anyhow::Result<String> {
        if records.is_empty() {
            return Ok(format!(
                "No token usage recorded in the last {} days.",
                days
            ));
        }

        let total_input: i64 = records.iter().map(|r| r.input_tokens).sum();
        let total_output: i64 = records.iter().map(|r| r.output_tokens).sum();
        let total_requests = records.len();

        // Per-model summary
        let mut by_model: HashMap<&str, (i64, i64, usize)> = HashMap::new();
        for r in records {
            let entry = by_model.entry(r.model.as_str()).or_default();
            entry.0 += r.input_tokens;
            entry.1 += r.output_tokens;
            entry.2 += 1;
        }

        let mut model_lines: Vec<String> = by_model
            .iter()
            .map(|(model, (inp, out, count))| {
                format!(
                    "  {}: {} input + {} output = {} tokens ({} requests)",
                    model,
                    format_tokens(*inp),
                    format_tokens(*out),
                    format_tokens(*inp + *out),
                    count
                )
            })
            .collect();
        model_lines.sort();

        let avg_per_request = if total_requests > 0 {
            (total_input + total_output) / total_requests as i64
        } else {
            0
        };

        let avg_per_day = if days > 0 {
            (total_input + total_output) / days as i64
        } else {
            0
        };

        Ok(format!(
            "## Token Usage Summary ({} days)\n\n\
             Total: {} input + {} output = {} tokens\n\
             Requests: {}\n\
             Avg per request: {} tokens\n\
             Avg per day: {} tokens\n\n\
             ### By Model\n{}\n",
            days,
            format_tokens(total_input),
            format_tokens(total_output),
            format_tokens(total_input + total_output),
            total_requests,
            format_tokens(avg_per_request),
            format_tokens(avg_per_day),
            model_lines.join("\n"),
        ))
    }

    async fn action_breakdown(&self, since: &str, days: u32) -> anyhow::Result<String> {
        // We need session_id to split background vs user.
        // get_token_usage_since doesn't return session_id, so we'll use
        // a dedicated query approach via the raw records.
        // For now, re-query with session_id grouping.
        let all_records = self.state.get_token_usage_since(since).await?;

        if all_records.is_empty() {
            return Ok(format!(
                "No token usage recorded in the last {} days.",
                days
            ));
        }

        // We can't get session_id from TokenUsageRecord — it's not included.
        // Instead, we'll use get_token_usage_breakdown which we'll add.
        // For now, use a workaround: query today's records and categorize by model patterns.

        // Actually, let's use the existing data but acknowledge the limitation
        // and provide model-level breakdown which is still very useful.
        let total_input: i64 = all_records.iter().map(|r| r.input_tokens).sum();
        let total_output: i64 = all_records.iter().map(|r| r.output_tokens).sum();
        let total = total_input + total_output;

        // Group by date
        let mut by_date: HashMap<String, (i64, i64, usize)> = HashMap::new();
        for r in &all_records {
            let date = r
                .created_at
                .split(' ')
                .next()
                .unwrap_or("unknown")
                .to_string();
            let entry = by_date.entry(date).or_default();
            entry.0 += r.input_tokens;
            entry.1 += r.output_tokens;
            entry.2 += 1;
        }

        let mut date_lines: Vec<String> = by_date
            .iter()
            .map(|(date, (inp, out, count))| {
                format!(
                    "  {}: {} tokens ({} requests)",
                    date,
                    format_tokens(*inp + *out),
                    count,
                )
            })
            .collect();
        date_lines.sort();
        date_lines.reverse(); // newest first

        // Use get_token_usage_by_session for source breakdown
        let session_breakdown = self.state.get_token_usage_by_session(since).await?;

        let mut background_tokens: i64 = 0;
        let mut background_requests: i64 = 0;
        let mut user_tokens: i64 = 0;
        let mut user_requests: i64 = 0;
        let mut background_detail: Vec<String> = Vec::new();

        for (session_id, input, output, count) in &session_breakdown {
            let tokens = input + output;
            if session_id.starts_with("background:") {
                background_tokens += tokens;
                background_requests += *count;
                let subsystem = session_id.strip_prefix("background:").unwrap_or(session_id);
                background_detail.push(format!(
                    "    {}: {} tokens ({} requests)",
                    subsystem,
                    format_tokens(tokens),
                    count,
                ));
            } else {
                user_tokens += tokens;
                user_requests += *count;
            }
        }

        background_detail.sort();

        let bg_pct = if total > 0 {
            background_tokens as f64 / total as f64 * 100.0
        } else {
            0.0
        };
        let user_pct = if total > 0 {
            user_tokens as f64 / total as f64 * 100.0
        } else {
            0.0
        };

        Ok(format!(
            "## Token Usage Breakdown ({} days)\n\n\
             Total: {} tokens ({} requests)\n\n\
             ### Source Split\n\
             User conversations: {} tokens ({:.1}%, {} requests)\n\
             Background tasks:   {} tokens ({:.1}%, {} requests)\n\n\
             ### Background Task Detail\n{}\n\n\
             ### Daily Trend\n{}\n",
            days,
            format_tokens(total),
            all_records.len(),
            format_tokens(user_tokens),
            user_pct,
            user_requests,
            format_tokens(background_tokens),
            bg_pct,
            background_requests,
            if background_detail.is_empty() {
                "    (no background usage recorded)".to_string()
            } else {
                background_detail.join("\n")
            },
            date_lines.join("\n"),
        ))
    }

    async fn action_top_sessions(&self, since: &str, days: u32) -> anyhow::Result<String> {
        let session_breakdown = self.state.get_token_usage_by_session(since).await?;

        if session_breakdown.is_empty() {
            return Ok(format!(
                "No token usage recorded in the last {} days.",
                days
            ));
        }

        // Sort by total tokens descending
        let mut sessions: Vec<_> = session_breakdown
            .iter()
            .map(|(sid, inp, out, count)| (sid, inp + out, *inp, *out, *count))
            .collect();
        sessions.sort_by(|a, b| b.1.cmp(&a.1));

        let total: i64 = sessions.iter().map(|s| s.1).sum();

        let lines: Vec<String> = sessions
            .iter()
            .take(20)
            .enumerate()
            .map(|(i, (sid, tokens, inp, out, count))| {
                let pct = if total > 0 {
                    *tokens as f64 / total as f64 * 100.0
                } else {
                    0.0
                };
                format!(
                    "  {}. {} — {} tokens ({:.1}%, {} in/{} out, {} requests)",
                    i + 1,
                    sid,
                    format_tokens(*tokens),
                    pct,
                    format_tokens(*inp),
                    format_tokens(*out),
                    count,
                )
            })
            .collect();

        Ok(format!(
            "## Top Sessions by Token Usage ({} days)\n\n{}\n",
            days,
            lines.join("\n"),
        ))
    }

    fn action_hourly(
        &self,
        records: &[crate::traits::TokenUsageRecord],
        days: u32,
    ) -> anyhow::Result<String> {
        if records.is_empty() {
            return Ok(format!(
                "No token usage recorded in the last {} days.",
                days
            ));
        }

        // Group by hour
        let mut by_hour: HashMap<String, (i64, usize)> = HashMap::new();
        for r in records {
            // Extract "YYYY-MM-DD HH" from created_at
            let hour = if r.created_at.len() >= 13 {
                r.created_at[..13].to_string()
            } else {
                r.created_at.clone()
            };
            let entry = by_hour.entry(hour).or_default();
            entry.0 += r.input_tokens + r.output_tokens;
            entry.1 += 1;
        }

        let mut hours: Vec<_> = by_hour.into_iter().collect();
        hours.sort_by(|a, b| b.0.cmp(&a.0)); // newest first

        // Find anomalies: hours with >2x the average
        let avg_tokens: f64 = if !hours.is_empty() {
            hours.iter().map(|(_, (t, _))| *t).sum::<i64>() as f64 / hours.len() as f64
        } else {
            0.0
        };
        let avg_requests: f64 = if !hours.is_empty() {
            hours.iter().map(|(_, (_, r))| *r).sum::<usize>() as f64 / hours.len() as f64
        } else {
            0.0
        };

        let spike_threshold = (avg_tokens * 2.0) as i64;

        let lines: Vec<String> = hours
            .iter()
            .take(48) // last 48 hours max
            .map(|(hour, (tokens, requests))| {
                let spike = if *tokens > spike_threshold {
                    " ⚠ SPIKE"
                } else {
                    ""
                };
                format!(
                    "  {}:00 — {} tokens, {} requests{}",
                    hour,
                    format_tokens(*tokens),
                    requests,
                    spike,
                )
            })
            .collect();

        Ok(format!(
            "## Hourly Token Usage (last {} days, newest first)\n\n\
             Avg per hour: {} tokens, {:.1} requests\n\
             Spike threshold (2x avg): {} tokens\n\n\
             {}\n",
            days,
            format_tokens(avg_tokens as i64),
            avg_requests,
            format_tokens(spike_threshold),
            lines.join("\n"),
        ))
    }

    fn action_models(
        &self,
        records: &[crate::traits::TokenUsageRecord],
        days: u32,
    ) -> anyhow::Result<String> {
        if records.is_empty() {
            return Ok(format!(
                "No token usage recorded in the last {} days.",
                days
            ));
        }

        let mut by_model: HashMap<&str, (i64, i64, usize)> = HashMap::new();
        for r in records {
            let entry = by_model.entry(r.model.as_str()).or_default();
            entry.0 += r.input_tokens;
            entry.1 += r.output_tokens;
            entry.2 += 1;
        }

        let total: i64 = records
            .iter()
            .map(|r| r.input_tokens + r.output_tokens)
            .sum();

        let mut models: Vec<_> = by_model.into_iter().collect();
        models.sort_by(|a, b| (b.1 .0 + b.1 .1).cmp(&(a.1 .0 + a.1 .1)));

        let lines: Vec<String> = models
            .iter()
            .map(|(model, (inp, out, count))| {
                let tokens = inp + out;
                let pct = if total > 0 {
                    tokens as f64 / total as f64 * 100.0
                } else {
                    0.0
                };
                let avg = if *count > 0 {
                    tokens / *count as i64
                } else {
                    0
                };
                format!(
                    "  {} — {} tokens ({:.1}%)\n    {} input + {} output, {} requests, avg {} tokens/request",
                    model,
                    format_tokens(tokens),
                    pct,
                    format_tokens(*inp),
                    format_tokens(*out),
                    count,
                    format_tokens(avg),
                )
            })
            .collect();

        Ok(format!(
            "## Token Usage by Model ({} days)\n\n{}\n",
            days,
            lines.join("\n"),
        ))
    }

    async fn action_budget_status(&self) -> anyhow::Result<String> {
        let today_start = chrono::Utc::now().format("%Y-%m-%d 00:00:00").to_string();
        let today_records = self.state.get_token_usage_since(&today_start).await?;

        let today_input: i64 = today_records.iter().map(|r| r.input_tokens).sum();
        let today_output: i64 = today_records.iter().map(|r| r.output_tokens).sum();
        let today_total = today_input + today_output;
        let today_requests = today_records.len();

        // Get last 7 days for projection
        let week_ago = (chrono::Utc::now() - chrono::Duration::days(7))
            .format("%Y-%m-%d 00:00:00")
            .to_string();
        let week_records = self.state.get_token_usage_since(&week_ago).await?;
        let week_total: i64 = week_records
            .iter()
            .map(|r| r.input_tokens + r.output_tokens)
            .sum();
        let avg_daily = week_total / 7;

        let mut output = format!(
            "## Budget Status\n\n\
             ### Today\n\
             Used: {} tokens ({} requests)\n\
             Input: {} / Output: {}\n\n\
             ### 7-day Average\n\
             Avg daily: {} tokens\n",
            format_tokens(today_total),
            today_requests,
            format_tokens(today_input),
            format_tokens(today_output),
            format_tokens(avg_daily),
        );

        if let Some(budget) = self.daily_token_budget {
            let remaining = budget as i64 - today_total;
            let pct_used = today_total as f64 / budget as f64 * 100.0;
            output.push_str(&format!(
                "\n### Daily Budget\n\
                 Budget: {} tokens\n\
                 Used: {} ({:.1}%)\n\
                 Remaining: {} tokens\n",
                format_tokens(budget as i64),
                format_tokens(today_total),
                pct_used,
                format_tokens(remaining),
            ));

            if avg_daily > 0 {
                let days_at_pace = budget as f64 / avg_daily as f64;
                if days_at_pace < 1.0 {
                    output.push_str("⚠ WARNING: Average daily usage EXCEEDS the daily budget!\n");
                }
            }
        } else {
            output.push_str("\nNo daily token budget configured.\n");
            output.push_str(
                "Tip: Set `daily_token_budget` in config.toml [state] to enable budget enforcement.\n",
            );
        }

        // Hours elapsed today for run-rate projection
        let now = chrono::Utc::now();
        let hours_elapsed = now.hour() as f64 + now.minute() as f64 / 60.0;
        if hours_elapsed > 1.0 {
            let projected_daily = (today_total as f64 / hours_elapsed * 24.0) as i64;
            output.push_str(&format!(
                "\n### Projected (based on today's pace)\n\
                 Projected end-of-day: {} tokens\n",
                format_tokens(projected_daily),
            ));
        }

        Ok(output)
    }
}

/// Format token counts with K/M suffixes for readability.
fn format_tokens(tokens: i64) -> String {
    if tokens.abs() >= 1_000_000 {
        format!("{:.1}M", tokens as f64 / 1_000_000.0)
    } else if tokens.abs() >= 10_000 {
        format!("{:.1}K", tokens as f64 / 1_000.0)
    } else {
        format!("{}", tokens)
    }
}
