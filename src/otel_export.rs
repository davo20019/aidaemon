//! Exports kernel runs to an OTLP backend when `[telemetry]` names one.

use std::sync::Arc;

use aidaemon_core as sdk;

use crate::config::TelemetryConfig;

#[cfg(feature = "otel")]
mod enabled {
    use std::collections::HashMap;
    use std::sync::{Arc, OnceLock};

    use aidaemon_core as sdk;
    use aidaemon_policies::{otlp_provider, OtelSink, Redactor};

    use crate::config::{resolve_from_keychain, TelemetryConfig};

    // The provider owns the export thread and must outlive every run.
    struct Exporter {
        _provider: opentelemetry_sdk::trace::SdkTracerProvider,
        sink: Arc<OtelSink>,
    }

    static EXPORTER: OnceLock<Exporter> = OnceLock::new();

    fn resolved_headers(
        headers: &HashMap<String, String>,
    ) -> anyhow::Result<HashMap<String, String>> {
        headers
            .iter()
            .map(|(name, value)| {
                let value = if value == "keychain" {
                    resolve_from_keychain(&aidaemon_secrets::names::otlp_header(name))?
                } else {
                    value.clone()
                };
                Ok((name.clone(), value))
            })
            .collect()
    }

    /// A telemetry problem is reported and never stops the daemon.
    pub(super) fn start(config: &TelemetryConfig) {
        let Some(endpoint) = config.otlp_endpoint.as_deref() else {
            return;
        };
        let headers = match resolved_headers(&config.otlp_headers) {
            Ok(headers) => headers,
            Err(error) => {
                tracing::warn!(%error, "Trace export is off: could not read its headers");
                return;
            }
        };
        let provider = match otlp_provider(endpoint, headers, "aidaemon") {
            Ok(provider) => provider,
            Err(error) => {
                tracing::warn!(%error, "Trace export is off");
                return;
            }
        };
        let mut sink = OtelSink::new(&provider);
        if config.capture_content {
            let redactor = Redactor::default();
            sink = sink
                .with_content(move |text: &str| redactor.redact(text))
                .with_wire_requests();
        }
        let _ = EXPORTER.set(Exporter {
            _provider: provider,
            sink: Arc::new(sink),
        });
        tracing::info!(
            endpoint,
            content = config.capture_content,
            "Trace export is on"
        );
    }

    pub(super) fn sink() -> Option<Arc<dyn sdk::EventSink>> {
        EXPORTER
            .get()
            .map(|exporter| exporter.sink.clone() as Arc<dyn sdk::EventSink>)
    }

    pub(super) fn observe_sent(run: sdk::RunId, messages: &[serde_json::Value]) {
        if let Some(exporter) = EXPORTER.get() {
            exporter.sink.observe_wire(run, messages);
        }
    }
}

#[cfg(not(feature = "otel"))]
mod enabled {
    use std::sync::Arc;

    use aidaemon_core as sdk;

    use crate::config::TelemetryConfig;

    pub(super) fn start(_config: &TelemetryConfig) {}

    pub(super) fn sink() -> Option<Arc<dyn sdk::EventSink>> {
        None
    }

    pub(super) fn observe_sent(_run: sdk::RunId, _messages: &[serde_json::Value]) {}
}

/// Start exporting, if the config asks for it. Call once at startup.
pub(crate) fn start(config: &TelemetryConfig) {
    enabled::start(config);
}

/// The messages actually sent for a model call. The daemon builds them from
/// the kernel's request (steering, directives, compaction, fitting), so only
/// this shows what the model received.
pub(crate) fn observe_sent(run: sdk::RunId, messages: &[serde_json::Value]) {
    enabled::observe_sent(run, messages);
}

/// The exporter to add to each agent, when export is on.
pub(crate) fn sink() -> Option<Arc<dyn sdk::EventSink>> {
    enabled::sink()
}
