//! Reading the HTTP decision backends' replies (Gemini, TypeSafe): the
//! pieces both read the same way.

use std::collections::BTreeMap;

use serde_json::Value;

/// A network failure, named for `api` ("decisions API", "TypeSafe API").
pub(crate) fn network_error(api: &str, error: &reqwest::Error) -> String {
    let kind = if error.is_timeout() {
        "timed out"
    } else if error.is_connect() {
        "could not connect"
    } else {
        "request failed"
    };
    format!("{api} {kind}")
}

/// At most 200 characters of an error text, ending in "..." when cut.
pub(crate) fn short(text: &str) -> String {
    const MAX: usize = 200;
    match text.char_indices().nth(MAX) {
        Some((cut, _)) => format!("{}...", &text[..cut]),
        None => text.to_string(),
    }
}

/// A finite number field as `f32`.
pub(crate) fn f32_field(value: &Value, field: &str) -> Option<f32> {
    value
        .get(field)
        .and_then(Value::as_f64)
        .filter(|n| n.is_finite())
        .map(|n| n as f32)
}

/// The `probabilities` object: each finite value clamped to 0..=1 (a NaN
/// or infinity is dropped, not passed on).
pub(crate) fn probabilities(value: &Value) -> Option<BTreeMap<String, f32>> {
    let map = value.get("probabilities")?.as_object()?;
    Some(
        map.iter()
            .filter_map(|(k, v)| {
                v.as_f64()
                    .filter(|p| p.is_finite())
                    .map(|p| (k.clone(), (p as f32).clamp(0.0, 1.0)))
            })
            .collect(),
    )
}
