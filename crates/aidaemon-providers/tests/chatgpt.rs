mod common;

use std::sync::{Arc, Mutex};

use aidaemon_core::{Message, ModelRequest, Provider, ProviderError};
use aidaemon_providers::{ChatGpt, ChatGptAuth, ChatGptCredentials};
use async_trait::async_trait;
use common::{MockResponse, MockServer, data_events, drain};
use serde_json::json;

/// Hands out `token-1`, then `token-2` after a rejection; records what was
/// reported.
#[derive(Default)]
struct RotatingAuth {
    refreshed_from: Mutex<Vec<String>>,
    rejected: Mutex<Vec<String>>,
}

#[async_trait]
impl ChatGptAuth for RotatingAuth {
    async fn credentials(&self) -> Result<ChatGptCredentials, String> {
        Ok(credentials("token-1"))
    }

    async fn refresh_after_rejection(&self, rejected: &str) -> Result<ChatGptCredentials, String> {
        self.refreshed_from.lock().unwrap().push(rejected.into());
        Ok(credentials("token-2"))
    }

    async fn mark_rejected(&self, token: &str, _detail: &str) {
        self.rejected.lock().unwrap().push(token.into());
    }
}

fn credentials(token: &str) -> ChatGptCredentials {
    ChatGptCredentials {
        access_token: token.into(),
        account_id: "acct-synthetic".into(),
    }
}

fn request() -> ModelRequest {
    ModelRequest {
        model: "gpt-6-sol".into(),
        messages: vec![Message::system("Be brief."), Message::user("Say hi")],
        ..Default::default()
    }
}

fn reply(text: &str) -> MockResponse {
    MockResponse::sse(
        data_events(&[
            json!({"type": "response.output_text.delta", "delta": text}),
            json!({"type": "response.output_item.done", "item": {"type": "message", "content": [{"text": text}]}}),
            json!({"type": "response.completed", "response": {"output": [], "usage": {"input_tokens": 12, "output_tokens": 2}}}),
        ]),
        16,
    )
}

fn provider(server: &MockServer, auth: Arc<RotatingAuth>) -> ChatGpt {
    ChatGpt::builder()
        .base_url(&server.base)
        .auth_arc(auth)
        .build()
        .unwrap()
}

#[tokio::test]
async fn streams_text_with_the_subscription_headers() {
    let server = MockServer::start(vec![reply("Hi there")]).await;
    let chatgpt = provider(&server, Arc::default());
    let request = request();
    let (deltas, done, error) = drain(chatgpt.stream(&request)).await;
    assert!(error.is_none(), "{error:?}");
    assert_eq!(deltas.concat(), "Hi there");
    assert_eq!(done[0].text, "Hi there");
    assert_eq!(done[0].usage.input_tokens, 12);

    let sent = &server.requests()[0];
    assert_eq!(sent.path, "/responses");
    assert_eq!(sent.headers["authorization"], "Bearer token-1");
    assert_eq!(sent.headers["chatgpt-account-id"], "acct-synthetic");
    assert_eq!(sent.headers["originator"], "aidaemon");
    assert_eq!(sent.body["instructions"], "Be brief.");
    assert_eq!(sent.body["store"], false);
}

/// A token revoked before its expiry: refresh once, replay the request.
#[tokio::test]
async fn a_rejected_token_is_refreshed_once_and_the_request_replayed() {
    let server = MockServer::start(vec![
        MockResponse::status(401, r#"{"detail":"token revoked"}"#),
        reply("after refresh"),
    ])
    .await;
    let auth = Arc::new(RotatingAuth::default());
    let response = provider(&server, auth.clone())
        .complete(&request())
        .await
        .unwrap();
    assert_eq!(response.text, "after refresh");
    assert_eq!(*auth.refreshed_from.lock().unwrap(), vec!["token-1"]);
    let sent = server.requests();
    assert_eq!(sent.len(), 2);
    assert_eq!(sent[1].headers["authorization"], "Bearer token-2");
    assert!(auth.rejected.lock().unwrap().is_empty());
}

#[tokio::test]
async fn a_refreshed_token_rejected_again_is_reported_and_fails() {
    let server = MockServer::start(vec![
        MockResponse::status(401, "{}"),
        MockResponse::status(401, "{}"),
    ])
    .await;
    let auth = Arc::new(RotatingAuth::default());
    let error = provider(&server, auth.clone())
        .complete(&request())
        .await
        .unwrap_err();
    assert!(matches!(error, ProviderError::Auth(m) if m.contains("sign in again")));
    assert_eq!(*auth.rejected.lock().unwrap(), vec!["token-2"]);
}

#[tokio::test]
async fn a_plan_limit_is_a_rate_limit_with_an_explanation() {
    let server = MockServer::start(vec![
        MockResponse::status(429, "{}").header("retry-after", "3"),
    ])
    .await;
    let error = provider(&server, Arc::default())
        .complete(&request())
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        ProviderError::RateLimited {
            retry_after: Some(_)
        }
    ));
}

fn reasoned_request() -> ModelRequest {
    let call = aidaemon_core::ToolCall::new("c1", "terminal", json!({})).with_provider_data(
        json!({ "chatgpt_reasoning_items": [{"type": "reasoning", "summary": [], "encrypted_content": "sealed"}] }),
    );
    ModelRequest {
        model: "gpt-6-sol".into(),
        messages: vec![
            Message::user("go"),
            Message::assistant(String::new(), vec![call]),
            Message::tool_result("c1", "done"),
        ],
        ..Default::default()
    }
}

/// Reasoning written by another model can be refused; the call must survive.
#[tokio::test]
async fn refused_reasoning_is_dropped_and_the_request_resent_once() {
    let server = MockServer::start(vec![
        MockResponse::status(400, r#"{"error":{"message":"could not be verified"}}"#),
        reply("recovered"),
    ])
    .await;
    let provider = provider(&server, Arc::new(RotatingAuth::default()));

    let response = provider.complete(&reasoned_request()).await.unwrap();

    assert_eq!(response.text, "recovered");
    let requests = server.requests();
    assert_eq!(requests.len(), 2);
    assert!(requests[0].body.to_string().contains("sealed"));
    assert!(!requests[1].body.to_string().contains("sealed"));
}

#[tokio::test]
async fn a_bad_request_with_no_reasoning_to_drop_is_not_resent() {
    let server = MockServer::start(vec![MockResponse::status(
        400,
        r#"{"error":{"message":"bad"}}"#,
    )])
    .await;
    let provider = provider(&server, Arc::new(RotatingAuth::default()));

    assert!(provider.complete(&request()).await.is_err());
    assert_eq!(server.requests().len(), 1);
}

#[tokio::test]
async fn a_single_attempt_call_is_never_resent() {
    let server = MockServer::start(vec![MockResponse::status(
        400,
        r#"{"error":{"message":"no"}}"#,
    )])
    .await;
    let provider = provider(&server, Arc::new(RotatingAuth::default()));
    let mut request = reasoned_request();
    request.options.single_attempt = true;

    assert!(provider.complete(&request).await.is_err());
    assert_eq!(server.requests().len(), 1);
}
