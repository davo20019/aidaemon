//! Typed tool-context extensions: application values every call can read.

use std::sync::{Arc, Mutex};

use aidaemon_core::testing::{call, spec, text, tool_calls, ScriptedProvider};
use aidaemon_core::{
    Agent, Extensions, Tool, ToolAnnotations, ToolContext, ToolOutput, ToolResult, ToolSpec,
};
use async_trait::async_trait;
use serde_json::{json, Value};

/// A synthetic application value carried to tools.
#[derive(Debug, Clone, PartialEq)]
struct Workspace {
    root: String,
}

/// One call's view of its context: the root, the workspace, and whether the
/// extension was present.
type SeenContext = (String, Option<Workspace>, bool);

/// Records what each call saw in its context.
struct Reader {
    seen: Arc<Mutex<Vec<SeenContext>>>,
}

#[async_trait]
impl Tool for Reader {
    fn spec(&self) -> ToolSpec {
        spec("reader", "reads the context", &[])
    }
    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::READ_ONLY
    }
    async fn call(&self, _args: Value, ctx: &ToolContext) -> ToolResult {
        self.seen.lock().unwrap().push((
            ctx.call_id.clone(),
            ctx.extensions.get::<Workspace>().cloned(),
            ctx.extensions.get::<u64>().is_some(),
        ));
        Ok(ToolOutput::text("read"))
    }
}

#[tokio::test]
async fn extensions_reach_every_call() {
    let mut extensions = Extensions::default();
    extensions.insert(Workspace {
        root: "/tmp/acme-project".into(),
    });
    let seen = Arc::new(Mutex::new(Vec::new()));
    let agent = Agent::builder("test-agent")
        .allow_unguarded()
        .model("scripted")
        .provider(ScriptedProvider::new([
            tool_calls([
                call("c1", "reader", json!({})),
                call("c2", "reader", json!({})),
            ]),
            tool_calls([call("c3", "reader", json!({}))]),
            text("done"),
        ]))
        .tool(Reader { seen: seen.clone() })
        .tool_extensions(extensions)
        .build()
        .unwrap();

    agent.run("s", "go").await.unwrap();

    let workspace = Workspace {
        root: "/tmp/acme-project".into(),
    };
    let mut seen = seen.lock().unwrap().clone();
    seen.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(
        seen,
        vec![
            ("c1".to_string(), Some(workspace.clone()), false),
            ("c2".to_string(), Some(workspace.clone()), false),
            ("c3".to_string(), Some(workspace), false),
        ]
    );
    // A context built outside the loop starts empty.
    assert!(ToolContext::new("s", "c")
        .extensions
        .get::<Workspace>()
        .is_none());
}
