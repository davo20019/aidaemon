use super::SpecialistRenderContext;

/// Shared executor frame included via `{{executor_base}}` in every
/// executor-flavored specialist .md (executor, code, research, review,
/// comms_draft, browser_verifier, artifact_writer, generic). Carries the
/// sub-agent header, mission/task headings, and the discipline rules so each
/// kind file only contains its role-specific tagline.
///
/// The file is intentionally body-only (no YAML frontmatter) and is NOT
/// loaded by the registry; it is spliced in by `render_template` before any
/// other placeholder substitution so the base's own `{{mission}}`/`{{task}}`/
/// `{{depth}}` markers resolve normally.
const EXECUTOR_BASE: &str = include_str!("../../../specialists/_executor_base.md");

pub fn render_template(template: &str, ctx: &SpecialistRenderContext) -> String {
    let bools = |b: bool| if b { "true" } else { "false" };
    template
        .replace("{{executor_base}}", EXECUTOR_BASE)
        .replace("{{mission}}", &ctx.mission)
        .replace("{{task}}", &ctx.task)
        .replace("{{depth}}", &ctx.depth.to_string())
        .replace("{{max_depth}}", &ctx.max_depth.to_string())
        .replace("{{max_iterations}}", &ctx.max_iterations.to_string())
        .replace("{{goal_id}}", &ctx.goal_id)
        .replace("{{working_dir}}", &ctx.working_dir)
        .replace("{{is_scheduled}}", bools(ctx.is_scheduled))
        .replace("{{parent_session_id}}", &ctx.parent_session_id)
        .replace("{{execution_mode}}", &ctx.execution_mode)
        .replace("{{delegation_rule}}", delegation_rule(ctx.can_spawn))
}

/// What a specialist's prompt says about delegating: exactly what the
/// spawn decision gave it, never a rule of the template's own.
pub fn delegation_rule(can_spawn: bool) -> &'static str {
    if can_spawn {
        "- You may delegate independent parts of this task with `spawn_agent` (several at once as `tasks`); do the rest yourself."
    } else {
        "- Do NOT spawn sub-agents."
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> SpecialistRenderContext {
        SpecialistRenderContext {
            mission: "build it".to_string(),
            task: "the task".to_string(),
            depth: 2,
            max_depth: 4,
            max_iterations: 24,
            goal_id: "goal_abc".to_string(),
            working_dir: "/tmp/proj".to_string(),
            is_scheduled: true,
            parent_session_id: "telegram:bot:1".to_string(),
            execution_mode: "EXEC".to_string(),
            can_spawn: false,
        }
    }

    #[test]
    fn substitutes_all_known_placeholders() {
        let tpl = "m={{mission}} t={{task}} d={{depth}}/{{max_depth}} mi={{max_iterations}} \
                   g={{goal_id}} wd={{working_dir}} sched={{is_scheduled}} ps={{parent_session_id}} \
                   em={{execution_mode}}";
        let out = render_template(tpl, &ctx());
        assert_eq!(
            out,
            "m=build it t=the task d=2/4 mi=24 g=goal_abc wd=/tmp/proj sched=true ps=telegram:bot:1 em=EXEC"
        );
    }

    #[test]
    fn leaves_unknown_placeholders_untouched() {
        let tpl = "hello {{unknown_var}} world";
        let out = render_template(tpl, &ctx());
        assert_eq!(out, "hello {{unknown_var}} world");
    }

    #[test]
    fn renders_when_strings_are_empty() {
        let mut c = ctx();
        c.working_dir.clear();
        let out = render_template("wd=[{{working_dir}}]", &c);
        assert_eq!(out, "wd=[]");
    }
}
