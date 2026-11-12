# aidaemon-skills

**Tier: implementation.** No stability promise while 0.x; may change in any minor version. Depend on it through [`aidaemon-sdk`](https://docs.rs/aidaemon-sdk) unless you need it alone.

Skills are markdown instructions an agent activates by trigger words or by name. Each lives in a skills directory as `<name>.md` (front matter `name`, `description`, `triggers`, then the body) or as a directory with `SKILL.md` and resource files. This crate loads and matches them (`SkillCache`, `match_skills`), renders the active ones for the system prompt, promotes procedures that keep succeeding into drafts for the owner's review (`promotion::SkillPromoter`, with the model behind `PromotionModel`), and gives the tools `use_skill`, `skill_resources` and `manage_skills` (add, install from a URL or registry, review drafts, enable, remove).

```rust,ignore
let skills = aidaemon_skills::SkillCache::new(skills_dir.clone());
let use_skill = aidaemon_skills::tools::UseSkillTool::new(skills_dir);
```

Drafts and settings go through `SkillsStore`; `InMemorySkillStore` keeps them in memory.

Part of the [aidaemon SDK](https://github.com/davo20019/aidaemon) (shared version 0.2.0). MIT licensed. Docs: https://docs.rs/aidaemon-skills
