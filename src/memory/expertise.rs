//! Expertise tracking module for calculating skill levels based on task performance.
#![allow(dead_code)] // Functions reserved for future memory system integration

/// Calculate expertise level based on success/failure counts.
///
/// Returns a tuple of (level_name, confidence_score).
/// - novice: <5 tasks or <70% success
/// - competent: >=5 tasks with >=70% success
/// - proficient: >=15 tasks with >=80% success
/// - expert: >=30 tasks with >=90% success
pub fn calculate_expertise_level(succeeded: i32, failed: i32) -> (&'static str, f32) {
    let total = succeeded + failed;
    if total == 0 {
        return ("novice", 0.0);
    }

    let success_rate = succeeded as f32 / total as f32;
    let volume_factor = (total as f32 / 50.0).min(1.0);
    let confidence = success_rate * volume_factor;

    let level = match (total, success_rate) {
        (t, r) if t >= 30 && r >= 0.9 => "expert",
        (t, r) if t >= 15 && r >= 0.8 => "proficient",
        (t, r) if t >= 5 && r >= 0.7 => "competent",
        _ => "novice",
    };

    (level, confidence)
}

/// Detect which domains a task context relates to.
///
/// This is a simple keyword-based detection. In the future, this could
/// use embeddings for more sophisticated domain classification.
pub fn detect_domains(task_context: &str) -> Vec<String> {
    let lower = task_context.to_lowercase();
    let mut domains = Vec::new();

    // Programming languages
    if lower.contains("rust") || lower.contains("cargo") || lower.contains(".rs") {
        domains.push("rust".to_string());
    }
    if lower.contains("python") || lower.contains(".py") || lower.contains("pip") {
        domains.push("python".to_string());
    }
    if lower.contains("javascript") || lower.contains("typescript") || lower.contains(".js") || lower.contains(".ts") || lower.contains("npm") || lower.contains("node") {
        domains.push("javascript".to_string());
    }
    if lower.contains("go ") || lower.contains("golang") || lower.contains(".go") {
        domains.push("go".to_string());
    }

    // DevOps/Infrastructure
    if lower.contains("docker") || lower.contains("container") {
        domains.push("docker".to_string());
    }
    if lower.contains("kubernetes") || lower.contains("k8s") || lower.contains("kubectl") {
        domains.push("kubernetes".to_string());
    }
    if lower.contains("terraform") || lower.contains("ansible") || lower.contains("infrastructure") {
        domains.push("infrastructure".to_string());
    }

    // Web development
    if lower.contains("html") || lower.contains("css") || lower.contains("frontend") || lower.contains("react") || lower.contains("vue") {
        domains.push("web-frontend".to_string());
    }
    if lower.contains("api") || lower.contains("backend") || lower.contains("server") || lower.contains("endpoint") {
        domains.push("web-backend".to_string());
    }

    // Databases
    if lower.contains("sql") || lower.contains("database") || lower.contains("postgres") || lower.contains("mysql") || lower.contains("sqlite") {
        domains.push("databases".to_string());
    }

    // Git/Version control
    if lower.contains("git") || lower.contains("commit") || lower.contains("branch") || lower.contains("merge") {
        domains.push("git".to_string());
    }

    // System administration
    if lower.contains("linux") || lower.contains("unix") || lower.contains("bash") || lower.contains("shell") || lower.contains("terminal") {
        domains.push("system-admin".to_string());
    }

    // If no specific domain detected, classify as general
    if domains.is_empty() {
        domains.push("general".to_string());
    }

    domains
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_novice_level() {
        let (level, _) = calculate_expertise_level(0, 0);
        assert_eq!(level, "novice");

        let (level, _) = calculate_expertise_level(2, 2);
        assert_eq!(level, "novice");
    }

    #[test]
    fn test_competent_level() {
        let (level, _) = calculate_expertise_level(7, 3); // 10 total, 70% success
        assert_eq!(level, "competent");
    }

    #[test]
    fn test_proficient_level() {
        let (level, _) = calculate_expertise_level(16, 4); // 20 total, 80% success
        assert_eq!(level, "proficient");
    }

    #[test]
    fn test_expert_level() {
        let (level, _) = calculate_expertise_level(45, 5); // 50 total, 90% success
        assert_eq!(level, "expert");
    }

    #[test]
    fn test_detect_domains() {
        let domains = detect_domains("Building a Rust CLI tool with cargo");
        assert!(domains.contains(&"rust".to_string()));

        let domains = detect_domains("Deploy to kubernetes using docker containers");
        assert!(domains.contains(&"docker".to_string()));
        assert!(domains.contains(&"kubernetes".to_string()));
    }
}
