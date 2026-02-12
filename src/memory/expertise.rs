//! Expertise tracking module for calculating skill levels based on task performance.

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

/// Normalize explicit domain hints into canonical expertise domains.
fn normalize_domain(domain: &str) -> Option<&'static str> {
    match domain.trim().to_ascii_lowercase().as_str() {
        "rust" => Some("rust"),
        "python" => Some("python"),
        "javascript" | "typescript" | "js" | "ts" => Some("javascript"),
        "go" | "golang" => Some("go"),
        "docker" => Some("docker"),
        "kubernetes" | "k8s" => Some("kubernetes"),
        "infrastructure" | "devops" => Some("infrastructure"),
        "web-frontend" | "frontend" => Some("web-frontend"),
        "web-backend" | "backend" => Some("web-backend"),
        "databases" | "database" | "db" => Some("databases"),
        "git" => Some("git"),
        "system-admin" | "sysadmin" | "shell" => Some("system-admin"),
        "general" => Some("general"),
        _ => None,
    }
}

/// Detect domains from explicit classifier hints only.
///
/// Keyword-based message parsing is intentionally disabled to avoid false
/// positives; callers must provide explicit domain hints.
pub fn detect_domains(explicit_domains: &[String]) -> Vec<String> {
    let mut domains = Vec::new();
    for domain in explicit_domains {
        if let Some(normalized) = normalize_domain(domain) {
            let normalized = normalized.to_string();
            if !domains.contains(&normalized) {
                domains.push(normalized);
            }
        }
    }

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
        let domains = detect_domains(&["Rust".to_string(), "typescript".to_string()]);
        assert!(domains.contains(&"rust".to_string()));
        assert!(domains.contains(&"javascript".to_string()));

        let domains = detect_domains(&["kubernetes".to_string(), "docker".to_string()]);
        assert!(domains.contains(&"docker".to_string()));
        assert!(domains.contains(&"kubernetes".to_string()));
    }

    #[test]
    fn test_detect_domains_defaults_general_when_no_explicit_hints() {
        let domains = detect_domains(&[]);
        assert_eq!(domains, vec!["general".to_string()]);
    }
}
