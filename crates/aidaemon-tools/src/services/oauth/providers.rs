use super::{OAuthProvider, OAuthType};

/// Get a built-in OAuth provider definition by name.
pub fn get_builtin_provider(name: &str) -> Option<OAuthProvider> {
    match name {
        "twitter" | "x" => Some(OAuthProvider {
            name: "twitter".to_string(),
            display_name: "Twitter/X".to_string(),
            auth_type: OAuthType::OAuth2Pkce,
            authorize_url: "https://twitter.com/i/oauth2/authorize".to_string(),
            token_url: "https://api.twitter.com/2/oauth2/token".to_string(),
            scopes: vec![
                "tweet.read".to_string(),
                "tweet.write".to_string(),
                "users.read".to_string(),
                "offline.access".to_string(),
            ],
            allowed_domains: vec!["api.twitter.com".to_string(), "api.x.com".to_string()],
        }),
        "github" => Some(OAuthProvider {
            name: "github".to_string(),
            display_name: "GitHub".to_string(),
            auth_type: OAuthType::OAuth2Pkce,
            authorize_url: "https://github.com/login/oauth/authorize".to_string(),
            token_url: "https://github.com/login/oauth/access_token".to_string(),
            scopes: vec!["read:user".to_string(), "repo".to_string()],
            allowed_domains: vec!["api.github.com".to_string()],
        }),
        "google" => Some(OAuthProvider {
            name: "google".to_string(),
            display_name: "Google (Gmail, Calendar, Tasks)".to_string(),
            auth_type: OAuthType::OAuth2Pkce,
            authorize_url: "https://accounts.google.com/o/oauth2/v2/auth".to_string(),
            token_url: "https://oauth2.googleapis.com/token".to_string(),
            scopes: vec![
                "https://www.googleapis.com/auth/gmail.modify".to_string(),
                "https://www.googleapis.com/auth/calendar".to_string(),
                "https://www.googleapis.com/auth/tasks".to_string(),
            ],
            allowed_domains: vec!["googleapis.com".to_string()],
        }),
        _ => None,
    }
}

/// List all built-in provider names.
pub fn builtin_provider_names() -> Vec<&'static str> {
    vec!["twitter", "github", "google"]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtin_twitter_provider() {
        let p = get_builtin_provider("twitter").unwrap();
        assert_eq!(p.name, "twitter");
        assert_eq!(p.auth_type, OAuthType::OAuth2Pkce);
        assert!(!p.authorize_url.is_empty());
        assert!(!p.token_url.is_empty());
        assert!(!p.scopes.is_empty());
        assert!(!p.allowed_domains.is_empty());
    }

    #[test]
    fn test_builtin_x_alias() {
        let p = get_builtin_provider("x").unwrap();
        assert_eq!(p.name, "twitter");
    }

    #[test]
    fn test_builtin_github_provider() {
        let p = get_builtin_provider("github").unwrap();
        assert_eq!(p.name, "github");
        assert!(p.allowed_domains.contains(&"api.github.com".to_string()));
    }

    #[test]
    fn test_unknown_provider_returns_none() {
        assert!(get_builtin_provider("unknown_service").is_none());
    }

    #[test]
    fn test_builtin_google_provider() {
        let p = get_builtin_provider("google").unwrap();
        assert_eq!(p.name, "google");
        assert_eq!(p.auth_type, OAuthType::OAuth2Pkce);
        assert!(p.scopes.iter().any(|s| s.contains("gmail")));
        assert!(p.scopes.iter().any(|s| s.contains("calendar")));
        assert!(p.scopes.iter().any(|s| s.contains("tasks")));
        assert!(p.allowed_domains.contains(&"googleapis.com".to_string()));
    }

    #[test]
    fn test_builtin_provider_names() {
        let names = builtin_provider_names();
        assert!(names.contains(&"twitter"));
        assert!(names.contains(&"github"));
        assert!(names.contains(&"google"));
    }
}
