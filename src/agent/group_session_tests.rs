use super::*;

#[test]
fn discord_guild_channel() {
    assert!(is_group_session("discord:ch:123456"));
    assert!(is_group_session("mybot:discord:ch:123456"));
}

#[test]
fn discord_dm() {
    assert!(!is_group_session("discord:dm:123456"));
    assert!(!is_group_session("mybot:discord:dm:123456"));
}

#[test]
fn slack_public_channel() {
    assert!(is_group_session("slack:C123456"));
    assert!(is_group_session("mybot:slack:C123456"));
    assert!(is_group_session("slack:C123456:1234567890.123"));
}

#[test]
fn slack_private_channel() {
    assert!(is_group_session("slack:G123456"));
    assert!(is_group_session("mybot:slack:G123456"));
}

#[test]
fn slack_dm() {
    assert!(!is_group_session("slack:D123456"));
    assert!(!is_group_session("mybot:slack:D123456"));
}

#[test]
fn telegram_sessions() {
    // Telegram uses numeric IDs — not detected as group
    assert!(!is_group_session("123456789"));
    assert!(!is_group_session("mybot:123456789"));
}
