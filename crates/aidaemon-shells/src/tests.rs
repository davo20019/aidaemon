use super::*;

#[test]
fn text_waits_for_a_character_cut_between_reads() {
    let mut carry = Utf8Carry::default();
    let e_acute = "é".as_bytes();
    assert_eq!(carry.push(&[b'a', e_acute[0]]), "a");
    assert_eq!(carry.push(&[e_acute[1], b'b']), "éb");
    assert_eq!(carry.push(&[0xff, b'c']), "\u{FFFD}c");
}

#[test]
fn the_backlog_keeps_whole_lines_from_the_end() {
    let (tx, _) = broadcast::channel(4);
    let mut output = Output {
        backlog: String::new(),
        limit: 12,
        parser: vt100::Parser::new_with_callbacks(
            nonzero(4),
            nonzero(20),
            0,
            TitleWatch::default(),
        ),
        tx,
    };
    output.write("first line\nsecond\nthird\n");
    assert_eq!(output.backlog, "third\n");
}

#[cfg(unix)]
mod live {
    use super::*;
    use std::time::Duration;

    /// Reads what the shell writes until `want` shows (a login shell can
    /// be slow to start on a busy machine).
    async fn until(output: &mut broadcast::Receiver<Arc<str>>, seen: &mut String, want: &str) {
        let wait = tokio::time::timeout(Duration::from_secs(30), async {
            while !seen.contains(want) {
                match output.recv().await {
                    Ok(text) => seen.push_str(&text),
                    Err(broadcast::error::RecvError::Lagged(_)) => {}
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        })
        .await;
        assert!(
            wait.is_ok(),
            "the shell never wrote {want:?}; it wrote {seen:?}"
        );
    }

    #[tokio::test]
    async fn a_conversation_shell_runs_in_its_folder_with_a_clean_environment() {
        let folder = tempfile::tempdir().unwrap();
        let shells = ShellRegistry::new(ShellsConfig::default());
        let info = shells.open("session-a", folder.path(), 80, 24).unwrap();
        assert!(info.running());
        // Opening it again is the same shell.
        assert_eq!(
            shells.open("session-a", folder.path(), 80, 24).unwrap().id,
            info.id
        );

        let mut attached = shells.attach(&info.id).unwrap();
        let mut seen = attached.backlog.clone();
        // Cargo sets CARGO_PKG_NAME for this test; the shell must not see it.
        // The sum keeps the typed line itself from matching.
        shells
            .write(
                &info.id,
                "printf 'at:%s env:%s sum:%s\\n' \"$(basename \"$PWD\")\" \"${CARGO_PKG_NAME:-clean}\" $((40+2))\n",
            )
            .unwrap();
        until(&mut attached.output, &mut seen, "sum:42").await;
        let name = folder
            .path()
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string();
        assert!(
            seen.contains(&format!("at:{name} env:clean sum:42")),
            "{seen:?}"
        );

        let screen = shells.screen(&info.id, 50).unwrap();
        assert!(
            screen
                .screen
                .iter()
                .chain(&screen.above)
                .any(|l| l.contains("sum:42")),
            "{screen:?}"
        );
        // A late attacher gets what was written.
        assert!(shells.attach(&info.id).unwrap().backlog.contains("sum:42"));
    }

    #[tokio::test]
    async fn an_ended_shell_says_so_and_opening_again_starts_a_new_one() {
        let folder = tempfile::tempdir().unwrap();
        let shells = ShellRegistry::new(ShellsConfig::default());
        let mut changes = shells.subscribe();
        let info = shells.open("session-b", folder.path(), 80, 24).unwrap();
        shells.write(&info.id, "exit 3\n").unwrap();
        let ended = tokio::time::timeout(Duration::from_secs(30), async {
            loop {
                if let Ok(ShellChange::Updated(now)) = changes.recv().await
                    && now.id == info.id
                    && now.exit_code.is_some()
                {
                    return now;
                }
            }
        })
        .await
        .expect("the shell never ended");
        assert_eq!(ended.exit_code, Some(3));
        assert_eq!(shells.write(&info.id, "x"), Err(ShellError::Ended));
        let again = shells.open("session-b", folder.path(), 80, 24).unwrap();
        assert_ne!(again.id, info.id);
        assert_eq!(shells.list().len(), 1);
    }

    #[tokio::test]
    async fn closing_a_conversation_closes_its_shell() {
        let folder = tempfile::tempdir().unwrap();
        let shells = ShellRegistry::new(ShellsConfig::default());
        let mut changes = shells.subscribe();
        let info = shells.open("session-c", folder.path(), 80, 24).unwrap();
        shells.close_session("session-c");
        assert!(shells.get(&info.id).is_none());
        let closed = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if let Ok(ShellChange::Closed { id, session_id }) = changes.recv().await {
                    return (id, session_id);
                }
            }
        })
        .await
        .unwrap();
        assert_eq!(closed, (info.id, "session-c".to_string()));
    }
}

#[test]
fn shells_turned_off_open_nothing() {
    let shells = ShellRegistry::new(ShellsConfig {
        enabled: false,
        ..ShellsConfig::default()
    });
    assert_eq!(
        shells.open("s", Path::new("."), 80, 24).err(),
        Some(ShellError::Disabled)
    );
}

#[test]
fn a_missing_folder_is_refused() {
    let shells = ShellRegistry::new(ShellsConfig::default());
    assert_eq!(
        shells
            .open("s", Path::new("/definitely/not/a/folder/here"), 80, 24)
            .err(),
        Some(ShellError::NoFolder)
    );
}
