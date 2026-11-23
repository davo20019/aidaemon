//! The local attach, share and start CLIs.

use super::*;

pub(crate) fn load_local_attach_endpoint() -> anyhow::Result<LocalAttachEndpoint> {
    let path = local_attach_endpoint_path();
    let raw = std::fs::read_to_string(&path).map_err(|e| {
        anyhow::anyhow!(
            "Failed to read local attach endpoint at {}: {}",
            path.display(),
            e
        )
    })?;
    let endpoint: LocalAttachEndpoint = serde_json::from_str(&raw).map_err(|e| {
        anyhow::anyhow!(
            "Invalid local attach endpoint JSON at {}: {}",
            path.display(),
            e
        )
    })?;
    if endpoint.host.trim().is_empty() || endpoint.port == 0 || endpoint.secret.trim().is_empty() {
        anyhow::bail!(
            "Local attach endpoint is incomplete at {}. Restart aidaemon.",
            path.display()
        );
    }
    Ok(endpoint)
}

#[cfg(unix)]
pub(crate) struct RawTerminalGuard {
    pub(crate) fd: i32,
    pub(crate) original: libc::termios,
}

#[cfg(unix)]
impl RawTerminalGuard {
    pub(crate) fn enable_if_tty() -> anyhow::Result<Option<Self>> {
        let fd = libc::STDIN_FILENO;
        // SAFETY: libc::isatty only reads fd metadata.
        if unsafe { libc::isatty(fd) } != 1 {
            return Ok(None);
        }
        // SAFETY: `termios` is plain old data and we initialize it via tcgetattr before use.
        let mut original = unsafe { std::mem::zeroed::<libc::termios>() };
        // SAFETY: tcgetattr/tcsetattr are called with a valid terminal fd and termios pointer.
        if unsafe { libc::tcgetattr(fd, &mut original) } != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        let mut raw = original;
        raw.c_lflag &= !(libc::ICANON | libc::ECHO | libc::ISIG);
        raw.c_iflag &= !(libc::IXON | libc::ICRNL);
        raw.c_oflag &= !(libc::OPOST);
        raw.c_cc[libc::VMIN] = 1;
        raw.c_cc[libc::VTIME] = 0;
        if unsafe { libc::tcsetattr(fd, libc::TCSANOW, &raw) } != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        Ok(Some(Self { fd, original }))
    }
}

#[cfg(unix)]
impl Drop for RawTerminalGuard {
    fn drop(&mut self) {
        // SAFETY: restoring previously captured termios back to the same fd.
        let _ = unsafe { libc::tcsetattr(self.fd, libc::TCSANOW, &self.original) };
    }
}

#[cfg(not(unix))]
pub(crate) struct RawTerminalGuard;

#[cfg(not(unix))]
impl RawTerminalGuard {
    pub(crate) fn enable_if_tty() -> anyhow::Result<Option<Self>> {
        Ok(None)
    }
}

pub(crate) fn send_local_attach_json_line(
    writer: &mut std::net::TcpStream,
    value: &Value,
) -> anyhow::Result<()> {
    let mut line = serde_json::to_string(value)?;
    line.push('\n');
    writer.write_all(line.as_bytes())?;
    writer.flush()?;
    Ok(())
}

pub(crate) fn send_local_attach_resize_and_redraw(
    writer: &mut std::net::TcpStream,
    cols: u16,
    rows: u16,
) -> anyhow::Result<()> {
    send_local_attach_json_line(
        writer,
        &json!({
            "type":"resize",
            "cols": cols,
            "rows": rows,
        }),
    )?;
    send_local_attach_json_line(writer, &json!({"type":"redraw"}))?;
    Ok(())
}

#[cfg(unix)]
pub(crate) fn spawn_local_attach_resize_watcher(mut writer: std::net::TcpStream) {
    std::thread::spawn(move || {
        let mut last_sent = local_terminal_size();
        let mut warmup_resyncs_left = 2u8;
        loop {
            std::thread::sleep(Duration::from_millis(250));
            let Some((cols, rows)) = local_terminal_size() else {
                continue;
            };
            let changed = last_sent != Some((cols, rows));
            if !changed && warmup_resyncs_left == 0 {
                continue;
            }
            if send_local_attach_resize_and_redraw(&mut writer, cols, rows).is_err() {
                break;
            }
            last_sent = Some((cols, rows));
            warmup_resyncs_left = warmup_resyncs_left.saturating_sub(1);
        }
    });
}

#[cfg(not(unix))]
pub(crate) fn spawn_local_attach_resize_watcher(_writer: std::net::TcpStream) {}

pub(crate) fn run_local_attach_stdout_loop(
    reader: &mut std::io::BufReader<std::net::TcpStream>,
) -> anyhow::Result<()> {
    let mut stdout = std::io::stdout();
    let mut line = String::new();
    loop {
        line.clear();
        let n = reader.read_line(&mut line)?;
        if n == 0 {
            break;
        }
        if line.len() > LOCAL_ATTACH_MAX_FRAME_BYTES {
            anyhow::bail!("received oversized frame from local attach endpoint");
        }
        let value: Value = match serde_json::from_str(line.trim()) {
            Ok(v) => v,
            Err(_) => continue,
        };
        match value.get("type").and_then(|v| v.as_str()).unwrap_or("") {
            "stdout" => {
                if let Some(data) = value.get("data").and_then(|v| v.as_str()) {
                    stdout.write_all(data.as_bytes())?;
                    stdout.flush()?;
                }
            }
            "status" => {
                if let Some(message) = value.get("message").and_then(|v| v.as_str()) {
                    eprintln!("\r\n[status] {}", message);
                }
            }
            "error" => {
                let message = value
                    .get("message")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown error");
                anyhow::bail!("{}", message);
            }
            "exit" => {
                let reason = value
                    .get("reason")
                    .and_then(|v| v.as_str())
                    .unwrap_or("session ended");
                eprintln!("\r\n[session] {}", reason);
                break;
            }
            _ => {}
        }
    }
    Ok(())
}

pub(crate) fn run_local_attach_stdin_pump(mut writer: std::net::TcpStream) -> anyhow::Result<()> {
    let mut stdin = std::io::stdin();
    let mut input = [0u8; 2048];
    loop {
        let n = stdin.read(&mut input)?;
        if n == 0 {
            let _ = send_local_attach_json_line(&mut writer, &json!({"type":"detach"}));
            break;
        }
        let chunk = &input[..n];
        if chunk.len() == 1 && chunk[0] == 0x1d {
            // Ctrl+]
            let _ = send_local_attach_json_line(&mut writer, &json!({"type":"detach"}));
            break;
        }
        let payload = json!({
            "type":"stdin",
            "data": b64_encode(chunk),
        });
        send_local_attach_json_line(&mut writer, &payload)?;
    }
    Ok(())
}

#[cfg(unix)]
pub(crate) fn local_terminal_size() -> Option<(u16, u16)> {
    // SAFETY: winsize is POD and ioctl fills it when fd is a terminal.
    let mut winsize = unsafe { std::mem::zeroed::<libc::winsize>() };
    // SAFETY: ioctl is called with valid fd and pointer to winsize.
    let rc = unsafe { libc::ioctl(libc::STDOUT_FILENO, libc::TIOCGWINSZ, &mut winsize) };
    if rc != 0 || winsize.ws_col == 0 || winsize.ws_row == 0 {
        return None;
    }
    Some((winsize.ws_col, winsize.ws_row))
}

#[cfg(not(unix))]
pub(crate) fn local_terminal_size() -> Option<(u16, u16)> {
    None
}

pub fn run_local_attach_cli(code: &str) -> anyhow::Result<()> {
    let code = code.trim();
    if code.is_empty() {
        anyhow::bail!("Usage: aidaemon attach <code>");
    }
    let endpoint = load_local_attach_endpoint()?;
    let address = format!("{}:{}", endpoint.host, endpoint.port);
    let stream = std::net::TcpStream::connect(&address).map_err(|e| {
        anyhow::anyhow!(
            "Failed to connect to local attach endpoint at {}: {}. Is aidaemon running?",
            address,
            e
        )
    })?;
    stream.set_nodelay(true).ok();

    let mut reader = std::io::BufReader::new(stream.try_clone()?);
    let mut writer = stream.try_clone()?;
    send_local_attach_json_line(
        &mut writer,
        &json!({
            "type":"attach",
            "code": code,
            "secret": endpoint.secret,
        }),
    )?;

    let mut first_line = String::new();
    let first_read = reader.read_line(&mut first_line)?;
    if first_read == 0 {
        anyhow::bail!("Local attach endpoint closed before handshake completed");
    }
    let first_value: Value = serde_json::from_str(first_line.trim())
        .map_err(|e| anyhow::anyhow!("invalid attach response: {}", e))?;
    run_local_attach_interactive(reader, writer, first_value, "Attached")
}

pub fn run_local_start_cli(
    agent: &str,
    cwd: Option<&Path>,
    agent_args: &[String],
) -> anyhow::Result<()> {
    let Some(agent) = normalize_agent(Some(agent)) else {
        anyhow::bail!(
            "Unknown agent `{}`. Supported: codex, claude, gemini, opencode.",
            agent
        );
    };
    let (agent_args, _) =
        aidaemon_tools::permission_bypass::normalize_terminal_agent_permission_aliases(
            Some(agent.as_str()),
            agent_args.to_vec(),
        );

    let endpoint = load_local_attach_endpoint()?;
    let address = format!("{}:{}", endpoint.host, endpoint.port);
    let stream = std::net::TcpStream::connect(&address).map_err(|e| {
        anyhow::anyhow!(
            "Failed to connect to local attach endpoint at {}: {}. Is aidaemon running?",
            address,
            e
        )
    })?;
    stream.set_nodelay(true).ok();

    let mut reader = std::io::BufReader::new(stream.try_clone()?);
    let mut writer = stream.try_clone()?;
    send_local_attach_json_line(
        &mut writer,
        &json!({
            "type":"start",
            "secret": endpoint.secret,
            "agent": agent,
            "cwd": cwd.map(|value| value.to_string_lossy().to_string()).unwrap_or_default(),
            "agent_args": agent_args,
        }),
    )?;

    let mut first_line = String::new();
    let first_read = reader.read_line(&mut first_line)?;
    if first_read == 0 {
        anyhow::bail!("Local attach endpoint closed before startup handshake completed");
    }
    let first_value: Value = serde_json::from_str(first_line.trim())
        .map_err(|e| anyhow::anyhow!("invalid local start response: {}", e))?;
    run_local_attach_interactive(reader, writer, first_value, "Started")
}

pub async fn request_local_start_session(
    agent: &str,
    cwd: Option<&str>,
    agent_args: &[String],
) -> anyhow::Result<String> {
    let Some(agent) = normalize_agent(Some(agent)) else {
        anyhow::bail!(
            "Unknown agent `{}`. Supported: codex, claude, gemini, opencode.",
            agent
        );
    };
    let (agent_args, _) =
        aidaemon_tools::permission_bypass::normalize_terminal_agent_permission_aliases(
            Some(agent.as_str()),
            agent_args.to_vec(),
        );

    let endpoint = load_local_attach_endpoint()?;
    let address = format!("{}:{}", endpoint.host, endpoint.port);
    let stream = TcpStream::connect(&address).await.map_err(|e| {
        anyhow::anyhow!(
            "Failed to connect to local attach endpoint at {}: {}. Is aidaemon running?",
            address,
            e
        )
    })?;
    stream.set_nodelay(true).ok();

    let (read_half, mut write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half);
    write_json_line(
        &mut write_half,
        &json!({
            "type":"start",
            "secret": endpoint.secret,
            "agent": agent,
            "cwd": cwd.unwrap_or(""),
            "agent_args": agent_args,
        }),
    )
    .await?;

    let mut line = String::new();
    let read = reader.read_line(&mut line).await?;
    if read == 0 {
        anyhow::bail!("Local attach endpoint closed before startup handshake completed");
    }
    let payload: Value = serde_json::from_str(line.trim())
        .map_err(|e| anyhow::anyhow!("invalid local start response: {}", e))?;
    match payload.get("type").and_then(|v| v.as_str()).unwrap_or("") {
        "attached" => payload
            .get("session_id")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .map(|v| v.to_string())
            .ok_or_else(|| anyhow::anyhow!("local start response missing session_id")),
        "error" => {
            let message = payload
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("local start failed");
            anyhow::bail!("{}", message);
        }
        _ => anyhow::bail!("Unexpected start response from local endpoint"),
    }
}

pub fn run_local_share_cli(session_id: Option<&str>) -> anyhow::Result<()> {
    let endpoint = load_local_attach_endpoint()?;
    let address = format!("{}:{}", endpoint.host, endpoint.port);
    let stream = std::net::TcpStream::connect(&address).map_err(|e| {
        anyhow::anyhow!(
            "Failed to connect to local attach endpoint at {}: {}. Is aidaemon running?",
            address,
            e
        )
    })?;
    stream.set_nodelay(true).ok();

    let mut reader = std::io::BufReader::new(stream.try_clone()?);
    let mut writer = stream.try_clone()?;
    let normalized_session_id = normalize_relay_session_id(session_id);
    send_local_attach_json_line(
        &mut writer,
        &json!({
            "type":"share",
            "secret": endpoint.secret,
            "session_id": normalized_session_id,
        }),
    )?;

    let mut line = String::new();
    let n = reader.read_line(&mut line)?;
    if n == 0 {
        anyhow::bail!("Local attach endpoint closed before share response");
    }
    let payload: Value = serde_json::from_str(line.trim())
        .map_err(|e| anyhow::anyhow!("invalid share response: {}", e))?;
    match payload.get("type").and_then(|v| v.as_str()).unwrap_or("") {
        "shared" => {
            let code = payload.get("code").and_then(|v| v.as_str()).unwrap_or("");
            if code.is_empty() {
                anyhow::bail!("Share response did not include a resume code");
            }
            let session_id = payload
                .get("session_id")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            println!("Session: {}", session_id);
            println!("Telegram command: /agent resume {}", code);
            if let Some(expires_at_unix) = payload.get("expires_at_unix").and_then(|v| v.as_i64()) {
                let now = chrono::Utc::now().timestamp();
                let secs_left = expires_at_unix.saturating_sub(now).max(0);
                println!("Expires in about {} minutes.", (secs_left + 59) / 60);
            } else {
                println!("Expires in about 5 minutes.");
            }
            Ok(())
        }
        "error" => {
            let message = payload
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("share failed");
            anyhow::bail!("{}", message);
        }
        _ => anyhow::bail!("Unexpected share response from local endpoint"),
    }
}

pub(crate) fn run_local_attach_interactive(
    mut reader: std::io::BufReader<std::net::TcpStream>,
    mut writer: std::net::TcpStream,
    first_value: Value,
    verb: &str,
) -> anyhow::Result<()> {
    match first_value
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("")
    {
        "attached" => {}
        "error" => {
            let message = first_value
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("attach failed");
            anyhow::bail!("{}", message);
        }
        _ => anyhow::bail!("Unexpected attach response from local endpoint"),
    }

    let session_id = first_value
        .get("session_id")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    eprintln!("{} session {}. Press Ctrl+] to detach.", verb, session_id);
    if let Some(code) = first_value.get("resume_code").and_then(|v| v.as_str())
        && !code.trim().is_empty()
    {
        eprintln!("Telegram command: /agent resume {}", code.trim());
    }

    if let Some((cols, rows)) = local_terminal_size() {
        let _ = send_local_attach_resize_and_redraw(&mut writer, cols, rows);
    } else {
        let _ = send_local_attach_json_line(&mut writer, &json!({"type":"redraw"}));
    }
    if let Ok(resize_writer) = writer.try_clone() {
        // Keep syncing local terminal dimensions while attached so full-screen TUIs
        // (Codex/Claude/Gemini/OpenCode) can reflow after handoff and on window resize.
        spawn_local_attach_resize_watcher(resize_writer);
    }

    let _raw_guard = RawTerminalGuard::enable_if_tty()?;
    let stdin_writer = writer.try_clone()?;
    std::thread::spawn(move || {
        let _ = run_local_attach_stdin_pump(stdin_writer);
    });

    run_local_attach_stdout_loop(&mut reader)
}
