//! The bridge's settings: every limit, with defaults, read from command-line
//! flags (`--name value`) or the environment (`AIDAEMON_BRIDGE_NAME`); a
//! flag beats the environment.

use std::net::SocketAddr;
use std::time::Duration;

/// Everything the bridge can be tuned by.
#[derive(Clone, Debug)]
pub struct Config {
    /// Plain-HTTP listen address. TLS belongs to the reverse proxy in front.
    pub listen: SocketAddr,
    /// Header the trusted reverse proxy puts the client address in (e.g.
    /// `X-Forwarded-For`; the last entry is used). `None`: the socket
    /// address is the client address and no header is believed.
    pub trusted_proxy_header: Option<String>,
    /// Largest browser frame before the daemon promotes the connection (§7).
    pub handshake_frame_max: usize,
    /// Largest browser frame after promotion (§7 DATA frame).
    pub data_frame_max: usize,
    /// Browser connections (handshaking or promoted) one daemon may carry.
    pub max_conns_per_daemon: usize,
    /// Of those, how many may be handshaking (not yet promoted) at once.
    pub max_handshaking_per_daemon: usize,
    /// Daemons logged in at once.
    pub max_daemons: usize,
    /// Browsers waiting on a nameplate at once.
    pub max_waiting: usize,
    /// A connection not promoted within this is closed (§7).
    pub handshake_timeout: Duration,
    /// How long a browser may wait for a daemon to join its nameplate (§3).
    pub wait_timeout: Duration,
    /// A daemon must answer the challenge within this.
    pub login_timeout: Duration,
    /// A browser connection with no frame either way for this is closed.
    pub idle_timeout: Duration,
    /// The bridge pings the daemon this often; three silent intervals close it.
    pub keepalive: Duration,
    /// A write to a peer (or a hand-off to the daemon's queue) that takes
    /// longer than this closes that peer: a slow side is dropped, not buffered.
    pub send_timeout: Duration,
    /// Frames queued for one browser before it counts as too slow.
    pub browser_queue: usize,
    /// Frames queued from all of one daemon's browsers toward it.
    pub daemon_queue: usize,
    /// New `/v1/connect` connections per client address per minute.
    pub connects_per_ip_per_minute: u32,
    /// Daemon logins per client address per minute.
    pub logins_per_ip_per_minute: u32,
    /// Nameplate waits per client address per minute.
    pub waits_per_ip_per_minute: u32,
    /// Pairing joins per client address per minute: a daemon joins once
    /// per pairing, so a stream of joins is someone guessing nameplates to
    /// take other people's waiting pages.
    pub joins_per_ip_per_minute: u32,
    /// Joins one daemon session may get wrong (no page waiting) before the
    /// bridge closes it.
    pub max_failed_joins_per_session: u32,
    /// Bytes forwarded per route (both directions) per day.
    pub daily_bytes_per_route: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            listen: SocketAddr::from(([127, 0, 0, 1], 8787)),
            trusted_proxy_header: None,
            handshake_frame_max: 1024,
            data_frame_max: 64 * 1024,
            max_conns_per_daemon: 8,
            max_handshaking_per_daemon: 4,
            max_daemons: 10_000,
            max_waiting: 10_000,
            handshake_timeout: Duration::from_secs(30),
            wait_timeout: Duration::from_secs(600),
            login_timeout: Duration::from_secs(10),
            idle_timeout: Duration::from_secs(600),
            keepalive: Duration::from_secs(30),
            send_timeout: Duration::from_secs(10),
            browser_queue: 64,
            daemon_queue: 256,
            connects_per_ip_per_minute: 30,
            logins_per_ip_per_minute: 10,
            waits_per_ip_per_minute: 10,
            joins_per_ip_per_minute: 6,
            max_failed_joins_per_session: 5,
            daily_bytes_per_route: 2 * 1024 * 1024 * 1024,
        }
    }
}

/// The options, as `(flag, what it sets)`, for `--help`.
pub const OPTIONS: &[(&str, &str)] = &[
    ("listen", "listen address (default 127.0.0.1:8787)"),
    (
        "trusted-proxy-header",
        "client-address header set by a trusted proxy, e.g. X-Forwarded-For (default: none, use the socket address)",
    ),
    ("handshake-frame-max", "bytes (default 1024)"),
    ("data-frame-max", "bytes (default 65536)"),
    ("max-conns-per-daemon", "default 8"),
    ("max-handshaking-per-daemon", "default 4"),
    ("max-daemons", "default 10000"),
    ("max-waiting", "default 10000"),
    ("handshake-timeout-secs", "default 30"),
    ("wait-timeout-secs", "default 600"),
    ("login-timeout-secs", "default 10"),
    ("idle-timeout-secs", "default 600"),
    ("keepalive-secs", "default 30"),
    ("send-timeout-secs", "default 10"),
    ("browser-queue", "frames (default 64)"),
    ("daemon-queue", "frames (default 256)"),
    ("connects-per-ip-per-minute", "default 30"),
    ("logins-per-ip-per-minute", "default 10"),
    ("waits-per-ip-per-minute", "default 10"),
    ("daily-bytes-per-route", "default 2147483648"),
];

impl Config {
    /// Read the settings from flags (`args` without the program name) and
    /// the environment (`env` looks one variable up).
    pub fn parse(
        args: impl IntoIterator<Item = String>,
        env: impl Fn(&str) -> Option<String>,
    ) -> Result<Self, String> {
        let mut values: Vec<(String, String)> = Vec::new();
        for (name, _) in OPTIONS {
            let var = format!(
                "AIDAEMON_BRIDGE_{}",
                name.replace('-', "_").to_ascii_uppercase()
            );
            if let Some(value) = env(&var) {
                values.push(((*name).to_string(), value));
            }
        }
        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            let Some(flag) = arg.strip_prefix("--") else {
                return Err(format!("unexpected argument {arg:?}"));
            };
            let (name, value) = match flag.split_once('=') {
                Some((name, value)) => (name.to_string(), value.to_string()),
                None => {
                    let value = args
                        .next()
                        .ok_or_else(|| format!("--{flag} needs a value"))?;
                    (flag.to_string(), value)
                }
            };
            if !OPTIONS.iter().any(|(known, _)| *known == name) {
                return Err(format!("unknown option --{name}"));
            }
            values.push((name, value));
        }
        let mut config = Self::default();
        for (name, value) in values {
            config.set(&name, &value)?;
        }
        config.check()?;
        Ok(config)
    }

    fn set(&mut self, name: &str, value: &str) -> Result<(), String> {
        fn num<T: std::str::FromStr>(name: &str, value: &str) -> Result<T, String> {
            value
                .parse()
                .map_err(|_| format!("--{name}: not a number: {value:?}"))
        }
        let secs = |value: &str| num::<u64>(name, value).map(Duration::from_secs);
        match name {
            "listen" => {
                self.listen = value
                    .parse()
                    .map_err(|_| format!("--listen: not an address: {value:?}"))?
            }
            "trusted-proxy-header" => {
                self.trusted_proxy_header = (!value.is_empty()).then(|| value.to_string())
            }
            "handshake-frame-max" => self.handshake_frame_max = num(name, value)?,
            "data-frame-max" => self.data_frame_max = num(name, value)?,
            "max-conns-per-daemon" => self.max_conns_per_daemon = num(name, value)?,
            "max-handshaking-per-daemon" => self.max_handshaking_per_daemon = num(name, value)?,
            "max-daemons" => self.max_daemons = num(name, value)?,
            "max-waiting" => self.max_waiting = num(name, value)?,
            "handshake-timeout-secs" => self.handshake_timeout = secs(value)?,
            "wait-timeout-secs" => self.wait_timeout = secs(value)?,
            "login-timeout-secs" => self.login_timeout = secs(value)?,
            "idle-timeout-secs" => self.idle_timeout = secs(value)?,
            "keepalive-secs" => self.keepalive = secs(value)?,
            "send-timeout-secs" => self.send_timeout = secs(value)?,
            "browser-queue" => self.browser_queue = num(name, value)?,
            "daemon-queue" => self.daemon_queue = num(name, value)?,
            "connects-per-ip-per-minute" => self.connects_per_ip_per_minute = num(name, value)?,
            "logins-per-ip-per-minute" => self.logins_per_ip_per_minute = num(name, value)?,
            "waits-per-ip-per-minute" => self.waits_per_ip_per_minute = num(name, value)?,
            "joins-per-ip-per-minute" => self.joins_per_ip_per_minute = num(name, value)?,
            "max-failed-joins-per-session" => self.max_failed_joins_per_session = num(name, value)?,
            "daily-bytes-per-route" => self.daily_bytes_per_route = num(name, value)?,
            _ => return Err(format!("unknown option --{name}")),
        }
        Ok(())
    }

    /// Refuse settings the bridge cannot work with.
    pub fn check(&self) -> Result<(), String> {
        if let Some(name) = &self.trusted_proxy_header
            && axum::http::HeaderName::from_bytes(name.as_bytes()).is_err()
        {
            return Err("--trusted-proxy-header is not a header name".into());
        }
        if self.handshake_frame_max == 0 || self.handshake_frame_max > self.data_frame_max {
            return Err("handshake-frame-max must be 1..=data-frame-max".into());
        }
        if self.max_handshaking_per_daemon == 0
            || self.max_handshaking_per_daemon > self.max_conns_per_daemon
        {
            return Err("max-handshaking-per-daemon must be 1..=max-conns-per-daemon".into());
        }
        if self.browser_queue == 0 || self.daemon_queue == 0 {
            return Err("queues must hold at least one frame".into());
        }
        for (name, duration) in [
            ("handshake-timeout-secs", self.handshake_timeout),
            ("wait-timeout-secs", self.wait_timeout),
            ("login-timeout-secs", self.login_timeout),
            ("idle-timeout-secs", self.idle_timeout),
            ("keepalive-secs", self.keepalive),
            ("send-timeout-secs", self.send_timeout),
        ] {
            if duration.is_zero() {
                return Err(format!("--{name} must be more than 0"));
            }
        }
        Ok(())
    }

    /// Largest daemon frame: a DATA frame plus its 4-byte connection id.
    pub fn daemon_frame_max(&self) -> usize {
        self.data_frame_max + 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_beat_environment() {
        let env = |var: &str| match var {
            "AIDAEMON_BRIDGE_LISTEN" => Some("0.0.0.0:9000".to_string()),
            "AIDAEMON_BRIDGE_MAX_DAEMONS" => Some("5".to_string()),
            _ => None,
        };
        let config = Config::parse(["--max-daemons".to_string(), "7".to_string()], env).unwrap();
        assert_eq!(config.listen, "0.0.0.0:9000".parse().unwrap());
        assert_eq!(config.max_daemons, 7);
    }

    #[test]
    fn refuses_unknown_and_bad_values() {
        assert!(Config::parse(["--nope=1".to_string()], |_| None).is_err());
        assert!(Config::parse(["--max-daemons=x".to_string()], |_| None).is_err());
        assert!(Config::parse(["--handshake-frame-max=0".to_string()], |_| None).is_err());
    }
}
