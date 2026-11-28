//! Pairing an e-ink reader (a Kindle) from the owner's trusted app.
//!
//! The reader shows a 25-character code: a 4-character nameplate (what the
//! bridge routes by), a 13-character secret (never sent to the bridge) and
//! 8 characters of its key's id (`device_id[0..5]`), all Crockford Base32.
//! The owner types it into the app, or scans the reader's QR code, a link
//! `https://app.aidaemon.ai/pair-eink#n=<nameplate>&s=<secret>&k=<device_id
//! hex>` the app turns into the same code.
//!
//! A link can come from anywhere (a message, a page), so pairing from one
//! also asks for the reader's check code: 4 characters derived from the
//! nameplate and secret that the reader shows next to its QR code. A link
//! someone else made names their reader, whose check code the owner's
//! Kindle does not show.

use sha2::{Digest, Sha256};

/// Crockford Base32, upper case.
pub const ALPHABET: &[u8; 32] = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ";

/// Characters in a pairing code.
pub const CODE_LEN: usize = 25;
/// Characters in a nameplate.
pub const NAMEPLATE_LEN: usize = 4;
/// Characters in the secret.
pub const SECRET_LEN: usize = 13;
/// Characters in a check code.
pub const CHECK_LEN: usize = 4;

/// The label the check code's hash starts with.
const CHECK_LABEL: &[u8] = b"aidaemon-eink-check-v1";

/// The path a reader's QR link opens.
pub const LINK_PATH: &str = "/pair-eink";

fn crockford(text: &str, len: usize) -> bool {
    text.len() == len && text.bytes().all(|b| ALPHABET.contains(&b))
}

/// 5 bytes as 8 Crockford Base32 characters.
fn encode_40(bytes: &[u8; 5]) -> String {
    let mut value = [0u8; 8];
    value[3..].copy_from_slice(bytes);
    let bits = u64::from_be_bytes(value);
    (0..8)
        .rev()
        .map(|i| ALPHABET[((bits >> (i * 5)) & 31) as usize] as char)
        .collect()
}

/// A code as the owner typed it, read the way the daemon reads it: spaces
/// and dashes ignored, any case. `None` if it is not a pairing code.
pub fn normalize_code(text: &str) -> Option<String> {
    let code: String = text
        .chars()
        .filter(|c| !c.is_whitespace() && *c != '-')
        .map(|c| c.to_ascii_uppercase())
        .collect();
    crockford(&code, CODE_LEN).then_some(code)
}

/// A code in groups the owner can read back against the reader's screen:
/// nameplate, secret, key id.
pub fn grouped(code: &str) -> String {
    match (code.get(..4), code.get(4..17), code.get(17..)) {
        (Some(a), Some(b), Some(c)) if code.len() == CODE_LEN => format!("{a}-{b}-{c}"),
        _ => code.to_owned(),
    }
}

/// The reader's check code for `nameplate` and `secret`: the first 4
/// Crockford Base32 characters of
/// `SHA-256("aidaemon-eink-check-v1" ‖ nameplate ‖ secret)` (ASCII bytes;
/// its first 5 bytes encoded as 8 characters, the first 4 kept: 20 bits).
pub fn check_code(nameplate: &str, secret: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(CHECK_LABEL);
    hasher.update(nameplate.as_bytes());
    hasher.update(secret.as_bytes());
    let digest = hasher.finalize();
    let mut first = [0u8; 5];
    first.copy_from_slice(&digest[..5]);
    encode_40(&first)[..CHECK_LEN].to_owned()
}

/// A reader's QR link, read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PairLink {
    /// The 25-character code it stands for, as the daemon takes it.
    pub code: String,
    /// The check code the reader shows beside its QR code.
    pub check: String,
}

/// Read a reader's QR link: `https://<site><prefix>/pair-eink#n=…&s=…&k=…`,
/// strictly. `n`: 4 Crockford characters; `s`: 13 upper-case Crockford
/// characters; `k`: the reader's `device_id`, 64 lower-case hex digits;
/// each once, nothing else. `None` for anything else.
pub fn parse_pair_link(url: &str) -> Option<PairLink> {
    // Machine format: an https URL, split at its fragment.
    let rest = url.trim().strip_prefix("https://")?;
    let (address, fragment) = rest.split_once('#')?;
    let path = address.split_once('/').map_or("", |(_, path)| path);
    let path = path.split_once('?').map_or(path, |(path, _)| path);
    let path = format!("/{}", path.trim_end_matches('/'));
    if !path.ends_with(LINK_PATH) {
        return None;
    }
    parse_pair_fragment(fragment)
}

/// [`parse_pair_link`]'s fragment part alone (`n=…&s=…&k=…`, no `#`).
pub fn parse_pair_fragment(fragment: &str) -> Option<PairLink> {
    let (mut n, mut s, mut k) = (None, None, None);
    for part in fragment.split('&') {
        let (key, value) = part.split_once('=')?;
        let slot = match key {
            "n" => &mut n,
            "s" => &mut s,
            "k" => &mut k,
            _ => return None,
        };
        if slot.replace(value).is_some() {
            return None;
        }
    }
    let (nameplate, secret, key) = (n?, s?, k?);
    if !crockford(nameplate, NAMEPLATE_LEN) || !crockford(secret, SECRET_LEN) {
        return None;
    }
    if key.len() != 64 || !key.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
        return None;
    }
    let mut first = [0u8; 5];
    for (i, byte) in first.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&key[i * 2..i * 2 + 2], 16).ok()?;
    }
    Some(PairLink {
        code: format!("{nameplate}{secret}{}", encode_40(&first)),
        check: check_code(nameplate, secret),
    })
}

/// What the owner typed as the check code, so far.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Check {
    /// Fewer than 4 characters yet.
    Incomplete,
    /// The reader's check code: pairing may go on.
    Matches,
    /// Not it: the link is dropped, and nothing pairs from it.
    Wrong,
}

impl PairLink {
    /// Judge the check code the owner typed (spaces and dashes ignored, any
    /// case).
    pub fn check_typed(&self, typed: &str) -> Check {
        let typed: String = typed
            .chars()
            .filter(|c| !c.is_whitespace() && *c != '-')
            .map(|c| c.to_ascii_uppercase())
            .collect();
        if typed.chars().count() < CHECK_LEN {
            Check::Incomplete
        } else if typed == self.check {
            Check::Matches
        } else {
            Check::Wrong
        }
    }
}

/// Where pairing a reader from this app stands.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum EinkPairing {
    #[default]
    Idle,
    /// Sent: the daemon waits for the reader's handshake through the bridge
    /// (up to two minutes).
    Pairing,
    /// Paired, under the name the reader gave.
    Paired(String),
    /// Refused or failed: why, as the daemon said it.
    Failed(String),
}

impl crate::client::Client {
    /// Pair the e-ink reader whose screen shows `code`, as the owner typed
    /// it (spaces, dashes, any case). False, with the reason in
    /// [`Self::eink_pairing`], when it is not a code; false too while a
    /// pairing is under way.
    pub fn pair_eink_device(&mut self, code: &str) -> bool {
        if self.eink_pairing == EinkPairing::Pairing {
            return false;
        }
        let Some(code) = normalize_code(code) else {
            self.eink_pairing = EinkPairing::Failed(
                "That is not the code your e-ink device shows: it has 25 letters and digits."
                    .into(),
            );
            return false;
        };
        self.eink_pairing = EinkPairing::Pairing;
        self.eink_pair_request = Some(self.send(
            aidaemon_client_protocol::Command::PairEinkDevice {
                code: grouped(&code),
            },
        ));
        true
    }

    /// Forget how the last pairing ended (the screen closed). A pairing
    /// under way keeps going: its answer still arrives.
    pub fn clear_eink_pairing(&mut self) {
        if self.eink_pairing != EinkPairing::Pairing {
            self.eink_pairing = EinkPairing::Idle;
        }
    }

    /// Set e-ink reader `device_id`'s access level; it applies at once (the
    /// app confirms a raise first). A refusal is in
    /// [`Self::eink_access_note`].
    pub fn set_eink_access(
        &mut self,
        device_id: &str,
        level: aidaemon_client_protocol::EinkAccess,
    ) {
        self.eink_access_note = None;
        self.eink_access_request =
            Some(self.send(aidaemon_client_protocol::Command::SetEinkAccess {
                device_id: device_id.to_owned(),
                level,
            }));
    }

    /// The answer to [`Self::set_eink_access`], if `frame` is it: the
    /// devices are read again to show the new level.
    fn eink_access_answer(&mut self, frame: &aidaemon_client_protocol::ServerFrame) -> bool {
        use aidaemon_client_protocol::{CommandResult, ServerFrame};
        let Some(asked) = self.eink_access_request.as_deref() else {
            return false;
        };
        match frame {
            ServerFrame::Ack {
                request_id,
                result: CommandResult::EinkAccessChanged { .. },
            } if request_id == asked => {
                self.eink_access_request = None;
                self.refresh_devices();
                true
            }
            ServerFrame::Error {
                request_id: Some(request_id),
                error,
            } if request_id == asked => {
                self.eink_access_request = None;
                self.eink_access_note = Some(error.message.clone());
                true
            }
            _ => false,
        }
    }

    /// The answer to [`Self::pair_eink_device`], if `frame` is it.
    pub(crate) fn eink_answer(&mut self, frame: &aidaemon_client_protocol::ServerFrame) -> bool {
        use aidaemon_client_protocol::{CommandResult, ServerFrame};
        if self.eink_access_answer(frame) {
            return true;
        }
        let Some(asked) = self.eink_pair_request.as_deref() else {
            return false;
        };
        match frame {
            ServerFrame::Ack {
                request_id, result, ..
            } if request_id == asked => {
                self.eink_pair_request = None;
                self.eink_pairing = match result {
                    CommandResult::EinkPaired { name } => EinkPairing::Paired(name.clone()),
                    _ => EinkPairing::Failed("The daemon answered something else.".into()),
                };
                self.refresh_devices();
                true
            }
            ServerFrame::Error {
                request_id: Some(request_id),
                error,
            } if request_id == asked => {
                self.eink_pair_request = None;
                self.eink_pairing = EinkPairing::Failed(error.message.clone());
                true
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const KEY: &str = "deadbeef42000000000000000000000000000000000000000000000000000001";

    fn link(fragment: &str) -> String {
        format!("https://app.aidaemon.ai/pair-eink#{fragment}")
    }

    /// Pinned: the e-ink page derives the same.
    #[test]
    fn check_code_vector() {
        assert_eq!(check_code("K7F2", "9QXM4HTAPZ6CR"), "FTJB");
        assert_eq!(check_code("0000", "0000000000000"), "X3N0");
    }

    #[test]
    fn a_link_reads_as_the_code_the_reader_shows() {
        let read = parse_pair_link(&link(&format!("n=K7F2&s=9QXM4HTAPZ6CR&k={KEY}"))).unwrap();
        // 0xde 0xad 0xbe 0xef 0x42 in Crockford Base32.
        assert_eq!(read.code, "K7F29QXM4HTAPZ6CRVTPVXVT2");
        assert_eq!(read.check, check_code("K7F2", "9QXM4HTAPZ6CR"));
        assert_eq!(
            normalize_code(&read.code).as_deref(),
            Some(read.code.as_str())
        );
        // Any order; the web app served under a prefix.
        let other = format!(
            "https://aidaemon.example.test/client/v1/next/pair-eink/#k={KEY}&s=9QXM4HTAPZ6CR&n=K7F2"
        );
        assert_eq!(parse_pair_link(&other), Some(read));
    }

    #[test]
    fn anything_malformed_is_refused() {
        let good = format!("n=K7F2&s=9QXM4HTAPZ6CR&k={KEY}");
        assert!(parse_pair_link(&link(&good)).is_some());
        for bad in [
            format!("n=k7f2&s=9QXM4HTAPZ6CR&k={KEY}"),
            format!("n=K7FU&s=9QXM4HTAPZ6CR&k={KEY}"),
            format!("n=K7F&s=9QXM4HTAPZ6CR&k={KEY}"),
            format!("n=K7F2&s=9qxm4htapz6cr&k={KEY}"),
            format!("n=K7F2&s=9QXM4HTAPZ6C&k={KEY}"),
            format!("n=K7F2&s=9QXM4HTAPZ6CR&k={}", KEY.to_uppercase()),
            format!("n=K7F2&s=9QXM4HTAPZ6CR&k={}", &KEY[..63]),
            format!("n=K7F2&s=9QXM4HTAPZ6CR&k={KEY}0"),
            format!("n=K7F2&s=9QXM4HTAPZ6CR&k={KEY}&n=K7F2"),
            format!("n=K7F2&s=9QXM4HTAPZ6CR&k={KEY}&x=1"),
            format!("n=K7F2&s=9QXM4HTAPZ6CR&k={KEY}&"),
            "n=K7F2&s=9QXM4HTAPZ6CR".to_owned(),
            format!("n=K7F2&s=9QXM4HTAPZ6CR&k=+{}", &KEY[1..]),
        ] {
            assert!(parse_pair_link(&link(&bad)).is_none(), "{bad} passed");
        }
        for bad_url in [
            format!("http://app.aidaemon.ai/pair-eink#{good}"),
            format!("https://app.aidaemon.ai/pair#{good}"),
            format!("https://app.aidaemon.ai/#{good}"),
            format!("https://app.aidaemon.ai/pair-einkx#{good}"),
            format!("https://app.aidaemon.ai/pair-eink?{good}"),
            format!("aidaemon-pair:#{good}"),
        ] {
            assert!(parse_pair_link(&bad_url).is_none(), "{bad_url} passed");
        }
    }

    #[test]
    fn typed_codes_read_however_they_are_typed() {
        let code = "K7F29QXM4HTAPZ6CRVTPVXVT2";
        assert_eq!(
            normalize_code("k7f2-9qxm4htapz6cr-vtpvxvt2").as_deref(),
            Some(code)
        );
        assert_eq!(
            normalize_code(" K7F2 9QXM4HTAPZ6CR VTPVXVT2 ").as_deref(),
            Some(code)
        );
        assert_eq!(grouped(code), "K7F2-9QXM4HTAPZ6CR-VTPVXVT2");
        assert!(normalize_code("K7F29QXM4HTAPZ6CRVTPVXVJ").is_none());
        assert!(normalize_code("K7FU9QXM4HTAPZ6CRVTPVXVT2").is_none());
        assert!(normalize_code("").is_none());
    }

    #[test]
    fn only_the_readers_check_code_matches() {
        let read = parse_pair_link(&link(&format!("n=K7F2&s=9QXM4HTAPZ6CR&k={KEY}"))).unwrap();
        assert_eq!(read.check_typed(""), Check::Incomplete);
        assert_eq!(read.check_typed(&read.check[..3]), Check::Incomplete);
        assert_eq!(read.check_typed(&read.check), Check::Matches);
        assert_eq!(read.check_typed(&read.check.to_lowercase()), Check::Matches);
        assert_eq!(read.check_typed("ZZZZ"), Check::Wrong);
        assert_eq!(read.check_typed(&format!("{}Z", read.check)), Check::Wrong);
    }

    fn answer(client: &mut crate::client::Client, json: String) {
        let frame = serde_json::from_str(&json).expect("a frame");
        client.update(
            crate::machine::Update::Frame(frame),
            &crate::client::Context::default(),
        );
    }

    #[test]
    fn the_client_pairs_a_typed_code_and_reads_the_answer() {
        use aidaemon_client_protocol::Command;
        let mut client = crate::client::Client::new();
        assert!(!client.pair_eink_device("not a code"));
        assert!(matches!(client.eink_pairing, EinkPairing::Failed(_)));
        assert!(client.take_outgoing().is_empty());

        assert!(client.pair_eink_device("k7f2 9qxm4htapz6cr-vtpvxvt2"));
        assert_eq!(client.eink_pairing, EinkPairing::Pairing);
        assert!(
            !client.pair_eink_device("K7F29QXM4HTAPZ6CRVTPVXVT2"),
            "one at a time"
        );
        let sent = client.take_outgoing();
        assert_eq!(sent.len(), 1);
        let (id, command) = sent[0].clone();
        assert_eq!(
            command,
            Command::PairEinkDevice {
                code: "K7F2-9QXM4HTAPZ6CR-VTPVXVT2".into()
            }
        );
        // Another command's answer is not this one's.
        answer(
            &mut client,
            r#"{"type":"error","request_id":"other","error":{"code":"forbidden","message":"no","retryable":false}}"#
                .into(),
        );
        assert_eq!(client.eink_pairing, EinkPairing::Pairing);
        client.clear_eink_pairing();
        assert_eq!(client.eink_pairing, EinkPairing::Pairing, "still under way");
        answer(
            &mut client,
            format!(
                r#"{{"type":"ack","request_id":"{id}","result":{{"type":"eink_paired","name":"Synthetic Kindle"}}}}"#
            ),
        );
        assert_eq!(
            client.eink_pairing,
            EinkPairing::Paired("Synthetic Kindle".into())
        );
        // The device list is read again, to show it.
        assert!(
            client
                .take_outgoing()
                .iter()
                .any(|(_, c)| *c == Command::ListDevices)
        );
        client.clear_eink_pairing();
        assert_eq!(client.eink_pairing, EinkPairing::Idle);

        assert!(client.pair_eink_device("K7F29QXM4HTAPZ6CRVTPVXVT2"));
        let (id, _) = client.take_outgoing().pop().unwrap();
        answer(
            &mut client,
            format!(
                r#"{{"type":"error","request_id":"{id}","error":{{"code":"conflict","message":"E-ink access is off","retryable":false}}}}"#
            ),
        );
        assert_eq!(
            client.eink_pairing,
            EinkPairing::Failed("E-ink access is off".into())
        );
    }

    #[test]
    fn the_client_sets_a_readers_level_and_reads_the_devices_again() {
        use aidaemon_client_protocol::{Command, EinkAccess};
        let mut client = crate::client::Client::new();
        client.set_eink_access("dv-k", EinkAccess::Full);
        let (id, command) = client.take_outgoing().pop().unwrap();
        assert_eq!(
            command,
            Command::SetEinkAccess {
                device_id: "dv-k".into(),
                level: EinkAccess::Full
            }
        );
        answer(
            &mut client,
            format!(
                r#"{{"type":"ack","request_id":"{id}","result":{{"type":"eink_access_changed","device_id":"dv-k","level":"full"}}}}"#
            ),
        );
        assert!(client.eink_access_note.is_none());
        assert!(
            client
                .take_outgoing()
                .iter()
                .any(|(_, c)| *c == Command::ListDevices)
        );
        client.set_eink_access("dv-k", EinkAccess::Full);
        let (id, _) = client.take_outgoing().pop().unwrap();
        answer(
            &mut client,
            format!(
                r#"{{"type":"error","request_id":"{id}","error":{{"code":"forbidden","message":"no","retryable":false}}}}"#
            ),
        );
        assert_eq!(client.eink_access_note.as_deref(), Some("no"));
    }
}
