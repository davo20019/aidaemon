//! The pairing code an e-ink device shows its owner, who types it into a
//! trusted app (or scans it as a QR code).
//!
//! 25 Crockford Base32 characters, read with spaces and dashes ignored and
//! in any case: a 4-character nameplate (what the bridge routes by), the
//! 13-character secret (65 bits, never sent to the bridge), and 8
//! characters encoding the first 5 bytes of the device's `device_id`, which
//! binds the pairing to the key that made the code.

use aidaemon_bridge_proto::ids;

/// A parsed pairing code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PairingCode {
    pub nameplate: String,
    pub secret: String,
    /// The leading bytes of the device's `device_id`.
    pub device_key_hash: Vec<u8>,
}

const ALPHABET: &[u8; 32] = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ";

/// Parse a code as typed: `None` if it is not one.
pub fn parse_code(text: &str) -> Option<PairingCode> {
    let normalized: String = text
        .chars()
        .filter(|c| !c.is_whitespace() && *c != '-')
        .map(|c| c.to_ascii_uppercase())
        .collect();
    if normalized.len() != 25 {
        return None;
    }
    let nameplate = normalized[..4].to_string();
    let secret = normalized[4..17].to_string();
    let hash = decode_base32(&normalized[17..25])?;
    if !ids::is_valid_nameplate(&nameplate) || !ids::is_valid_secret(secret.as_bytes()) {
        return None;
    }
    Some(PairingCode {
        nameplate,
        secret,
        device_key_hash: hash,
    })
}

/// 8 Crockford Base32 characters to 5 bytes.
fn decode_base32(text: &str) -> Option<Vec<u8>> {
    let mut bits: u64 = 0;
    for c in text.bytes() {
        let value = ALPHABET.iter().position(|a| *a == c)? as u64;
        bits = (bits << 5) | value;
    }
    Some(bits.to_be_bytes()[3..8].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode_base32(bytes: &[u8; 5]) -> String {
        let mut value = [0u8; 8];
        value[3..].copy_from_slice(bytes);
        let bits = u64::from_be_bytes(value);
        (0..8)
            .rev()
            .map(|i| ALPHABET[((bits >> (i * 5)) & 31) as usize] as char)
            .collect()
    }

    #[test]
    fn a_code_reads_back_however_it_is_typed() {
        let hash = [0xde, 0xad, 0xbe, 0xef, 0x42];
        let code = format!("K7F2 9QXM4HTAPZ6CR {}", encode_base32(&hash));
        let parsed = parse_code(&code).unwrap();
        assert_eq!(parsed.nameplate, "K7F2");
        assert_eq!(parsed.secret, "9QXM4HTAPZ6CR");
        assert_eq!(parsed.device_key_hash, hash);
        let dashed = code.to_lowercase().replace(' ', "-");
        assert_eq!(parse_code(&dashed), Some(parsed));
    }

    #[test]
    fn anything_else_is_refused() {
        assert!(parse_code("").is_none());
        assert!(parse_code("K7F29QXM4HTAPZ6CR12345").is_none());
        assert!(
            parse_code("K7FU9QXM4HTAPZ6CR12345678").is_none(),
            "U is not Crockford"
        );
    }
}
