use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use hmac::{Hmac, Mac};
use p256::ecdsa::signature::Verifier;
use p256::ecdsa::{Signature, VerifyingKey};
use rand::RngCore;
use sha2::{Digest, Sha256};

type HmacSha256 = Hmac<Sha256>;

const INSTANCE_KEYCHAIN_FIELD: &str = "node_gateway_instance_key_v1";

pub fn load_or_create_instance_key() -> anyhow::Result<[u8; 32]> {
    if let Ok(encoded) = crate::config::resolve_from_keychain(INSTANCE_KEYCHAIN_FIELD) {
        let decoded = URL_SAFE_NO_PAD
            .decode(encoded.trim())
            .map_err(|_| anyhow::anyhow!("Stored Node Gateway instance key is malformed"))?;
        anyhow::ensure!(
            decoded.len() == 32,
            "Stored Node Gateway instance key has invalid length"
        );
        let mut key = [0_u8; 32];
        key.copy_from_slice(&decoded);
        return Ok(key);
    }

    let mut key = [0_u8; 32];
    rand::thread_rng().fill_bytes(&mut key);
    crate::config::store_in_keychain(INSTANCE_KEYCHAIN_FIELD, &URL_SAFE_NO_PAD.encode(key))?;
    Ok(key)
}

pub fn random_secret(bytes: usize) -> String {
    let mut data = vec![0_u8; bytes];
    rand::thread_rng().fill_bytes(&mut data);
    URL_SAFE_NO_PAD.encode(data)
}

pub fn keyed_digest(key: &[u8; 32], purpose: &str, value: &[u8]) -> Vec<u8> {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC accepts 32-byte key");
    mac.update(b"aidaemon-node-v1\0");
    mac.update(purpose.as_bytes());
    mac.update(b"\0");
    mac.update(value);
    mac.finalize().into_bytes().to_vec()
}

pub fn constant_time_digest_matches(
    key: &[u8; 32],
    purpose: &str,
    value: &[u8],
    expected: &[u8],
) -> bool {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC accepts 32-byte key");
    mac.update(b"aidaemon-node-v1\0");
    mac.update(purpose.as_bytes());
    mac.update(b"\0");
    mac.update(value);
    mac.verify_slice(expected).is_ok()
}

pub fn decode_public_key(encoded: &str) -> anyhow::Result<Vec<u8>> {
    let bytes = URL_SAFE_NO_PAD
        .decode(encoded)
        .map_err(|_| anyhow::anyhow!("public_key_sec1 is not valid base64url"))?;
    anyhow::ensure!(
        (33..=65).contains(&bytes.len()),
        "public key length is invalid"
    );
    VerifyingKey::from_sec1_bytes(&bytes)
        .map_err(|_| anyhow::anyhow!("public key is not a valid P-256 SEC1 point"))?;
    Ok(bytes)
}

pub fn public_key_fingerprint(public_key_sec1: &[u8]) -> String {
    let digest = Sha256::digest(public_key_sec1);
    digest[..8]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<Vec<_>>()
        .join(":")
}

pub fn canonical_session_challenge(
    credential_id: &str,
    challenge_id: &str,
    nonce: &str,
    protocol_major: u16,
    instance_id: &str,
    boot_id: &str,
) -> Vec<u8> {
    format!(
        "AIDAEMON-NODE-V1\n{credential_id}\n{challenge_id}\n{nonce}\n{protocol_major}\n{instance_id}\n{boot_id}"
    )
    .into_bytes()
}

pub fn canonical_credential_rotation(
    node_id: &str,
    node_session_id: &str,
    rotation_id: &str,
    nonce: &str,
    instance_id: &str,
    new_public_key_sec1: &str,
) -> Vec<u8> {
    format!(
        "AIDAEMON-NODE-ROTATE-V1\n{node_id}\n{node_session_id}\n{rotation_id}\n{nonce}\n{instance_id}\n{new_public_key_sec1}"
    )
    .into_bytes()
}

pub fn verify_session_signature(
    public_key_sec1: &[u8],
    canonical: &[u8],
    signature_der_b64: &str,
) -> anyhow::Result<()> {
    let key = VerifyingKey::from_sec1_bytes(public_key_sec1)
        .map_err(|_| anyhow::anyhow!("registered public key is invalid"))?;
    let signature_bytes = URL_SAFE_NO_PAD
        .decode(signature_der_b64)
        .map_err(|_| anyhow::anyhow!("signature is not valid base64url"))?;
    anyhow::ensure!(signature_bytes.len() <= 80, "signature is too large");
    let signature = Signature::from_der(&signature_bytes)
        .map_err(|_| anyhow::anyhow!("signature is not valid DER ECDSA"))?;
    key.verify(canonical, &signature)
        .map_err(|_| anyhow::anyhow!("signature verification failed"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use p256::ecdsa::signature::Signer;
    use p256::ecdsa::SigningKey;

    #[test]
    fn challenge_signature_round_trips_and_is_context_bound() {
        let signing_key = SigningKey::random(&mut rand::thread_rng());
        let verifying_key = signing_key.verifying_key();
        let canonical =
            canonical_session_challenge("cred", "challenge", "nonce", 1, "instance", "boot");
        let signature: Signature = signing_key.sign(&canonical);
        let encoded = URL_SAFE_NO_PAD.encode(signature.to_der().as_bytes());
        verify_session_signature(
            verifying_key.to_encoded_point(true).as_bytes(),
            &canonical,
            &encoded,
        )
        .unwrap();
        assert!(verify_session_signature(
            verifying_key.to_encoded_point(true).as_bytes(),
            b"different",
            &encoded,
        )
        .is_err());
    }

    #[test]
    fn keyed_digests_are_purpose_bound() {
        let key = [9_u8; 32];
        let digest = keyed_digest(&key, "session", b"secret");
        assert!(constant_time_digest_matches(
            &key, "session", b"secret", &digest
        ));
        assert!(!constant_time_digest_matches(
            &key, "offer", b"secret", &digest
        ));
    }
}
