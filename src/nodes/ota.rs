use std::sync::Arc;

use sha2::{Digest, Sha256};

use crate::config::NodeOtaConfig;

use super::auth;
use super::protocol::{FirmwareReleaseManifest, FirmwareUpdateOffer};

pub const FIRMWARE_RELEASE_SCHEMA: &str = "aidaemon.firmware.release.v1";
pub const K10_BOARD_ID: &str = "unihiker_k10";

#[derive(Clone)]
pub struct FirmwareRelease {
    manifest: FirmwareReleaseManifest,
    image: Arc<[u8]>,
}

impl FirmwareRelease {
    pub fn load(config: &NodeOtaConfig) -> anyhow::Result<Self> {
        anyhow::ensure!(config.enabled, "Node OTA is disabled");
        let manifest_path = shellexpand::tilde(config.manifest_path.trim()).into_owned();
        let image_path = shellexpand::tilde(config.image_path.trim()).into_owned();
        let manifest: FirmwareReleaseManifest =
            serde_json::from_slice(&std::fs::read(&manifest_path)?)?;
        let image = std::fs::read(&image_path)?;
        Self::validate(
            manifest,
            image,
            &config.release_public_key_sec1,
            config.max_image_bytes,
        )
    }

    fn validate(
        manifest: FirmwareReleaseManifest,
        image: Vec<u8>,
        release_public_key_sec1: &str,
        maximum_image_bytes: usize,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            manifest.schema == FIRMWARE_RELEASE_SCHEMA,
            "firmware manifest schema is unsupported"
        );
        super::domain::validate_identifier("firmware release id", &manifest.release_id, 8, 80)?;
        anyhow::ensure!(
            (1..=32).contains(&manifest.version.len())
                && manifest.version.bytes().all(|byte| {
                    byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'+')
                }),
            "firmware version is invalid"
        );
        anyhow::ensure!(
            manifest.board == K10_BOARD_ID,
            "firmware release targets an unsupported board"
        );
        anyhow::ensure!(manifest.sequence > 0, "firmware sequence must be positive");
        anyhow::ensure!(
            !image.is_empty() && image.len() <= maximum_image_bytes,
            "firmware image exceeds the configured OTA slot bound"
        );
        anyhow::ensure!(
            manifest.size_bytes == image.len() as u64,
            "firmware image size does not match its manifest"
        );
        anyhow::ensure!(
            manifest.sha256.len() == 64
                && manifest.sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
                && manifest.sha256 == manifest.sha256.to_ascii_lowercase(),
            "firmware manifest SHA-256 is invalid"
        );
        let actual_digest = format!("{:x}", Sha256::digest(&image));
        anyhow::ensure!(
            actual_digest == manifest.sha256,
            "firmware image digest does not match its manifest"
        );
        let public_key = auth::decode_public_key(release_public_key_sec1)?;
        auth::verify_session_signature(
            &public_key,
            &manifest.canonical_bytes(),
            &manifest.signature_der,
        )
        .map_err(|_| anyhow::anyhow!("firmware release signature verification failed"))?;
        Ok(Self {
            manifest,
            image: image.into(),
        })
    }

    pub fn offer(&self) -> FirmwareUpdateOffer {
        FirmwareUpdateOffer {
            schema: self.manifest.schema.clone(),
            release_id: self.manifest.release_id.clone(),
            version: self.manifest.version.clone(),
            board: self.manifest.board.clone(),
            sequence: self.manifest.sequence,
            size_bytes: self.manifest.size_bytes,
            sha256: self.manifest.sha256.clone(),
            signature_der: self.manifest.signature_der.clone(),
            download_path: format!("/node/v1/firmware/{}", self.manifest.release_id),
        }
    }

    pub fn image_for(&self, release_id: &str) -> anyhow::Result<Arc<[u8]>> {
        anyhow::ensure!(
            release_id == self.manifest.release_id,
            "firmware release is unavailable"
        );
        Ok(self.image.clone())
    }

    pub fn manifest(&self) -> &FirmwareReleaseManifest {
        &self.manifest
    }

    #[cfg(test)]
    pub(crate) fn signed_for_test(image: &[u8], version: &str, sequence: u64) -> Self {
        use base64::engine::general_purpose::URL_SAFE_NO_PAD;
        use base64::Engine;
        use p256::ecdsa::signature::Signer;
        use p256::ecdsa::{Signature, SigningKey};

        let key = SigningKey::random(&mut rand::thread_rng());
        let public_key =
            URL_SAFE_NO_PAD.encode(key.verifying_key().to_encoded_point(true).as_bytes());
        let mut manifest = FirmwareReleaseManifest {
            schema: FIRMWARE_RELEASE_SCHEMA.to_string(),
            release_id: "release_test_0001".to_string(),
            version: version.to_string(),
            board: K10_BOARD_ID.to_string(),
            sequence,
            size_bytes: image.len() as u64,
            sha256: format!("{:x}", Sha256::digest(image)),
            signature_der: String::new(),
        };
        let signature: Signature = key.sign(&manifest.canonical_bytes());
        manifest.signature_der = URL_SAFE_NO_PAD.encode(signature.to_der().as_bytes());
        Self::validate(manifest, image.to_vec(), &public_key, 2_621_440).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use base64::Engine;
    use p256::ecdsa::signature::Signer;
    use p256::ecdsa::{Signature, SigningKey};

    fn signed_release(image: &[u8]) -> (FirmwareReleaseManifest, String) {
        let key = SigningKey::random(&mut rand::thread_rng());
        let public_key =
            URL_SAFE_NO_PAD.encode(key.verifying_key().to_encoded_point(true).as_bytes());
        let mut manifest = FirmwareReleaseManifest {
            schema: FIRMWARE_RELEASE_SCHEMA.to_string(),
            release_id: "release_test_0001".to_string(),
            version: "0.4.0".to_string(),
            board: K10_BOARD_ID.to_string(),
            sequence: 1,
            size_bytes: image.len() as u64,
            sha256: format!("{:x}", Sha256::digest(image)),
            signature_der: String::new(),
        };
        let signature: Signature = key.sign(&manifest.canonical_bytes());
        manifest.signature_der = URL_SAFE_NO_PAD.encode(signature.to_der().as_bytes());
        (manifest, public_key)
    }

    #[test]
    fn accepts_only_matching_signed_immutable_image() {
        let image = b"synthetic ESP application image";
        let (manifest, public_key) = signed_release(image);
        let release =
            FirmwareRelease::validate(manifest.clone(), image.to_vec(), &public_key, 2_621_440)
                .unwrap();
        assert_eq!(release.offer().sequence, 1);
        assert_eq!(
            release.image_for("release_test_0001").unwrap().as_ref(),
            image
        );

        let mut tampered_manifest = manifest;
        tampered_manifest.version = "0.4.1".to_string();
        assert!(FirmwareRelease::validate(
            tampered_manifest,
            image.to_vec(),
            &public_key,
            2_621_440,
        )
        .is_err());
    }

    #[test]
    fn rejects_tampered_image_and_wrong_board() {
        let image = b"synthetic ESP application image";
        let (manifest, public_key) = signed_release(image);
        assert!(FirmwareRelease::validate(
            manifest.clone(),
            b"tampered".to_vec(),
            &public_key,
            2_621_440,
        )
        .is_err());
        let mut wrong_board = manifest;
        wrong_board.board = "different_board".to_string();
        assert!(
            FirmwareRelease::validate(wrong_board, image.to_vec(), &public_key, 2_621_440,)
                .is_err()
        );
    }
}
