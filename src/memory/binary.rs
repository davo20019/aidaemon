//! Binary encoding/decoding for f32 embedding vectors.
//!
//! Stores embeddings as flat little-endian f32 bytes (384 dims × 4 bytes = 1,536 bytes)
//! instead of JSON text (~4,700 bytes), saving ~67% storage with faster serialization.

use anyhow::{bail, Result};

/// Expected embedding dimension (AllMiniLM-L6-v2).
const EMBEDDING_DIM: usize = 384;

/// Expected binary blob size: 384 × 4 bytes.
const BINARY_BLOB_SIZE: usize = EMBEDDING_DIM * 4;

/// Encode an f32 embedding vector as flat little-endian bytes.
pub fn encode_embedding(vec: &[f32]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(vec.len() * 4);
    for &val in vec {
        buf.extend_from_slice(&val.to_le_bytes());
    }
    buf
}

/// Decode an embedding blob, auto-detecting format:
/// - If length == 1,536 → binary little-endian f32
/// - If starts with `[` → legacy JSON (backward compat)
/// - Otherwise → error
pub fn decode_embedding(blob: &[u8]) -> Result<Vec<f32>> {
    if blob.len() == BINARY_BLOB_SIZE {
        // Binary format: flat little-endian f32
        let mut vec = Vec::with_capacity(EMBEDDING_DIM);
        for chunk in blob.chunks_exact(4) {
            vec.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(vec)
    } else if blob.first() == Some(&b'[') {
        // Legacy JSON format
        let vec: Vec<f32> = serde_json::from_slice(blob)?;
        Ok(vec)
    } else {
        bail!(
            "Unknown embedding format: length={}, first byte={:?}",
            blob.len(),
            blob.first()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_binary() {
        let original: Vec<f32> = (0..384).map(|i| i as f32 * 0.001).collect();
        let encoded = encode_embedding(&original);
        assert_eq!(encoded.len(), 1536);
        let decoded = decode_embedding(&encoded).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_decode_legacy_json() {
        let original: Vec<f32> = (0..384).map(|i| i as f32 * 0.001).collect();
        let json_blob = serde_json::to_vec(&original).unwrap();
        assert!(json_blob.len() > 1536); // JSON is bigger
        let decoded = decode_embedding(&json_blob).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_invalid_format() {
        let bad_blob = vec![0u8; 100];
        assert!(decode_embedding(&bad_blob).is_err());
    }

    #[test]
    fn test_empty_embedding() {
        let empty: Vec<f32> = vec![];
        let encoded = encode_embedding(&empty);
        assert!(encoded.is_empty());
        // Empty blob won't match binary (len != 1536) or JSON (doesn't start with [)
        // So it should error
        assert!(decode_embedding(&encoded).is_err());
    }

    #[test]
    fn test_special_float_values() {
        let mut vec: Vec<f32> = (0..384).map(|i| i as f32).collect();
        vec[0] = f32::NEG_INFINITY;
        vec[1] = f32::INFINITY;
        vec[2] = 0.0;
        vec[3] = -0.0;
        let encoded = encode_embedding(&vec);
        let decoded = decode_embedding(&encoded).unwrap();
        assert_eq!(vec.len(), decoded.len());
        assert!(decoded[0].is_infinite() && decoded[0].is_sign_negative());
        assert!(decoded[1].is_infinite() && decoded[1].is_sign_positive());
    }
}
