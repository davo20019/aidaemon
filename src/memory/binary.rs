//! Embedding blobs, encoded and decoded by the SDK memory crate: flat
//! little-endian f32 bytes (384 dims × 4 bytes = 1,536 bytes), and reading
//! the legacy JSON-array format older databases hold.

pub use aidaemon_memory::encode_embedding;

/// Expected embedding dimension (AllMiniLM-L6-v2).
const EMBEDDING_DIM: usize = 384;

/// Decode a 384-dimension embedding blob, binary or legacy JSON.
pub fn decode_embedding(blob: &[u8]) -> anyhow::Result<Vec<f32>> {
    decode_embedding_with_dim(blob, EMBEDDING_DIM)
}

/// Decode a vector validated against the dimension stored with its index
/// row, so a future embedding model can use another dimension.
pub fn decode_embedding_with_dim(blob: &[u8], expected_dim: usize) -> anyhow::Result<Vec<f32>> {
    aidaemon_memory::decode_embedding_with_dim(blob, expected_dim).map_err(anyhow::Error::from)
}
