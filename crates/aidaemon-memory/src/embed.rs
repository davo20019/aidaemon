//! Text embedders: the [`Embedder`] extension point, a deterministic
//! offline [`HashEmbedder`], and (feature `fastembed`) a local ONNX model.

use async_trait::async_trait;

use crate::error::Result;
use crate::scoring::tokens;

/// Turns text into vectors. Implement this to use an embedding API or
/// another local model; [`crate::Memory`] only sees `Arc<dyn Embedder>`.
///
/// Vectors should be L2-normalized so cosine similarity is a dot product,
/// but [`cosine`] normalizes anyway.
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Stable identifier persisted beside every vector. Changing the model
    /// changes this id, and old vectors are then ignored (lexical recall
    /// still works) instead of being compared across models.
    fn model_id(&self) -> &str;

    /// Length of every returned vector.
    fn dimensions(&self) -> usize;

    /// One vector per input text, in order.
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

#[async_trait]
impl<E: Embedder + ?Sized> Embedder for std::sync::Arc<E> {
    fn model_id(&self) -> &str {
        (**self).model_id()
    }
    fn dimensions(&self) -> usize {
        (**self).dimensions()
    }
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        (**self).embed(texts).await
    }
}

/// Cosine similarity; 0.0 for mismatched lengths or zero vectors.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na <= f32::EPSILON || nb <= f32::EPSILON {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Little-endian f32 bytes, the BLOB format used by `SqliteMemoryStore`.
pub fn encode_embedding(vector: &[f32]) -> Vec<u8> {
    vector.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Inverse of [`encode_embedding`]; `None` for a malformed blob.
pub fn decode_embedding(bytes: &[u8]) -> Option<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) {
        return None;
    }
    Some(
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

/// Deterministic hashed bag-of-words embedder. No model, no network, same
/// output on every machine (FNV-1a hashing). Similarity reflects shared
/// words only, so it suits tests and offline use, not production recall.
#[derive(Debug, Clone)]
pub struct HashEmbedder {
    dims: usize,
    id: String,
}

impl HashEmbedder {
    pub fn new(dims: usize) -> Self {
        let dims = dims.max(8);
        Self {
            dims,
            id: format!("hash-bow-v1/{dims}"),
        }
    }

    /// Embed one text synchronously.
    pub fn embed_one(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0.0f32; self.dims];
        for token in tokens(text) {
            let h = fnv1a(token.as_bytes());
            let index = (h % self.dims as u64) as usize;
            let sign = if (h >> 63) == 0 { 1.0 } else { -1.0 };
            v[index] += sign;
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            v.iter_mut().for_each(|x| *x /= norm);
        }
        v
    }
}

impl Default for HashEmbedder {
    fn default() -> Self {
        Self::new(256)
    }
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        hash ^= u64::from(*b);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

#[async_trait]
impl Embedder for HashEmbedder {
    fn model_id(&self) -> &str {
        &self.id
    }

    fn dimensions(&self) -> usize {
        self.dims
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.embed_one(t)).collect())
    }
}

#[cfg(feature = "fastembed")]
pub use fast::FastEmbedder;

#[cfg(feature = "fastembed")]
mod fast {
    use std::path::PathBuf;
    use std::sync::Arc;

    use async_trait::async_trait;
    use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
    use tokio::sync::OnceCell;

    use super::Embedder;
    use crate::error::{MemoryError, Result};

    /// Local AllMiniLM-L6-v2 (384 dimensions) via fastembed, the daemon's
    /// embedding model. The model is downloaded and loaded lazily on the
    /// first `embed` call; loading and inference run on `spawn_blocking`.
    pub struct FastEmbedder {
        cache_dir: Option<PathBuf>,
        show_download_progress: bool,
        model: OnceCell<Arc<TextEmbedding>>,
    }

    impl FastEmbedder {
        /// Model id persisted beside vectors (same as the daemon's).
        pub const MODEL_ID: &'static str = "fastembed/all-MiniLM-L6-v2";

        pub fn new() -> Self {
            Self {
                cache_dir: None,
                show_download_progress: false,
                model: OnceCell::new(),
            }
        }

        /// Where model files are cached (fastembed's default otherwise).
        pub fn cache_dir(mut self, dir: impl Into<PathBuf>) -> Self {
            self.cache_dir = Some(dir.into());
            self
        }

        pub fn show_download_progress(mut self, show: bool) -> Self {
            self.show_download_progress = show;
            self
        }

        async fn model(&self) -> Result<Arc<TextEmbedding>> {
            let cache_dir = self.cache_dir.clone();
            let progress = self.show_download_progress;
            self.model
                .get_or_try_init(|| async move {
                    tokio::task::spawn_blocking(move || {
                        let mut options = InitOptions::new(EmbeddingModel::AllMiniLML6V2)
                            .with_show_download_progress(progress);
                        if let Some(dir) = cache_dir {
                            options = options.with_cache_dir(dir);
                        }
                        TextEmbedding::try_new(options)
                            .map(Arc::new)
                            .map_err(|e| MemoryError::Embedding(e.to_string()))
                    })
                    .await
                    .map_err(|e| MemoryError::Embedding(e.to_string()))?
                })
                .await
                .cloned()
        }
    }

    impl Default for FastEmbedder {
        fn default() -> Self {
            Self::new()
        }
    }

    #[async_trait]
    impl Embedder for FastEmbedder {
        fn model_id(&self) -> &str {
            Self::MODEL_ID
        }

        fn dimensions(&self) -> usize {
            384
        }

        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            if texts.is_empty() {
                return Ok(Vec::new());
            }
            let model = self.model().await?;
            let texts = texts.to_vec();
            tokio::task::spawn_blocking(move || model.embed(texts, None))
                .await
                .map_err(|e| MemoryError::Embedding(e.to_string()))?
                .map_err(|e| MemoryError::Embedding(e.to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn hash_embedder_is_deterministic_and_normalized() {
        let a = HashEmbedder::new(128);
        let b = HashEmbedder::new(128);
        let texts = vec!["Alice Rivera likes hiking".to_string()];
        let va = a.embed(&texts).await.unwrap();
        let vb = b.embed(&texts).await.unwrap();
        assert_eq!(va, vb);
        assert_eq!(va[0].len(), 128);
        let norm: f32 = va[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
        let related = a.embed_one("hiking trips for Alice");
        let unrelated = a.embed_one("quarterly invoice totals");
        assert!(cosine(&va[0], &related) > cosine(&va[0], &unrelated));
        assert_eq!(a.embed_one(""), vec![0.0; 128]);
    }

    #[test]
    fn blobs_round_trip() {
        let v = vec![0.25f32, -1.5, 3.0];
        assert_eq!(decode_embedding(&encode_embedding(&v)), Some(v));
        assert_eq!(decode_embedding(&[1, 2, 3]), None);
    }

    #[cfg(feature = "fastembed")]
    #[tokio::test]
    #[ignore = "downloads the AllMiniLM-L6-v2 model"]
    async fn fastembed_embeds_with_384_dims() {
        let e = FastEmbedder::new();
        let out = e.embed(&["Mia and Zoe".to_string()]).await.unwrap();
        assert_eq!(out[0].len(), 384);
    }
}
