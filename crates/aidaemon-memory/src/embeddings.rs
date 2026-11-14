use std::sync::Arc;

use crate::Embedder;
use crate::embed::FastEmbedder;

/// The model the daemon runs unless `[memory] embedding_model` names
/// another: AllMiniLM-L6-v2, quantized (384 dimensions, about a quarter of
/// the full model's size, nearly the same vectors).
pub const DAEMON_EMBEDDING_MODEL: &str = "AllMiniLML6V2Q";

/// Id of the deterministic test vectors. Not a real model's id, so they are
/// never mistaken for one's.
const DETERMINISTIC_MODEL_ID: &str = "deterministic/hashed-384";

/// Length of the deterministic test vectors.
const DETERMINISTIC_DIMENSIONS: usize = 384;

/// Which models a service runs. Chosen by the caller's constructor, never by
/// a cargo feature: a test build and the shipped daemon differ only in what
/// their code asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedderKind {
    /// A local fastembed model.
    FastEmbed,
    /// Hashed lexical vectors, offline. For tests: nothing is downloaded or
    /// loaded.
    Deterministic,
}

#[derive(Clone)]
enum Backend {
    FastEmbed(Arc<FastEmbedder>),
    Deterministic,
}

#[derive(Clone)]
pub struct EmbeddingService {
    backend: Backend,
}

impl EmbeddingService {
    /// The daemon's default model ([`DAEMON_EMBEDDING_MODEL`]), not loaded
    /// until first used.
    pub fn fastembed() -> anyhow::Result<Self> {
        Self::fastembed_named(DAEMON_EMBEDDING_MODEL)
    }

    /// A fastembed model by name (see [`FastEmbedder::named`]), not loaded
    /// until first used.
    pub fn fastembed_named(name: &str) -> anyhow::Result<Self> {
        let embedder = FastEmbedder::named(name)?.show_download_progress(true);
        Ok(Self {
            backend: Backend::FastEmbed(Arc::new(embedder)),
        })
    }

    /// Deterministic offline vectors: nothing is downloaded
    /// or loaded, and the same text always gets the same vector.
    pub fn deterministic() -> Self {
        Self {
            backend: Backend::Deterministic,
        }
    }

    /// Which models this service runs.
    pub fn kind(&self) -> EmbedderKind {
        match self.backend {
            Backend::FastEmbed(_) => EmbedderKind::FastEmbed,
            Backend::Deterministic => EmbedderKind::Deterministic,
        }
    }

    /// Compute embedding for a single string.
    /// Runs on a blocking thread to avoid blocking the async runtime.
    pub async fn embed(&self, text: String) -> anyhow::Result<Vec<f32>> {
        self.embed_batch(vec![text])
            .await?
            .pop()
            .ok_or_else(|| anyhow::anyhow!("the embedding model returned no vector"))
    }

    /// Compute embeddings for multiple strings.
    pub async fn embed_batch(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        match &self.backend {
            Backend::Deterministic => Ok(texts
                .iter()
                .map(|text| deterministic_embedding(text))
                .collect()),
            Backend::FastEmbed(embedder) => Ok(embedder.embed(&texts).await?),
        }
    }
}

#[async_trait::async_trait]
impl crate::Embedder for EmbeddingService {
    fn model_id(&self) -> &str {
        match &self.backend {
            Backend::FastEmbed(embedder) => embedder.model_id(),
            Backend::Deterministic => DETERMINISTIC_MODEL_ID,
        }
    }

    fn dimensions(&self) -> usize {
        match &self.backend {
            Backend::FastEmbed(embedder) => embedder.dimensions(),
            Backend::Deterministic => DETERMINISTIC_DIMENSIONS,
        }
    }

    async fn embed(&self, texts: &[String]) -> crate::Result<Vec<Vec<f32>>> {
        self.embed_batch(texts.to_vec())
            .await
            .map_err(|error| crate::MemoryError::Embedding(format!("{error:#}")))
    }
}

fn deterministic_embedding(text: &str) -> Vec<f32> {
    use std::hash::{Hash, Hasher};

    const DIMENSIONS: usize = DETERMINISTIC_DIMENSIONS;
    const SEMANTIC_DIMENSIONS: usize = 2;
    const LEXICAL_DIMENSIONS: usize = DIMENSIONS - SEMANTIC_DIMENSIONS;
    let mut vector = vec![0.0_f32; DIMENSIONS];
    for token in text
        .split(|character: char| !character.is_alphanumeric())
        .filter(|token| !token.is_empty())
    {
        let normalized = token.to_ascii_lowercase();
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        normalized.hash(&mut hasher);
        let hash = hasher.finish();
        let index = hash as usize % LEXICAL_DIMENSIONS;
        let sign = if hash & (1 << 63) == 0 { 1.0 } else { -1.0 };
        vector[index] += sign;

        // Preserve the small set of semantic relationships exercised by the
        // retrieval contract tests without loading or downloading a model.
        // Each occurrence contributes so repeated natural-language mentions
        // retain roughly the same cosine behavior as the production encoder.
        let concept_index = match normalized.as_str() {
            "spouse" | "partner" | "wife" | "husband" => Some(LEXICAL_DIMENSIONS),
            "kubernetes" | "k8s" | "container" | "containers" | "orchestration" | "deployment"
            | "infrastructure" | "devops" => Some(LEXICAL_DIMENSIONS + 1),
            _ => None,
        };
        if let Some(index) = concept_index {
            vector[index] += 1.0;
        }
    }
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut vector {
            *value /= norm;
        }
    }
    vector
}
