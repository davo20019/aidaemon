use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use std::sync::{Arc, Mutex};
use tokio::sync::OnceCell;
use tokio::task;
use tracing::info;

/// Stable identifier persisted beside every derived embedding. Changing the
/// model creates a new index generation instead of silently mixing vectors.
pub const EMBEDDING_MODEL_ID: &str = "fastembed/all-MiniLM-L6-v2";

/// fastembed 5.x's `TextEmbedding::embed` requires `&mut self`. We wrap the
/// model in a `std::sync::Mutex` so blocking-thread callers can take it,
/// embed, and release. Calls serialize, but embedding itself is CPU-bound
/// and short, so this is acceptable for our usage pattern.
type SharedModel = Arc<Mutex<TextEmbedding>>;

/// Which models a service runs. Chosen by the caller's constructor, never by
/// a cargo feature: a test build and the shipped daemon differ only in what
/// their code asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedderKind {
    /// AllMiniLM-L6-v2 embeddings.
    FastEmbed,
    /// Hashed lexical vectors, offline. For tests: nothing is downloaded or
    /// loaded.
    Deterministic,
}

#[derive(Clone)]
enum Backend {
    FastEmbed { model: Arc<OnceCell<SharedModel>> },
    Deterministic,
}

#[derive(Clone)]
pub struct EmbeddingService {
    backend: Backend,
}

impl EmbeddingService {
    /// The daemon's real model: AllMiniLM-L6-v2 embeddings, not loaded until
    /// first used.
    pub fn fastembed() -> anyhow::Result<Self> {
        Ok(Self {
            backend: Backend::FastEmbed {
                model: Arc::new(OnceCell::new()),
            },
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
            Backend::FastEmbed { .. } => EmbedderKind::FastEmbed,
            Backend::Deterministic => EmbedderKind::Deterministic,
        }
    }

    /// Returns the model, initializing it on first call.
    async fn get_model(model: &OnceCell<SharedModel>) -> anyhow::Result<SharedModel> {
        let model = model
            .get_or_try_init(|| async {
                task::spawn_blocking(|| {
                    let mut options = TextInitOptions::default();
                    options.model_name = EmbeddingModel::AllMiniLML6V2;
                    options.show_download_progress = true;
                    let model = TextEmbedding::try_new(options)?;
                    info!("Embedding model loaded (AllMiniLML6V2)");
                    Ok::<_, anyhow::Error>(Arc::new(Mutex::new(model)))
                })
                .await?
            })
            .await?;
        Ok(model.clone())
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
        let model = match &self.backend {
            Backend::Deterministic => {
                return Ok(texts
                    .iter()
                    .map(|text| deterministic_embedding(text))
                    .collect());
            }
            Backend::FastEmbed { model } => Self::get_model(model).await?,
        };
        task::spawn_blocking(move || {
            let mut guard = model
                .lock()
                .map_err(|e| anyhow::anyhow!("embedding model mutex poisoned: {e}"))?;
            Ok(guard.embed(texts, None)?)
        })
        .await?
    }
}

#[async_trait::async_trait]
impl crate::Embedder for EmbeddingService {
    fn model_id(&self) -> &str {
        EMBEDDING_MODEL_ID
    }

    fn dimensions(&self) -> usize {
        384
    }

    async fn embed(&self, texts: &[String]) -> crate::Result<Vec<Vec<f32>>> {
        self.embed_batch(texts.to_vec())
            .await
            .map_err(|error| crate::MemoryError::Embedding(format!("{error:#}")))
    }
}

fn deterministic_embedding(text: &str) -> Vec<f32> {
    use std::hash::{Hash, Hasher};

    const DIMENSIONS: usize = 384;
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
