use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use fastembed::{RerankInitOptions, RerankerModel, TextRerank};
use std::sync::{Arc, Mutex};
use tokio::sync::OnceCell;
use tokio::task;
use tracing::info;

use crate::background_model::BackgroundModel;

/// Stable identifier persisted beside every derived embedding. Changing the
/// model creates a new index generation instead of silently mixing vectors.
pub const EMBEDDING_MODEL_ID: &str = "fastembed/all-MiniLM-L6-v2";

/// fastembed 5.x's `TextEmbedding::embed` requires `&mut self`. We wrap the
/// model in a `std::sync::Mutex` so blocking-thread callers can take it,
/// embed, and release. Calls serialize, but embedding itself is CPU-bound
/// and short, so this is acceptable for our usage pattern.
type SharedModel = Arc<Mutex<TextEmbedding>>;

/// Cross-encoder reranker for the explicit memory-search path. A multilingual
/// model so it handles the user's mixed EN/ES facts. Loaded lazily (and only if
/// reranking is actually used) since it is a sizeable extra download.
const RERANKER_MODEL: RerankerModel = RerankerModel::JINARerankerV2BaseMultilingual;

/// Which models a service runs. Chosen by the caller's constructor, never by
/// a cargo feature: a test build and the shipped daemon differ only in what
/// their code asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedderKind {
    /// AllMiniLM-L6-v2 embeddings and the cross-encoder reranker.
    FastEmbed,
    /// Hashed lexical vectors, offline, with no reranker (searches keep
    /// bi-encoder order). For tests: nothing is downloaded or loaded.
    Deterministic,
}

#[derive(Clone)]
enum Backend {
    FastEmbed {
        model: Arc<OnceCell<SharedModel>>,
        /// Loaded in the background on first use and kept even when the
        /// search that started the load stops waiting; one rerank runs at a
        /// time, and a rerank whose search gave up never holds later
        /// searches in a queue.
        reranker: BackgroundModel<TextRerank>,
    },
    Deterministic,
}

#[derive(Clone)]
pub struct EmbeddingService {
    backend: Backend,
}

impl EmbeddingService {
    /// The daemon's real models: AllMiniLM-L6-v2 embeddings and the
    /// multilingual reranker. Neither is loaded until first used.
    pub fn fastembed() -> anyhow::Result<Self> {
        Ok(Self {
            backend: Backend::FastEmbed {
                model: Arc::new(OnceCell::new()),
                reranker: BackgroundModel::new("reranker", || {
                    let options =
                        RerankInitOptions::new(RERANKER_MODEL).with_show_download_progress(true);
                    let model = TextRerank::try_new(options)?;
                    info!("Reranker model loaded ({:?})", RERANKER_MODEL);
                    Ok(model)
                }),
            },
        })
    }

    /// Deterministic offline vectors and no reranker: nothing is downloaded
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
            Backend::FastEmbed { model, .. } => Self::get_model(model).await?,
        };
        task::spawn_blocking(move || {
            let mut guard = model
                .lock()
                .map_err(|e| anyhow::anyhow!("embedding model mutex poisoned: {e}"))?;
            Ok(guard.embed(texts, None)?)
        })
        .await?
    }

    /// Cross-encoder rerank: score each document against the query and return
    /// `(original_index, score)` pairs sorted by descending relevance. The
    /// indices map back into the input `documents` vec. Runs on a blocking
    /// thread. Errors (including "busy with a rerank whose search gave up",
    /// and a deterministic service, which has no reranker) propagate so
    /// callers can fall back to bi-encoder order.
    pub async fn rerank(
        &self,
        query: String,
        documents: Vec<String>,
    ) -> anyhow::Result<Vec<(usize, f32)>> {
        if documents.is_empty() {
            return Ok(vec![]);
        }
        let reranker = match &self.backend {
            Backend::FastEmbed { reranker, .. } => reranker,
            Backend::Deterministic => anyhow::bail!("a deterministic embedder has no reranker"),
        };
        reranker
            .run(move |model| {
                let results = model.rerank(query, documents, false, None)?;
                let mut ranked: Vec<(usize, f32)> =
                    results.into_iter().map(|r| (r.index, r.score)).collect();
                ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                Ok(ranked)
            })
            .await
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

#[async_trait::async_trait]
impl crate::sqlite::Reranker for EmbeddingService {
    async fn rerank(
        &self,
        query: String,
        documents: Vec<String>,
    ) -> anyhow::Result<Vec<(usize, f32)>> {
        EmbeddingService::rerank(self, query, documents).await
    }

    /// Calibrated for `RERANKER_MODEL` (raw logits) by the ignored
    /// `reranker_floor_calibration` test in `aidaemon-memory` on keyword-list
    /// queries like the ones the model sends: a single bar of -2.25 dropped
    /// 7 relevant facts; these bars drop 3 with about the same noise and
    /// return nothing for questions no fact answers. The related gap comes
    /// from live searches, where relevant related facts (a parent link, a
    /// sibling) sat within 2 of the top hit and irrelevant ones 2.9 or more
    /// below it.
    fn relevance_floor(&self) -> Option<crate::search::RelevanceFloor> {
        Some(crate::search::RelevanceFloor {
            lexical: -3.0,
            semantic: -2.25,
            related_gap: Some(2.5),
        })
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
