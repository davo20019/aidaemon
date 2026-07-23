use fastembed::{
    EmbeddingModel, InitOptions, RerankInitOptions, RerankerModel, TextEmbedding, TextRerank,
};
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
type SharedReranker = Arc<Mutex<TextRerank>>;

/// Cross-encoder reranker for the explicit memory-search path. A multilingual
/// model so it handles the user's mixed EN/ES facts. Loaded lazily (and only if
/// reranking is actually used) since it is a sizeable extra download.
const RERANKER_MODEL: RerankerModel = RerankerModel::JINARerankerV2BaseMultiligual;

#[derive(Clone)]
pub struct EmbeddingService {
    model: Arc<OnceCell<SharedModel>>,
    reranker: Arc<OnceCell<SharedReranker>>,
}

impl EmbeddingService {
    /// Creates the service without loading the model.
    /// The model is loaded lazily on the first embedding request.
    pub fn new() -> anyhow::Result<Self> {
        Ok(Self {
            model: Arc::new(OnceCell::new()),
            reranker: Arc::new(OnceCell::new()),
        })
    }

    /// Returns the model, initializing it on first call.
    async fn get_model(&self) -> anyhow::Result<SharedModel> {
        let model = self
            .model
            .get_or_try_init(|| async {
                task::spawn_blocking(|| {
                    let mut options = InitOptions::default();
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
        let model = self.get_model().await?;
        task::spawn_blocking(move || {
            let guard = model
                .lock()
                .map_err(|e| anyhow::anyhow!("embedding model mutex poisoned: {e}"))?;
            let embeddings = guard.embed(vec![text], None)?;
            Ok(embeddings[0].clone())
        })
        .await?
    }

    /// Compute embeddings for multiple strings.
    #[allow(dead_code)]
    pub async fn embed_batch(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        let model = self.get_model().await?;
        task::spawn_blocking(move || {
            let guard = model
                .lock()
                .map_err(|e| anyhow::anyhow!("embedding model mutex poisoned: {e}"))?;
            guard.embed(texts, None)
        })
        .await?
    }

    /// Returns the cross-encoder reranker, initializing it on first call.
    async fn get_reranker(&self) -> anyhow::Result<SharedReranker> {
        let reranker = self
            .reranker
            .get_or_try_init(|| async {
                task::spawn_blocking(|| {
                    let options =
                        RerankInitOptions::new(RERANKER_MODEL).with_show_download_progress(true);
                    let model = TextRerank::try_new(options)?;
                    info!("Reranker model loaded ({:?})", RERANKER_MODEL);
                    Ok::<_, anyhow::Error>(Arc::new(Mutex::new(model)))
                })
                .await?
            })
            .await?;
        Ok(reranker.clone())
    }

    /// Cross-encoder rerank: score each document against the query and return
    /// `(original_index, score)` pairs sorted by descending relevance. The
    /// indices map back into the input `documents` vec. Runs on a blocking
    /// thread. Errors propagate so callers can fall back to bi-encoder order.
    pub async fn rerank(
        &self,
        query: String,
        documents: Vec<String>,
    ) -> anyhow::Result<Vec<(usize, f32)>> {
        if documents.is_empty() {
            return Ok(vec![]);
        }
        // Never download/load the heavyweight reranker during unit tests: callers
        // treat this error as "rerank unavailable" and fall back to bi-encoder
        // order, so test behavior stays deterministic and offline. The reranker
        // is exercised by live/ignored tests and in production builds.
        if cfg!(test) {
            anyhow::bail!("reranker disabled in test builds");
        }
        let reranker = self.get_reranker().await?;
        task::spawn_blocking(move || {
            let guard = reranker
                .lock()
                .map_err(|e| anyhow::anyhow!("reranker model mutex poisoned: {e}"))?;
            let results = guard.rerank(query, documents, false, None)?;
            let mut ranked: Vec<(usize, f32)> =
                results.into_iter().map(|r| (r.index, r.score)).collect();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            Ok(ranked)
        })
        .await?
    }
}
