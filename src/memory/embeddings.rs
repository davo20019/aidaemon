use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use std::sync::{Arc, Mutex};
use tokio::sync::OnceCell;
use tokio::task;
use tracing::info;

/// fastembed 5.x's `TextEmbedding::embed` requires `&mut self`. We wrap the
/// model in a `std::sync::Mutex` so blocking-thread callers can take it,
/// embed, and release. Calls serialize, but embedding itself is CPU-bound
/// and short, so this is acceptable for our usage pattern.
type SharedModel = Arc<Mutex<TextEmbedding>>;

#[derive(Clone)]
pub struct EmbeddingService {
    model: Arc<OnceCell<SharedModel>>,
}

impl EmbeddingService {
    /// Creates the service without loading the model.
    /// The model is loaded lazily on the first embedding request.
    pub fn new() -> anyhow::Result<Self> {
        Ok(Self {
            model: Arc::new(OnceCell::new()),
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
}
