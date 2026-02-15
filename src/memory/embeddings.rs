use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use std::sync::Arc;
use tokio::sync::OnceCell;
use tokio::task;
use tracing::info;

#[derive(Clone)]
pub struct EmbeddingService {
    model: Arc<OnceCell<Arc<TextEmbedding>>>,
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
    async fn get_model(&self) -> anyhow::Result<Arc<TextEmbedding>> {
        let model = self
            .model
            .get_or_try_init(|| async {
                task::spawn_blocking(|| {
                    let mut options = InitOptions::default();
                    options.model_name = EmbeddingModel::AllMiniLML6V2;
                    options.show_download_progress = true;
                    let model = TextEmbedding::try_new(options)?;
                    info!("Embedding model loaded (AllMiniLML6V2)");
                    Ok::<_, anyhow::Error>(Arc::new(model))
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
            let embeddings = model.embed(vec![text], None)?;
            Ok(embeddings[0].clone())
        })
        .await?
    }

    /// Compute embeddings for multiple strings.
    #[allow(dead_code)]
    pub async fn embed_batch(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        let model = self.get_model().await?;
        task::spawn_blocking(move || model.embed(texts, None)).await?
    }
}
