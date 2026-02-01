use fastembed::{InitOptions, TextEmbedding, EmbeddingModel};
use std::sync::Arc;
use tokio::task;

#[derive(Clone)]
pub struct EmbeddingService {
    model: Arc<TextEmbedding>,
}

impl EmbeddingService {
    pub fn new() -> anyhow::Result<Self> {
        let mut options = InitOptions::default();
        options.model_name = EmbeddingModel::AllMiniLML6V2;
        options.show_download_progress = true;

        let model = TextEmbedding::try_new(options)?;
        Ok(Self {
            model: Arc::new(model),
        })
    }

    /// Compute embedding for a single string.
    /// Runs on a blocking thread to avoid blocking the async runtime.
    pub async fn embed(&self, text: String) -> anyhow::Result<Vec<f32>> {
        let model = self.model.clone();
        task::spawn_blocking(move || {
            let embeddings = model.embed(vec![text], None)?;
            Ok(embeddings[0].clone())
        })
        .await?
    }

    /// Compute embeddings for multiple strings.
    pub async fn embed_batch(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        let model = self.model.clone();
        task::spawn_blocking(move || {
            model.embed(texts, None).map_err(anyhow::Error::from)
        })
        .await?
    }
}
