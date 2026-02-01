mod error;
mod openai_compatible;

pub use error::{ProviderError, ProviderErrorKind};
pub use openai_compatible::OpenAiCompatibleProvider;
