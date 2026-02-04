//! Agent Lightning - Rust port of Microsoft's Agent Lightning RL framework
//!
//! Provides environment-independent reinforcement learning training through
//! structured span recording, storage, and algorithm interfaces.

pub mod algorithm;
pub mod collector;
#[cfg(feature = "plugin")]
pub mod plugin;
pub mod span;
pub mod store;
pub mod trainer;

// Re-export main types
pub use algorithm::{LightningAlgorithm, TrainingResult};
pub use collector::SpanCollector;
#[cfg(feature = "plugin")]
pub use plugin::{LightningPlugin, LightningRequest, LightningResponse, TrainingStats};
pub use span::{ActionSpan, ObservationSpan, RewardSpan, Span};
pub use store::LightningStore;
pub use trainer::{Trainer, TrainerConfig};

/// Result type alias
pub type Result<T> = std::result::Result<T, Error>;

/// Error types for Lightning operations
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Storage error: {0}")]
    Storage(#[from] sled::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Bincode error: {0}")]
    Bincode(#[from] bincode::Error),

    #[error("Training error: {0}")]
    Training(String),

    #[error("Invalid span: {0}")]
    InvalidSpan(String),
}
