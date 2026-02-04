//! Core components for Agent Lightning
//!
//! Contains foundational types: Span, Store, Trainer, and Algorithm traits.

pub mod algorithm;
pub mod collector;
pub mod span;
pub mod store;
pub mod trainer;

// Re-export core types
pub use algorithm::{LightningAlgorithm, TrainingResult, LlmBackend};
pub use collector::SpanCollector;
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

    #[error("State error: {0}")]
    State(String),
}
