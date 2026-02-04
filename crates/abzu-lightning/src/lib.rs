//! Agent Lightning - Reinforcement Learning Platform
//!
//! This crate serves as the facade for the Modular Brain Architecture.
//! It re-exports core components and provides a Runtime Factory for 
//! hot-swappable algorithms.

// Re-export core components
pub use abzu_lightning_core::*;

pub mod factory;
pub mod harness;

#[cfg(feature = "plugin")]
pub mod plugin;
#[cfg(feature = "plugin")]
pub use plugin::LightningPlugin;

// Re-export new runtime components
pub use factory::{BrainFactory, AlgorithmConfig};
pub use harness::TrainingHarness;
