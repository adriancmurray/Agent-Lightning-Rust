# abzu-lightning

Rust port of Microsoft's [Agent Lightning](https://github.com/microsoft/agent-lightning) reinforcement learning framework for the Abzu ecosystem.

## Overview

Agent Lightning provides environment-independent RL training through structured span recording. This Rust implementation eliminates Python fragility, provides native Abzu integration, and enables single-binary deployment.

## Features

- 🎯 **Span Collection**: Structured recording of agent interactions (Observations, Actions, Rewards)
- 💾 **Lightning Store**: Embedded Sled-based persistent storage with automatic indexing
- 🧠 **Algorithm Interface**: Trait-based RL algorithm abstraction
- 🔄 **Async Trainer**: Configurable training loop with batching and metrics
- 🔌 **Plugin Integration**: Optional Abzu plugin for dashboard integration

## Quick Start

```rust
use abzu_lightning::{
    LightningStore, ObservationSpan, RewardSpan, Span,
    algorithm::RewardAggregator, Trainer, TrainerConfig,
};
use serde_json::json;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize store
    let store = Arc::new(LightningStore::open("~/.abzu/lightning")?);
    
    // Emit spans
    let obs = Span::Observation(
        ObservationSpan::new(json!({"step": 1}))
            .with_task("demo")
            .with_agent("agent-1")
    );
    store.insert_span(&obs)?;
    
    // Train
    let config = TrainerConfig {
        task_id: Some("demo".into()),
        batch_size: 50,
        ..Default::default()
    };
    
    let mut trainer = Trainer::new(store.clone(), config);
    let mut algo = RewardAggregator::new();
    let results = trainer.run(&mut algo).await?;
    
    Ok(())
}
```

## Architecture

```
abzu-lightning/
├── span.rs        # Span types (Observation, Action, Reward)
├── collector.rs   # SpanCollector trait + implementations
├── store.rs       # Sled-based persistent storage
├── algorithm.rs   # LightningAlgorithm trait
├── trainer.rs     # Training loop orchestration
└── plugin.rs      # Abzu plugin (optional, feature-gated)
```

## Features

- **Default**: Core Lightning functionality (no dependencies on Abzu runtime)
- **`plugin`**: Enables Abzu plugin integration (requires `abzu-runtime`)

```toml
[dependencies]
abzu-lightning = { path = "crates/abzu-lightning" }

# Or with plugin support:
abzu-lightning = { path = "crates/abzu-lightning", features = ["plugin"] }
```

## Testing

```bash
# Run all tests
cargo test -p abzu-lightning

# Run specific module tests
cargo test -p abzu-lightning span::tests
```

All 14 unit tests pass with comprehensive coverage of span types, storage, algorithms, and training.

## References

- [Microsoft Agent Lightning](https://github.com/microsoft/agent-lightning)
- [Research Paper](https://arxiv.org/abs/2508.03680)
