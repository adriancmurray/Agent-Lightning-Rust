use abzu_lightning_core::{LightningAlgorithm, algorithm::RewardAggregator, Result, Error};
use serde::Deserialize;

/// Configuration for the Algorithm Selection
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AlgorithmConfig {
    /// Proximal Policy Optimization
    Ppo {
        input_dim: usize,
        action_dim: usize,
        learning_rate: f64,
        clip_range: f64,
        device: String,
    },
    /// Group Relative Policy Optimization (Future)
    Grpo {
        group_size: usize,
        beta: f64,
    },
    /// Agent Policy Optimization (Future)
    Apo {
        param: f64,
    },
    /// Simple Reward Aggregation (Baseline)
    Aggregator {
        window_size: usize,
    },
    /// Custom/Experimental
    Custom {
        name: String,
        params: serde_json::Value,
    }
}

/// The Brain Factory: Hydrates a static config into a dynamic brain
pub struct BrainFactory;

impl BrainFactory {
    pub fn build(config: &AlgorithmConfig) -> Result<Box<dyn LightningAlgorithm>> {
        match config {
            AlgorithmConfig::Ppo { input_dim, action_dim, learning_rate, clip_range, device } => {
                #[cfg(feature = "ppo")]
                {
                    use abzu_lightning_ppo::PpoAlgorithm;
                    let brain = PpoAlgorithm::new(*input_dim, *action_dim, *learning_rate, *clip_range, device)
                        .map_err(|e| Error::Training(format!("PPO init failed: {}", e)))?;
                    Ok(Box::new(brain))
                }
                #[cfg(not(feature = "ppo"))]
                {
                    // Suppress unused variables warning if PPO disabled
                    let _ = (learning_rate, clip_range, device, input_dim, action_dim);
                    Err(Error::Training("PPO feature not enabled. Add 'ppo' feature to abzu-lightning.".to_string()))
                }
            }
            AlgorithmConfig::Grpo { .. } => {
                Err(Error::Training("GRPO algorithm not yet implemented".to_string()))
            }
            AlgorithmConfig::Apo { .. } => {
                Err(Error::Training("APO algorithm not yet implemented".to_string()))
            }
            AlgorithmConfig::Aggregator { window_size } => {
                let algo = RewardAggregator::new(Some(*window_size));
                Ok(Box::new(algo))
            }
            AlgorithmConfig::Custom { name, .. } => {
                Err(Error::Training(format!("Unknown custom algorithm: {}", name)))
            }
        }
    }
}
