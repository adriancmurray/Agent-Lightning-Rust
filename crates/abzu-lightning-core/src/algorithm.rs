//! Algorithm trait and training result types

use crate::{Result, Span};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use async_trait::async_trait;

// Verification: Async trait enabled
/// LLM Backend interface for Prompt Optimization
#[async_trait]
pub trait LlmBackend: Send + Sync {
    async fn generate(&self, prompt: &str) -> anyhow::Result<String>;
}


/// Result of a training iteration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResult {
    /// Metrics produced by this training iteration
    pub metrics: HashMap<String, f64>,
    
    /// Optional updated policy/weights (as binary blob)
    pub updated_weights: Option<Vec<u8>>,
    
    /// Number of spans processed
    pub spans_processed: usize,
}

impl TrainingResult {
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
            updated_weights: None,
            spans_processed: 0,
        }
    }

    /// Add a metric
    pub fn with_metric(mut self, key: impl Into<String>, value: f64) -> Self {
        self.metrics.insert(key.into(), value);
        self
    }

    /// Set updated weights
    pub fn with_weights(mut self, weights: Vec<u8>) -> Self {
        self.updated_weights = Some(weights);
        self
    }

    /// Set spans processed
    pub fn with_spans_processed(mut self, count: usize) -> Self {
        self.spans_processed = count;
        self
    }
}

impl Default for TrainingResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for reinforcement learning algorithms
#[async_trait]
pub trait LightningAlgorithm: Send + Sync {
    /// Train on a batch of spans
    /// Returns None if no training happened (e.g. APO filtering)
    async fn train(&mut self, spans: &[Span]) -> Result<Option<TrainingResult>>;

    /// Update the policy with new weights
    fn update_policy(&mut self, weights: &[u8]) -> Result<()>;

    /// Get current policy/weights (if applicable)
    fn get_weights(&self) -> Option<Vec<u8>> {
        None
    }

    /// Reset the algorithm state
    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Basic reward aggregation algorithm (for initial implementation)
#[derive(Debug, Clone)]
pub struct RewardAggregator {
    window_size: Option<usize>,
    rewards: std::collections::VecDeque<f64>,
    sum: f64,
}

impl RewardAggregator {
    /// Create a new aggregator with optional sliding window size
    pub fn new(window_size: Option<usize>) -> Self {
        Self {
            window_size,
            rewards: std::collections::VecDeque::new(),
            sum: 0.0,
        }
    }

    pub fn mean_reward(&self) -> f64 {
        if self.rewards.is_empty() {
            0.0
        } else {
            self.sum / self.rewards.len() as f64
        }
    }

    pub fn total_reward(&self) -> f64 {
        self.sum
    }

    pub fn count(&self) -> usize {
        self.rewards.len()
    }
}

impl Default for RewardAggregator {
    fn default() -> Self {
        Self::new(None)
    }
}

#[async_trait]
impl LightningAlgorithm for RewardAggregator {
    async fn train(&mut self, spans: &[Span]) -> Result<Option<TrainingResult>> {
        let mut local_total = 0.0;
        let mut local_count = 0;

        for span in spans {
            if let Span::Reward(reward_span) = span {
                let reward = reward_span.reward;
                local_total += reward;
                local_count += 1;
                
                self.rewards.push_back(reward);
                self.sum += reward;

                if let Some(limit) = self.window_size {
                    while self.rewards.len() > limit {
                        if let Some(popped) = self.rewards.pop_front() {
                            self.sum -= popped;
                        }
                    }
                }
            }
        }

        let result = TrainingResult::new()
            .with_metric("mean_reward", self.mean_reward())
            .with_metric("total_reward", self.sum)
            .with_metric("batch_mean_reward", if local_count > 0 { local_total / local_count as f64 } else { 0.0 })
            .with_metric("reward_window_count", self.rewards.len() as f64)
            .with_spans_processed(spans.len());

        Ok(Some(result))
    }

    fn update_policy(&mut self, _weights: &[u8]) -> Result<()> {
        // No-op for reward aggregator
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.rewards.clear();
        self.sum = 0.0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RewardSpan, ObservationSpan};
    use serde_json::json;

    #[tokio::test]
    async fn test_reward_aggregator() {
        let mut algo = RewardAggregator::default();

        let spans = vec![
            Span::Observation(ObservationSpan::new(json!({}))),
            Span::Reward(RewardSpan::new(1.0)),
            Span::Reward(RewardSpan::new(0.5)),
            Span::Reward(RewardSpan::new(-0.5)),
        ];

        // Must await
        let result_opt = algo.train(&spans).await.unwrap();
        let result = result_opt.unwrap();

        assert_eq!(result.spans_processed, 4);
        assert_eq!(algo.count(), 3);
        assert_eq!(algo.total_reward(), 1.0);
        assert!((algo.mean_reward() - 0.333333).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_algorithm_reset() {
        let mut algo = RewardAggregator::default();
        let spans = vec![Span::Reward(RewardSpan::new(1.0))];

        algo.train(&spans).await.unwrap();
        assert_eq!(algo.count(), 1);

        algo.reset().unwrap();
        assert_eq!(algo.count(), 0);
        assert_eq!(algo.total_reward(), 0.0);
    }
}
