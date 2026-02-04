//! Training loop orchestration

use crate::{LightningAlgorithm, LightningStore, Result, TrainingResult};
use std::sync::Arc;
use tokio::time::{interval, Duration};
use tracing::{debug, info, warn};

/// Configuration for the trainer
#[derive(Debug, Clone)]
pub struct TrainerConfig {
    /// Task ID to train on (if None, trains on all tasks)
    pub task_id: Option<String>,
    
    /// Agent ID to train on (if None, trains on all agents)
    pub agent_id: Option<String>,
    
    /// Batch size (number of spans per training iteration)
    pub batch_size: usize,
    
    /// Training interval (seconds between training iterations)
    pub interval_secs: u64,
    
    /// Maximum number of iterations (None = run indefinitely)
    pub max_iterations: Option<usize>,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            task_id: None,
            agent_id: None,
            batch_size: 100,
            interval_secs: 10,
            max_iterations: None,
        }
    }
}

/// Training loop orchestrator
pub struct Trainer {
    store: Arc<LightningStore>,
    config: TrainerConfig,
    last_processed_index: usize,
}

impl Trainer {
    /// Create a new trainer
    pub fn new(store: Arc<LightningStore>, config: TrainerConfig) -> Self {
        Self {
            store,
            config,
            last_processed_index: 0,
        }
    }

    /// Run a single training iteration
    pub async fn train_iteration<A: LightningAlgorithm>(
        &mut self,
        algorithm: &mut A,
    ) -> Result<Option<TrainingResult>> {
        // Query spans based on config
        let spans = if let Some(task_id) = &self.config.task_id {
            debug!("Querying spans for task: {}", task_id);
            self.store.query_task(task_id)?
        } else if let Some(agent_id) = &self.config.agent_id {
            debug!("Querying spans for agent: {}", agent_id);
            self.store.query_agent(agent_id)?
        } else {
            // Get all spans (limited by batch size)
            debug!("Querying all spans");
            vec![] // TODO: implement full span query
        };

        // Check if we have new spans to process
        if spans.len() <= self.last_processed_index {
            debug!("No new spans to process");
            return Ok(None);
        }

        // Get new spans since last iteration
        let new_spans: Vec<_> = spans
            .into_iter()
            .skip(self.last_processed_index)
            .take(self.config.batch_size)
            .collect();

        if new_spans.is_empty() {
            return Ok(None);
        }

        info!("Training on {} new spans", new_spans.len());
        let result = algorithm.train(&new_spans)?;

        // Update last processed index
        self.last_processed_index += new_spans.len();

        // Store updated weights if provided
        if let Some(weights) = &result.updated_weights {
            let key = format!("weights_iter_{}", self.last_processed_index);
            self.store.store_resource(&key, weights)?;
            debug!("Stored updated weights: {}", key);
        }

        Ok(Some(result))
    }

    /// Run the training loop
    pub async fn run<A: LightningAlgorithm>(
        &mut self,
        algorithm: &mut A,
    ) -> Result<Vec<TrainingResult>> {
        let mut results = Vec::new();
        let mut iteration_count = 0;
        let mut ticker = interval(Duration::from_secs(self.config.interval_secs));

        info!("Starting training loop");

        loop {
            ticker.tick().await;

            match self.train_iteration(algorithm).await {
                Ok(Some(result)) => {
                    info!("Training iteration {} complete: {:?}", iteration_count, result.metrics);
                    results.push(result);
                    iteration_count += 1;

                    // Check if we've reached max iterations
                    if let Some(max) = self.config.max_iterations {
                        if iteration_count >= max {
                            info!("Reached max iterations ({}), stopping", max);
                            break;
                        }
                    }
                }
                Ok(None) => {
                    debug!("No new spans, waiting...");
                }
                Err(e) => {
                    warn!("Training iteration error: {}", e);
                    // Continue despite errors
                }
            }
        }

        info!("Training loop complete, {} iterations", iteration_count);
        Ok(results)
    }

    /// Reset the trainer state
    pub fn reset(&mut self) {
        self.last_processed_index = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{algorithm::RewardAggregator, RewardSpan, Span};

    #[tokio::test]
    async fn test_single_iteration() {
        let store = Arc::new(LightningStore::memory().unwrap());
        
        // Insert test spans
        let task_id = "test-task";
        for i in 0..5 {
            let span = Span::Reward(
                RewardSpan::new(i as f64).with_task(task_id)
            );
            store.insert_span(&span).unwrap();
        }

        let config = TrainerConfig {
            task_id: Some(task_id.to_string()),
            batch_size: 3,
            ..Default::default()
        };

        let mut trainer = Trainer::new(store.clone(), config);
        let mut algo = RewardAggregator::new();

        // First iteration should process 3 spans
        let result = trainer.train_iteration(&mut algo).await.unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().spans_processed, 3);

        // Second iteration should process remaining 2 spans
        let result = trainer.train_iteration(&mut algo).await.unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().spans_processed, 2);

        // Third iteration should have no new spans
        let result = trainer.train_iteration(&mut algo).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_run_with_max_iterations() {
        let store = Arc::new(LightningStore::memory().unwrap());
        
        // Insert many spans
        let task_id = "test-task";
        for i in 0..20 {
            let span = Span::Reward(
                RewardSpan::new(i as f64).with_task(task_id)
            );
            store.insert_span(&span).unwrap();
        }

        let config = TrainerConfig {
            task_id: Some(task_id.to_string()),
            batch_size: 5,
            interval_secs: 1, // Minimum 1 second interval
            max_iterations: Some(3),
            ..Default::default()
        };

        let mut trainer = Trainer::new(store.clone(), config);
        let mut algo = RewardAggregator::new();

        let results = trainer.run(&mut algo).await.unwrap();
        assert_eq!(results.len(), 3);
    }
}
