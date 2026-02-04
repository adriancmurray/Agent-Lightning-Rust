//! Training loop orchestration

use crate::{LightningAlgorithm, LightningStore, Result, TrainingResult};
use chrono::Utc;
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
    last_processed_cursor: Option<(chrono::DateTime<Utc>, uuid::Uuid)>,
}

impl Trainer {
    /// Create a new trainer
    pub fn new(store: Arc<LightningStore>, config: TrainerConfig) -> Self {
        Self {
            store,
            config,
            last_processed_cursor: None,
        }
    }

    /// Run a single training iteration
    pub async fn train_iteration<A: LightningAlgorithm>(
        &mut self,
        algorithm: &mut A,
    ) -> Result<Option<TrainingResult>> {
        // Query spans based on config
        let mut spans = if let Some(task_id) = &self.config.task_id {
            debug!("Querying spans for task: {}", task_id);
            self.store.query_task_since(task_id, self.last_processed_cursor)?
        } else if let Some(agent_id) = &self.config.agent_id {
            // TODO: implementing query_agent_since would be symmetric
            // For now, minimal support only for tasks which is primary use case
            debug!("Querying spans for agent: {}", agent_id);
            vec![]
        } else {
             // TODO: implement global query
             vec![]
        };

        if spans.is_empty() {
            debug!("No new spans to process");
            return Ok(None);
        }

        // Apply batch size limit
        if spans.len() > self.config.batch_size {
            spans.truncate(self.config.batch_size);
        }

        let count = spans.len();
        info!("Training on {} new spans", count);
        
        // Update last processed cursor from the last span in the batch
        if let Some(last_span) = spans.last() {
             self.last_processed_cursor = Some((last_span.timestamp(), last_span.id()));
        }

        let result = algorithm.train(&spans)?;

        // Store updated weights if provided
        if let Some(weights) = &result.updated_weights {
            let key = format!("weights_{}", Utc::now().timestamp());
            self.store.store_resource(&key, weights)?;
            debug!("Stored updated weights: {}", key);
        }

        Ok(Some(result))
    }
    
    // ... remainder of file unchanged ...

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
        self.last_processed_cursor = None;
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
