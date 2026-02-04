use abzu_lightning_core::{LightningStore, Trainer, TrainerConfig, LightningAlgorithm};
use crate::{BrainFactory, AlgorithmConfig};
use std::sync::Arc;
use tokio::sync::RwLock;

/// A self-contained, thread-safe training harness
pub struct TrainingHarness {
    trainer: Arc<RwLock<Trainer>>,
    algorithm: Arc<RwLock<Box<dyn LightningAlgorithm>>>,
    running: Arc<std::sync::atomic::AtomicBool>,
}

impl TrainingHarness {
    /// Bootstrap the harness from config
    pub fn new(
        store_path: impl AsRef<std::path::Path>,
        algo_config: AlgorithmConfig,
        trainer_config: TrainerConfig
    ) -> anyhow::Result<Self> {
        // Initialize Store
        let store = Arc::new(LightningStore::open(store_path)?);

        // Hydrate Algorithm via Factory
        let algorithm = BrainFactory::build(&algo_config)?;
        
        // Initialize Trainer
        let trainer = Trainer::new(store, trainer_config);

        Ok(Self {
            trainer: Arc::new(RwLock::new(trainer)),
            algorithm: Arc::new(RwLock::new(algorithm)),
            running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        })
    }

    /// Spawn the training loop in the background
    pub fn spawn(&self) {
        let trainer_ref = self.trainer.clone();
        let algo_ref = self.algorithm.clone();
        let running = self.running.clone();

        running.store(true, std::sync::atomic::Ordering::SeqCst);

        tokio::spawn(async move {
            tracing::info!("Background training loop started");
            
            while running.load(std::sync::atomic::Ordering::Relaxed) {
                let mut trainer = trainer_ref.write().await;
                let mut algo = algo_ref.write().await;

                match trainer.train_iteration(&mut **algo).await {
                    Ok(Some(res)) => {
                        if res.spans_processed > 0 {
                            tracing::debug!("Trained on {} spans", res.spans_processed);
                        }
                    },
                    Ok(None) => {
                        // No data, sleep briefly to avoid busy loop
                    },
                    Err(e) => {
                        tracing::error!("Training iteration failed: {}", e);
                    }
                }
                
                // Release locks before sleeping
                drop(trainer);
                drop(algo);

                // Default sleep to prevent CPU burn, can be tuned or config integration later
                tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;
            }
            tracing::info!("Background training loop stopped");
        });
    }

    /// Stop the training loop
    pub fn stop(&self) {
        self.running.store(false, std::sync::atomic::Ordering::SeqCst);
    }

    /// Hot-swap the brain at runtime
    pub async fn swap_algorithm(&self, new_config: AlgorithmConfig) -> anyhow::Result<()> {
        let new_brain = BrainFactory::build(&new_config)?;
        let mut algo_guard = self.algorithm.write().await;
        *algo_guard = new_brain;
        tracing::info!("Algorithm hot-swapped successfully");
        Ok(())
    }
}
