mod model;

use crate::model::ActorCritic;
use abzu_lightning_core::{LightningAlgorithm, Result, Error, Span, TrainingResult};
use async_trait::async_trait;
use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap, ops};

const GAMMA: f32 = 0.99;
const LAMBDA_GAE: f32 = 0.95;
const VALUE_COEF: f32 = 0.5;
const ENTROPY_COEF: f32 = 0.01;
const EPOCHS: usize = 4;

pub struct PpoAlgorithm {
    vars: VarMap,
    model: ActorCritic,
    optimizer: AdamW,
    device: Device,
    clip_ratio: f64, // Config
    input_dim: usize,
    action_dim: usize,
}

impl PpoAlgorithm {
    pub fn new(
        input_dim: usize,
        action_dim: usize,
        learning_rate: f64,
        clip_range: f64,
        device_str: &str,
    ) -> Result<Self> {
        let device = if device_str == "cuda" {
            Device::new_cuda(0).map_err(|e| Error::Training(format!("CUDA error: {}", e)))?
        } else if device_str == "metal" {
            Device::new_metal(0).map_err(|e| Error::Training(format!("Metal error: {}", e)))?
        } else {
            Device::Cpu
        };

        let vars = VarMap::new();
        let vb = VarBuilder::from_varmap(&vars, DType::F32, &device);
        
        // Initialize Model
        let model = ActorCritic::new(vb, input_dim, action_dim)
            .map_err(|e| Error::Training(format!("Model creation failed: {}", e)))?;

        // Initialize Optimizer
        let params = ParamsAdamW {
            lr: learning_rate,
            ..Default::default()
        };
        let optimizer = AdamW::new(vars.all_vars(), params)
            .map_err(|e| Error::Training(format!("Optimizer creation failed: {}", e)))?;

        Ok(Self {
            vars,
            model,
            optimizer,
            device,
            clip_ratio: clip_range,
            input_dim,
            action_dim,
        })
    }

    fn process_batch(&self, spans: &[Span]) -> std::result::Result<Batch, String> {
        let mut obs_vec: Vec<f32> = Vec::new();
        let mut act_vec: Vec<f32> = Vec::new(); // actions are indices
        let mut rew_vec: Vec<f32> = Vec::new(); // rewards
        let mut next_obs_vec: Vec<f32> = Vec::new();
        let mut done_vec: Vec<f32> = Vec::new();

        let mut current_obs: Option<Vec<f32>> = None;
        let mut current_act: Option<u32> = None;
        
        for span in spans {
            match span {
                Span::Observation(o) => {
                    let vec = self.extract_features(&o.data)?;
                    if let (Some(s), Some(a)) = (current_obs.take(), current_act.take()) {
                         obs_vec.extend_from_slice(&s);
                         act_vec.push(a as f32);
                         rew_vec.push(0.0); 
                         next_obs_vec.extend_from_slice(&vec);
                         done_vec.push(0.0);
                    }
                    current_obs = Some(vec);
                }
                Span::Action(a) => {
                     if let Some(val) = a.data.get("action").and_then(|v: &serde_json::Value| v.as_u64()) {
                         current_act = Some(val as u32);
                     }
                }
                Span::Reward(r) => {
                    if !rew_vec.is_empty() {
                         let last_idx = rew_vec.len() - 1;
                         rew_vec[last_idx] += r.reward as f32;
                    }
                }
                _ => {}
            }
        }
        
        let count = act_vec.len();
        if count == 0 {
             return Err("No valid transitions found in batch".to_string());
        }

        let obs_tensor = Tensor::from_vec(obs_vec, (count, self.input_dim), &self.device).map_err(|e| e.to_string())?;
        let next_obs_tensor = Tensor::from_vec(next_obs_vec, (count, self.input_dim), &self.device).map_err(|e| e.to_string())?;
        let act_tensor = Tensor::from_vec(act_vec, (count,), &self.device).map_err(|e| e.to_string())?;
        let rew_tensor = Tensor::from_vec(rew_vec, (count,), &self.device).map_err(|e| e.to_string())?;
        let done_tensor = Tensor::from_vec(done_vec, (count,), &self.device).map_err(|e| e.to_string())?;

        Ok(Batch {
            obs: obs_tensor,
            act: act_tensor,
            rew: rew_tensor,
            next_obs: next_obs_tensor,
            done: done_tensor,
            size: count,
        })
    }

    fn extract_features(&self, value: &serde_json::Value) -> std::result::Result<Vec<f32>, String> {
        if let Some(arr) = value.get("features").and_then(|v| v.as_array()) {
            if arr.len() != self.input_dim {
                return Err(format!("Feature dim mismatch: expected {}, got {}", self.input_dim, arr.len()));
            }
            let vec: std::result::Result<Vec<f32>, _> = arr.iter().map(|v| v.as_f64().ok_or(()).map(|f| f as f32)).collect();
            vec.map_err(|_| "Invalid feature value".to_string())
        } else {
             Err("Observation missing 'features' array".to_string())
        }
    }

    /// Calculate log probs for specific actions using one-hot encoding
    fn get_log_probs(&self, logits: &Tensor, actions: &Tensor) -> std::result::Result<Tensor, Error> {
        let n_batch = logits.dim(0).map_err(|e| Error::Training(e.to_string()))?;
        // actions is [B], logits is [B, A]
        
        // One-hot encode actions
        // actions tensor is f32, need u32 for one_hot? Candle one_hot needs specific logic?
        // Actually, candle doesn't have `one_hot` easily directly?
        // Let's check `candle_nn::encoding::one_hot`? 
        // No, standard way is manual or `gather`.
        // Since `gather` is missing, let's look for `gather` equivalent logic.
        // `logits.gather(indices, dim)` IS available in recent Candle versions!
        // Let's try `gather`.
        
        let actions_u32 = actions.to_dtype(DType::U32).map_err(|e| Error::Training(e.to_string()))?;
        let log_probs_all = ops::log_softmax(logits, 1).map_err(|e| Error::Training(e.to_string()))?;
        
        // gather expects actions to be [B, 1] for dim 1 gather?
        let actions_unsq = actions_u32.unsqueeze(1).map_err(|e| Error::Training(e.to_string()))?;
        let selected_log_probs = log_probs_all.gather(&actions_unsq, 1).map_err(|e| Error::Training(e.to_string()))?;
        
        selected_log_probs.squeeze(1).map_err(|e| Error::Training(e.to_string()))
    }
}

struct Batch {
    obs: Tensor,
    act: Tensor,
    rew: Tensor,
    next_obs: Tensor,
    done: Tensor,
    size: usize,
}

#[async_trait]
impl LightningAlgorithm for PpoAlgorithm {
    async fn train(&mut self, spans: &[Span]) -> Result<Option<TrainingResult>> {
        let batch = match self.process_batch(spans) {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!("PPO batch processing failed: {}", e);
                return Ok(None); 
            }
        };

        // 1. Initial Forward Pass (No grad) - Calculate Old Log Probs
        // ... (Logic unchanged generally, just wrapped in async fn)
        // Note: Candle operations are blocking. In a real production setup this should be 
        // wrapped in block_in_place or spawn_blocking. 
        // For local loop it's acceptable.

        let old_log_probs = {
            let (logits, _values) = self.model.forward(&batch.obs).map_err(|e| Error::Training(format!("Forward: {}", e)))?;
            let probs = self.get_log_probs(&logits, &batch.act).map_err(|e| Error::Training(format!("LogProbs: {}", e)))?;
            probs.detach() // Detach from graph explicitly
        };

        // 2. GAE Calculation
        let (_logits, values) = self.model.forward(&batch.obs).map_err(|e| Error::Training(format!("Forward: {}", e)))?;
        let (_next_logits, next_values) = self.model.forward(&batch.next_obs).map_err(|e| Error::Training(format!("Next Forward: {}", e)))?;
        
        let values_vec = values.squeeze(1).map_err(|e| Error::Training(e.to_string()))?.to_vec1::<f32>().map_err(|e| Error::Training(e.to_string()))?;
        let next_values_vec = next_values.squeeze(1).map_err(|e| Error::Training(e.to_string()))?.to_vec1::<f32>().map_err(|e| Error::Training(e.to_string()))?;
        let rewards_vec = batch.rew.to_vec1::<f32>().map_err(|e| Error::Training(e.to_string()))?;
        let dones_vec = batch.done.to_vec1::<f32>().map_err(|e| Error::Training(e.to_string()))?;

        let mut advantages = vec![0.0f32; batch.size];
        let mut returns = vec![0.0f32; batch.size];
        let mut next_adv = 0.0f32;

        for t in (0..batch.size).rev() {
            let delta = rewards_vec[t] + GAMMA * next_values_vec[t] * (1.0 - dones_vec[t]) - values_vec[t];
            advantages[t] = delta + GAMMA * LAMBDA_GAE * (1.0 - dones_vec[t]) * next_adv;
            next_adv = advantages[t];
            returns[t] = advantages[t] + values_vec[t];
        }
        
        // Tensorize
        let adv_tensor = Tensor::from_vec(advantages, (batch.size,), &self.device).map_err(|e| Error::Training(e.to_string()))?;
        let ret_tensor = Tensor::from_vec(returns, (batch.size,), &self.device).map_err(|e| Error::Training(e.to_string()))?;

        // Normalize Advantages
        let adv_mean = adv_tensor.mean_all().map_err(|e| Error::Training(e.to_string()))?;
        let adv_std = adv_tensor.var(0).map_err(|e| Error::Training(e.to_string()))?.sqrt().map_err(|e| Error::Training(e.to_string()))?;
        // Add small epsilon to std prevents NaN
        let adv_normalized = adv_tensor.sub(&adv_mean).map_err(|e| Error::Training(e.to_string()))?
            .div(&adv_std).map_err(|e| Error::Training(e.to_string()))?;

        // Constants Tensors
        let value_coef_tensor = Tensor::new(VALUE_COEF, &self.device).map_err(|e| Error::Training(e.to_string()))?;
        let entropy_coef_tensor = Tensor::new(ENTROPY_COEF, &self.device).map_err(|e| Error::Training(e.to_string()))?;
        let _one_tensor = Tensor::new(1.0f32, &self.device).map_err(|e| Error::Training(e.to_string()))?;

        let mut total_loss_val = 0.0;
        
        for _ in 0..EPOCHS {
            let (logits, values) = self.model.forward(&batch.obs).map_err(|e| Error::Training(e.to_string()))?;
            let values = values.squeeze(1).map_err(|e| Error::Training(e.to_string()))?;
            
            // New Log Probs
            let new_log_probs = self.get_log_probs(&logits, &batch.act)?;

            // Ratio = (new - old).exp()
            let ratio = (new_log_probs.sub(&old_log_probs).map_err(|e| Error::Training(e.to_string()))?)
                .exp().map_err(|e| Error::Training(e.to_string()))?;
            
            // Surrogate 1 = ratio * adv
            let surr1 = ratio.mul(&adv_normalized).map_err(|e| Error::Training(e.to_string()))?;
            
            // Surrogate 2 = clamp(ratio) * adv
            let clip = self.clip_ratio as f32;
            let ratio_clamped = ratio.clamp(1.0 - clip, 1.0 + clip).map_err(|e| Error::Training(e.to_string()))?;
            let surr2 = ratio_clamped.mul(&adv_normalized).map_err(|e| Error::Training(e.to_string()))?;
            
            // Policy Loss = -min(surr1, surr2).mean()
            let policy_loss = surr1.minimum(&surr2).map_err(|e| Error::Training(e.to_string()))?
                .neg().map_err(|e| Error::Training(e.to_string()))?
                .mean_all().map_err(|e| Error::Training(e.to_string()))?;
            
            // Value Loss
            let v_loss = (values.sub(&ret_tensor).map_err(|e| Error::Training(e.to_string()))?
                        .powf(2.0).map_err(|e| Error::Training(e.to_string()))?)
                        .mean_all().map_err(|e| Error::Training(e.to_string()))?;
            
            // Entropy Loss
            let probs = ops::softmax(&logits, 1).map_err(|e| Error::Training(e.to_string()))?;
            let log_probs_all = ops::log_softmax(&logits, 1).map_err(|e| Error::Training(e.to_string()))?;
            let entropy = (probs.mul(&log_probs_all).map_err(|e| Error::Training(e.to_string()))?)
                .sum(1).map_err(|e| Error::Training(e.to_string()))?
                .neg().map_err(|e| Error::Training(e.to_string()))?
                .mean_all().map_err(|e| Error::Training(e.to_string()))?;

            // Total Loss
            let loss = policy_loss
                .add(&v_loss.mul(&value_coef_tensor).map_err(|e| Error::Training(e.to_string()))?).map_err(|e| Error::Training(e.to_string()))?
                .sub(&entropy.mul(&entropy_coef_tensor).map_err(|e| Error::Training(e.to_string()))?).map_err(|e| Error::Training(e.to_string()))?;

            self.optimizer.backward_step(&loss).map_err(|e| Error::Training(format!("Backward: {}", e)))?;
            
            total_loss_val = loss.to_scalar::<f32>().map_err(|e| Error::Training(e.to_string()))? as f64;
        }

        let result = TrainingResult::new()
            .with_metric("loss", total_loss_val)
            .with_metric("mean_reward", rewards_vec.iter().sum::<f32>() as f64 / batch.size as f64)
            .with_spans_processed(spans.len());

        Ok(Some(result))
    }

    fn update_policy(&mut self, _weights: &[u8]) -> Result<()> {
        Ok(())
    }
}
