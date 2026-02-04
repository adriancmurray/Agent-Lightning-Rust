mod model;

use crate::model::Actor;
use abzu_lightning_core::{LightningAlgorithm, Result, Error, Span, TrainingResult};
use async_trait::async_trait;
use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap, ops};

// Hyperparameters
const EPOCHS: usize = 4;
const CLIP_RATIO: f64 = 0.2;
const ENTROPY_COEF: f32 = 0.01;
// KL penalty is often used in GRPO but we stick to PPO-clip style for now

pub struct GrpoAlgorithm {
    vars: VarMap,
    model: Actor,
    optimizer: AdamW,
    device: Device,
    input_dim: usize,
    action_dim: usize,
    group_size: usize, 
}

impl GrpoAlgorithm {
    pub fn new(
        input_dim: usize, 
        action_dim: usize,
        learning_rate: f64,
        group_size: usize,
    ) -> Result<Self> {
        let device = Device::Cpu; // Default to CPU for now
        let vars = VarMap::new();
        let vb = VarBuilder::from_varmap(&vars, DType::F32, &device);
        
        // Actor Only
        let model = Actor::new(vb, input_dim, action_dim)
            .map_err(|e| Error::Training(format!("Actor creation failed: {}", e)))?;

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
            input_dim,
            action_dim,
            group_size,
        })
    }

    /// Calculate log probs for specific actions
    fn get_log_probs(&self, logits: &Tensor, actions: &Tensor) -> std::result::Result<Tensor, Error> {
        let actions_u32 = actions.to_dtype(DType::U32).map_err(|e| Error::Training(e.to_string()))?;
        let log_probs_all = ops::log_softmax(logits, 1).map_err(|e| Error::Training(e.to_string()))?;
        let actions_unsq = actions_u32.unsqueeze(1).map_err(|e| Error::Training(e.to_string()))?;
        let selected_log_probs = log_probs_all.gather(&actions_unsq, 1).map_err(|e| Error::Training(e.to_string()))?;
        selected_log_probs.squeeze(1).map_err(|e| Error::Training(e.to_string()))
    }

    fn process_batch(&self, spans: &[Span]) -> std::result::Result<Batch, String> {
        // Reuse PPO's batch processing logic basically
        // Or simplify. Let's do simple matching.
        let mut obs_vec: Vec<f32> = Vec::new();
        let mut act_vec: Vec<f32> = Vec::new();
        let mut rew_vec: Vec<f32> = Vec::new();

        let mut current_obs: Option<Vec<f32>> = None;
        let mut current_act: Option<u32> = None;
        
        for span in spans {
            match span {
                Span::Observation(o) => {
                    if let Some(arr) = o.data.get("features").and_then(|v| v.as_array()) {
                        let vec: Vec<f32> = arr.iter().filter_map(|v| v.as_f64()).map(|f| f as f32).collect();
                        if vec.len() == self.input_dim {
                             if let (Some(s), Some(a)) = (current_obs.take(), current_act.take()) {
                                 // Close previous transition
                                 obs_vec.extend(s);
                                 act_vec.push(a as f32);
                                 rew_vec.push(0.0); // No reward was found?
                             }
                             current_obs = Some(vec);
                        }
                    }
                }
                Span::Action(a) => {
                     if let Some(val) = a.data.get("action").and_then(|v| v.as_u64()) {
                         current_act = Some(val as u32);
                     }
                }
                Span::Reward(r) => {
                    // Back-assign reward to last transition
                    if !rew_vec.is_empty() {
                         let last = rew_vec.len() - 1;
                         rew_vec[last] += r.reward as f32;
                    }
                }
                _ => {}
            }
        }
        // Handle last pending? No next obs needed for GRPO really unless we do GAE.
        // We do simple Advantage = (R - mean) / std.
        // So we just need the completed triples.
        if let (Some(s), Some(a)) = (current_obs, current_act) {
            obs_vec.extend(s);
            act_vec.push(a as f32);
            rew_vec.push(0.0); // Pending last reward?
        }

        let count = act_vec.len();
        if count == 0 { return Err("Empty batch".into()); }

        let obs = Tensor::from_vec(obs_vec, (count, self.input_dim), &self.device).map_err(|e| e.to_string())?;
        let act = Tensor::from_vec(act_vec, (count,), &self.device).map_err(|e| e.to_string())?;
        let rew = Tensor::from_vec(rew_vec, (count,), &self.device).map_err(|e| e.to_string())?;

        Ok(Batch { obs, act, rew, size: count })
    }
}

struct Batch {
    obs: Tensor,
    act: Tensor,
    rew: Tensor,
    size: usize,
}

#[async_trait]
impl LightningAlgorithm for GrpoAlgorithm {
    async fn train(&mut self, spans: &[Span]) -> Result<Option<TrainingResult>> {
        let batch = match self.process_batch(spans) {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!("GRPO batch error: {}", e);
                return Ok(None);
            }
        };

        // 1. Calculate Advantages using Group Normalization
        // Here we assume the batch IS the group.
        // Adv = (R - mean(R)) / (std(R) + epsilon)
        let rew_mean = batch.rew.mean_all().map_err(|e| Error::Training( e.to_string()))?;
        let rew_std = batch.rew.var(0).map_err(|e| Error::Training(e.to_string()))?
            .sqrt().map_err(|e| Error::Training(e.to_string()))?;
        
        let advantages = batch.rew.sub(&rew_mean).map_err(|e| Error::Training(e.to_string()))?
            .div(&rew_std.add(&Tensor::new(1e-8f32, &self.device).unwrap()).unwrap())
            .map_err(|e| Error::Training(e.to_string()))?;

        // 2. Initial Log Probs
        // (Block in place if needed)
        let old_logits = self.model.forward(&batch.obs).map_err(|e| Error::Training(e.to_string()))?;
        let old_log_probs = self.get_log_probs(&old_logits, &batch.act)?
            .detach();

        let mut total_loss_val = 0.0;
        let entropy_coef_t = Tensor::new(ENTROPY_COEF, &self.device).map_err(|e| Error::Training(e.to_string()))?;

        // 3. Optimization Loop
        for _ in 0..EPOCHS {
            let logits = self.model.forward(&batch.obs).map_err(|e| Error::Training(e.to_string()))?;
            let new_log_probs = self.get_log_probs(&logits, &batch.act)?;

            // Ratio = (new - old).exp()
            let ratio = (new_log_probs.sub(&old_log_probs).map_err(|e| Error::Training(e.to_string()))?)
                .exp().map_err(|e| Error::Training(e.to_string()))?;

            // PPO Clipping Objective
            let surr1 = ratio.mul(&advantages).map_err(|e| Error::Training(e.to_string()))?;
            let clip = CLIP_RATIO as f32;
            let ratio_clamped = ratio.clamp(1.0 - clip, 1.0 + clip).map_err(|e| Error::Training(e.to_string()))?;
            let surr2 = ratio_clamped.mul(&advantages).map_err(|e| Error::Training(e.to_string()))?;
            
            let policy_loss = surr1.minimum(&surr2).map_err(|e| Error::Training(e.to_string()))?
                .neg().map_err(|e| Error::Training(e.to_string()))?
                .mean_all().map_err(|e| Error::Training(e.to_string()))?;

            // KLD/Entropy term
            let probs = ops::softmax(&logits, 1).map_err(|e| Error::Training(e.to_string()))?;
            let log_probs_all = ops::log_softmax(&logits, 1).map_err(|e| Error::Training(e.to_string()))?;
            let entropy = (probs.mul(&log_probs_all).map_err(|e| Error::Training(e.to_string()))?)
                .sum(1).map_err(|e| Error::Training(e.to_string()))?
                .neg().map_err(|e| Error::Training(e.to_string()))?
                .mean_all().map_err(|e| Error::Training(e.to_string()))?;

            let loss = policy_loss.sub(&entropy.mul(&entropy_coef_t).map_err(|e| Error::Training(e.to_string()))?)
                .map_err(|e| Error::Training(e.to_string()))?;
            
            self.optimizer.backward_step(&loss).map_err(|e| Error::Training(e.to_string()))?;
            
            total_loss_val = loss.to_scalar::<f32>().map_err(|e| Error::Training(e.to_string()))? as f64;
        }

        let avg_rew = batch.rew.mean_all().map_err(|e| Error::Training(e.to_string()))?.to_scalar::<f32>().unwrap_or(0.0) as f64;

        Ok(Some(TrainingResult::new()
            .with_metric("loss", total_loss_val)
            .with_metric("mean_reward", avg_rew)
            .with_spans_processed(batch.size)))
    }

    fn update_policy(&mut self, _weights: &[u8]) -> Result<()> {
        Ok(())
    }
}
