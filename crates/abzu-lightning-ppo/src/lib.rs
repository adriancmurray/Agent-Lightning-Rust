mod model;

use crate::model::ActorCritic;
use abzu_lightning_core::{LightningAlgorithm, Result, Error, Span, TrainingResult};
use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};

const GAMMA: f32 = 0.99;
const LAMBDA_GAE: f32 = 0.95;
const VALUE_COEF: f32 = 0.5;
const EPOCHS: usize = 4;

pub struct PpoAlgorithm {
    vars: VarMap,
    model: ActorCritic,
    optimizer: AdamW,
    device: Device,
    clip_ratio: f64,
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
        
        let model = ActorCritic::new(vb, input_dim, action_dim)
            .map_err(|e| Error::Training(format!("Model creation failed: {}", e)))?;

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
        let mut rew_vec: Vec<f32> = Vec::new();
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
}

struct Batch {
    obs: Tensor,
    act: Tensor,
    rew: Tensor,
    next_obs: Tensor,
    done: Tensor,
    size: usize,
}

impl LightningAlgorithm for PpoAlgorithm {
    fn train(&mut self, spans: &[Span]) -> Result<TrainingResult> {
        let batch = match self.process_batch(spans) {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!("PPO batch processing failed: {}", e);
                return Ok(TrainingResult::new()); 
            }
        };

        // GAE Calculation (CPU side with f32)
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
        
        // Tensorize Returns for loss
        let ret_tensor = Tensor::from_vec(returns, (batch.size,), &self.device).map_err(|e| Error::Training(e.to_string()))?;

        // Value Coefficient Tensor
        let value_coef_tensor = Tensor::new(VALUE_COEF, &self.device).map_err(|e| Error::Training(e.to_string()))?;

        // K Epochs
        let mut total_loss_val = 0.0;
        
        for _ in 0..EPOCHS {
            let (_logits, values) = self.model.forward(&batch.obs).map_err(|e| Error::Training(e.to_string()))?;
            let values = values.squeeze(1).map_err(|e| Error::Training(e.to_string()))?;
            
            // Value Loss
            let v_loss = (values.sub(&ret_tensor).map_err(|e| Error::Training(e.to_string()))?
                        .powf(2.0).map_err(|e| Error::Training(e.to_string()))?)
                        .mean_all().map_err(|e| Error::Training(e.to_string()))?;
            
            // Total Loss (Currently only V-Loss enabled for scaffolding)
            let loss = v_loss.mul(&value_coef_tensor).map_err(|e| Error::Training(e.to_string()))?;

            self.optimizer.backward_step(&loss).map_err(|e| Error::Training(format!("Backward: {}", e)))?;
            
            total_loss_val = loss.to_scalar::<f32>().map_err(|e| Error::Training(e.to_string()))? as f64;
        }

        let result = TrainingResult::new()
            .with_metric("loss", total_loss_val)
            .with_metric("mean_reward", rewards_vec.iter().sum::<f32>() as f64 / batch.size as f64)
            .with_spans_processed(spans.len());

        Ok(result)
    }

    fn update_policy(&mut self, _weights: &[u8]) -> Result<()> {
        Ok(())
    }
}
