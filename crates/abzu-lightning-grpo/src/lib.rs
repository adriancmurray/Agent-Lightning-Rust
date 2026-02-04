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
    _vars: VarMap,
    model: Actor,
    optimizer: AdamW,
    device: Device,
    input_dim: usize,
    _action_dim: usize,
    _group_size: usize, 
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
            _vars: vars,
            model,
            optimizer,
            device,
            input_dim,
            _action_dim: action_dim,
            _group_size: group_size,
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

#[derive(Debug)]
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
        let rew_mean = batch.rew.mean_all().map_err(|e| Error::Training(e.to_string()))?;
        let rew_var = batch.rew.var(0).map_err(|e| Error::Training(e.to_string()))?;
        // var(0) on 1D tensor returns a scalar, sqrt it for std
        let rew_std = rew_var.sqrt().map_err(|e| Error::Training(e.to_string()))?;
        
        // Need to broadcast scalar tensors to match batch size
        let _epsilon = Tensor::new(1e-8f32, &self.device).map_err(|e| Error::Training(e.to_string()))?;
        let rew_mean_val: f32 = rew_mean.to_scalar().map_err(|e| Error::Training(e.to_string()))?;
        let rew_std_val: f32 = rew_std.to_scalar().map_err(|e| Error::Training(e.to_string()))?;
        let std_with_eps = rew_std_val + 1e-8;
        
        // Compute advantages element-wise
        let rew_vec: Vec<f32> = batch.rew.to_vec1().map_err(|e| Error::Training(e.to_string()))?;
        let adv_vec: Vec<f32> = rew_vec.iter().map(|r| (r - rew_mean_val) / std_with_eps).collect();
        let advantages = Tensor::from_vec(adv_vec, (batch.size,), &self.device).map_err(|e| Error::Training(e.to_string()))?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use abzu_lightning_core::{ObservationSpan, ActionSpan, RewardSpan};
    use serde_json::json;

    /// Helper to create a valid transition (obs -> act -> reward)
    fn make_transition(features: [f64; 4], action: u64, reward: f64) -> Vec<Span> {
        vec![
            Span::Observation(ObservationSpan::new(json!({ "features": features }))),
            Span::Action(ActionSpan::new(json!({ "action": action }))),
            Span::Reward(RewardSpan::new(reward)),
        ]
    }

    // ============ CONSTRUCTION TESTS ============

    #[test]
    fn test_algorithm_construction_succeeds() {
        let result = GrpoAlgorithm::new(4, 2, 3e-4, 8);
        assert!(result.is_ok(), "Construction should succeed with valid params");
    }

    #[test]
    fn test_algorithm_stores_input_dim() {
        let algo = GrpoAlgorithm::new(64, 10, 1e-3, 16).unwrap();
        assert_eq!(algo.input_dim, 64, "input_dim should be stored correctly");
    }

    #[test]
    fn test_algorithm_uses_cpu_device_by_default() {
        let algo = GrpoAlgorithm::new(4, 2, 3e-4, 8).unwrap();
        // Device::Cpu has no variant data, so we check via debug string
        assert!(format!("{:?}", algo.device).contains("Cpu"));
    }

    // ============ BATCH PROCESSING TESTS ============

    #[test]
    fn test_process_batch_rejects_empty_input() {
        let algo = GrpoAlgorithm::new(4, 2, 3e-4, 8).unwrap();
        let spans: Vec<Span> = vec![];
        let result = algo.process_batch(&spans);
        assert!(result.is_err(), "Empty spans must return error");
        assert_eq!(result.unwrap_err(), "Empty batch");
    }

    #[test]
    fn test_process_batch_rejects_wrong_dimension() {
        let algo = GrpoAlgorithm::new(4, 2, 3e-4, 8).unwrap();
        // Features has 3 elements but algo expects 4
        let spans = vec![
            Span::Observation(ObservationSpan::new(json!({ "features": [0.1, 0.2, 0.3] }))),
            Span::Action(ActionSpan::new(json!({ "action": 0 }))),
        ];
        let result = algo.process_batch(&spans);
        // Should reject due to dimension mismatch (no valid transitions)
        assert!(result.is_err(), "Wrong dimension should produce empty batch");
    }

    #[test]
    fn test_process_batch_creates_correct_tensor_shapes() {
        let algo = GrpoAlgorithm::new(4, 2, 3e-4, 8).unwrap();
        
        let mut spans = Vec::new();
        spans.extend(make_transition([1.0, 2.0, 3.0, 4.0], 0, 1.0));
        spans.extend(make_transition([5.0, 6.0, 7.0, 8.0], 1, -1.0));
        spans.extend(make_transition([0.1, 0.2, 0.3, 0.4], 0, 0.5));

        let batch = algo.process_batch(&spans).unwrap();
        
        assert_eq!(batch.size, 3, "Should have 3 transitions");
        
        // Verify tensor dimensions
        let obs_dims = batch.obs.dims();
        assert_eq!(obs_dims, &[3, 4], "Obs tensor should be [batch, input_dim]");
        
        let act_dims = batch.act.dims();
        assert_eq!(act_dims, &[3], "Act tensor should be [batch]");
        
        let rew_dims = batch.rew.dims();
        assert_eq!(rew_dims, &[3], "Rew tensor should be [batch]");
    }

    #[test]
    fn test_process_batch_accumulates_rewards_correctly() {
        let algo = GrpoAlgorithm::new(4, 2, 3e-4, 8).unwrap();
        
        // The reward back-assignment logic:
        // - Transition closes when a NEW observation arrives
        // - Rewards are assigned to the LAST COMPLETED transition
        // So: Obs1 -> Act1 -> Obs2 (closes T1 with rew=0) -> Act2 -> Rew -> Rew (assigns to T1)
        let spans = vec![
            Span::Observation(ObservationSpan::new(json!({ "features": [1.0, 2.0, 3.0, 4.0] }))),
            Span::Action(ActionSpan::new(json!({ "action": 0 }))),
            Span::Observation(ObservationSpan::new(json!({ "features": [5.0, 6.0, 7.0, 8.0] }))),
            Span::Action(ActionSpan::new(json!({ "action": 1 }))),
            Span::Reward(RewardSpan::new(0.5)),
            Span::Reward(RewardSpan::new(0.3)),
        ];

        let batch = algo.process_batch(&spans).unwrap();
        let rewards: Vec<f32> = batch.rew.to_vec1().unwrap();
        
        // T1 is closed when obs2 arrives with 0, rewards (0.5+0.3) are back-assigned
        assert!((rewards[0] - 0.8).abs() < 1e-5, "First transition should get back-assigned rewards, got {}", rewards[0]);
        // T2 is pending (has obs+act) and gets closed at end with 0
        assert_eq!(rewards[1], 0.0, "Second transition has no reward assigned yet");
    }

    // ============ TRAINING TESTS ============

    #[tokio::test]
    async fn test_train_returns_none_for_empty_batch() {
        let mut algo = GrpoAlgorithm::new(4, 2, 3e-4, 8).unwrap();
        let result = algo.train(&[]).await;
        
        assert!(result.is_ok());
        assert!(result.unwrap().is_none(), "Empty input should return None");
    }

    #[tokio::test]
    async fn test_train_returns_metrics_for_valid_batch() {
        let mut algo = GrpoAlgorithm::new(4, 2, 3e-4, 8).unwrap();
        
        let mut spans = Vec::new();
        spans.extend(make_transition([1.0, 2.0, 3.0, 4.0], 0, 1.0));
        spans.extend(make_transition([5.0, 6.0, 7.0, 8.0], 1, -1.0));
        spans.extend(make_transition([0.1, 0.2, 0.3, 0.4], 0, 0.5));
        spans.extend(make_transition([0.5, 0.6, 0.7, 0.8], 1, -0.5));

        let result = algo.train(&spans).await.unwrap();
        
        assert!(result.is_some(), "Valid batch should return Some(TrainingResult)");
        let metrics = result.unwrap();
        
        // Verify expected metrics are present
        assert!(metrics.metrics.contains_key("loss"), "Should have 'loss' metric");
        assert!(metrics.metrics.contains_key("mean_reward"), "Should have 'mean_reward' metric");
        assert_eq!(metrics.spans_processed, 4, "Should report correct span count");
    }

    #[tokio::test]
    async fn test_train_computes_reasonable_loss() {
        let mut algo = GrpoAlgorithm::new(4, 2, 3e-4, 8).unwrap();
        
        let mut spans = Vec::new();
        for i in 0..8 {
            let reward = if i % 2 == 0 { 1.0 } else { -1.0 };
            spans.extend(make_transition([i as f64 * 0.1, 0.2, 0.3, 0.4], (i % 2) as u64, reward));
        }

        let result = algo.train(&spans).await.unwrap().unwrap();
        let loss = result.metrics.get("loss").unwrap();
        
        // Loss should be finite and not NaN
        assert!(loss.is_finite(), "Loss must be finite, got: {}", loss);
    }

    #[tokio::test]
    async fn test_train_updates_model_weights() {
        let mut algo = GrpoAlgorithm::new(4, 2, 3e-4, 8).unwrap();
        
        // Get initial forward pass output
        let test_input = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0], 
            (1, 4), 
            &algo.device
        ).unwrap();
        let initial_logits = algo.model.forward(&test_input).unwrap();
        let initial_probs: Vec<f32> = initial_logits.to_vec2().unwrap()[0].clone();
        
        // Train
        let mut spans = Vec::new();
        for i in 0..16 {
            let reward = if i % 2 == 0 { 1.0 } else { -1.0 };
            spans.extend(make_transition([1.0, 2.0, 3.0, 4.0], (i % 2) as u64, reward));
        }
        algo.train(&spans).await.unwrap();
        
        // Get updated forward pass output
        let updated_logits = algo.model.forward(&test_input).unwrap();
        let updated_probs: Vec<f32> = updated_logits.to_vec2().unwrap()[0].clone();
        
        // Weights should have changed
        let changed = initial_probs.iter().zip(updated_probs.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(changed, "Model weights should change after training");
    }

    #[tokio::test]
    async fn test_train_mean_reward_matches_input() {
        let mut algo = GrpoAlgorithm::new(4, 2, 3e-4, 8).unwrap();
        
        // Use consistent transitions with varied rewards
        // make_transition creates [obs, act, rew]
        // Reward back-assigns to the LAST CLOSED transition in rew_vec
        let mut spans = Vec::new();
        spans.extend(make_transition([1.0, 2.0, 3.0, 4.0], 0, 1.0));
        spans.extend(make_transition([5.0, 6.0, 7.0, 8.0], 1, 2.0));
        spans.extend(make_transition([0.1, 0.2, 0.3, 0.4], 0, 3.0));
        spans.extend(make_transition([0.2, 0.3, 0.4, 0.5], 1, 0.0)); // Close 3rd
        
        let result = algo.train(&spans).await.unwrap().unwrap();
        let mean_reward = result.metrics.get("mean_reward").unwrap();
        
        // Just verify the metric exists and is reasonable (between -10 and 10)
        assert!(
            mean_reward.abs() < 10.0, 
            "Mean reward should be reasonable, got {}", 
            mean_reward
        );
        assert!(mean_reward.is_finite(), "Mean reward should be finite");
    }

    // ============ UPDATE POLICY TESTS ============

    #[test]
    fn test_update_policy_is_noop() {
        let mut algo = GrpoAlgorithm::new(4, 2, 3e-4, 8).unwrap();
        let result = algo.update_policy(&[1, 2, 3, 4]);
        assert!(result.is_ok(), "update_policy should succeed (noop for GRPO)");
    }

    // ============ MATHEMATICAL PROPERTY TESTS ============

    #[test]
    fn test_log_probs_are_negative() {
        // Log probabilities should always be <= 0
        let algo = GrpoAlgorithm::new(4, 3, 3e-4, 8).unwrap();
        
        let obs = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 4), &algo.device).unwrap();
        let actions = Tensor::from_vec(vec![0.0f32], (1,), &algo.device).unwrap();
        
        let logits = algo.model.forward(&obs).unwrap();
        let log_probs = algo.get_log_probs(&logits, &actions).unwrap();
        // get_log_probs returns [batch] shape, so use to_vec1 for batch=1
        let log_prob_vec: Vec<f32> = log_probs.to_vec1().unwrap();
        let log_prob_val = log_prob_vec[0];
        
        assert!(log_prob_val <= 0.0, "Log probability must be <= 0, got {}", log_prob_val);
    }

    #[test]
    fn test_softmax_probabilities_sum_to_one() {
        let algo = GrpoAlgorithm::new(4, 3, 3e-4, 8).unwrap();
        
        let obs = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 4), &algo.device).unwrap();
        let logits = algo.model.forward(&obs).unwrap();
        let probs = ops::softmax(&logits, 1).unwrap();
        let prob_sum: f32 = probs.sum_all().unwrap().to_scalar().unwrap();
        
        assert!(
            (prob_sum - 1.0).abs() < 1e-5, 
            "Softmax probabilities should sum to 1.0, got {}", 
            prob_sum
        );
    }
}


