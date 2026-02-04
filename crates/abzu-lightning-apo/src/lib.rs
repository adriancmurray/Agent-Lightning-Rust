use abzu_lightning_core::{LightningAlgorithm, Result, Error, Span, TrainingResult, LlmBackend};
use async_trait::async_trait;
use std::sync::Arc;
use tracing::info;

/// A trace of a single interaction
#[derive(Debug, Clone)]
struct InteractionTrace {
    input: String,
    action: String,
    reward: f64,
}

pub struct ApoAlgorithm {
    current_prompt: String,
    backend: Arc<dyn LlmBackend>,
    failures: Vec<InteractionTrace>,
    batch_size: usize, // Number of failures to accumulate before optimizing
    
    // State for parsing spans
    pending_obs: Option<String>,
    pending_act: Option<String>,
}

impl ApoAlgorithm {
    pub fn new(initial_prompt: String, backend: Arc<dyn LlmBackend>) -> Self {
        Self {
            current_prompt: initial_prompt,
            backend,
            failures: Vec::new(),
            batch_size: 3, // Initial default
            pending_obs: None,
            pending_act: None,
        }
    }

    fn construct_optimization_prompt(&self, failures: &[InteractionTrace]) -> String {
        let mut prompt = String::new();
        prompt.push_str("You are an Automatic Prompt Optimizer.\n");
        prompt.push_str("Your goal is to improve the System Instruction for an AI Agent to prevent future failures.\n\n");
        
        prompt.push_str("### Current System Instruction:\n");
        prompt.push_str(&format!("\"{}\"\n\n", self.current_prompt));
        
        prompt.push_str("### Failure Traces (Negative Reward):\n");
        for (i, trace) in failures.iter().enumerate() {
            prompt.push_str(&format!("{}. Input: \"{}\"\n", i + 1, trace.input));
            prompt.push_str(&format!("   Action: \"{}\"\n", trace.action));
            prompt.push_str(&format!("   Reward: {}\n", trace.reward));
        }
        
        prompt.push_str("\n### Task:\n");
        prompt.push_str("Analyze these failures. Identify why the agent made the wrong decision based on the current instruction.\n");
        prompt.push_str("Write a NEW, improved System Instruction that fixes these specific issues while maintaining general capability.\n");
        prompt.push_str("Output ONLY the new System Instruction text. Do not include reasoning or markdown formatting.");
        
        prompt
    }
}

#[async_trait]
impl LightningAlgorithm for ApoAlgorithm {
    async fn train(&mut self, spans: &[Span]) -> Result<Option<TrainingResult>> {
        let mut _new_failures = 0;

        for span in spans {
            match span {
                Span::Observation(o) => {
                    // Try to get textual representation
                    if let Some(text) = o.data.get("text").and_then(|v| v.as_str()) {
                        self.pending_obs = Some(text.to_string());
                    } else {
                        // Fallback to json string
                        self.pending_obs = Some(o.data.to_string());
                    }
                    self.pending_act = None;
                }
                Span::Action(a) => {
                    if let Some(text) = a.data.get("text").and_then(|v| v.as_str()) {
                         self.pending_act = Some(text.to_string());
                    } else if let Some(action) = a.data.get("action") {
                         self.pending_act = Some(action.to_string());
                    } else {
                         self.pending_act = Some(a.data.to_string());
                    }
                }
                Span::Reward(r) => {
                    if let (Some(obs), Some(act)) = (&self.pending_obs, &self.pending_act) {
                        // Check if failure (negative reward)
                        if r.reward < 0.0 {
                            self.failures.push(InteractionTrace {
                                input: obs.clone(),
                                action: act.clone(),
                                reward: r.reward,
                            });
                            _new_failures += 1;
                        }
                    }
                }
            }
        }

        if self.failures.len() >= self.batch_size {
            info!("APO: optimising prompt with {} failures...", self.failures.len());
            
            let optimization_prompt = self.construct_optimization_prompt(&self.failures);
            
            match self.backend.generate(&optimization_prompt).await {
                Ok(new_prompt) => {
                    let cleaned_prompt = new_prompt.trim().replace("```", ""); // Basic cleanup
                    info!("APO: Optimization successful. New prompt length: {}", cleaned_prompt.len());
                    
                    self.current_prompt = cleaned_prompt.clone();
                    self.failures.clear(); // Clear buffer after optimization
                    
                    let result = TrainingResult::new()
                        .with_metric("failures_processed", self.batch_size as f64)
                        .with_weights(self.current_prompt.as_bytes().to_vec())
                        .with_spans_processed(spans.len());
                        
                    return Ok(Some(result));
                },
                Err(e) => {
                    tracing::error!("APO Backend Error: {}", e);
                    // Keep failures to retry? Or clear to avoid stuck loop?
                    // Keep for now.
                    return Ok(None);
                }
            }
        }

        // Return None if no update occurred
        Ok(None)
    }

    fn update_policy(&mut self, weights: &[u8]) -> Result<()> {
        if let Ok(s) = String::from_utf8(weights.to_vec()) {
            self.current_prompt = s;
            Ok(())
        } else {
            Err(Error::State("Invalid UTF-8 weights for APO".to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use abzu_lightning_core::{ObservationSpan, ActionSpan, RewardSpan, Span};
    use serde_json::json;
    use std::sync::Mutex;

    /// Mock LLM Backend for testing APO without real API calls
    struct MockLlmBackend {
        response: Mutex<String>,
        call_count: Mutex<usize>,
    }

    impl MockLlmBackend {
        fn new(response: &str) -> Arc<Self> {
            Arc::new(Self {
                response: Mutex::new(response.to_string()),
                call_count: Mutex::new(0),
            })
        }

        fn get_call_count(&self) -> usize {
            *self.call_count.lock().unwrap()
        }
    }

    #[async_trait]
    impl LlmBackend for MockLlmBackend {
        async fn generate(&self, _prompt: &str) -> anyhow::Result<String> {
            *self.call_count.lock().unwrap() += 1;
            Ok(self.response.lock().unwrap().clone())
        }
    }

    /// Helper to create failure-inducing spans
    fn make_failure(input: &str, action: &str, reward: f64) -> Vec<Span> {
        vec![
            Span::Observation(ObservationSpan::new(json!({ "text": input }))),
            Span::Action(ActionSpan::new(json!({ "text": action }))),
            Span::Reward(RewardSpan::new(reward)),
        ]
    }

    // ============ CONSTRUCTION TESTS ============

    #[test]
    fn test_apo_creates_with_initial_prompt() {
        let backend = MockLlmBackend::new("improved prompt");
        let algo = ApoAlgorithm::new("initial".to_string(), backend);
        assert_eq!(algo.current_prompt, "initial");
    }

    #[test]
    fn test_apo_starts_with_empty_failures() {
        let backend = MockLlmBackend::new("improved");
        let algo = ApoAlgorithm::new("prompt".to_string(), backend);
        assert!(algo.failures.is_empty());
    }

    // ============ FAILURE ACCUMULATION TESTS ============

    #[tokio::test]
    async fn test_negative_reward_adds_failure() {
        let backend = MockLlmBackend::new("improved");
        let mut algo = ApoAlgorithm::new("prompt".to_string(), backend);
        
        // Single failure - should accumulate but not trigger optimization
        let spans = make_failure("input", "wrong action", -1.0);
        algo.train(&spans).await.unwrap();
        
        assert_eq!(algo.failures.len(), 1);
    }

    #[tokio::test]
    async fn test_positive_reward_does_not_add_failure() {
        let backend = MockLlmBackend::new("improved");
        let mut algo = ApoAlgorithm::new("prompt".to_string(), backend);
        
        let spans = make_failure("input", "good action", 1.0); // Positive reward
        algo.train(&spans).await.unwrap();
        
        assert_eq!(algo.failures.len(), 0, "Positive rewards should not add failures");
    }

    #[tokio::test]
    async fn test_zero_reward_does_not_add_failure() {
        let backend = MockLlmBackend::new("improved");
        let mut algo = ApoAlgorithm::new("prompt".to_string(), backend);
        
        let spans = make_failure("input", "neutral action", 0.0);
        algo.train(&spans).await.unwrap();
        
        assert_eq!(algo.failures.len(), 0, "Zero rewards should not add failures");
    }

    // ============ OPTIMIZATION TESTS ============

    #[tokio::test]
    async fn test_optimization_triggers_at_batch_size() {
        let backend = MockLlmBackend::new("IMPROVED PROMPT");
        let mut algo = ApoAlgorithm::new("original".to_string(), backend.clone());
        algo.batch_size = 2; // Lower threshold for testing
        
        // Add failures one by one
        let spans1 = make_failure("input1", "bad1", -1.0);
        let result1 = algo.train(&spans1).await.unwrap();
        assert!(result1.is_none(), "Should not optimize with 1 failure");
        
        let spans2 = make_failure("input2", "bad2", -0.5);
        let result2 = algo.train(&spans2).await.unwrap();
        assert!(result2.is_some(), "Should optimize after reaching batch_size");
        
        // Verify prompt was updated
        assert_eq!(algo.current_prompt, "IMPROVED PROMPT");
        assert_eq!(backend.get_call_count(), 1);
    }

    #[tokio::test]
    async fn test_failures_cleared_after_optimization() {
        let backend = MockLlmBackend::new("new prompt");
        let mut algo = ApoAlgorithm::new("original".to_string(), backend);
        algo.batch_size = 1;
        
        let spans = make_failure("input", "bad", -1.0);
        algo.train(&spans).await.unwrap();
        
        assert!(algo.failures.is_empty(), "Failures should be cleared after optimization");
    }

    #[tokio::test]
    async fn test_optimization_result_contains_new_weights() {
        let backend = MockLlmBackend::new("new prompt content");
        let mut algo = ApoAlgorithm::new("original".to_string(), backend);
        algo.batch_size = 1;
        
        let spans = make_failure("input", "bad", -1.0);
        let result = algo.train(&spans).await.unwrap().unwrap();
        
        assert!(result.updated_weights.is_some());
        let weights = result.updated_weights.unwrap();
        assert_eq!(String::from_utf8(weights).unwrap(), "new prompt content");
    }

    // ============ PROMPT CONSTRUCTION TESTS ============

    #[test]
    fn test_optimization_prompt_includes_current_prompt() {
        let backend = MockLlmBackend::new("improved");
        let algo = ApoAlgorithm::new("You are a helpful assistant.".to_string(), backend);
        
        let traces = vec![InteractionTrace {
            input: "test input".to_string(),
            action: "test action".to_string(),
            reward: -1.0,
        }];
        
        let prompt = algo.construct_optimization_prompt(&traces);
        assert!(prompt.contains("You are a helpful assistant."));
    }

    #[test]
    fn test_optimization_prompt_includes_failure_details() {
        let backend = MockLlmBackend::new("improved");
        let algo = ApoAlgorithm::new("prompt".to_string(), backend);
        
        let traces = vec![
            InteractionTrace {
                input: "user said hello".to_string(),
                action: "responded rudely".to_string(),
                reward: -2.0,
            },
        ];
        
        let prompt = algo.construct_optimization_prompt(&traces);
        assert!(prompt.contains("user said hello"));
        assert!(prompt.contains("responded rudely"));
        assert!(prompt.contains("-2"));
    }

    // ============ UPDATE POLICY TESTS ============

    #[test]
    fn test_update_policy_sets_prompt() {
        let backend = MockLlmBackend::new("x");
        let mut algo = ApoAlgorithm::new("old".to_string(), backend);
        
        algo.update_policy(b"new prompt from weights").unwrap();
        assert_eq!(algo.current_prompt, "new prompt from weights");
    }

    #[test]
    fn test_update_policy_rejects_invalid_utf8() {
        let backend = MockLlmBackend::new("x");
        let mut algo = ApoAlgorithm::new("prompt".to_string(), backend);
        
        let invalid_utf8 = vec![0xff, 0xfe, 0x00, 0x01];
        let result = algo.update_policy(&invalid_utf8);
        
        assert!(result.is_err());
    }

    // ============ EDGE CASE TESTS ============

    #[tokio::test]
    async fn test_empty_spans_returns_none() {
        let backend = MockLlmBackend::new("improved");
        let mut algo = ApoAlgorithm::new("prompt".to_string(), backend);
        
        let result = algo.train(&[]).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_spans_without_complete_trace_ignored() {
        let backend = MockLlmBackend::new("improved");
        let mut algo = ApoAlgorithm::new("prompt".to_string(), backend);
        
        // Only observation, no action/reward pair
        let spans = vec![
            Span::Observation(ObservationSpan::new(json!({ "text": "hello" }))),
            Span::Reward(RewardSpan::new(-1.0)), // Reward without preceding action
        ];
        algo.train(&spans).await.unwrap();
        
        assert!(algo.failures.is_empty(), "Incomplete traces should not add failures");
    }
}

