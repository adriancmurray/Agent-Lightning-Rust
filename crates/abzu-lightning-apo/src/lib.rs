use abzu_lightning_core::{LightningAlgorithm, Result, Error, Span, TrainingResult, LlmBackend};
use async_trait::async_trait;
use std::sync::Arc;
use tracing::{info, warn};

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
        let mut new_failures = 0;

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
                            new_failures += 1;
                        }
                    }
                    // Reset state after reward? Usually yes strictly sequential
                    // self.pending_obs = None;
                    // self.pending_act = None; 
                    // Keeping them assumes simple loop.
                }
                _ => {}
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
