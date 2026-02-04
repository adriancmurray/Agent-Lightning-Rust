//! Span collector trait and implementations

use crate::{ActionSpan, ObservationSpan, RewardSpan, Result, Span};
use serde_json::Value;

/// Trait for collecting and emitting spans during agent execution
pub trait SpanCollector: Send + Sync {
    /// Emit an observation span
    fn emit_observation(&self, data: Value) -> Result<()>;

    /// Emit an action span
    fn emit_action(&self, data: Value) -> Result<()>;

    /// Emit a reward span
    fn emit_reward(&self, reward: f64) -> Result<()>;

    /// Emit a raw span (for advanced usage)
    fn emit_span(&self, span: Span) -> Result<()>;
}

/// In-memory span collector for testing
#[derive(Debug, Default, Clone)]
pub struct MemoryCollector {
    spans: std::sync::Arc<std::sync::Mutex<Vec<Span>>>,
}

impl MemoryCollector {
    /// Create a new memory collector
    pub fn new() -> Self {
        Self {
            spans: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
        }
    }

    /// Get all collected spans
    pub fn get_spans(&self) -> Vec<Span> {
        self.spans.lock().unwrap().clone()
    }

    /// Clear all collected spans
    pub fn clear(&self) {
        self.spans.lock().unwrap().clear();
    }
}

impl SpanCollector for MemoryCollector {
    fn emit_observation(&self, data: Value) -> Result<()> {
        let span = Span::Observation(ObservationSpan::new(data));
        self.emit_span(span)
    }

    fn emit_action(&self, data: Value) -> Result<()> {
        let span = Span::Action(ActionSpan::new(data));
        self.emit_span(span)
    }

    fn emit_reward(&self, reward: f64) -> Result<()> {
        let span = Span::Reward(RewardSpan::new(reward));
        self.emit_span(span)
    }

    fn emit_span(&self, span: Span) -> Result<()> {
        self.spans.lock().unwrap().push(span);
        Ok(())
    }
}

/// Builder for creating context-aware span collectors
#[derive(Debug, Clone)]
pub struct CollectorContext {
    task_id: Option<String>,
    agent_id: Option<String>,
}

impl CollectorContext {
    /// Create a new collector context
    pub fn new() -> Self {
        Self {
            task_id: None,
            agent_id: None,
        }
    }

    /// Set task ID
    pub fn with_task(mut self, task_id: impl Into<String>) -> Self {
        self.task_id = Some(task_id.into());
        self
    }

    /// Set agent ID
    pub fn with_agent(mut self, agent_id: impl Into<String>) -> Self {
        self.agent_id = Some(agent_id.into());
        self
    }

    /// Apply context to a span
    pub fn apply(&self, mut span: Span) -> Span {
        match &mut span {
            Span::Observation(s) => {
                if s.task_id.is_none() {
                    s.task_id = self.task_id.clone();
                }
                if s.agent_id.is_none() {
                    s.agent_id = self.agent_id.clone();
                }
            }
            Span::Action(s) => {
                if s.task_id.is_none() {
                    s.task_id = self.task_id.clone();
                }
                if s.agent_id.is_none() {
                    s.agent_id = self.agent_id.clone();
                }
            }
            Span::Reward(s) => {
                if s.task_id.is_none() {
                    s.task_id = self.task_id.clone();
                }
                if s.agent_id.is_none() {
                    s.agent_id = self.agent_id.clone();
                }
            }
        }
        span
    }
}

impl Default for CollectorContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Contextual span collector that wraps another collector with context
pub struct ContextualCollector<C: SpanCollector> {
    inner: C,
    context: CollectorContext,
}

impl<C: SpanCollector> ContextualCollector<C> {
    /// Create a new contextual collector
    pub fn new(inner: C, context: CollectorContext) -> Self {
        Self { inner, context }
    }
}

impl<C: SpanCollector> SpanCollector for ContextualCollector<C> {
    fn emit_observation(&self, data: Value) -> Result<()> {
        let span = Span::Observation(ObservationSpan::new(data));
        let span = self.context.apply(span);
        self.inner.emit_span(span)
    }

    fn emit_action(&self, data: Value) -> Result<()> {
        let span = Span::Action(ActionSpan::new(data));
        let span = self.context.apply(span);
        self.inner.emit_span(span)
    }

    fn emit_reward(&self, reward: f64) -> Result<()> {
        let span = Span::Reward(RewardSpan::new(reward));
        let span = self.context.apply(span);
        self.inner.emit_span(span)
    }

    fn emit_span(&self, span: Span) -> Result<()> {
        let span = self.context.apply(span);
        self.inner.emit_span(span)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_memory_collector() {
        let collector = MemoryCollector::new();
        
        collector.emit_observation(json!({"test": 1})).unwrap();
        collector.emit_action(json!({"test": 2})).unwrap();
        collector.emit_reward(1.0).unwrap();

        let spans = collector.get_spans();
        assert_eq!(spans.len(), 3);
        assert!(matches!(spans[0], Span::Observation(_)));
        assert!(matches!(spans[1], Span::Action(_)));
        assert!(matches!(spans[2], Span::Reward(_)));
    }

    #[test]
    fn test_contextual_collector() {
        let base = MemoryCollector::new();
        let context = CollectorContext::new()
            .with_task("test-task")
            .with_agent("test-agent");
        let collector = ContextualCollector::new(base.clone(), context);

        collector.emit_observation(json!({"test": true})).unwrap();

        let spans = base.get_spans();
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].task_id(), Some("test-task"));
        assert_eq!(spans[0].agent_id(), Some("test-agent"));
    }
}
