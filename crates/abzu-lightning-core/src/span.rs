//! Span types for capturing agent interactions

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

/// Main span enum representing different types of agent interactions
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Span {
    Observation(ObservationSpan),
    Action(ActionSpan),
    Reward(RewardSpan),
}

impl Span {
    /// Get the timestamp of this span
    pub fn timestamp(&self) -> DateTime<Utc> {
        match self {
            Span::Observation(s) => s.timestamp,
            Span::Action(s) => s.timestamp,
            Span::Reward(s) => s.timestamp,
        }
    }

    /// Get the span ID
    pub fn id(&self) -> Uuid {
        match self {
            Span::Observation(s) => s.id,
            Span::Action(s) => s.id,
            Span::Reward(s) => s.id,
        }
    }

    /// Get the task ID if present
    pub fn task_id(&self) -> Option<&str> {
        match self {
            Span::Observation(s) => s.task_id.as_deref(),
            Span::Action(s) => s.task_id.as_deref(),
            Span::Reward(s) => s.task_id.as_deref(),
        }
    }

    /// Get the agent ID if present
    pub fn agent_id(&self) -> Option<&str> {
        match self {
            Span::Observation(s) => s.agent_id.as_deref(),
            Span::Action(s) => s.agent_id.as_deref(),
            Span::Reward(s) => s.agent_id.as_deref(),
        }
    }
}

/// Observation span - captures environment state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationSpan {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub task_id: Option<String>,
    pub agent_id: Option<String>,
    
    /// Observation data (flexible JSON structure)
    pub data: Value,
    
    /// Optional metadata
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

impl ObservationSpan {
    /// Create a new observation span
    pub fn new(data: Value) -> Self {
        Self {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            task_id: None,
            agent_id: None,
            data,
            metadata: None,
        }
    }

    /// Builder: Set task ID
    pub fn with_task(mut self, task_id: impl Into<String>) -> Self {
        self.task_id = Some(task_id.into());
        self
    }

    /// Builder: Set agent ID
    pub fn with_agent(mut self, agent_id: impl Into<String>) -> Self {
        self.agent_id = Some(agent_id.into());
        self
    }

    /// Builder: Set metadata
    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

/// Action span - captures agent decision/action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionSpan {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub task_id: Option<String>,
    pub agent_id: Option<String>,
    
    /// Action data (flexible JSON structure)
    pub data: Value,
    
    /// Optional metadata
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

impl ActionSpan {
    /// Create a new action span
    pub fn new(data: Value) -> Self {
        Self {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            task_id: None,
            agent_id: None,
            data,
            metadata: None,
        }
    }

    /// Builder: Set task ID
    pub fn with_task(mut self, task_id: impl Into<String>) -> Self {
        self.task_id = Some(task_id.into());
        self
    }

    /// Builder: Set agent ID
    pub fn with_agent(mut self, agent_id: impl Into<String>) -> Self {
        self.agent_id = Some(agent_id.into());
        self
    }

    /// Builder: Set metadata
    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

/// Reward span - captures feedback signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardSpan {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub task_id: Option<String>,
    pub agent_id: Option<String>,
    
    /// Reward value (typically -1.0 to 1.0 or similar range)
    pub reward: f64,
    
    /// Optional reward source/reason
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    
    /// Optional metadata
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

impl RewardSpan {
    /// Create a new reward span
    pub fn new(reward: f64) -> Self {
        Self {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            task_id: None,
            agent_id: None,
            reward,
            source: None,
            metadata: None,
        }
    }

    /// Builder: Set task ID
    pub fn with_task(mut self, task_id: impl Into<String>) -> Self {
        self.task_id = Some(task_id.into());
        self
    }

    /// Builder: Set agent ID
    pub fn with_agent(mut self, agent_id: impl Into<String>) -> Self {
        self.agent_id = Some(agent_id.into());
        self
    }

    /// Builder: Set reward source
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Builder: Set metadata
    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_observation_span_creation() {
        let span = ObservationSpan::new(json!({"state": "ready"}))
            .with_task("task-123")
            .with_agent("agent-1");

        assert_eq!(span.task_id, Some("task-123".to_string()));
        assert_eq!(span.agent_id, Some("agent-1".to_string()));
        assert_eq!(span.data, json!({"state": "ready"}));
    }

    #[test]
    fn test_action_span_creation() {
        let span = ActionSpan::new(json!({"action": "move_forward"}))
            .with_task("task-123")
            .with_agent("agent-1");

        assert_eq!(span.task_id, Some("task-123".to_string()));
        assert_eq!(span.agent_id, Some("agent-1".to_string()));
    }

    #[test]
    fn test_reward_span_creation() {
        let span = RewardSpan::new(1.0)
            .with_task("task-123")
            .with_agent("agent-1")
            .with_source("success");

        assert_eq!(span.reward, 1.0);
        assert_eq!(span.source, Some("success".to_string()));
    }

    #[test]
    fn test_span_serialization() {
        let obs = Span::Observation(ObservationSpan::new(json!({"test": true})));
        let json = serde_json::to_string(&obs).unwrap();
        let deserialized: Span = serde_json::from_str(&json).unwrap();
        
        assert!(matches!(deserialized, Span::Observation(_)));
    }
}
