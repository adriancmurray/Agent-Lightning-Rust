//! Abzu plugin integration for Lightning

use crate::{
    LightningStore, Result, Span, SpanCollector, TrainerConfig, 
    algorithm::RewardAggregator,
};
use abzu_runtime::plugins::{
    PluginManifest, PluginProvider, PluginStatus, SettingDefinition, SettingKind, StatusField,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Lightning plugin for Abzu
pub struct LightningPlugin {
    store: Arc<LightningStore>,
    state: Arc<RwLock<PluginState>>,
}

struct PluginState {
    enabled: bool,
    batch_size: usize,
    interval_secs: u64,
}

impl LightningPlugin {
    /// Create a new Lightning plugin with the given store
    pub fn new(store: Arc<LightningStore>) -> Self {
        Self {
            store,
            state: Arc::new(RwLock::new(PluginState {
                enabled: true,
                batch_size: 100,
                interval_secs: 10,
            })),
        }
    }

    /// Emit a span (convenience method for RPC)
    pub async fn emit_span(&self, span: Span) -> Result<()> {
        self.store.insert_span(&span)
    }

    /// Query task spans
    pub async fn query_task(&self, task_id: &str) -> Result<Vec<Span>> {
        self.store.query_task(task_id)
    }

    /// Query agent spans
    pub async fn query_agent(&self, agent_id: &str) -> Result<Vec<Span>> {
        self.store.query_agent(agent_id)
    }

    /// Get training statistics
    pub async fn get_stats(&self) -> Result<TrainingStats> {
        let tasks = self.store.list_tasks()?;
        let agents = self.store.list_agents()?;
        
        let mut total_spans = 0;
        for task in &tasks {
            total_spans += self.store.query_task(task)?.len();
        }

        Ok(TrainingStats {
            total_tasks: tasks.len(),
            total_agents: agents.len(),
            total_spans,
        })
    }
}

#[async_trait]
impl PluginProvider for LightningPlugin {
    fn manifest(&self) -> PluginManifest {
        PluginManifest {
            id: "lightning".to_string(),
            name: "Agent Lightning".to_string(),
            icon: "⚡".to_string(),
            color: "#FFD700".to_string(),
            enabled: true,
            settings: vec![
                SettingDefinition {
                    key: "batch_size".to_string(),
                    label: "Batch Size".to_string(),
                    kind: SettingKind::Slider {
                        min: 10.0,
                        max: 1000.0,
                        step: Some(10.0),
                    },
                    default_value: Some("100".to_string()),
                    current_value: None,
                },
                SettingDefinition {
                    key: "interval_secs".to_string(),
                    label: "Training Interval (seconds)".to_string(),
                    kind: SettingKind::Slider {
                        min: 1.0,
                        max: 300.0,
                        step: Some(1.0),
                    },
                    default_value: Some("10".to_string()),
                    current_value: None,
                },
            ],
            status_fields: vec![
                StatusField {
                    key: "total_tasks".to_string(),
                    label: "Tasks".to_string(),
                    color: None,
                    show_dot: false,
                },
                StatusField {
                    key: "total_spans".to_string(),
                    label: "Spans".to_string(),
                    color: None,
                    show_dot: false,
                },
            ],
        }
    }

    async fn status(&self) -> PluginStatus {
        let stats = self.get_stats().await.unwrap_or_default();
        let mut fields = HashMap::new();
        
        fields.insert("total_tasks".to_string(), serde_json::json!(stats.total_tasks));
        fields.insert("total_agents".to_string(), serde_json::json!(stats.total_agents));
        fields.insert("total_spans".to_string(), serde_json::json!(stats.total_spans));

        PluginStatus {
            plugin_id: "lightning".to_string(),
            healthy: true,
            fields,
        }
    }

    async fn update_setting(&self, key: &str, value: &str) -> std::result::Result<(), String> {
        let mut state = self.state.write().await;
        
        match key {
            "batch_size" => {
                let size = value.parse::<usize>()
                    .map_err(|e| format!("Invalid batch_size: {}", e))?;
                state.batch_size = size;
                Ok(())
            }
            "interval_secs" => {
                let secs = value.parse::<u64>()
                    .map_err(|e| format!("Invalid interval_secs: {}", e))?;
                state.interval_secs = secs;
                Ok(())
            }
            _ => Err(format!("Unknown setting key: {}", key)),
        }
    }

    async fn get_setting(&self, key: &str) -> Option<String> {
        let state = self.state.read().await;
        
        match key {
            "batch_size" => Some(state.batch_size.to_string()),
            "interval_secs" => Some(state.interval_secs.to_string()),
            _ => None,
        }
    }
}

/// Training statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrainingStats {
    pub total_tasks: usize,
    pub total_agents: usize,
    pub total_spans: usize,
}

/// RPC request types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "method", rename_all = "snake_case")]
pub enum LightningRequest {
    EmitSpan { span: Span },
    QueryTask { task_id: String },
    QueryAgent { agent_id: String },
    GetStats,
}

/// RPC response types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum LightningResponse {
    Success,
    Spans(Vec<Span>),
    Stats(TrainingStats),
}

impl LightningPlugin {
    /// Handle an RPC request
    pub async fn handle_rpc(&self, request: LightningRequest) -> Result<LightningResponse> {
        match request {
            LightningRequest::EmitSpan { span } => {
                self.emit_span(span).await?;
                Ok(LightningResponse::Success)
            }
            LightningRequest::QueryTask { task_id } => {
                let spans = self.query_task(&task_id).await?;
                Ok(LightningResponse::Spans(spans))
            }
            LightningRequest::QueryAgent { agent_id } => {
                let spans = self.query_agent(&agent_id).await?;
                Ok(LightningResponse::Spans(spans))
            }
            LightningRequest::GetStats => {
                let stats = self.get_stats().await?;
                Ok(LightningResponse::Stats(stats))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ObservationSpan;
    use serde_json::json;

    #[tokio::test]
    async fn test_plugin_manifest() {
        let store = Arc::new(LightningStore::memory().unwrap());
        let plugin = LightningPlugin::new(store);
        
        let manifest = plugin.manifest();
        assert_eq!(manifest.id, "lightning");
        assert_eq!(manifest.name, "Agent Lightning");
        assert!(manifest.enabled);
    }

    #[tokio::test]
    async fn test_plugin_settings() {
        let store = Arc::new(LightningStore::memory().unwrap());
        let plugin = LightningPlugin::new(store);
        
        plugin.update_setting("batch_size", "200").await.unwrap();
        let value = plugin.get_setting("batch_size").await;
        assert_eq!(value, Some("200".to_string()));
    }

    #[tokio::test]
    async fn test_rpc_emit_and_query() {
        let store = Arc::new(LightningStore::memory().unwrap());
        let plugin = LightningPlugin::new(store);
        
        let span = Span::Observation(
            ObservationSpan::new(json!({"test": true}))
                .with_task("test-task")
        );
        
        let req = LightningRequest::EmitSpan { span };
        let res = plugin.handle_rpc(req).await.unwrap();
        assert!(matches!(res, LightningResponse::Success));
        
        let req = LightningRequest::QueryTask { task_id: "test-task".to_string() };
        let res = plugin.handle_rpc(req).await.unwrap();
        assert!(matches!(res, LightningResponse::Spans(_)));
    }
}
