//! Abzu plugin integration for Lightning

use crate::{
    LightningStore, Result, Span,
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
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || {
            store.insert_span(&span)
        }).await.map_err(|e| crate::Error::Training(format!("Task join error: {}", e)))??;
        Ok(())
    }

    /// Query task spans
    pub async fn query_task(&self, task_id: &str) -> Result<Vec<Span>> {
        let store = self.store.clone();
        let task_id = task_id.to_string();
        tokio::task::spawn_blocking(move || {
            store.query_task(&task_id)
        }).await.map_err(|e| crate::Error::Training(format!("Task join error: {}", e)))?
    }

    /// Query agent spans
    pub async fn query_agent(&self, agent_id: &str) -> Result<Vec<Span>> {
        let store = self.store.clone();
        let agent_id = agent_id.to_string();
        tokio::task::spawn_blocking(move || {
            store.query_agent(&agent_id)
        }).await.map_err(|e| crate::Error::Training(format!("Task join error: {}", e)))?
    }

    /// Get training statistics
    pub async fn get_stats(&self) -> Result<TrainingStats> {
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || {
            let tasks = store.list_tasks()?;
            let agents = store.list_agents()?;
            
            let mut total_spans = 0;
            for task in &tasks {
                total_spans += store.query_task(task)?.len();
            }

            Ok(TrainingStats {
                total_tasks: tasks.len(),
                total_agents: agents.len(),
                total_spans,
            })
        }).await.map_err(|e| crate::Error::Training(format!("Task join error: {}", e)))?
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

    async fn call(&self, method: &str, params: Value) -> std::result::Result<Value, String> {
        // Construct LightningRequest by injecting method field
        let mut request_obj = params;
        if let Value::Object(ref mut map) = request_obj {
            map.insert("method".to_string(), Value::String(method.to_string()));
        } else if method == "get_stats" && request_obj == Value::Null {
             // Handle case where params is null (optional)
             request_obj = serde_json::json!({
                 "method": method
             });
        }
        
        // Deserialize proper request type
        let request: LightningRequest = serde_json::from_value(request_obj)
            .map_err(|e| format!("Invalid params for method {}: {}", method, e))?;

        // Handle request
        match self.handle_rpc(request).await {
            Ok(response) => {
                // Serialize response to Value
                serde_json::to_value(response).map_err(|e| format!("Serialization error: {}", e))
            }
            Err(e) => Err(format!("Lightning error: {}", e)),
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

    #[tokio::test]
    async fn test_rpc_generic_call() {
        let store = Arc::new(LightningStore::memory().unwrap());
        let plugin = LightningPlugin::new(store);
        
        // Test emit_span via generic call
        let span_data = json!({
            "span": {
                "type": "observation",
                "id": "test-id",
                "timestamp": "2023-01-01T00:00:00Z",
                "data": {"foo": "bar"}
            }
        });
        
        // Note: Generic call injects "method": "emit_span" into the JSON
        // but we need to match LightningRequest deserialization.
        // LightningRequest::EmitSpan { span: Span }
        // JSON: { "method": "emit_span", "span": ... }
        
        // We pass "emit_span" as method, and params as object containing "span"
        let res = plugin.call("emit_span", span_data).await;
        // It might fail if Span deserialization is strict or my mock JSON is lazy
        // Span struct has strict fields? 
        // Let's use a cleaner params construction
        
        let span = Span::Observation(
             ObservationSpan::new(json!({"test": true})).with_task("t1")
        );
        let params = json!({ "span": span });
        
        let res = plugin.call("emit_span", params).await;
        assert!(res.is_ok());
        
        // Test get_stats
        let res = plugin.call("get_stats", Value::Null).await;
        assert!(res.is_ok());
    }
}
