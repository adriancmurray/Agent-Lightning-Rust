//! Lightning Store - persistent storage for spans and resources

use crate::{Result, Span};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Lightning Store for persisting spans and training resources
pub struct LightningStore {
    db: sled::Db,
}

impl LightningStore {
    /// Open or create a Lightning store at the specified path
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let db = sled::open(path)?;
        Ok(Self { db })
    }

    /// Open an in-memory store (for testing)
    pub fn memory() -> Result<Self> {
        let config = sled::Config::new().temporary(true);
        let db = config.open()?;
        Ok(Self { db })
    }

    /// Insert a span into the store
    pub fn insert_span(&self, span: &Span) -> Result<()> {
        // Generate key: timestamp_spanid
        // Use big-endian for correct lex sorting of timestamps if using raw bytes, 
        // but string format is also sortable if fixed width. 
        // Using nanoseconds ensures high precision.
        let ts = span.timestamp().timestamp_nanos_opt().unwrap_or(0);
        let key = format!("{:019}_{}", ts, span.id());

        // Serialize span using serde_json because bincode struggles with serde_json::Value
        let value = serde_json::to_vec(&span)?;

        // Insert into main spans tree
        self.db.insert(key.as_bytes(), value)?;

        // Index by task_id if present
        if let Some(task_id) = span.task_id() {
            let task_tree = self.db.open_tree(format!("task:{}", task_id))?;
            // Use time-ordered key for index
            task_tree.insert(key.as_bytes(), key.as_bytes())?;
        }

        // Index by agent_id if present
        if let Some(agent_id) = span.agent_id() {
            let agent_tree = self.db.open_tree(format!("agent:{}", agent_id))?;
            // Use time-ordered key for index
            agent_tree.insert(key.as_bytes(), key.as_bytes())?;
        }

        Ok(())
    }

    /// Query all spans for a specific task
    pub fn query_task(&self, task_id: &str) -> Result<Vec<Span>> {
        self.query_task_since(task_id, None, usize::MAX)
    }

    /// Query spans for a task since a specific cursor (timestamp, uuid)
    pub fn query_task_since(
        &self, 
        task_id: &str, 
        cursor: Option<(DateTime<Utc>, uuid::Uuid)>,
        limit: usize,
    ) -> Result<Vec<Span>> {
        let task_tree = self.db.open_tree(format!("task:{}", task_id))?;
        let mut spans = Vec::new();

        let start_key = if let Some((dt, id)) = cursor {
            let ts = dt.timestamp_nanos_opt().unwrap_or(0);
            // Start strictly after the cursor: append a byte larger than any valid key suffix
            format!("{:019}_{}\0", ts, id) 
        } else {
            "".to_string()
        };

        for item in task_tree.range(start_key.as_bytes()..).take(limit) {
            let (_index_key, main_key) = item?;
            if let Some(span_data) = self.db.get(&main_key)? {
                let span: Span = serde_json::from_slice(&span_data)?;
                spans.push(span);
            }
        }

        Ok(spans)
    }

    /// Query all spans for a specific agent
    pub fn query_agent(&self, agent_id: &str) -> Result<Vec<Span>> {
        let agent_tree = self.db.open_tree(format!("agent:{}", agent_id))?;
        let mut spans = Vec::new();

        for item in agent_tree.iter() {
            let (_index_key, main_key) = item?;
            if let Some(span_data) = self.db.get(&main_key)? {
                let span: Span = serde_json::from_slice(&span_data)?;
                spans.push(span);
            }
        }

        Ok(spans)
    }

    /// Query spans within a time range
    pub fn query_time_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<Span>> {
        let start_ts = start.timestamp_nanos_opt().unwrap_or(0);
        let end_ts = end.timestamp_nanos_opt().unwrap_or(i64::MAX);
        
        // Keys are formatted as {:019}_{uuid}
        let start_key = format!("{:019}_", start_ts);
        let end_key = format!("{:019}_\x7f", end_ts); // \x7f is higher than any uuid char

        let mut spans = Vec::new();

        for item in self.db.range(start_key.as_bytes()..end_key.as_bytes()) {
            let (_key, value) = item?;
            let span: Span = serde_json::from_slice(&value)?;
            spans.push(span);
        }

        Ok(spans)
    }

    /// Get all tasks with recorded spans
    pub fn list_tasks(&self) -> Result<Vec<String>> {
        let mut tasks = Vec::new();
        
        for name in self.db.tree_names() {
            let name_str = String::from_utf8_lossy(&name);
            if let Some(task_id) = name_str.strip_prefix("task:") {
                tasks.push(task_id.to_string());
            }
        }

        Ok(tasks)
    }

    /// Get all agents with recorded spans
    pub fn list_agents(&self) -> Result<Vec<String>> {
        let mut agents = Vec::new();
        
        for name in self.db.tree_names() {
            let name_str = String::from_utf8_lossy(&name);
            if let Some(agent_id) = name_str.strip_prefix("agent:") {
                agents.push(agent_id.to_string());
            }
        }

        Ok(agents)
    }

    /// Store a resource (e.g., model weights, config)
    pub fn store_resource(&self, key: &str, data: &[u8]) -> Result<()> {
        let resource_tree = self.db.open_tree("resources")?;
        resource_tree.insert(key.as_bytes(), data)?;
        Ok(())
    }

    /// Load a resource
    pub fn load_resource(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let resource_tree = self.db.open_tree("resources")?;
        Ok(resource_tree.get(key.as_bytes())?.map(|v| v.to_vec()))
    }

    /// List all resource keys
    pub fn list_resources(&self) -> Result<Vec<String>> {
        let resource_tree = self.db.open_tree("resources")?;
        let mut keys = Vec::new();
        
        for item in resource_tree.iter() {
            let (key, _) = item?;
            keys.push(String::from_utf8_lossy(&key).to_string());
        }

        Ok(keys)
    }

    /// Store training metadata
    pub fn store_metadata(&self, key: &str, metadata: &TrainingMetadata) -> Result<()> {
        let meta_tree = self.db.open_tree("metadata")?;
        let data = serde_json::to_vec(metadata)?;
        meta_tree.insert(key.as_bytes(), data)?;
        Ok(())
    }

    /// Load training metadata
    pub fn load_metadata(&self, key: &str) -> Result<Option<TrainingMetadata>> {
        let meta_tree = self.db.open_tree("metadata")?;
        if let Some(data) = meta_tree.get(key.as_bytes())? {
            let metadata: TrainingMetadata = serde_json::from_slice(&data)?;
            Ok(Some(metadata))
        } else {
            Ok(None)
        }
    }

    /// Flush all pending writes
    pub fn flush(&self) -> Result<usize> {
        Ok(self.db.flush()?)
    }

    /// Get database size in bytes
    pub fn size_on_disk(&self) -> Result<u64> {
        Ok(self.db.size_on_disk()?)
    }
}

/// Training metadata for tracking training runs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetadata {
    pub run_id: String,
    pub task_id: Option<String>,
    pub agent_id: Option<String>,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub total_spans: usize,
    pub metrics: serde_json::Value,
}

impl TrainingMetadata {
    pub fn new(run_id: impl Into<String>) -> Self {
        Self {
            run_id: run_id.into(),
            task_id: None,
            agent_id: None,
            start_time: Utc::now(),
            end_time: None,
            total_spans: 0,
            metrics: serde_json::Value::Object(Default::default()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ObservationSpan, RewardSpan};
    use serde_json::json;

    #[test]
    fn test_store_insert_and_query() {
        let store = LightningStore::memory().unwrap();

        // Insert spans
        let obs1 = Span::Observation(
            ObservationSpan::new(json!({"step": 1}))
                .with_task("test-task")
                .with_agent("agent-1"),
        );
        let obs2 = Span::Observation(
            ObservationSpan::new(json!({"step": 2}))
                .with_task("test-task")
                .with_agent("agent-1"),
        );
        let reward = Span::Reward(
            RewardSpan::new(1.0)
                .with_task("test-task")
                .with_agent("agent-1"),
        );

        store.insert_span(&obs1).unwrap();
        store.insert_span(&obs2).unwrap();
        store.insert_span(&reward).unwrap();

        // Query by task
        let spans = store.query_task("test-task").unwrap();
        assert_eq!(spans.len(), 3);

        // Query by agent
        let spans = store.query_agent("agent-1").unwrap();
        assert_eq!(spans.len(), 3);
    }

    #[test]
    fn test_list_tasks_and_agents() {
        let store = LightningStore::memory().unwrap();

        let span1 = Span::Observation(
            ObservationSpan::new(json!({}))
                .with_task("task-1")
                .with_agent("agent-1"),
        );
        let span2 = Span::Observation(
            ObservationSpan::new(json!({}))
                .with_task("task-2")
                .with_agent("agent-2"),
        );

        store.insert_span(&span1).unwrap();
        store.insert_span(&span2).unwrap();

        let tasks = store.list_tasks().unwrap();
        assert_eq!(tasks.len(), 2);

        let agents = store.list_agents().unwrap();
        assert_eq!(agents.len(), 2);
    }

    #[test]
    fn test_resource_storage() {
        let store = LightningStore::memory().unwrap();

        let data = b"test resource data";
        store.store_resource("model.bin", data).unwrap();

        let loaded = store.load_resource("model.bin").unwrap();
        assert_eq!(loaded, Some(data.to_vec()));

        let resources = store.list_resources().unwrap();
        assert_eq!(resources.len(), 1);
        assert_eq!(resources[0], "model.bin");
    }

    #[test]
    fn test_metadata_storage() {
        let store = LightningStore::memory().unwrap();

        let mut metadata = TrainingMetadata::new("run-123");
        metadata.task_id = Some("task-1".to_string());
        metadata.total_spans = 100;

        store.store_metadata("run-123", &metadata).unwrap();

        let loaded = store.load_metadata("run-123").unwrap();
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().total_spans, 100);
    }
}
