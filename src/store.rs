use std::cmp::Ordering;
use std::env;
use std::fs;
use std::io::{Error as IoError, ErrorKind};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use chrono::{DateTime, Utc};
use rusqlite::{Connection, params};
use serde::Serialize;

use crate::embedding::{blend, cosine_similarity, decode_embedding, embed_text, encode_embedding};
use crate::expiration::{fingerprint_for_condition, is_expired, validate_expiration};
use crate::model::{ExpirationCondition, MemoryMode, normalize_tags};

const SCHEMA_VERSION: i64 = 2;
const SIMILAR_MEMORY_THRESHOLD: f32 = 0.72;

pub struct MemoryStore {
    connection: Connection,
}

#[derive(Debug, Clone)]
pub struct SetMemory {
    pub content: String,
    pub mode: MemoryMode,
    pub mode_ref: Option<String>,
    pub tags: Vec<String>,
    pub expiration_condition: Option<ExpirationCondition>,
    pub expiration_value: Option<String>,
    pub metadata: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SearchOptions {
    pub query: String,
    pub positive_tags: Vec<String>,
    pub negative_tags: Vec<String>,
    pub limit: usize,
    pub offset: usize,
    pub mode: Option<MemoryMode>,
    pub mode_ref: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Alert {
    pub session_ref: String,
    pub content: String,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            query: String::new(),
            positive_tags: Vec::new(),
            negative_tags: Vec::new(),
            limit: 10,
            offset: 0,
            mode: None,
            mode_ref: None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct MemorySearchResult {
    pub id: i64,
    pub content: String,
    pub mode: MemoryMode,
    pub mode_ref: Option<String>,
    pub tags: Vec<String>,
    pub score: f32,
    pub positive_score: f32,
    pub negative_score: f32,
    pub usage_count: i64,
    pub metadata: Option<String>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TagSummary {
    pub tag: String,
    pub count: i64,
}

impl MemoryStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent).with_context(|| {
                format!("failed to create database directory {}", parent.display())
            })?;
        }

        let connection = Connection::open(path)
            .with_context(|| format!("failed to open database {}", path.display()))?;
        Self::from_connection(connection)
    }

    pub fn in_memory() -> Result<Self> {
        Self::from_connection(Connection::open_in_memory()?)
    }

    fn from_connection(connection: Connection) -> Result<Self> {
        let mut store = Self { connection };
        store.migrate()?;
        Ok(store)
    }

    pub fn set(&mut self, input: SetMemory) -> Result<i64> {
        let input = normalize_set_memory(input)?;
        let now = Utc::now();
        let created_at = now.to_rfc3339();
        let content_embedding = embed_text(&input.content);
        let tag_text = input.tags.join(" ");
        let tag_embedding = embed_text(&tag_text);
        let combined_embedding = blend(&content_embedding, &tag_embedding);
        let file_fingerprint = fingerprint_for_condition(
            input.expiration_condition,
            input.expiration_value.as_deref(),
        )?;
        let related_updates = self.similar_memory_updates(&combined_embedding, now)?;

        let transaction = self.connection.transaction()?;
        transaction.execute(
            "INSERT INTO memories (
                content, mode, mode_ref, tags_json, expiration_condition, expiration_value,
                metadata, content_embedding, tag_embedding, combined_embedding,
                positive_score, negative_score, usage_count, created_at, updated_at,
                file_fingerprint
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, 0.0, 0.0, 0, ?11, ?11, ?12)",
            params![
                input.content,
                input.mode.as_str(),
                input.mode_ref,
                serde_json::to_string(&input.tags)?,
                input.expiration_condition.map(ExpirationCondition::as_str),
                input.expiration_value,
                input.metadata,
                encode_embedding(&content_embedding),
                encode_embedding(&tag_embedding),
                encode_embedding(&combined_embedding),
                created_at,
                file_fingerprint,
            ],
        )?;
        let id = transaction.last_insert_rowid();

        for tag in input.tags {
            transaction.execute(
                "INSERT OR IGNORE INTO memory_tags (memory_id, tag) VALUES (?1, ?2)",
                params![id, tag],
            )?;
        }

        for (memory_id, penalty) in related_updates {
            transaction.execute(
                "UPDATE memories
                 SET negative_score = negative_score + ?1, updated_at = ?2
                 WHERE id = ?3",
                params![penalty, created_at, memory_id],
            )?;
        }

        transaction.commit()?;
        Ok(id)
    }

    pub fn get(&mut self, options: SearchOptions) -> Result<Vec<MemorySearchResult>> {
        let options = normalize_search_options(options);
        let now = Utc::now();
        let query_embedding = embed_text(&options.query);
        let query_lower = options.query.to_ascii_lowercase();
        let mut scored = Vec::new();

        for memory in self.load_memories()? {
            if !memory.matches_scope(options.mode, options.mode_ref.as_deref()) {
                continue;
            }

            if memory.is_expired(now) {
                continue;
            }

            if !options
                .positive_tags
                .iter()
                .all(|tag| memory.tags.iter().any(|memory_tag| memory_tag == tag))
            {
                continue;
            }

            let score = score_memory(&memory, &query_embedding, &query_lower, &options);
            scored.push((memory, score));
        }

        scored.sort_by(|(left_memory, left_score), (right_memory, right_score)| {
            right_score
                .partial_cmp(left_score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| right_memory.id.cmp(&left_memory.id))
        });

        let returned = scored
            .into_iter()
            .skip(options.offset)
            .take(options.limit)
            .enumerate()
            .map(|(rank, (memory, score))| (rank, memory, score))
            .collect::<Vec<_>>();

        self.record_retrievals(&returned, now)?;

        Ok(returned
            .into_iter()
            .map(|(_, memory, score)| MemorySearchResult {
                id: memory.id,
                content: memory.content,
                mode: memory.mode,
                mode_ref: memory.mode_ref,
                tags: memory.tags,
                score,
                positive_score: memory.positive_score,
                negative_score: memory.negative_score,
                usage_count: memory.usage_count,
                metadata: memory.metadata,
                created_at: memory.created_at,
            })
            .collect())
    }

    pub fn list_tags(&self, filter: Option<&str>) -> Result<Vec<TagSummary>> {
        let now = Utc::now();
        let mut summaries = std::collections::BTreeMap::<String, i64>::new();
        let filter = filter.map(str::trim).filter(|filter| !filter.is_empty());
        let filter_lower = filter.map(str::to_ascii_lowercase);
        let filter_embedding = filter.map(embed_text);

        for memory in self.load_memories()? {
            if memory.is_expired(now) {
                continue;
            }

            for tag in memory.tags {
                if let Some(filter_lower) = &filter_lower {
                    let tag_matches_text = tag.contains(filter_lower);
                    let tag_matches_embedding = filter_embedding
                        .as_ref()
                        .map(|filter_embedding| {
                            cosine_similarity(filter_embedding, &embed_text(&tag)) >= 0.2
                        })
                        .unwrap_or(false);

                    if !tag_matches_text && !tag_matches_embedding {
                        continue;
                    }
                }

                *summaries.entry(tag).or_default() += 1;
            }
        }

        Ok(summaries
            .into_iter()
            .map(|(tag, count)| TagSummary { tag, count })
            .collect())
    }

    pub fn set_alert(
        &mut self,
        session_ref: impl Into<String>,
        content: impl Into<String>,
    ) -> Result<i64> {
        let session_ref = normalize_required_text(session_ref.into(), "alert session_ref")?;
        let content = normalize_required_text(content.into(), "alert content")?;

        self.connection.execute(
            "INSERT INTO alerts (session_ref, content) VALUES (?1, ?2)",
            params![session_ref, content],
        )?;

        Ok(self.connection.last_insert_rowid())
    }

    pub fn get_alerts(&mut self, session_ref: impl Into<String>) -> Result<Vec<Alert>> {
        let session_ref = normalize_required_text(session_ref.into(), "alert session_ref")?;
        let transaction = self.connection.transaction()?;
        let alerts = {
            let mut statement = transaction.prepare(
                "SELECT session_ref, content FROM alerts WHERE session_ref = ?1 ORDER BY id",
            )?;
            let rows = statement.query_map(params![session_ref], |row| {
                Ok(Alert {
                    session_ref: row.get(0)?,
                    content: row.get(1)?,
                })
            })?;

            rows.collect::<Result<Vec<_>, _>>()?
        };

        transaction.execute(
            "DELETE FROM alerts WHERE session_ref = ?1",
            params![session_ref],
        )?;
        transaction.commit()?;
        Ok(alerts)
    }

    fn migrate(&mut self) -> Result<()> {
        self.connection.pragma_update(None, "foreign_keys", "ON")?;
        let version: i64 = self
            .connection
            .pragma_query_value(None, "user_version", |row| row.get(0))?;

        if version > SCHEMA_VERSION {
            bail!(
                "database schema version {version} is newer than this binary supports ({SCHEMA_VERSION})"
            );
        }

        if version == 0 {
            let transaction = self.connection.transaction()?;
            transaction.execute_batch(
                "CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    mode_ref TEXT,
                    tags_json TEXT NOT NULL,
                    expiration_condition TEXT,
                    expiration_value TEXT,
                    metadata TEXT,
                    content_embedding TEXT NOT NULL,
                    tag_embedding TEXT NOT NULL,
                    combined_embedding TEXT NOT NULL,
                    positive_score REAL NOT NULL DEFAULT 0.0,
                    negative_score REAL NOT NULL DEFAULT 0.0,
                    usage_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    file_fingerprint TEXT
                );

                CREATE TABLE IF NOT EXISTS memory_tags (
                    memory_id INTEGER NOT NULL,
                    tag TEXT NOT NULL,
                    PRIMARY KEY (memory_id, tag),
                    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(mode, mode_ref);
                CREATE INDEX IF NOT EXISTS idx_memory_tags_tag ON memory_tags(tag);
                PRAGMA user_version = 1;",
            )?;
            transaction.commit()?;
        }

        let version: i64 = self
            .connection
            .pragma_query_value(None, "user_version", |row| row.get(0))?;

        if version == 1 {
            let transaction = self.connection.transaction()?;
            transaction.execute_batch(
                "CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_ref TEXT NOT NULL,
                    content TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_alerts_session_ref ON alerts(session_ref, id);
                PRAGMA user_version = 2;",
            )?;
            transaction.commit()?;
        }

        Ok(())
    }

    fn load_memories(&self) -> Result<Vec<MemoryRecord>> {
        let mut statement = self.connection.prepare(
            "SELECT
                id, content, mode, mode_ref, tags_json, expiration_condition, expiration_value,
                metadata, combined_embedding, positive_score, negative_score, usage_count,
                created_at, file_fingerprint
             FROM memories",
        )?;

        let rows = statement.query_map([], |row| {
            let mode: String = row.get(2)?;
            let expiration_condition: Option<String> = row.get(5)?;
            let created_at: String = row.get(12)?;

            Ok(MemoryRecord {
                id: row.get(0)?,
                content: row.get(1)?,
                mode: mode
                    .parse()
                    .map_err(|error| conversion_error(error, "mode"))?,
                mode_ref: row.get(3)?,
                tags: serde_json::from_str::<Vec<String>>(&row.get::<_, String>(4)?)
                    .unwrap_or_default(),
                expiration_condition: expiration_condition
                    .as_deref()
                    .map(str::parse::<ExpirationCondition>)
                    .transpose()
                    .map_err(|error| conversion_error(error, "expiration_condition"))?,
                expiration_value: row.get(6)?,
                metadata: row.get(7)?,
                combined_embedding: decode_embedding(&row.get::<_, String>(8)?),
                positive_score: row.get(9)?,
                negative_score: row.get(10)?,
                usage_count: row.get(11)?,
                created_at: DateTime::parse_from_rfc3339(&created_at)
                    .map(|datetime| datetime.with_timezone(&Utc))
                    .map_err(|error| rusqlite::Error::ToSqlConversionFailure(Box::new(error)))?,
                file_fingerprint: row.get(13)?,
            })
        })?;

        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    fn similar_memory_updates(
        &self,
        combined_embedding: &[f32],
        now: DateTime<Utc>,
    ) -> Result<Vec<(i64, f32)>> {
        let mut updates = Vec::new();

        for memory in self.load_memories()? {
            if memory.is_expired(now) {
                continue;
            }

            let similarity = cosine_similarity(combined_embedding, &memory.combined_embedding);
            if similarity >= SIMILAR_MEMORY_THRESHOLD {
                updates.push((memory.id, similarity));
            }
        }

        Ok(updates)
    }

    fn record_retrievals(
        &mut self,
        returned: &[(usize, MemoryRecord, f32)],
        now: DateTime<Utc>,
    ) -> Result<()> {
        if returned.is_empty() {
            return Ok(());
        }

        let updated_at = now.to_rfc3339();
        let transaction = self.connection.transaction()?;
        for (rank, memory, _) in returned {
            let gain = 1.0_f32 / (*rank as f32 + 1.0);
            transaction.execute(
                "UPDATE memories
                 SET positive_score = positive_score + ?1,
                     usage_count = usage_count + 1,
                     updated_at = ?2
                 WHERE id = ?3",
                params![gain, updated_at, memory.id],
            )?;
        }
        transaction.commit()?;
        Ok(())
    }
}

fn conversion_error(error: anyhow::Error, field: &'static str) -> rusqlite::Error {
    rusqlite::Error::ToSqlConversionFailure(Box::new(IoError::new(
        ErrorKind::InvalidData,
        format!("invalid {field}: {error}"),
    )))
}

pub fn default_database_path() -> PathBuf {
    PathBuf::from(".mii-memory.db")
}

pub fn infer_mode_ref(mode: MemoryMode, explicit: Option<String>) -> Result<Option<String>> {
    if mode == MemoryMode::Global {
        return Ok(None);
    }

    if let Some(explicit) = explicit
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
    {
        return Ok(Some(explicit));
    }

    match mode {
        MemoryMode::Global => Ok(None),
        MemoryMode::Workspace => Ok(Some(
            env::current_dir()
                .context("failed to infer workspace mode_ref from current directory")?
                .to_string_lossy()
                .into_owned(),
        )),
        MemoryMode::Session => Ok(Some(
            env::var("MII_MEMORY_SESSION")
                .or_else(|_| env::var("MCP_SESSION_ID"))
                .unwrap_or_else(|_| "default".to_string()),
        )),
    }
}

fn normalize_set_memory(mut input: SetMemory) -> Result<SetMemory> {
    input.content = input.content.trim().to_string();
    if input.content.is_empty() {
        bail!("memory content cannot be empty");
    }

    input.tags = normalize_tags(&input.tags);
    if input.tags.is_empty() {
        bail!("at least one tag is required");
    }

    input.mode_ref = infer_mode_ref(input.mode, input.mode_ref)?;

    match (
        input.expiration_condition,
        input.expiration_value.as_deref(),
    ) {
        (Some(condition), Some(value)) => validate_expiration(condition, value)?,
        (Some(condition), None) => bail!("expiration condition {condition} requires a value"),
        (None, Some(_)) => bail!("expiration value was provided without an expiration condition"),
        (None, None) => {}
    }

    Ok(input)
}

fn normalize_required_text(mut value: String, field: &'static str) -> Result<String> {
    value = value.trim().to_string();
    if value.is_empty() {
        bail!("{field} cannot be empty");
    }

    Ok(value)
}

fn normalize_search_options(mut options: SearchOptions) -> SearchOptions {
    options.query = options.query.trim().to_string();
    options.positive_tags = normalize_tags(&options.positive_tags);
    options.negative_tags = normalize_tags(&options.negative_tags);
    options.limit = options.limit.max(1);
    options
}

fn score_memory(
    memory: &MemoryRecord,
    query_embedding: &[f32],
    query_lower: &str,
    options: &SearchOptions,
) -> f32 {
    let semantic = cosine_similarity(query_embedding, &memory.combined_embedding) * 10.0;
    let content_lower = memory.content.to_ascii_lowercase();
    let text_bonus = if !query_lower.is_empty() && content_lower.contains(query_lower) {
        2.0
    } else {
        0.0
    };
    let tag_text_bonus =
        if !query_lower.is_empty() && memory.tags.iter().any(|tag| tag.contains(query_lower)) {
            1.0
        } else {
            0.0
        };
    let positive_tag_bonus = options.positive_tags.len() as f32 * 0.35;
    let negative_tag_penalty = options
        .negative_tags
        .iter()
        .filter(|negative_tag| memory.tags.iter().any(|tag| tag == *negative_tag))
        .count() as f32
        * 4.0;

    semantic + text_bonus + tag_text_bonus + positive_tag_bonus + memory.positive_score
        - memory.negative_score
        - negative_tag_penalty
}

#[derive(Debug, Clone)]
struct MemoryRecord {
    id: i64,
    content: String,
    mode: MemoryMode,
    mode_ref: Option<String>,
    tags: Vec<String>,
    expiration_condition: Option<ExpirationCondition>,
    expiration_value: Option<String>,
    metadata: Option<String>,
    combined_embedding: Vec<f32>,
    positive_score: f32,
    negative_score: f32,
    usage_count: i64,
    created_at: DateTime<Utc>,
    file_fingerprint: Option<String>,
}

impl MemoryRecord {
    fn matches_scope(&self, mode: Option<MemoryMode>, mode_ref: Option<&str>) -> bool {
        if mode.is_some_and(|mode| self.mode != mode) {
            return false;
        }

        if let Some(mode_ref) = mode_ref {
            return self.mode_ref.as_deref() == Some(mode_ref);
        }

        true
    }

    fn is_expired(&self, now: DateTime<Utc>) -> bool {
        is_expired(
            self.expiration_condition,
            self.expiration_value.as_deref(),
            self.created_at,
            self.usage_count,
            self.file_fingerprint.as_deref(),
            now,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn memory(content: &str, tags: &[&str]) -> SetMemory {
        SetMemory {
            content: content.to_string(),
            mode: MemoryMode::Global,
            mode_ref: None,
            tags: tags.iter().map(|tag| tag.to_string()).collect(),
            expiration_condition: None,
            expiration_value: None,
            metadata: None,
        }
    }

    #[test]
    fn set_get_and_list_tags_round_trip() -> Result<()> {
        let mut store = MemoryStore::in_memory()?;
        store.set(memory("Rust sqlite memory backend", &["rust", "sqlite"]))?;

        let results = store.get(SearchOptions {
            query: "sqlite backend".to_string(),
            positive_tags: vec!["rust".to_string()],
            ..SearchOptions::default()
        })?;

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "Rust sqlite memory backend");

        let tags = store.list_tags(Some("sql"))?;
        assert_eq!(tags[0].tag, "sqlite");
        Ok(())
    }

    #[test]
    fn usage_expiration_hides_memory_after_limit() -> Result<()> {
        let mut store = MemoryStore::in_memory()?;
        let mut input = memory("single use memory", &["temporary"]);
        input.expiration_condition = Some(ExpirationCondition::Usage);
        input.expiration_value = Some("1".to_string());
        store.set(input)?;

        let first = store.get(SearchOptions {
            query: "single".to_string(),
            ..SearchOptions::default()
        })?;
        let second = store.get(SearchOptions {
            query: "single".to_string(),
            ..SearchOptions::default()
        })?;

        assert_eq!(first.len(), 1);
        assert!(second.is_empty());
        Ok(())
    }

    #[test]
    fn file_pristine_expiration_tracks_changes() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let file_path = directory.path().join("tracked.txt");
        fs::write(&file_path, "first")?;

        let mut store = MemoryStore::in_memory()?;
        let mut input = memory("tracked file state", &["file"]);
        input.expiration_condition = Some(ExpirationCondition::FilePristine);
        input.expiration_value = Some(file_path.to_string_lossy().into_owned());
        store.set(input)?;

        assert_eq!(
            store
                .get(SearchOptions {
                    query: "tracked".to_string(),
                    ..SearchOptions::default()
                })?
                .len(),
            1
        );

        let mut file = fs::OpenOptions::new().append(true).open(&file_path)?;
        writeln!(file, "changed")?;

        assert!(
            store
                .get(SearchOptions {
                    query: "tracked".to_string(),
                    ..SearchOptions::default()
                })?
                .is_empty()
        );
        Ok(())
    }

    #[test]
    fn alerts_are_session_scoped_and_one_shot() -> Result<()> {
        let mut store = MemoryStore::in_memory()?;
        store.set_alert("session-a", "remember the summary")?;
        store.set_alert("session-b", "other session")?;

        let first = store.get_alerts("session-a")?;
        let second = store.get_alerts("session-a")?;
        let other = store.get_alerts("session-b")?;

        assert_eq!(first.len(), 1);
        assert_eq!(first[0].content, "remember the summary");
        assert!(second.is_empty());
        assert_eq!(other.len(), 1);
        assert_eq!(other[0].content, "other session");
        Ok(())
    }

    #[test]
    fn default_database_path_matches_spec() {
        assert_eq!(default_database_path(), PathBuf::from(".mii-memory.db"));
    }
}
