use std::fmt;
use std::str::FromStr;

use anyhow::{anyhow, bail};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
#[value(rename_all = "snake_case")]
pub enum MemoryMode {
    #[default]
    Global,
    Workspace,
    Session,
}

impl MemoryMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Global => "global",
            Self::Workspace => "workspace",
            Self::Session => "session",
        }
    }
}

impl fmt::Display for MemoryMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for MemoryMode {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "global" => Ok(Self::Global),
            "workspace" => Ok(Self::Workspace),
            "session" => Ok(Self::Session),
            other => bail!("unsupported memory mode: {other}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
#[value(rename_all = "snake_case")]
pub enum ExpirationCondition {
    Time,
    Usage,
    FileExist,
    FilePristine,
    Period,
}

impl ExpirationCondition {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Time => "time",
            Self::Usage => "usage",
            Self::FileExist => "file_exist",
            Self::FilePristine => "file_pristine",
            Self::Period => "period",
        }
    }
}

impl fmt::Display for ExpirationCondition {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for ExpirationCondition {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "time" => Ok(Self::Time),
            "usage" => Ok(Self::Usage),
            "file_exist" | "file-exist" => Ok(Self::FileExist),
            "file_pristine" | "file-pristine" => Ok(Self::FilePristine),
            "period" => Ok(Self::Period),
            other => Err(anyhow!("unsupported expiration condition: {other}")),
        }
    }
}

pub fn normalize_tags(tags: &[String]) -> Vec<String> {
    let mut normalized = Vec::new();

    for tag in tags {
        let tag = tag.trim().to_ascii_lowercase();
        if tag.is_empty() || normalized.contains(&tag) {
            continue;
        }
        normalized.push(tag);
    }

    normalized
}
