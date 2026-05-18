use std::io::{self, BufReader};
use std::path::PathBuf;
use std::str::FromStr;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use serde::Serialize;

use crate::explorer;
use crate::mcp;
use crate::model::{ExpirationCondition, MemoryMode};
use crate::store::{MemoryStore, SearchOptions, SetMemory, default_database_path};

#[derive(Debug, Parser)]
#[command(version, about = "A smart memory management system for agents")]
pub struct Cli {
    #[arg(long, global = true, env = "MII_MEMORY_DB", value_name = "PATH")]
    db: Option<PathBuf>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Set(SetCommand),
    Get(GetCommand),
    ListTags(ListTagsCommand),
    Alert(AlertCommand),
    Alerts(AlertsCommand),
    Mcp,
    Explorer(ExplorerCommand),
}

#[derive(Debug, Parser)]
struct AlertCommand {
    #[command(subcommand)]
    command: AlertSubcommand,
}

#[derive(Debug, Subcommand)]
enum AlertSubcommand {
    Set(AlertSetCommand),
}

#[derive(Debug, Parser)]
struct AlertSetCommand {
    session_ref: String,
    content: String,
}

#[derive(Debug, Parser)]
struct AlertsCommand {
    session_ref: String,
}

#[derive(Debug, Parser)]
struct SetCommand {
    content: String,

    #[arg(long, value_enum, default_value_t = MemoryMode::Global)]
    mode: MemoryMode,

    #[arg(value_name = "MODE_REF")]
    mode_ref: Option<String>,

    #[arg(short = 't', long = "tag", required = true)]
    tags: Vec<String>,

    #[arg(
        long = "expiration-condition",
        value_names = ["CONDITION", "VALUE"],
        num_args = 2
    )]
    expiration: Option<Vec<String>>,

    #[arg(long)]
    metadata: Option<String>,
}

#[derive(Debug, Parser)]
struct GetCommand {
    query: String,

    #[arg(short = 't', long = "tag", alias = "p-tag")]
    positive_tags: Vec<String>,

    #[arg(long = "n-tag")]
    negative_tags: Vec<String>,

    #[arg(long, default_value_t = 10)]
    limit: usize,

    #[arg(long, default_value_t = 0)]
    offset: usize,

    #[arg(long, value_enum)]
    mode: Option<MemoryMode>,

    #[arg(long)]
    mode_ref: Option<String>,
}

#[derive(Debug, Parser)]
struct ListTagsCommand {
    #[arg(long)]
    filter: Option<String>,

    #[arg(long)]
    json: bool,
}

#[derive(Debug, Parser)]
struct ExplorerCommand {
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    #[arg(long, default_value_t = 4117)]
    port: u16,
}

#[derive(Debug, Serialize)]
struct SetOutput {
    id: i64,
}

pub fn run() -> Result<()> {
    let cli = Cli::parse();
    let database_path = cli.db.unwrap_or_else(default_database_path);
    let mut store = MemoryStore::open(&database_path)?;

    match cli.command {
        Command::Set(command) => {
            let id = store.set(command.try_into()?)?;
            println!("{}", serde_json::to_string(&SetOutput { id })?);
        }
        Command::Get(command) => {
            for result in store.get(command.into())? {
                println!("{}", serde_json::to_string(&result)?);
            }
        }
        Command::ListTags(command) => {
            let tags = store.list_tags(command.filter.as_deref())?;
            for tag in tags {
                if command.json {
                    println!("{}", serde_json::to_string(&tag)?);
                } else {
                    println!("{}", tag.tag);
                }
            }
        }
        Command::Alert(command) => match command.command {
            AlertSubcommand::Set(command) => {
                let id = store.set_alert(command.session_ref, command.content)?;
                println!("{}", serde_json::to_string(&SetOutput { id })?);
            }
        },
        Command::Alerts(command) => {
            for alert in store.get_alerts(command.session_ref)? {
                println!("{}", serde_json::to_string(&alert)?);
            }
        }
        Command::Mcp => {
            let input = BufReader::new(io::stdin().lock());
            let output = io::stdout().lock();
            mcp::serve(store, input, output)?;
        }
        Command::Explorer(command) => {
            drop(store);
            explorer::serve(database_path, &command.host, command.port)?;
        }
    }

    Ok(())
}

impl TryFrom<SetCommand> for SetMemory {
    type Error = anyhow::Error;

    fn try_from(command: SetCommand) -> Result<Self> {
        let (expiration_condition, expiration_value) = parse_expiration_pair(command.expiration)?;

        Ok(Self {
            content: command.content,
            mode: command.mode,
            mode_ref: command.mode_ref,
            tags: command.tags,
            expiration_condition,
            expiration_value,
            metadata: command.metadata,
        })
    }
}

impl From<GetCommand> for SearchOptions {
    fn from(command: GetCommand) -> Self {
        Self {
            query: command.query,
            positive_tags: command.positive_tags,
            negative_tags: command.negative_tags,
            limit: command.limit,
            offset: command.offset,
            mode: command.mode,
            mode_ref: command.mode_ref,
        }
    }
}

pub fn parse_expiration_pair(
    expiration: Option<Vec<String>>,
) -> Result<(Option<ExpirationCondition>, Option<String>)> {
    let Some(expiration) = expiration else {
        return Ok((None, None));
    };

    let [condition, value] = expiration.as_slice() else {
        bail!("--expiration-condition expects CONDITION and VALUE");
    };

    Ok((
        Some(
            ExpirationCondition::from_str(condition)
                .with_context(|| format!("invalid expiration condition {condition}"))?,
        ),
        Some(value.clone()),
    ))
}
