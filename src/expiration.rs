use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use chrono::{DateTime, Duration, NaiveDate, TimeZone, Utc};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::model::ExpirationCondition;

pub fn validate_expiration(condition: ExpirationCondition, value: &str) -> Result<()> {
    match condition {
        ExpirationCondition::Time => {
            parse_duration(value).or_else(|_| parse_instant(value).map(|_| Duration::zero()))?;
        }
        ExpirationCondition::Usage => {
            parse_usage_limit(value)?;
        }
        ExpirationCondition::FileExist | ExpirationCondition::FilePristine => {
            if value.trim().is_empty() {
                bail!("file-based expiration requires a path");
            }
        }
        ExpirationCondition::Period => {
            parse_period(value)?;
        }
    }

    Ok(())
}

pub fn fingerprint_for_condition(
    condition: Option<ExpirationCondition>,
    value: Option<&str>,
) -> Result<Option<String>> {
    if condition != Some(ExpirationCondition::FilePristine) {
        return Ok(None);
    }

    let path = value.ok_or_else(|| anyhow!("file_pristine expiration requires a path"))?;
    fingerprint_file(path)
}

pub fn is_expired(
    condition: Option<ExpirationCondition>,
    value: Option<&str>,
    created_at: DateTime<Utc>,
    usage_count: i64,
    file_fingerprint: Option<&str>,
    now: DateTime<Utc>,
) -> bool {
    let Some(condition) = condition else {
        return false;
    };
    let Some(value) = value else {
        return true;
    };

    match condition {
        ExpirationCondition::Time => time_expired(value, created_at, now),
        ExpirationCondition::Usage => usage_expired(value, usage_count),
        ExpirationCondition::FileExist => !Path::new(value).exists(),
        ExpirationCondition::FilePristine => file_changed(value, file_fingerprint),
        ExpirationCondition::Period => period_expired(value, now),
    }
}

fn time_expired(value: &str, created_at: DateTime<Utc>, now: DateTime<Utc>) -> bool {
    if let Ok(expires_at) = parse_instant(value) {
        return now >= expires_at;
    }

    parse_duration(value)
        .map(|duration| now >= created_at + duration)
        .unwrap_or(true)
}

fn usage_expired(value: &str, usage_count: i64) -> bool {
    parse_usage_limit(value)
        .map(|limit| usage_count >= limit)
        .unwrap_or(true)
}

fn file_changed(value: &str, original_fingerprint: Option<&str>) -> bool {
    let Some(original_fingerprint) = original_fingerprint else {
        return true;
    };

    match fingerprint_file(value) {
        Ok(Some(current_fingerprint)) => current_fingerprint != original_fingerprint,
        Ok(None) | Err(_) => true,
    }
}

fn period_expired(value: &str, now: DateTime<Utc>) -> bool {
    parse_period(value)
        .map(|period| !period.contains(now))
        .unwrap_or(true)
}

fn parse_usage_limit(value: &str) -> Result<i64> {
    let limit = value
        .trim()
        .parse::<i64>()
        .with_context(|| format!("invalid usage expiration value: {value}"))?;

    if limit <= 0 {
        bail!("usage expiration value must be greater than zero");
    }

    Ok(limit)
}

fn parse_duration(value: &str) -> Result<Duration> {
    let value = value.trim();
    if value.is_empty() {
        bail!("duration cannot be empty");
    }

    let split_at = value
        .find(|character: char| !character.is_ascii_digit())
        .unwrap_or(value.len());
    let (number, unit) = value.split_at(split_at);
    let amount = number
        .parse::<i64>()
        .with_context(|| format!("invalid duration value: {value}"))?;

    if amount < 0 {
        bail!("duration must not be negative");
    }

    match unit.trim().to_ascii_lowercase().as_str() {
        "" | "s" | "sec" | "secs" | "second" | "seconds" => Ok(Duration::seconds(amount)),
        "m" | "min" | "mins" | "minute" | "minutes" => Ok(Duration::minutes(amount)),
        "h" | "hr" | "hrs" | "hour" | "hours" => Ok(Duration::hours(amount)),
        "d" | "day" | "days" => Ok(Duration::days(amount)),
        "w" | "week" | "weeks" => Ok(Duration::weeks(amount)),
        other => bail!("unsupported duration unit: {other}"),
    }
}

fn parse_instant(value: &str) -> Result<DateTime<Utc>> {
    let value = value.trim();

    if let Ok(datetime) = DateTime::parse_from_rfc3339(value) {
        return Ok(datetime.with_timezone(&Utc));
    }

    let date = NaiveDate::parse_from_str(value, "%Y-%m-%d")
        .with_context(|| format!("invalid instant value: {value}"))?;
    let datetime = date
        .and_hms_opt(0, 0, 0)
        .ok_or_else(|| anyhow!("invalid date value: {value}"))?;
    Ok(Utc.from_utc_datetime(&datetime))
}

fn parse_period(value: &str) -> Result<Period> {
    let value = value.trim();

    if let Ok(json) = serde_json::from_str::<Value>(value) {
        let start = json
            .get("start")
            .and_then(Value::as_str)
            .map(parse_instant)
            .transpose()?;
        let end = json
            .get("end")
            .and_then(Value::as_str)
            .map(parse_instant)
            .transpose()?;
        return Ok(Period { start, end });
    }

    let (start, end) = value
        .split_once("..")
        .or_else(|| value.split_once(','))
        .ok_or_else(|| anyhow!("period must be formatted as start..end, start,end, or JSON"))?;

    Ok(Period {
        start: parse_optional_instant(start)?,
        end: parse_optional_instant(end)?,
    })
}

fn parse_optional_instant(value: &str) -> Result<Option<DateTime<Utc>>> {
    let value = value.trim();
    if value.is_empty() {
        return Ok(None);
    }

    parse_instant(value).map(Some)
}

fn fingerprint_file(path: impl AsRef<Path>) -> Result<Option<String>> {
    let path = path.as_ref();
    if !path.exists() {
        return Ok(None);
    }

    if !path.is_file() {
        let metadata = path
            .metadata()
            .with_context(|| format!("failed to read metadata for {}", path.display()))?;
        let mut hasher = Sha256::new();
        hasher.update(metadata.len().to_le_bytes());
        if let Ok(modified) = metadata.modified()
            && let Ok(duration) = modified.duration_since(std::time::UNIX_EPOCH)
        {
            hasher.update(duration.as_nanos().to_le_bytes());
        }
        return Ok(Some(hex::encode(hasher.finalize())));
    }

    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 8192];

    loop {
        let read = reader
            .read(&mut buffer)
            .with_context(|| format!("failed to read {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }

    Ok(Some(hex::encode(hasher.finalize())))
}

#[derive(Debug, Clone, Copy)]
struct Period {
    start: Option<DateTime<Utc>>,
    end: Option<DateTime<Utc>>,
}

impl Period {
    fn contains(self, now: DateTime<Utc>) -> bool {
        if self.start.is_some_and(|start| now < start) {
            return false;
        }

        if self.end.is_some_and(|end| now > end) {
            return false;
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_expiration_allows_exact_number_of_reads() {
        let now = Utc::now();

        assert!(!is_expired(
            Some(ExpirationCondition::Usage),
            Some("2"),
            now,
            1,
            None,
            now,
        ));
        assert!(is_expired(
            Some(ExpirationCondition::Usage),
            Some("2"),
            now,
            2,
            None,
            now,
        ));
    }

    #[test]
    fn period_accepts_open_start() {
        let now = Utc.with_ymd_and_hms(2026, 5, 18, 12, 0, 0).unwrap();

        assert!(!is_expired(
            Some(ExpirationCondition::Period),
            Some("..2026-05-19T00:00:00Z"),
            now,
            0,
            None,
            now,
        ));
    }
}
