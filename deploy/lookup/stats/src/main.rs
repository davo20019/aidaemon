//! Daily count of the daemons publishing to the project's address lookup
//! (`dns.aidaemon.ai`).
//!
//! Every daemon on the project's relays republishes its signed address record
//! every 5 minutes under a key that stays the same across restarts, so the
//! number of keys the lookup server heard from in a window is the number of
//! daemons that ran in it. This reads a copy of the server's packet store,
//! counts those keys for the last day and the last 7 days, and appends one
//! line of counts to a CSV file. No key, address or record leaves the
//! function: the file holds counts only.
//!
//! ```text
//! lookup-stats count --db <signed-packets-1.db> --out <daily.csv>
//!                    [--exclude <file>] [--work <dir>]
//! lookup-stats key <remote-key>   # the public key of a daemon's remote-key file
//! ```
//!
//! `--exclude` lists public keys (hex, one per line, `#` comments) left out of
//! the count, such as the maintainers' own daemons.

use std::collections::HashSet;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use redb::{Database, ReadableDatabase, ReadableTable, TableDefinition};

/// iroh-dns-server's packet table: public key -> `<8 bytes last seen><packet>`,
/// where last seen is the server's receipt time in microseconds (big endian).
const PACKETS: TableDefinition<&[u8; 32], &[u8]> = TableDefinition::new("signed-packets-1");

const DAY_US: u64 = 24 * 3600 * 1_000_000;
const CSV_HEADER: &str = "time_utc,daily,weekly";

type Error = Box<dyn std::error::Error>;

fn main() {
    if let Err(error) = run(std::env::args().skip(1).collect()) {
        eprintln!("lookup-stats: {error}");
        std::process::exit(1);
    }
}

fn run(args: Vec<String>) -> Result<(), Error> {
    match args.first().map(String::as_str) {
        Some("count") => {
            let db = flag(&args, "--db")?.ok_or("--db is required")?;
            let out = flag(&args, "--out")?.ok_or("--out is required")?;
            let exclude = match flag(&args, "--exclude")? {
                Some(path) => read_exclude(Path::new(&path))?,
                None => HashSet::new(),
            };
            let work = flag(&args, "--work")?.unwrap_or_else(|| std::env::temp_dir().display().to_string());
            let now = now_us();
            let counts = count_copy(Path::new(&db), Path::new(&work), &exclude, now)?;
            append(Path::new(&out), now, counts)?;
            println!("daily={} weekly={}", counts.daily, counts.weekly);
            Ok(())
        }
        Some("key") => {
            let path = args.get(1).ok_or("key needs a remote-key file")?;
            println!("{}", hex(&public_key(&std::fs::read_to_string(path)?)?));
            Ok(())
        }
        _ => Err("usage: lookup-stats count --db <file> --out <csv> [--exclude <file>] [--work <dir>] | lookup-stats key <remote-key>".into()),
    }
}

fn flag(args: &[String], name: &str) -> Result<Option<String>, Error> {
    match args.iter().position(|arg| arg == name) {
        Some(at) => Ok(Some(
            args.get(at + 1)
                .ok_or_else(|| format!("{name} needs a value"))?
                .clone(),
        )),
        None => Ok(None),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Counts {
    daily: u64,
    weekly: u64,
}

/// Counts from a private copy of the store: the server holds the file open and
/// locked, and opening a copy taken mid-write may need redb's repair, which
/// writes. A copy torn by a concurrent commit fails to open; take another.
fn count_copy(
    db: &Path,
    work: &Path,
    exclude: &HashSet<[u8; 32]>,
    now: u64,
) -> Result<Counts, Error> {
    let copy = work.join(format!("lookup-stats-{}.db", std::process::id()));
    let mut last_error = None;
    for attempt in 0..3 {
        if attempt > 0 {
            std::thread::sleep(Duration::from_secs(5));
        }
        let result = std::fs::copy(db, &copy)
            .map_err(Error::from)
            .and_then(|_| count(&copy, exclude, now));
        let _ = std::fs::remove_file(&copy);
        match result {
            Ok(counts) => return Ok(counts),
            Err(error) => last_error = Some(error),
        }
    }
    Err(format!(
        "could not read {}: {}",
        db.display(),
        last_error.expect("three attempts")
    )
    .into())
}

fn count(db: &Path, exclude: &HashSet<[u8; 32]>, now: u64) -> Result<Counts, Error> {
    let db = Database::open(db)?;
    let read = db.begin_read()?;
    let table = read.open_table(PACKETS)?;
    let mut counts = Counts {
        daily: 0,
        weekly: 0,
    };
    for row in table.iter()? {
        let (key, value) = row?;
        if exclude.contains(key.value()) {
            continue;
        }
        let Some(seen) = value.value().get(..8) else {
            continue;
        };
        let age = now.saturating_sub(u64::from_be_bytes(seen.try_into()?));
        if age <= DAY_US {
            counts.daily += 1;
        }
        if age <= 7 * DAY_US {
            counts.weekly += 1;
        }
    }
    Ok(counts)
}

fn append(out: &Path, now: u64, counts: Counts) -> Result<(), Error> {
    let new = !out.exists();
    let mut file = OpenOptions::new().create(true).append(true).open(out)?;
    if new {
        writeln!(file, "{CSV_HEADER}")?;
    }
    writeln!(
        file,
        "{},{},{}",
        utc_minute(now),
        counts.daily,
        counts.weekly
    )?;
    Ok(())
}

fn read_exclude(path: &Path) -> Result<HashSet<[u8; 32]>, Error> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(HashSet::new()),
        Err(error) => return Err(error.into()),
    };
    text.lines()
        .map(|line| line.split('#').next().unwrap_or("").trim())
        .filter(|line| !line.is_empty())
        .map(|line| {
            unhex32(line)
                .ok_or_else(|| format!("{}: not a 64-hex-digit key: {line}", path.display()).into())
        })
        .collect()
}

/// The public key of a daemon's `remote-key` file (its secret key in hex).
fn public_key(secret_hex: &str) -> Result<[u8; 32], Error> {
    let secret = unhex32(secret_hex.trim()).ok_or("the file holds no 64-hex-digit key")?;
    Ok(ed25519_dalek::SigningKey::from_bytes(&secret)
        .verifying_key()
        .to_bytes())
}

fn unhex32(text: &str) -> Option<[u8; 32]> {
    let bytes = text.as_bytes();
    if bytes.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, pair) in bytes.chunks(2).enumerate() {
        out[i] = u8::from_str_radix(std::str::from_utf8(pair).ok()?, 16).ok()?;
    }
    Some(out)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn now_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}

/// `YYYY-MM-DDTHH:MMZ` for a time in microseconds since the epoch.
fn utc_minute(us: u64) -> String {
    let secs = us / 1_000_000;
    let (days, rem) = (secs / 86_400, secs % 86_400);
    // Civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}Z",
        rem / 3600,
        rem % 3600 / 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store(rows: &[([u8; 32], u64)]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::create(dir.path().join("packets.db")).unwrap();
        let write = db.begin_write().unwrap();
        {
            let mut table = write.open_table(PACKETS).unwrap();
            for (key, seen) in rows {
                let mut value = seen.to_be_bytes().to_vec();
                value.extend_from_slice(&[0u8; 120]); // stands in for the signed packet
                table.insert(key, value.as_slice()).unwrap();
            }
        }
        write.commit().unwrap();
        dir
    }

    #[test]
    fn counts_keys_seen_in_the_last_day_and_week_minus_excluded() {
        let now = 1_000 * DAY_US;
        let dir = store(&[
            ([1; 32], now - 3600 * 1_000_000), // an hour ago
            ([2; 32], now - 2 * DAY_US),       // two days ago
            ([3; 32], now - 8 * DAY_US),       // older than a week
            ([4; 32], now - 60 * 1_000_000),   // excluded
        ]);
        let exclude = HashSet::from([[4; 32]]);
        let work = tempfile::tempdir().unwrap();
        let counts =
            count_copy(&dir.path().join("packets.db"), work.path(), &exclude, now).unwrap();
        assert_eq!(
            counts,
            Counts {
                daily: 1,
                weekly: 2
            }
        );
        assert_eq!(
            std::fs::read_dir(work.path()).unwrap().count(),
            0,
            "the copy is removed"
        );
    }

    #[test]
    fn appends_a_header_once_then_counts() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("daily.csv");
        append(
            &out,
            0,
            Counts {
                daily: 1,
                weekly: 2,
            },
        )
        .unwrap();
        append(
            &out,
            86_400 * 1_000_000 + 61 * 1_000_000,
            Counts {
                daily: 3,
                weekly: 4,
            },
        )
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(&out).unwrap(),
            "time_utc,daily,weekly\n1970-01-01T00:00Z,1,2\n1970-01-02T00:01Z,3,4\n"
        );
    }

    #[test]
    fn dates_cross_leap_days() {
        // 2024-02-29T12:34Z
        assert_eq!(utc_minute(1_709_210_040 * 1_000_000), "2024-02-29T12:34Z");
    }

    #[test]
    fn exclude_file_takes_comments_and_rejects_junk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("exclude");
        std::fs::write(&path, format!("# mine\n{}  # laptop\n\n", "ab".repeat(32))).unwrap();
        assert_eq!(read_exclude(&path).unwrap(), HashSet::from([[0xab; 32]]));
        std::fs::write(&path, "nope\n").unwrap();
        assert!(read_exclude(&path).is_err());
        assert!(
            read_exclude(&dir.path().join("missing"))
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn public_key_matches_ed25519() {
        let secret = [7u8; 32];
        let expected = ed25519_dalek::SigningKey::from_bytes(&secret)
            .verifying_key()
            .to_bytes();
        assert_eq!(
            public_key(&format!("{}\n", hex(&secret))).unwrap(),
            expected
        );
    }
}
