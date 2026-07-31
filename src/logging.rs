use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{self, IsTerminal, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::Context;
use tracing_subscriber::fmt::writer::BoxMakeWriter;
use tracing_subscriber::EnvFilter;

const DEFAULT_MAX_LOG_BYTES: u64 = 10 * 1024 * 1024;
const DEFAULT_LOG_BACKUPS: usize = 4;

struct SizeRotatingFile {
    path: PathBuf,
    file: File,
    current_len: u64,
    max_bytes: u64,
    backup_count: usize,
}

impl SizeRotatingFile {
    fn open(path: PathBuf, max_bytes: u64, backup_count: usize) -> io::Result<Self> {
        if path
            .metadata()
            .is_ok_and(|metadata| metadata.len() >= max_bytes)
        {
            rotate_existing_files(&path, backup_count)?;
            trim_file_to_tail(&backup_path(&path, 1), max_bytes)?;
        }
        let file = open_private_append_file(&path)?;
        let current_len = file.metadata()?.len();
        Ok(Self {
            path,
            file,
            current_len,
            max_bytes,
            backup_count,
        })
    }

    fn rotate(&mut self) -> io::Result<()> {
        self.file.flush()?;
        rotate_existing_files(&self.path, self.backup_count)?;
        trim_file_to_tail(&backup_path(&self.path, 1), self.max_bytes)?;
        self.file = open_private_append_file(&self.path)?;
        self.current_len = 0;
        Ok(())
    }
}

impl Write for SizeRotatingFile {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        if self.current_len > 0
            && self.current_len.saturating_add(buffer.len() as u64) > self.max_bytes
        {
            self.rotate()?;
        }
        let written = self.file.write(buffer)?;
        self.current_len = self.current_len.saturating_add(written as u64);
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}

fn backup_path(path: &Path, index: usize) -> PathBuf {
    let mut value = OsString::from(path.as_os_str());
    value.push(format!(".{index}"));
    PathBuf::from(value)
}

fn rotate_existing_files(path: &Path, backup_count: usize) -> io::Result<()> {
    if backup_count == 0 {
        if path.exists() {
            fs::remove_file(path)?;
        }
        return Ok(());
    }

    let oldest = backup_path(path, backup_count);
    if oldest.exists() {
        fs::remove_file(&oldest)?;
    }
    for index in (1..backup_count).rev() {
        let source = backup_path(path, index);
        if source.exists() {
            let destination = backup_path(path, index + 1);
            if destination.exists() {
                fs::remove_file(&destination)?;
            }
            fs::rename(source, destination)?;
        }
    }
    if path.exists() {
        let first_backup = backup_path(path, 1);
        fs::rename(path, &first_backup)?;
        set_private_permissions(&first_backup)?;
    }
    Ok(())
}

fn trim_file_to_tail(path: &Path, max_bytes: u64) -> io::Result<()> {
    let length = match path.metadata() {
        Ok(metadata) => metadata.len(),
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error),
    };
    if length <= max_bytes {
        set_private_permissions(path)?;
        return Ok(());
    }

    let mut source = File::open(path)?;
    source.seek(SeekFrom::End(-(max_bytes as i64)))?;
    let mut tail = Vec::with_capacity(max_bytes as usize);
    source.read_to_end(&mut tail)?;

    let temporary = backup_path(path, 0);
    if temporary.exists() {
        fs::remove_file(&temporary)?;
    }
    let mut destination = open_private_truncate_file(&temporary)?;
    destination.write_all(&tail)?;
    destination.flush()?;
    drop(destination);
    fs::remove_file(path)?;
    fs::rename(&temporary, path)?;
    set_private_permissions(path)
}

fn open_private_append_file(path: &Path) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options.create(true).append(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let file = options.open(path)?;
    set_private_file_permissions(&file)?;
    Ok(file)
}

fn open_private_truncate_file(path: &Path) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options.create(true).truncate(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let file = options.open(path)?;
    set_private_file_permissions(&file)?;
    Ok(file)
}

fn set_private_file_permissions(file: &File) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = file.metadata()?.permissions();
        if permissions.mode() & 0o077 != 0 {
            permissions.set_mode(0o600);
            file.set_permissions(permissions)?;
        }
    }
    Ok(())
}

fn set_private_permissions(path: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}

fn log_directory() -> anyhow::Result<PathBuf> {
    if let Ok(value) = std::env::var("AIDAEMON_LOG_DIR") {
        let value = value.trim();
        if !value.is_empty() {
            return Ok(PathBuf::from(value));
        }
    }
    let home = dirs::home_dir().context("home directory is unavailable for daemon logging")?;
    #[cfg(target_os = "macos")]
    {
        Ok(home.join("Library").join("Logs").join("aidaemon"))
    }
    #[cfg(not(target_os = "macos"))]
    {
        let state_root = std::env::var_os("XDG_STATE_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| home.join(".local").join("state"));
        Ok(state_root.join("aidaemon"))
    }
}

fn bounded_env_u64(name: &str, default: u64, minimum: u64, maximum: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .map(|value| value.clamp(minimum, maximum))
        .unwrap_or(default)
}

pub(crate) fn init(default_filter: &str) -> anyhow::Result<()> {
    let ansi = io::stderr().is_terminal();
    let cli_command = std::env::args_os().nth(1).is_some();
    let writer = if ansi || cli_command {
        BoxMakeWriter::new(io::stderr)
    } else {
        match rotating_file_writer() {
            Ok(writer) => writer,
            Err(error) => {
                eprintln!(
                    "Warning: rotating daemon log is unavailable; using stderr for this run: {error:#}"
                );
                BoxMakeWriter::new(io::stderr)
            }
        }
    };

    tracing_subscriber::fmt()
        .with_ansi(ansi)
        .with_writer(writer)
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_filter)),
        )
        .try_init()
        .map_err(|error| anyhow::anyhow!("failed to initialize tracing: {error}"))
}

fn rotating_file_writer() -> anyhow::Result<BoxMakeWriter> {
    let log_dir = log_directory()?;
    fs::create_dir_all(&log_dir)
        .with_context(|| format!("failed to create log directory {}", log_dir.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&log_dir, fs::Permissions::from_mode(0o700))?;
    }
    let max_bytes = bounded_env_u64(
        "AIDAEMON_LOG_MAX_BYTES",
        DEFAULT_MAX_LOG_BYTES,
        1024 * 1024,
        1024 * 1024 * 1024,
    );
    let backup_count =
        bounded_env_u64("AIDAEMON_LOG_BACKUPS", DEFAULT_LOG_BACKUPS as u64, 1, 20) as usize;
    let file = SizeRotatingFile::open(log_dir.join("stdout.log"), max_bytes, backup_count)
        .context("failed to initialize rotating daemon log")?;
    Ok(BoxMakeWriter::new(Mutex::new(file)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_rotation_is_bounded_and_preserves_recent_backups() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("stdout.log");
        let mut writer = SizeRotatingFile::open(path.clone(), 10, 2).unwrap();

        writer.write_all(b"first-log").unwrap();
        writer.write_all(b"second-log").unwrap();
        writer.write_all(b"third-log").unwrap();
        writer.flush().unwrap();

        assert_eq!(fs::read_to_string(&path).unwrap(), "third-log");
        assert_eq!(
            fs::read_to_string(backup_path(&path, 1)).unwrap(),
            "second-log"
        );
        assert_eq!(
            fs::read_to_string(backup_path(&path, 2)).unwrap(),
            "first-log"
        );
        assert!(!backup_path(&path, 3).exists());
    }

    #[test]
    fn oversized_existing_log_rotates_before_startup_append() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("stdout.log");
        fs::write(&path, b"existing-large-log").unwrap();

        let mut writer = SizeRotatingFile::open(path.clone(), 10, 2).unwrap();
        writer.write_all(b"fresh").unwrap();
        writer.flush().unwrap();

        assert_eq!(fs::read_to_string(&path).unwrap(), "fresh");
        let backup = fs::read_to_string(backup_path(&path, 1)).unwrap();
        assert!(backup.len() <= 10);
        assert!(backup.ends_with("large-log"));
    }
}
