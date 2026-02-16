use super::*;

#[test]
fn test_extracts_existing_file() {
    // Use a file that definitely exists
    let text = format!("Report generated at {}", file!());
    // file!() returns a relative path, so this won't match (only absolute paths)
    let paths = extract_file_paths_from_text(&text);
    assert!(paths.is_empty(), "Relative paths should not match");
}

#[test]
fn test_extracts_absolute_path_with_extension() {
    let tmp = std::env::temp_dir().join("aidaemon_test_extract.txt");
    std::fs::write(&tmp, "test content").unwrap();

    let text = format!("Final report generated at {}", tmp.display());
    let paths = extract_file_paths_from_text(&text);
    assert_eq!(paths.len(), 1);
    assert_eq!(paths[0], tmp.to_string_lossy());

    std::fs::remove_file(&tmp).unwrap();
}

#[test]
fn test_ignores_nonexistent_paths() {
    let text = "Report at /tmp/nonexistent_aidaemon_xyz_12345.md";
    let paths = extract_file_paths_from_text(text);
    assert!(paths.is_empty());
}

#[test]
fn test_blocks_sensitive_paths() {
    let tmp = std::env::temp_dir().join(".ssh");
    std::fs::create_dir_all(&tmp).ok();
    let sensitive = tmp.join("id_rsa.pub");
    std::fs::write(&sensitive, "fake key").unwrap();

    let text = format!("Key at {}", sensitive.display());
    let paths = extract_file_paths_from_text(&text);
    assert!(paths.is_empty(), "Paths under .ssh should be blocked");

    std::fs::remove_file(&sensitive).unwrap();
}

#[test]
fn test_blocks_pem_extension() {
    let tmp = std::env::temp_dir().join("aidaemon_test.pem");
    std::fs::write(&tmp, "fake cert").unwrap();

    let text = format!("Cert at {}", tmp.display());
    let paths = extract_file_paths_from_text(&text);
    assert!(paths.is_empty(), ".pem files should be blocked");

    std::fs::remove_file(&tmp).unwrap();
}

#[test]
fn test_multiple_paths() {
    let tmp1 = std::env::temp_dir().join("aidaemon_test_a.md");
    let tmp2 = std::env::temp_dir().join("aidaemon_test_b.csv");
    std::fs::write(&tmp1, "report").unwrap();
    std::fs::write(&tmp2, "data").unwrap();

    let text = format!("Generated {} and also {}", tmp1.display(), tmp2.display());
    let paths = extract_file_paths_from_text(&text);
    assert_eq!(paths.len(), 2);

    std::fs::remove_file(&tmp1).unwrap();
    std::fs::remove_file(&tmp2).unwrap();
}

#[test]
fn test_no_paths_in_text() {
    let text = "Goal completed successfully. All tasks done.";
    let paths = extract_file_paths_from_text(text);
    assert!(paths.is_empty());
}

#[test]
fn test_ignores_directories() {
    // /tmp exists but is a directory, not a file — and has no extension
    let text = "Output in /tmp directory";
    let paths = extract_file_paths_from_text(text);
    assert!(paths.is_empty());
}
