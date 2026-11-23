//! Image upload ids, names, prompts and storage.

use super::*;
use tempfile::tempdir;

#[test]
fn test_normalize_image_mime() {
    assert_eq!(normalize_image_mime(Some("image/png")), Some("image/png"));
    assert_eq!(normalize_image_mime(Some("image/jpg")), Some("image/jpeg"));
    assert_eq!(
        normalize_image_mime(Some("image/jpeg; charset=utf-8")),
        Some("image/jpeg")
    );
    assert_eq!(normalize_image_mime(Some("image/webp")), Some("image/webp"));
    assert_eq!(normalize_image_mime(Some("application/pdf")), None);
}

#[test]
fn test_normalize_upload_id() {
    assert_eq!(
        normalize_upload_id(Some(" img_01-abc ")).as_deref(),
        Some("img_01-abc")
    );
    assert!(normalize_upload_id(Some("$$$")).is_none());
}

#[test]
fn test_sanitize_upload_filename_adds_extension() {
    assert_eq!(
        sanitize_upload_filename("screen shot", "image/png"),
        "screen shot.png"
    );
    assert_eq!(
        sanitize_upload_filename("photo.jpg", "image/jpeg"),
        "photo.jpg"
    );
}

#[test]
fn test_build_upload_prompt_includes_context() {
    let path = PathBuf::from("/tmp/inbox/abc_screenshot.png");
    let prompt = build_upload_prompt(
        "screenshot.png",
        &path,
        "image/png",
        120_000,
        Some("error appears after login"),
    );
    assert!(prompt.contains("screenshot.png"));
    assert!(prompt.contains("/tmp/inbox/abc_screenshot.png"));
    assert!(prompt.contains("error appears after login"));
}

#[test]
fn test_store_uploaded_image_prefers_session_workspace_path() {
    let session_dir = tempdir().expect("session tempdir");
    let fallback_dir = tempdir().expect("fallback tempdir");

    let stored = store_uploaded_image(
        session_dir.path(),
        fallback_dir.path(),
        "screenshot.png",
        b"png-bytes",
    )
    .expect("stored upload");

    assert!(stored.saved_inside_session_workspace);
    assert!(stored.fallback_reason.is_none());
    assert!(
        stored
            .path
            .starts_with(session_upload_dir(session_dir.path()))
    );
    assert_eq!(
        std::fs::read(&stored.path).expect("read stored image"),
        b"png-bytes"
    );
}

#[test]
fn test_store_uploaded_image_falls_back_when_session_workspace_path_is_blocked() {
    let session_dir = tempdir().expect("session tempdir");
    let fallback_dir = tempdir().expect("fallback tempdir");
    std::fs::write(session_dir.path().join(".aidaemon"), "not a directory")
        .expect("block workspace upload dir");

    let stored = store_uploaded_image(
        session_dir.path(),
        fallback_dir.path(),
        "screenshot.png",
        b"png-bytes",
    )
    .expect("stored upload with fallback");

    assert!(!stored.saved_inside_session_workspace);
    assert!(stored.fallback_reason.is_some());
    assert!(stored.path.starts_with(fallback_dir.path()));
    assert_eq!(
        std::fs::read(&stored.path).expect("read fallback image"),
        b"png-bytes"
    );
}
