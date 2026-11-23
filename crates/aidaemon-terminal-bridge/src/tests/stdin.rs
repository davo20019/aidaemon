//! Client stdin decoding and UTF-8 stream chunks.

use super::*;

#[test]
fn test_normalize_stdin_for_pty_decodes_single_enter_escape() {
    assert_eq!(normalize_stdin_for_pty("\\r"), "\r");
}

#[test]
fn test_normalize_stdin_for_pty_decodes_ansi_escape() {
    assert_eq!(normalize_stdin_for_pty("\\u001b[C"), "\u{001b}[C");
}

#[test]
fn test_normalize_stdin_for_pty_decodes_multiple_control_escapes() {
    assert_eq!(normalize_stdin_for_pty("status\\r\\t"), "status\r\t");
}

#[test]
fn test_normalize_stdin_for_pty_decodes_trailing_submit_escape() {
    assert_eq!(normalize_stdin_for_pty("npm test\\r"), "npm test\r");
}

#[test]
fn test_normalize_stdin_for_pty_preserves_non_control_backslashes() {
    assert_eq!(normalize_stdin_for_pty("echo \\q"), "echo \\q");
    assert_eq!(normalize_stdin_for_pty("C:\\temp\\logs"), "C:\\temp\\logs");
}

#[test]
fn test_normalize_stdin_for_pty_preserves_non_ascii_hex_escapes() {
    assert_eq!(normalize_stdin_for_pty("\\x80"), "\\x80");
}

#[test]
fn test_decode_utf8_stream_chunk_preserves_split_multibyte_chars() {
    let mut carry = Vec::new();
    let part1 = decode_utf8_stream_chunk(&mut carry, &[0xE2, 0x94]);
    assert!(part1.is_empty());
    let part2 = decode_utf8_stream_chunk(&mut carry, &[0x80, b'\n']);
    assert_eq!(part2, "─\n");
    assert!(carry.is_empty());
}

#[test]
fn test_decode_utf8_stream_chunk_normalizes_c1_csi() {
    let mut carry = Vec::new();
    let out = decode_utf8_stream_chunk(&mut carry, &[0x9B, b'3', b'1', b'm']);
    assert_eq!(out, "\u{001b}[31m");
    assert!(carry.is_empty());
}
