use super::*;

#[test]
fn test_touch_heartbeat_updates_timestamp() {
    let hb = Arc::new(AtomicU64::new(0));
    touch_heartbeat(&Some(hb.clone()));
    let val = hb.load(Ordering::Relaxed);
    assert!(val > 0, "heartbeat should be updated to current time");
}

#[test]
fn test_touch_heartbeat_none_is_noop() {
    // Should not panic
    touch_heartbeat(&None);
}
