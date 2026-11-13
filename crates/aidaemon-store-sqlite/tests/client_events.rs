#![cfg(feature = "client-events")]

use aidaemon_store_sqlite::SqliteClientEventLog;
use sqlx::sqlite::SqlitePoolOptions;

async fn log() -> SqliteClientEventLog {
    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .min_connections(1)
        .idle_timeout(None)
        .max_lifetime(None)
        .connect("sqlite::memory:")
        .await
        .unwrap();
    SqliteClientEventLog::open(pool).await.unwrap()
}

const OWNER: &str = "owner";

#[tokio::test]
async fn seq_is_gap_free_per_owner_and_reads_in_order() {
    let log = log().await;
    for n in 1..=3u64 {
        let seq = log
            .append(OWNER, n * 10, Some("app:s1"), "user_message", None, "{}")
            .await
            .unwrap();
        assert_eq!(seq, n);
    }
    assert_eq!(
        log.append("someone-else", 5, None, "x", None, "{}")
            .await
            .unwrap(),
        1
    );
    let rows = log.after(OWNER, 1, 10).await.unwrap();
    assert_eq!(rows.iter().map(|r| r.seq).collect::<Vec<_>>(), [2, 3]);
    assert_eq!(rows[0].session_id.as_deref(), Some("app:s1"));
    assert_eq!(log.bounds(OWNER).await.unwrap(), Some((1, 3)));
    assert_eq!(log.bounds("nobody").await.unwrap(), None);
}

#[tokio::test]
async fn opening_twice_keeps_the_log() {
    let log = log().await;
    log.append(OWNER, 1, None, "a", None, "{}").await.unwrap();
    let again = SqliteClientEventLog::open(log.pool().clone())
        .await
        .unwrap();
    assert_eq!(again.bounds(OWNER).await.unwrap(), Some((1, 1)));
}

#[tokio::test]
async fn pruning_keeps_the_newest_so_numbering_continues() {
    let log = log().await;
    for n in 1..=5u64 {
        log.append(OWNER, n, None, "e", None, "{}").await.unwrap();
    }
    // Everything is older than the cutoff: all but the newest go.
    assert_eq!(log.prune(OWNER, 100, 1000).await.unwrap(), 4);
    assert_eq!(log.bounds(OWNER).await.unwrap(), Some((5, 5)));
    assert_eq!(
        log.append(OWNER, 6, None, "e", None, "{}").await.unwrap(),
        6
    );
    // A size cap keeps the newest `keep` rows.
    for n in 7..=10u64 {
        log.append(OWNER, n, None, "e", None, "{}").await.unwrap();
    }
    log.prune(OWNER, 0, 3).await.unwrap();
    assert_eq!(log.bounds(OWNER).await.unwrap(), Some((8, 10)));
}

#[tokio::test]
async fn open_refs_lists_what_no_close_event_answered() {
    let log = log().await;
    log.append(OWNER, 1, None, "approval_requested", Some("a1"), "{}")
        .await
        .unwrap();
    log.append(OWNER, 2, None, "approval_requested", Some("a2"), "{}")
        .await
        .unwrap();
    log.append(OWNER, 3, None, "approval_resolved", Some("a1"), "{}")
        .await
        .unwrap();
    let open = log
        .open_refs(OWNER, "approval_requested", "approval_resolved")
        .await
        .unwrap();
    assert_eq!(open.len(), 1);
    assert_eq!(open[0].ref_id.as_deref(), Some("a2"));
    let latest = log
        .latest_of_kind(OWNER, "approval_requested")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(latest.seq, 2);
}

#[tokio::test]
async fn the_first_reply_to_a_request_is_kept() {
    let log = log().await;
    assert_eq!(log.remembered_reply(OWNER, "r1").await.unwrap(), None);
    let first = log
        .remember_reply(OWNER, "r1", 10, r#"{"n":1}"#)
        .await
        .unwrap();
    let second = log
        .remember_reply(OWNER, "r1", 11, r#"{"n":2}"#)
        .await
        .unwrap();
    assert_eq!(first, r#"{"n":1}"#);
    assert_eq!(second, r#"{"n":1}"#);
    assert_eq!(log.forget_replies_before(11).await.unwrap(), 1);
    assert_eq!(log.remembered_reply(OWNER, "r1").await.unwrap(), None);
}
