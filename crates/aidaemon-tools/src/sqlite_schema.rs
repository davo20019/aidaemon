//! Additive schema upgrades shared by the SQLite stores (state, work,
//! memory, events): a column added only where it is missing, with any other
//! failure reported instead of swallowed.

use sqlx::{SqliteConnection, SqlitePool};

/// Whether table `name` exists.
pub async fn table_exists(conn: &mut SqliteConnection, name: &str) -> sqlx::Result<bool> {
    Ok(sqlx::query_scalar::<_, i64>(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
    )
    .bind(name)
    .fetch_optional(conn)
    .await?
    .is_some())
}

/// Whether table `table` has column `column` (false when the table is
/// missing).
pub async fn column_exists(
    conn: &mut SqliteConnection,
    table: &str,
    column: &str,
) -> sqlx::Result<bool> {
    Ok(
        sqlx::query_scalar::<_, i64>("SELECT 1 FROM pragma_table_info(?) WHERE name = ?")
            .bind(table)
            .bind(column)
            .fetch_optional(conn)
            .await?
            .is_some(),
    )
}

/// Run `statement`, an `ALTER TABLE <table> ADD COLUMN <column> ...` this
/// crate's callers write, only when the table exists and lacks the column.
/// `Ok(true)` when the column was added; `Ok(false)` when there was nothing
/// to do. A failing ALTER is an error: before this, every failure (not only
/// "duplicate column") was dropped, so a column could silently never exist.
pub async fn add_column_if_missing(pool: &SqlitePool, statement: &str) -> sqlx::Result<bool> {
    let mut conn = pool.acquire().await?;
    add_column_if_missing_on(&mut conn, statement).await
}

/// [`add_column_if_missing`] on one connection (inside a transaction).
pub async fn add_column_if_missing_on(
    conn: &mut SqliteConnection,
    statement: &str,
) -> sqlx::Result<bool> {
    // Our own statement text (a machine format), not user input.
    let words: Vec<&str> = statement.split_whitespace().collect();
    let (table, column) = match words.as_slice() {
        [alter, kw_table, table, add, kw_column, column, ..]
            if alter.eq_ignore_ascii_case("ALTER")
                && kw_table.eq_ignore_ascii_case("TABLE")
                && add.eq_ignore_ascii_case("ADD")
                && kw_column.eq_ignore_ascii_case("COLUMN") =>
        {
            (*table, *column)
        }
        _ => {
            return Err(sqlx::Error::Protocol(format!(
                "not an ALTER TABLE ... ADD COLUMN statement: {statement}"
            )));
        }
    };
    if !table_exists(conn, table).await? || column_exists(conn, table, column).await? {
        return Ok(false);
    }
    sqlx::query(sqlx::AssertSqlSafe(statement))
        .execute(conn)
        .await?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_column_is_added_once_and_a_missing_table_is_left_alone() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        sqlx::query("CREATE TABLE goals (id TEXT)")
            .execute(&pool)
            .await
            .unwrap();
        let add = "ALTER TABLE goals ADD COLUMN notes TEXT NOT NULL DEFAULT ''";
        assert!(add_column_if_missing(&pool, add).await.unwrap());
        assert!(!add_column_if_missing(&pool, add).await.unwrap());
        assert!(
            !add_column_if_missing(&pool, "ALTER TABLE absent ADD COLUMN x TEXT")
                .await
                .unwrap()
        );
        // A real failure is reported, not dropped.
        assert!(
            add_column_if_missing(
                &pool,
                "ALTER TABLE goals ADD COLUMN bad INTEGER PRIMARY KEY"
            )
            .await
            .is_err()
        );
    }
}
