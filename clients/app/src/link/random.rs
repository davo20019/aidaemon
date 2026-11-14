//! Entropy in a browser for a Rust-only app: getrandom's custom backend,
//! filled by the page's Web Crypto through NSH's page script (the same
//! import NSH's own JS runtime uses). Without it, request ids and nothing
//! else would have no randomness to draw on.

#[link(wasm_import_module = "env")]
unsafe extern "C" {
    fn __nsh_random_fill(ptr: *mut u8, len: usize) -> i32;
}

/// getrandom's custom backend, shared by its 0.3 and 0.4 releases.
#[unsafe(no_mangle)]
unsafe extern "Rust" fn __getrandom_v03_custom(
    ptr: *mut u8,
    len: usize,
) -> Result<(), getrandom::Error> {
    // SAFETY: getrandom passes a writable range of exactly `len` bytes,
    // which the page fills synchronously before returning.
    if unsafe { __nsh_random_fill(ptr, len) } == 0 {
        Ok(())
    } else {
        Err(getrandom::Error::UNSUPPORTED)
    }
}
