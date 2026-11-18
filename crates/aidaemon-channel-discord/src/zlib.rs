//! Discord's `zlib-stream` transport compression: one zlib context for the
//! whole connection, a payload complete when the buffered frames end in the
//! `Z_SYNC_FLUSH` marker.

use flate2::{Decompress, FlushDecompress};

const SYNC_FLUSH: [u8; 4] = [0x00, 0x00, 0xff, 0xff];
/// Largest compressed payload buffered.
const MAX_COMPRESSED: usize = 16 * 1024 * 1024;
/// Largest payload inflated (a big guild's GUILD_CREATE fits well within).
const MAX_INFLATED: usize = 64 * 1024 * 1024;
const INFLATE_STEP: usize = 64 * 1024;

pub(crate) struct ZlibStream {
    inflate: Decompress,
    buffer: Vec<u8>,
}

impl ZlibStream {
    pub(crate) fn new() -> Self {
        Self {
            inflate: Decompress::new(true),
            buffer: Vec::new(),
        }
    }

    /// Take one binary frame. Returns the payload it finished, if any. On
    /// an error the context is lost, so the connection has to be replaced.
    pub(crate) fn push(&mut self, frame: &[u8]) -> Result<Option<Vec<u8>>, String> {
        self.buffer.extend_from_slice(frame);
        if self.buffer.len() > MAX_COMPRESSED {
            return Err("compressed payload too large".into());
        }
        if !self.buffer.ends_with(&SYNC_FLUSH) {
            return Ok(None);
        }
        let mut out = Vec::with_capacity(self.buffer.len().saturating_mul(4).min(INFLATE_STEP));
        let mut consumed = 0;
        loop {
            if out.capacity() == out.len() {
                out.reserve(INFLATE_STEP);
            }
            let (in_before, out_before) = (self.inflate.total_in(), self.inflate.total_out());
            self.inflate
                .decompress_vec(&self.buffer[consumed..], &mut out, FlushDecompress::Sync)
                .map_err(|e| format!("inflate failed: {e}"))?;
            consumed += (self.inflate.total_in() - in_before) as usize;
            if out.len() > MAX_INFLATED {
                return Err("inflated payload too large".into());
            }
            let wrote = self.inflate.total_out() != out_before;
            // Done once all input is in and the output did not fill up, so
            // nothing is left inside the inflater.
            let room_left = out.len() < out.capacity();
            if (consumed == self.buffer.len() && room_left)
                || (!wrote && self.inflate.total_in() == in_before)
            {
                break;
            }
        }
        self.buffer.clear();
        Ok(Some(out))
    }
}
