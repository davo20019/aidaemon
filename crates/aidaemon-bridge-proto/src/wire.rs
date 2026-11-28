//! `lp()` fields and a bounds-checked reader (SPEC §1).

use crate::error::{Error, Result};

/// Append `lp(x)`: a 4-byte big-endian length, then `x`.
///
/// Every field this crate writes is far below 4 GiB (frames are capped at
/// 64 KiB, messages at 16 MiB), so the length always fits.
pub(crate) fn put_lp(out: &mut Vec<u8>, x: &[u8]) {
    let len = u32::try_from(x.len()).expect("lp field over 4 GiB");
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(x);
}

/// `lp(a) ‖ lp(b) ‖ …` as one buffer.
pub(crate) fn lp_concat(fields: &[&[u8]]) -> Vec<u8> {
    let total = fields.iter().map(|f| 4 + f.len()).sum();
    let mut out = Vec::with_capacity(total);
    for field in fields {
        put_lp(&mut out, field);
    }
    out
}

/// Reads fields from a peer's bytes; every read checks the bounds.
pub(crate) struct Reader<'a> {
    rest: &'a [u8],
}

impl<'a> Reader<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { rest: bytes }
    }

    pub(crate) fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.rest.len() < n {
            return Err(Error::Short);
        }
        let (head, tail) = self.rest.split_at(n);
        self.rest = tail;
        Ok(head)
    }

    pub(crate) fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    pub(crate) fn u16(&mut self) -> Result<u16> {
        let b = self.take(2)?;
        Ok(u16::from_be_bytes([b[0], b[1]]))
    }

    pub(crate) fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// One `lp()` field of any length that fits.
    pub(crate) fn lp(&mut self) -> Result<&'a [u8]> {
        let len = self.u32()? as usize;
        self.take(len)
    }

    /// One `lp()` field that must be exactly `N` bytes.
    pub(crate) fn lp_fixed<const N: usize>(&mut self, field: &'static str) -> Result<[u8; N]> {
        let bytes = self.lp()?;
        bytes.try_into().map_err(|_| Error::FieldLength {
            field,
            len: bytes.len(),
        })
    }

    /// Refuse anything left over.
    pub(crate) fn finish(self) -> Result<()> {
        if self.rest.is_empty() {
            Ok(())
        } else {
            Err(Error::TrailingBytes)
        }
    }
}
