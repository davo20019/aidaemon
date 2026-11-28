//! Frames (SPEC §4), DATA sealing with per-direction counters, and
//! fragmentation with the FINAL flag and reassembly limits (SPEC §7).

use aes_gcm::aead::{Aead, Payload};
use aes_gcm::{Aes256Gcm, KeyInit};

use crate::error::{Error, Result};

/// The only protocol version.
pub const VERSION: u8 = 1;
/// `version ‖ type ‖ flags ‖ counter`.
pub const HEADER_LEN: usize = 11;
/// AES-GCM tag length.
pub const TAG_LEN: usize = 16;
/// DATA flag: last fragment of a message.
pub const FLAG_FINAL: u8 = 0x01;

/// Largest handshake frame, header included (before `promote`).
pub const MAX_HANDSHAKE_FRAME: usize = 1024;
/// Largest DATA frame, header and tag included.
pub const MAX_DATA_FRAME: usize = 64 * 1024;
/// Largest plaintext one DATA frame carries.
pub const MAX_FRAGMENT: usize = MAX_DATA_FRAME - HEADER_LEN - TAG_LEN;
/// Largest reassembled request (page → daemon).
pub const MAX_REQUEST: usize = 16 * 1024 * 1024;
/// Largest reassembled response or control (daemon → page).
pub const MAX_RESPONSE: usize = 4 * 1024 * 1024;

/// Frame types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FrameType {
    Hello = 0x01,
    HelloReply = 0x02,
    Finish = 0x03,
    Data = 0x10,
    Abort = 0x7F,
}

impl FrameType {
    fn from_byte(b: u8) -> Result<Self> {
        Ok(match b {
            0x01 => FrameType::Hello,
            0x02 => FrameType::HelloReply,
            0x03 => FrameType::Finish,
            0x10 => FrameType::Data,
            0x7F => FrameType::Abort,
            other => return Err(Error::UnknownType(other)),
        })
    }

    /// HELLO, HELLO_REPLY or FINISH.
    pub fn is_handshake(self) -> bool {
        matches!(
            self,
            FrameType::Hello | FrameType::HelloReply | FrameType::Finish
        )
    }
}

/// A decoded frame; `body` borrows the input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Frame<'a> {
    pub kind: FrameType,
    pub flags: u8,
    pub counter: u64,
    pub body: &'a [u8],
}

impl Frame<'_> {
    /// The 11 header bytes (DATA's associated data).
    pub fn header(&self) -> [u8; HEADER_LEN] {
        header(self.kind, self.flags, self.counter)
    }
}

fn header(kind: FrameType, flags: u8, counter: u64) -> [u8; HEADER_LEN] {
    let mut h = [0u8; HEADER_LEN];
    h[0] = VERSION;
    h[1] = kind as u8;
    h[2] = flags;
    h[3..].copy_from_slice(&counter.to_be_bytes());
    h
}

/// Encode a frame. Callers use [`handshake_frame`], [`abort`] or a
/// [`Sealer`]; this does not check the type's rules.
pub fn encode(kind: FrameType, flags: u8, counter: u64, body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_LEN + body.len());
    out.extend_from_slice(&header(kind, flags, counter));
    out.extend_from_slice(body);
    out
}

/// A handshake frame: flags 0, counter 0.
pub fn handshake_frame(kind: FrameType, body: &[u8]) -> Vec<u8> {
    debug_assert!(kind.is_handshake());
    encode(kind, 0, 0, body)
}

/// The ABORT frame: empty body, flags 0, counter 0.
pub fn abort() -> Vec<u8> {
    encode(FrameType::Abort, 0, 0, &[])
}

/// Decode a frame no longer than `max_len`, enforcing every per-type rule
/// the header alone can show: version 1, a known type, handshake and ABORT
/// frames with flags 0 and counter 0, ABORT with an empty body, DATA with
/// only the FINAL flag and room for a tag.
pub fn decode(bytes: &[u8], max_len: usize) -> Result<Frame<'_>> {
    if bytes.len() > max_len {
        return Err(Error::FrameTooLarge {
            len: bytes.len(),
            max: max_len,
        });
    }
    if bytes.len() < HEADER_LEN {
        return Err(Error::Truncated);
    }
    if bytes[0] != VERSION {
        return Err(Error::Version(bytes[0]));
    }
    let kind = FrameType::from_byte(bytes[1])?;
    let flags = bytes[2];
    let counter = u64::from_be_bytes(bytes[3..HEADER_LEN].try_into().expect("8 bytes"));
    let body = &bytes[HEADER_LEN..];
    match kind {
        FrameType::Data => {
            if flags & !FLAG_FINAL != 0 {
                return Err(Error::Flags(flags));
            }
            if body.len() < TAG_LEN {
                return Err(Error::Short);
            }
        }
        _ => {
            if flags != 0 {
                return Err(Error::Flags(flags));
            }
            if counter != 0 {
                return Err(Error::HandshakeCounter(counter));
            }
            if kind == FrameType::Abort && !body.is_empty() {
                return Err(Error::TrailingBytes);
            }
        }
    }
    Ok(Frame {
        kind,
        flags,
        counter,
        body,
    })
}

/// Decode a handshake frame of exactly `expected` type (1 KiB limit).
/// ABORT becomes [`Error::Aborted`]; any other type is unexpected.
pub fn decode_handshake(bytes: &[u8], expected: FrameType) -> Result<&[u8]> {
    let frame = decode(bytes, MAX_HANDSHAKE_FRAME)?;
    if frame.kind == FrameType::Abort {
        return Err(Error::Aborted);
    }
    if frame.kind != expected {
        return Err(Error::UnexpectedFrame(frame.kind as u8));
    }
    Ok(frame.body)
}

fn nonce(counter: u64) -> aes_gcm::aead::Nonce<Aes256Gcm> {
    let mut n = [0u8; 12];
    n[4..].copy_from_slice(&counter.to_be_bytes());
    n.into()
}

fn cipher(key: &[u8; 32]) -> Aes256Gcm {
    Aes256Gcm::new_from_slice(key).expect("32-byte key")
}

/// Seals outgoing DATA frames for one direction. Counters start at 0 and
/// go up by one per frame.
pub struct Sealer {
    cipher: Aes256Gcm,
    next: u64,
}

impl Sealer {
    /// A sealer with this direction's key (`k_d2p` on the daemon,
    /// `k_p2d` on the page).
    pub fn new(key: &[u8; 32]) -> Self {
        Self {
            cipher: cipher(key),
            next: 0,
        }
    }

    /// The counter the next frame will carry.
    pub fn next_counter(&self) -> u64 {
        self.next
    }

    /// Seal one fragment of at most [`MAX_FRAGMENT`] bytes.
    pub fn seal_fragment(&mut self, plaintext: &[u8], final_fragment: bool) -> Result<Vec<u8>> {
        if plaintext.len() > MAX_FRAGMENT {
            return Err(Error::FrameTooLarge {
                len: HEADER_LEN + plaintext.len() + TAG_LEN,
                max: MAX_DATA_FRAME,
            });
        }
        let counter = self.next;
        let next = counter.checked_add(1).ok_or(Error::CounterExhausted)?;
        let flags = if final_fragment { FLAG_FINAL } else { 0 };
        let aad = header(FrameType::Data, flags, counter);
        let sealed = self
            .cipher
            .encrypt(
                &nonce(counter),
                Payload {
                    msg: plaintext,
                    aad: &aad,
                },
            )
            .map_err(|_| Error::Decrypt)?;
        self.next = next;
        let mut out = Vec::with_capacity(HEADER_LEN + sealed.len());
        out.extend_from_slice(&aad);
        out.extend_from_slice(&sealed);
        Ok(out)
    }

    /// Seal a whole message as one or more frames, the last one FINAL.
    /// An empty message is one empty FINAL frame.
    pub fn seal_message(&mut self, message: &[u8]) -> Result<Vec<Vec<u8>>> {
        if message.is_empty() {
            return Ok(vec![self.seal_fragment(&[], true)?]);
        }
        let mut frames = Vec::with_capacity(message.len().div_ceil(MAX_FRAGMENT));
        let mut chunks = message.chunks(MAX_FRAGMENT).peekable();
        while let Some(chunk) = chunks.next() {
            frames.push(self.seal_fragment(chunk, chunks.peek().is_none())?);
        }
        Ok(frames)
    }
}

/// Opens incoming DATA frames for one direction, enforcing the next
/// counter, and reassembles messages up to a limit. After any error it
/// refuses everything ([`Error::Closed`]): the connection is over.
pub struct Opener {
    cipher: Aes256Gcm,
    next: u64,
    max_message: usize,
    pending: Vec<u8>,
    failed: bool,
}

impl Opener {
    /// An opener with this direction's key and reassembly limit
    /// ([`MAX_REQUEST`] on the daemon, [`MAX_RESPONSE`] on the page).
    pub fn new(key: &[u8; 32], max_message: usize) -> Self {
        Self {
            cipher: cipher(key),
            next: 0,
            max_message,
            pending: Vec::new(),
            failed: false,
        }
    }

    /// The counter the next frame must carry.
    pub fn next_counter(&self) -> u64 {
        self.next
    }

    /// Open one frame. `Ok(Some(message))` when it completes a message,
    /// `Ok(None)` for a non-final fragment.
    pub fn open(&mut self, bytes: &[u8]) -> Result<Option<Vec<u8>>> {
        if self.failed {
            return Err(Error::Closed);
        }
        let result = self.open_inner(bytes);
        if result.is_err() {
            self.failed = true;
            self.pending = Vec::new();
        }
        result
    }

    fn open_inner(&mut self, bytes: &[u8]) -> Result<Option<Vec<u8>>> {
        let frame = decode(bytes, MAX_DATA_FRAME)?;
        match frame.kind {
            FrameType::Data => {}
            FrameType::Abort => return Err(Error::Aborted),
            other => return Err(Error::UnexpectedFrame(other as u8)),
        }
        if frame.counter != self.next {
            return Err(Error::Counter {
                expected: self.next,
                got: frame.counter,
            });
        }
        let plaintext = self
            .cipher
            .decrypt(
                &nonce(frame.counter),
                Payload {
                    msg: frame.body,
                    aad: &frame.header(),
                },
            )
            .map_err(|_| Error::Decrypt)?;
        self.next = self.next.checked_add(1).ok_or(Error::CounterExhausted)?;
        if self.pending.len() + plaintext.len() > self.max_message {
            return Err(Error::MessageTooLarge {
                max: self.max_message,
            });
        }
        if frame.flags & FLAG_FINAL == 0 {
            self.pending.extend_from_slice(&plaintext);
            return Ok(None);
        }
        if self.pending.is_empty() {
            return Ok(Some(plaintext));
        }
        let mut message = std::mem::take(&mut self.pending);
        message.extend_from_slice(&plaintext);
        Ok(Some(message))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const KEY: [u8; 32] = [7u8; 32];

    fn pair() -> (Sealer, Opener) {
        (Sealer::new(&KEY), Opener::new(&KEY, MAX_REQUEST))
    }

    #[test]
    fn round_trip_and_counters() {
        let (mut s, mut o) = pair();
        for i in 0..3u8 {
            let f = s.seal_fragment(&[i; 5], true).unwrap();
            assert_eq!(o.open(&f).unwrap(), Some(vec![i; 5]));
        }
        assert_eq!(s.next_counter(), 3);
        assert_eq!(o.next_counter(), 3);
    }

    #[test]
    fn fragmentation_reassembles() {
        let (mut s, mut o) = pair();
        let msg: Vec<u8> = (0..(MAX_FRAGMENT * 2 + 10)).map(|i| i as u8).collect();
        let frames = s.seal_message(&msg).unwrap();
        assert_eq!(frames.len(), 3);
        assert!(frames.iter().all(|f| f.len() <= MAX_DATA_FRAME));
        assert_eq!(frames[0].len(), MAX_DATA_FRAME);
        assert_eq!(o.open(&frames[0]).unwrap(), None);
        assert_eq!(o.open(&frames[1]).unwrap(), None);
        assert_eq!(o.open(&frames[2]).unwrap(), Some(msg));
        let empty = s.seal_message(&[]).unwrap();
        assert_eq!(empty.len(), 1);
        assert_eq!(o.open(&empty[0]).unwrap(), Some(vec![]));
    }

    #[test]
    fn reassembly_limit() {
        let mut s = Sealer::new(&KEY);
        let mut o = Opener::new(&KEY, 100);
        let a = s.seal_fragment(&[1; 60], false).unwrap();
        let b = s.seal_fragment(&[1; 60], true).unwrap();
        assert_eq!(o.open(&a).unwrap(), None);
        assert_eq!(o.open(&b), Err(Error::MessageTooLarge { max: 100 }));
        assert_eq!(o.open(&b), Err(Error::Closed));
        // Exactly at the limit is fine.
        let mut s = Sealer::new(&KEY);
        let mut o = Opener::new(&KEY, 100);
        let a = s.seal_fragment(&[1; 100], true).unwrap();
        assert!(o.open(&a).unwrap().is_some());
    }

    #[test]
    fn oversize_fragment_refused_on_both_sides() {
        let (mut s, mut o) = pair();
        assert!(matches!(
            s.seal_fragment(&vec![0; MAX_FRAGMENT + 1], true),
            Err(Error::FrameTooLarge { .. })
        ));
        let big = encode(FrameType::Data, 1, 0, &vec![0; MAX_DATA_FRAME]);
        assert!(matches!(o.open(&big), Err(Error::FrameTooLarge { .. })));
    }

    #[test]
    fn counter_gap_and_repeat_refused() {
        let (mut s, mut o) = pair();
        let f0 = s.seal_fragment(b"a", true).unwrap();
        let f1 = s.seal_fragment(b"b", true).unwrap();
        // Gap: frame 1 before frame 0.
        assert_eq!(
            o.open(&f1),
            Err(Error::Counter {
                expected: 0,
                got: 1
            })
        );
        assert_eq!(o.open(&f0), Err(Error::Closed));
        // Repeat.
        let (_, mut o) = pair();
        o.open(&f0).unwrap();
        assert_eq!(
            o.open(&f0),
            Err(Error::Counter {
                expected: 1,
                got: 0
            })
        );
    }

    #[test]
    fn tampering_refused() {
        let (mut s, _) = pair();
        let f = s.seal_fragment(b"hello world", true).unwrap();
        for i in 0..f.len() {
            let mut bad = f.clone();
            bad[i] ^= 0x01;
            let (_, mut o) = pair();
            assert!(o.open(&bad).is_err(), "byte {i} flip accepted");
        }
        // Flipping FINAL off (header is AAD) fails decryption.
        let mut bad = f.clone();
        bad[2] = 0;
        let (_, mut o) = pair();
        assert_eq!(o.open(&bad), Err(Error::Decrypt));
        // Wrong key.
        let mut o = Opener::new(&[8u8; 32], MAX_REQUEST);
        assert_eq!(o.open(&f), Err(Error::Decrypt));
        // Truncated tag.
        let (_, mut o) = pair();
        assert_eq!(o.open(&f[..HEADER_LEN + 10]), Err(Error::Short));
    }

    #[test]
    fn header_rules() {
        let body = [0u8; 16];
        let mut f = encode(FrameType::Data, 0, 0, &body);
        f[0] = 2;
        assert_eq!(decode(&f, 2048), Err(Error::Version(2)));
        let mut f = encode(FrameType::Data, 0, 0, &body);
        f[1] = 0x11;
        assert_eq!(decode(&f, 2048), Err(Error::UnknownType(0x11)));
        let f = encode(FrameType::Data, 0x02, 0, &body);
        assert_eq!(decode(&f, 2048), Err(Error::Flags(0x02)));
        let f = encode(FrameType::Hello, 0x01, 0, &body);
        assert_eq!(decode(&f, 2048), Err(Error::Flags(0x01)));
        let f = encode(FrameType::Finish, 0, 1, &body);
        assert_eq!(decode(&f, 2048), Err(Error::HandshakeCounter(1)));
        let f = encode(FrameType::Abort, 0, 0, &[1]);
        assert_eq!(decode(&f, 2048), Err(Error::TrailingBytes));
        assert_eq!(decode(&[1, 1, 0], 2048), Err(Error::Truncated));
        let f = handshake_frame(FrameType::Hello, &[0; 1014]);
        assert!(decode(&f, MAX_HANDSHAKE_FRAME).is_err());
        let f = handshake_frame(FrameType::Hello, &[0; 1013]);
        assert!(decode(&f, MAX_HANDSHAKE_FRAME).is_ok());
        assert!(decode(&abort(), 2048).is_ok());
    }

    #[test]
    fn handshake_frame_after_handshake_refused() {
        let (_, mut o) = pair();
        let f = handshake_frame(FrameType::Hello, b"x");
        assert_eq!(o.open(&f), Err(Error::UnexpectedFrame(0x01)));
        let (_, mut o) = pair();
        assert_eq!(o.open(&abort()), Err(Error::Aborted));
    }

    #[test]
    fn decode_handshake_rules() {
        let f = handshake_frame(FrameType::Finish, b"x");
        assert_eq!(decode_handshake(&f, FrameType::Finish).unwrap(), b"x");
        assert_eq!(
            decode_handshake(&f, FrameType::Hello),
            Err(Error::UnexpectedFrame(0x03))
        );
        assert_eq!(
            decode_handshake(&abort(), FrameType::Hello),
            Err(Error::Aborted)
        );
        let data = encode(FrameType::Data, 1, 0, &[0; 16]);
        assert_eq!(
            decode_handshake(&data, FrameType::Hello),
            Err(Error::UnexpectedFrame(0x10))
        );
    }
}
