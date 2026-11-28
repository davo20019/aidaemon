//! Messages inside DATA (SPEC §8): request, response and control.
//!
//! The spec fixes the layout and the header allowlists; the other checks
//! here (method, path, header value characters, no repeated header, status
//! range, UTF-8 JSON) are this implementation's strict reading, because the
//! daemon hands these values to its HTTP gateway.

use crate::error::{Error, Result};
use crate::wire::{Reader, put_lp};

/// Request message type byte.
pub const TYPE_REQUEST: u8 = 1;
/// Response message type byte.
pub const TYPE_RESPONSE: u8 = 2;
/// Control message type byte.
pub const TYPE_CONTROL: u8 = 3;

/// Headers a request may carry.
pub const REQUEST_HEADERS: &[&str] = &["authorization", "content-type", "x-aidaemon-request-id"];
/// Headers a response may carry.
pub const RESPONSE_HEADERS: &[&str] = &["content-type"];

/// Longest method.
pub const MAX_METHOD: usize = 16;
/// Longest path and query.
pub const MAX_PATH: usize = 8 * 1024;
/// Longest header value.
pub const MAX_HEADER_VALUE: usize = 8 * 1024;

/// A request, page → daemon.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request {
    pub request_id: u32,
    pub method: String,
    pub path_and_query: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

/// A response, daemon → page.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Response {
    pub request_id: u32,
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

/// What the daemon may send the page.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DaemonMessage {
    Response(Response),
    /// The control message's JSON bytes (valid UTF-8).
    Control(Vec<u8>),
}

fn check_method(method: &str) -> Result<()> {
    if method.is_empty()
        || method.len() > MAX_METHOD
        || !method.bytes().all(|b| b.is_ascii_uppercase())
    {
        return Err(Error::BadMessage("method"));
    }
    Ok(())
}

fn check_path(path: &str) -> Result<()> {
    let b = path.as_bytes();
    // Machine format (URL path): must be origin-form; `//` would read as an
    // authority to a URL joiner.
    if b.is_empty()
        || b.len() > MAX_PATH
        || b[0] != b'/'
        || b.starts_with(b"//")
        || !b.iter().all(|&c| (0x21..=0x7E).contains(&c))
    {
        return Err(Error::BadMessage("path_and_query"));
    }
    Ok(())
}

fn check_headers(headers: &[(String, String)], allowed: &[&str]) -> Result<()> {
    for (i, (name, value)) in headers.iter().enumerate() {
        if name.is_empty() || !name.bytes().all(|b| b.is_ascii_lowercase() || b == b'-') {
            return Err(Error::BadMessage("header name"));
        }
        if !allowed.contains(&name.as_str()) {
            return Err(Error::HeaderNotAllowed(name.clone()));
        }
        if headers[..i].iter().any(|(n, _)| n == name) {
            return Err(Error::BadMessage("repeated header"));
        }
        if value.len() > MAX_HEADER_VALUE
            || !value
                .bytes()
                .all(|b| b == b'\t' || (0x20..=0x7E).contains(&b))
        {
            return Err(Error::BadMessage("header value"));
        }
    }
    Ok(())
}

fn check_status(status: u16) -> Result<()> {
    if (100..=599).contains(&status) {
        Ok(())
    } else {
        Err(Error::BadMessage("status"))
    }
}

fn utf8(bytes: &[u8], what: &'static str) -> Result<String> {
    String::from_utf8(bytes.to_vec()).map_err(|_| Error::BadMessage(what))
}

fn put_headers(out: &mut Vec<u8>, headers: &[(String, String)]) {
    out.extend_from_slice(&(headers.len() as u16).to_be_bytes());
    for (name, value) in headers {
        put_lp(out, name.as_bytes());
        put_lp(out, value.as_bytes());
    }
}

fn read_headers(r: &mut Reader<'_>) -> Result<Vec<(String, String)>> {
    let n = r.u16()? as usize;
    // Each header is at least 8 bytes; never trust `n` for an allocation.
    let mut headers = Vec::with_capacity(n.min(8));
    for _ in 0..n {
        let name = utf8(r.lp()?, "header name")?;
        let value = utf8(r.lp()?, "header value")?;
        headers.push((name, value));
    }
    Ok(headers)
}

impl Request {
    /// Encode after checking every rule; a page must not send what the
    /// daemon would refuse.
    pub fn encode(&self) -> Result<Vec<u8>> {
        check_method(&self.method)?;
        check_path(&self.path_and_query)?;
        check_headers(&self.headers, REQUEST_HEADERS)?;
        if self.headers.len() > u16::MAX as usize {
            return Err(Error::BadMessage("too many headers"));
        }
        let mut out = Vec::with_capacity(64 + self.body.len());
        out.push(TYPE_REQUEST);
        out.extend_from_slice(&self.request_id.to_be_bytes());
        put_lp(&mut out, self.method.as_bytes());
        put_lp(&mut out, self.path_and_query.as_bytes());
        put_headers(&mut out, &self.headers);
        put_lp(&mut out, &self.body);
        Ok(out)
    }

    /// Decode a request (the only message a page may send).
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let mut r = Reader::new(bytes);
        let kind = r.u8()?;
        if kind != TYPE_REQUEST {
            return Err(Error::BadMessage("not a request"));
        }
        let request_id = r.u32()?;
        let method = utf8(r.lp()?, "method")?;
        let path_and_query = utf8(r.lp()?, "path_and_query")?;
        let headers = read_headers(&mut r)?;
        let body = r.lp()?.to_vec();
        r.finish()?;
        check_method(&method)?;
        check_path(&path_and_query)?;
        check_headers(&headers, REQUEST_HEADERS)?;
        Ok(Self {
            request_id,
            method,
            path_and_query,
            headers,
            body,
        })
    }

    /// The value of a header, if present.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_str())
    }
}

impl Response {
    /// Encode after checking every rule.
    pub fn encode(&self) -> Result<Vec<u8>> {
        check_status(self.status)?;
        check_headers(&self.headers, RESPONSE_HEADERS)?;
        let mut out = Vec::with_capacity(32 + self.body.len());
        out.push(TYPE_RESPONSE);
        out.extend_from_slice(&self.request_id.to_be_bytes());
        out.extend_from_slice(&self.status.to_be_bytes());
        put_headers(&mut out, &self.headers);
        put_lp(&mut out, &self.body);
        Ok(out)
    }

    fn decode_after_type(r: &mut Reader<'_>) -> Result<Self> {
        let request_id = r.u32()?;
        let status = r.u16()?;
        let headers = read_headers(r)?;
        let body = r.lp()?.to_vec();
        check_status(status)?;
        check_headers(&headers, RESPONSE_HEADERS)?;
        Ok(Self {
            request_id,
            status,
            headers,
            body,
        })
    }
}

/// Encode a control message: `u8 3 ‖ lp(json)`.
pub fn encode_control(json: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(5 + json.len());
    out.push(TYPE_CONTROL);
    put_lp(&mut out, json);
    out
}

/// Decode what the daemon sent: a response or a control message.
pub fn decode_from_daemon(bytes: &[u8]) -> Result<DaemonMessage> {
    let mut r = Reader::new(bytes);
    let message = match r.u8()? {
        TYPE_RESPONSE => DaemonMessage::Response(Response::decode_after_type(&mut r)?),
        TYPE_CONTROL => {
            let json = r.lp()?;
            if json.is_empty() || std::str::from_utf8(json).is_err() {
                return Err(Error::BadMessage("control json"));
            }
            DaemonMessage::Control(json.to_vec())
        }
        _ => return Err(Error::BadMessage("not a response or control")),
    };
    r.finish()?;
    Ok(message)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> Request {
        Request {
            request_id: 7,
            method: "POST".into(),
            path_and_query: "/client/v1/chat?x=1".into(),
            headers: vec![
                ("authorization".into(), "Bearer t".into()),
                ("content-type".into(), "application/json".into()),
            ],
            body: b"{}".to_vec(),
        }
    }

    #[test]
    fn request_round_trip() {
        let r = request();
        let bytes = r.encode().unwrap();
        assert_eq!(Request::decode(&bytes).unwrap(), r);
        assert_eq!(r.header("content-type"), Some("application/json"));
    }

    #[test]
    fn response_and_control_round_trip() {
        let resp = Response {
            request_id: 7,
            status: 200,
            headers: vec![("content-type".into(), "text/plain".into())],
            body: b"ok".to_vec(),
        };
        let bytes = resp.encode().unwrap();
        assert_eq!(
            decode_from_daemon(&bytes).unwrap(),
            DaemonMessage::Response(resp)
        );
        let c = encode_control(b"{\"type\":\"x\"}");
        assert_eq!(
            decode_from_daemon(&c).unwrap(),
            DaemonMessage::Control(b"{\"type\":\"x\"}".to_vec())
        );
    }

    #[test]
    fn header_allowlists() {
        let mut r = request();
        r.headers.push(("cookie".into(), "a".into()));
        assert_eq!(r.encode(), Err(Error::HeaderNotAllowed("cookie".into())));
        let mut r = request();
        r.headers = vec![("Content-Type".into(), "a".into())];
        assert_eq!(r.encode(), Err(Error::BadMessage("header name")));
        let mut r = request();
        r.headers.push(("authorization".into(), "again".into()));
        assert_eq!(r.encode(), Err(Error::BadMessage("repeated header")));
        let mut r = request();
        r.headers = vec![("content-type".into(), "a\r\nx: y".into())];
        assert_eq!(r.encode(), Err(Error::BadMessage("header value")));
        let resp = Response {
            request_id: 1,
            status: 200,
            headers: vec![("authorization".into(), "x".into())],
            body: vec![],
        };
        assert!(matches!(resp.encode(), Err(Error::HeaderNotAllowed(_))));
        // A disallowed header in decoded bytes is refused too.
        let mut bytes = vec![TYPE_REQUEST, 0, 0, 0, 1];
        put_lp(&mut bytes, b"GET");
        put_lp(&mut bytes, b"/");
        bytes.extend_from_slice(&1u16.to_be_bytes());
        put_lp(&mut bytes, b"cookie");
        put_lp(&mut bytes, b"a");
        put_lp(&mut bytes, b"");
        assert!(matches!(
            Request::decode(&bytes),
            Err(Error::HeaderNotAllowed(_))
        ));
        let mut bytes = vec![TYPE_RESPONSE, 0, 0, 0, 1, 0, 200];
        bytes.extend_from_slice(&1u16.to_be_bytes());
        put_lp(&mut bytes, b"set-cookie");
        put_lp(&mut bytes, b"a");
        put_lp(&mut bytes, b"");
        assert!(matches!(
            decode_from_daemon(&bytes),
            Err(Error::HeaderNotAllowed(_))
        ));
    }

    #[test]
    fn method_path_status_rules() {
        for m in ["", "get", "GE T", "AAAAAAAAAAAAAAAAA"] {
            let mut r = request();
            r.method = m.into();
            assert_eq!(r.encode(), Err(Error::BadMessage("method")), "{m}");
        }
        for p in ["", "x", "//evil.example/x", "/a b", "/\u{e9}"] {
            let mut r = request();
            r.path_and_query = p.into();
            assert_eq!(r.encode(), Err(Error::BadMessage("path_and_query")), "{p}");
        }
        for s in [0u16, 99, 600] {
            let resp = Response {
                request_id: 1,
                status: s,
                headers: vec![],
                body: vec![],
            };
            assert_eq!(resp.encode(), Err(Error::BadMessage("status")));
        }
    }

    #[test]
    fn structure_refusals() {
        let bytes = request().encode().unwrap();
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert_eq!(Request::decode(&trailing), Err(Error::TrailingBytes));
        for cut in 0..bytes.len() {
            assert!(Request::decode(&bytes[..cut]).is_err(), "cut {cut}");
        }
        let mut wrong = bytes.clone();
        wrong[0] = TYPE_RESPONSE;
        assert_eq!(
            Request::decode(&wrong),
            Err(Error::BadMessage("not a request"))
        );
        assert!(decode_from_daemon(&bytes).is_err());
        // A huge declared header count with no data is refused, not
        // allocated.
        let mut bytes = vec![TYPE_RESPONSE, 0, 0, 0, 1, 0, 200, 0xFF, 0xFF];
        assert_eq!(decode_from_daemon(&bytes), Err(Error::Short));
        bytes.truncate(7);
        bytes.extend_from_slice(&[0, 0, 0xFF, 0xFF, 0xFF, 0xFF]);
        assert_eq!(decode_from_daemon(&bytes), Err(Error::Short));
        let mut c = encode_control(b"{}");
        c.push(1);
        assert_eq!(decode_from_daemon(&c), Err(Error::TrailingBytes));
        assert_eq!(
            decode_from_daemon(&encode_control(b"")),
            Err(Error::BadMessage("control json"))
        );
        assert_eq!(
            decode_from_daemon(&encode_control(&[0xFF])),
            Err(Error::BadMessage("control json"))
        );
    }
}
