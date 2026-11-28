use super::*;
use crate::ids::route_id;
use crate::message::{DaemonMessage, Request, Response, decode_from_daemon};
use p256::NonZeroScalar;

const SECRET: &[u8] = b"7K3M9QXR2TVW4";
const NAME: &str = "Synthetic Kindle";

fn key(seed: u8) -> SigningKey {
    let mut bytes = [0u8; 32];
    bytes[31] = seed;
    bytes[0] = 0x11;
    SigningKey::from_slice(&bytes).unwrap()
}

fn eph(seed: u8) -> SecretKey {
    let mut bytes = [0u8; 32];
    bytes[31] = seed;
    bytes[0] = 0x22;
    SecretKey::from_slice(&bytes).unwrap()
}

fn pub_bytes(k: &SigningKey) -> [u8; 65] {
    ids::verifying_key_bytes(k.verifying_key())
}

fn err<T>(r: Result<T>) -> Error {
    match r {
        Ok(_) => panic!("expected an error"),
        Err(e) => e,
    }
}

/// Re-frame a body as a handshake frame of `kind`.
fn body_of(frame_bytes: &[u8]) -> Vec<u8> {
    frame_bytes[frame::HEADER_LEN..].to_vec()
}

fn session_parts() -> (SigningKey, SigningKey, PageSession, Vec<u8>) {
    let identity = key(1);
    let device = key(2);
    let (page, hello) = PageSession::start(&device, &pub_bytes(&identity)).unwrap();
    (identity, device, page, hello)
}

fn exchange(daemon: &mut DaemonChannel, page: &mut PageChannel) {
    let req = Request {
        request_id: 1,
        method: "GET".into(),
        path_and_query: "/client/v1/state".into(),
        headers: vec![("authorization".into(), "Bearer synthetic".into())],
        body: vec![],
    };
    for f in page.sealer.seal_message(&req.encode().unwrap()).unwrap() {
        if let Some(m) = daemon.opener.open(&f).unwrap() {
            assert_eq!(Request::decode(&m).unwrap(), req);
        }
    }
    let resp = Response {
        request_id: 1,
        status: 200,
        headers: vec![("content-type".into(), "application/json".into())],
        body: vec![b'x'; 200_000],
    };
    let frames = daemon.sealer.seal_message(&resp.encode().unwrap()).unwrap();
    assert_eq!(frames.len(), 4);
    let mut got = None;
    for f in frames {
        got = page.opener.open(&f).unwrap();
    }
    assert_eq!(
        decode_from_daemon(&got.unwrap()).unwrap(),
        DaemonMessage::Response(resp)
    );
}

#[test]
fn session_round_trip() {
    let (identity, device, page, hello) = session_parts();
    let hello = parse_session_hello(&hello).unwrap();
    assert_eq!(hello.device_id(), &device_id(&pub_bytes(&device)));
    let (daemon, reply) = hello.respond(&identity, Some(&pub_bytes(&device))).unwrap();
    let (await_ready, finish) = page.on_reply(&reply).unwrap();
    let mut daemon_ch = daemon.finish(&finish).unwrap();
    let ready = daemon_ch.ready_frame().unwrap();
    let mut page_ch = await_ready.on_ready(&ready).unwrap();
    assert_eq!(daemon_ch.transcript_hash, page_ch.transcript_hash);
    assert_eq!(daemon_ch.device_pub, pub_bytes(&device));
    exchange(&mut daemon_ch, &mut page_ch);
}

#[test]
fn pairing_round_trip() {
    let identity = key(1);
    let device = key(2);
    let (page, hello) = PagePairing::start(&device, NAME, SECRET).unwrap();
    let hello = parse_pairing_hello(&hello).unwrap();
    assert_eq!(hello.device_name(), NAME);
    let full_hash = device_id(hello.device_pub());
    let (daemon, reply) = hello.respond(&identity, SECRET, &full_hash[..5]).unwrap();
    let (mut paired_page, finish) = page.on_reply(&reply).unwrap();
    let mut paired = daemon.finish(&finish).unwrap();
    assert_eq!(paired.device_name, NAME);
    assert_eq!(paired.pairing_id, paired_page.pairing_id);
    assert_eq!(paired.daemon_id, daemon_id(&pub_bytes(&identity)));
    assert_eq!(paired_page.identity_pub, pub_bytes(&identity));
    let route = route_id(&pub_bytes(&key(3)));
    let control = paired.paired_control(&route, "dv-0123456789abcdef0123");
    let frames = paired.channel.sealer.seal_message(&control).unwrap();
    let msg = paired_page
        .channel
        .opener
        .open(&frames[0])
        .unwrap()
        .unwrap();
    let DaemonMessage::Control(json) = decode_from_daemon(&msg).unwrap() else {
        panic!("not control");
    };
    assert_eq!(
        json,
        paired_json(&paired_page.daemon_id, &route, "dv-0123456789abcdef0123")
    );
    exchange(&mut paired.channel, &mut paired_page.channel);
}

#[test]
fn hello_refusals() {
    let (_, _, _, hello) = session_parts();
    // Wrong mode for the connection kind.
    assert_eq!(
        err(parse_pairing_hello(&hello)),
        Error::WrongMode(MODE_SESSION)
    );
    let (_, phello) = PagePairing::start(&key(2), NAME, SECRET).unwrap();
    assert_eq!(
        err(parse_session_hello(&phello)),
        Error::WrongMode(MODE_PAIRING)
    );
    // Unknown mode byte.
    let mut b = body_of(&hello);
    b[4] = 3;
    let f = frame::handshake_frame(FrameType::Hello, &b);
    assert_eq!(err(parse_hello(&f)), Error::WrongMode(3));
    // Trailing bytes.
    let mut b = body_of(&hello);
    b.push(0);
    let f = frame::handshake_frame(FrameType::Hello, &b);
    assert_eq!(err(parse_hello(&f)), Error::TrailingBytes);
    // Truncated.
    let b = body_of(&hello);
    let f = frame::handshake_frame(FrameType::Hello, &b[..b.len() - 1]);
    assert_eq!(err(parse_hello(&f)), Error::Short);
    // Wrong type, ABORT, flags, version.
    let f = frame::handshake_frame(FrameType::Finish, &body_of(&hello));
    assert_eq!(err(parse_hello(&f)), Error::UnexpectedFrame(0x03));
    assert_eq!(err(parse_hello(&frame::abort())), Error::Aborted);
    let mut f = hello.clone();
    f[2] = 1;
    assert_eq!(err(parse_hello(&f)), Error::Flags(1));
    let mut f = hello.clone();
    f[0] = 0;
    assert_eq!(err(parse_hello(&f)), Error::Version(0));
    // Oversize.
    let f = frame::handshake_frame(FrameType::Hello, &[0u8; 1100]);
    assert!(matches!(err(parse_hello(&f)), Error::FrameTooLarge { .. }));
}

fn session_hello_with_point(e_k: &[u8]) -> Vec<u8> {
    frame::handshake_frame(
        FrameType::Hello,
        &lp_concat(&[&[MODE_SESSION], e_k, &[0u8; 32], &[0u8; 32]]),
    )
}

#[test]
fn point_refusals() {
    let good = point_bytes(&eph(5).public_key());
    // Compressed point (33 bytes).
    let compressed = eph(5).public_key().to_sec1_point(true);
    use p256::elliptic_curve::sec1::ToSec1Point;
    let f = session_hello_with_point(compressed.as_bytes());
    assert_eq!(
        err(parse_hello(&f)),
        Error::FieldLength {
            field: "E_k",
            len: 33
        }
    );
    // 65 bytes but a compressed/hybrid tag.
    let mut tagged = good;
    tagged[0] = 0x06;
    assert_eq!(
        err(parse_hello(&session_hello_with_point(&tagged))),
        Error::BadPoint("E_k")
    );
    // Off the curve.
    let mut off = good;
    off[64] ^= 1;
    assert_eq!(
        err(parse_hello(&session_hello_with_point(&off))),
        Error::BadPoint("E_k")
    );
    // Good point parses.
    assert!(parse_hello(&session_hello_with_point(&good)).is_ok());
}

#[test]
fn session_daemon_refusals() {
    let (identity, device, _, hello) = session_parts();
    let h = parse_session_hello(&hello).unwrap();
    assert_eq!(err(h.respond(&identity, None)), Error::UnknownDevice);
    let h = parse_session_hello(&hello).unwrap();
    assert_eq!(
        err(h.respond(&identity, Some(&pub_bytes(&key(9))))),
        Error::DeviceIdMismatch
    );
    let h = parse_session_hello(&hello).unwrap();
    assert_eq!(
        err(h.respond(&identity, Some(&pub_bytes(&device)[..64]))),
        Error::BadPoint("device_pub")
    );
}

#[test]
fn session_page_refusals() {
    // Wrong identity pinned: the daemon's signature fails.
    let identity = key(1);
    let device = key(2);
    let (page, hello) = PageSession::start(&device, &pub_bytes(&key(7))).unwrap();
    let (_, reply) = parse_session_hello(&hello)
        .unwrap()
        .respond(&identity, Some(&pub_bytes(&device)))
        .unwrap();
    assert_eq!(err(page.on_reply(&reply)), Error::BadSignature);

    // Any flipped byte in the reply body is refused.
    let (identity, device, _, _) = session_parts();
    for i in frame::HEADER_LEN.. {
        let (page, hello) = PageSession::start(&device, &pub_bytes(&identity)).unwrap();
        let (_, reply) = parse_session_hello(&hello)
            .unwrap()
            .respond(&identity, Some(&pub_bytes(&device)))
            .unwrap();
        if i >= reply.len() {
            break;
        }
        let mut bad = reply.clone();
        bad[i] ^= 0x80;
        assert!(page.on_reply(&bad).is_err(), "byte {i}");
    }

    // Trailing bytes on the reply.
    let (page, hello) = PageSession::start(&device, &pub_bytes(&identity)).unwrap();
    let (_, reply) = parse_session_hello(&hello)
        .unwrap()
        .respond(&identity, Some(&pub_bytes(&device)))
        .unwrap();
    let mut b = body_of(&reply);
    b.extend_from_slice(&[0, 0, 0, 0]);
    let f = frame::handshake_frame(FrameType::HelloReply, &b);
    assert_eq!(err(page.on_reply(&f)), Error::TrailingBytes);

    // Pinned identity must be an uncompressed point.
    assert_eq!(
        err(PageSession::start(&device, &[4u8; 65])),
        Error::BadPoint("identity_pub")
    );
}

fn session_to_finish() -> (DaemonSession, Vec<u8>, Vec<u8>, SigningKey) {
    let (identity, device, page, hello) = session_parts();
    let (daemon, reply) = parse_session_hello(&hello)
        .unwrap()
        .respond(&identity, Some(&pub_bytes(&device)))
        .unwrap();
    let (_, finish) = page.on_reply(&reply).unwrap();
    (daemon, finish, reply, device)
}

#[test]
fn session_finish_refusals() {
    // Tampered signature.
    let (daemon, finish, _, _) = session_to_finish();
    let mut bad = finish.clone();
    let last = bad.len() - 1;
    bad[last] ^= 1;
    assert_eq!(err(daemon.finish(&bad)), Error::BadSignature);
    // Signature over the right transcript with the pairing prefix.
    let (daemon, _, _, device) = session_to_finish();
    let wrong_prefix = sign(
        &device,
        &signed_message(PAIRING_SIG_PREFIX, Role::Device, &daemon.transcript),
    );
    let f = frame::handshake_frame(FrameType::Finish, &lp_concat(&[&wrong_prefix]));
    assert_eq!(err(daemon.finish(&f)), Error::BadSignature);
    // Signature with the daemon role label (a reflected sig_d).
    let (daemon, _, _, device) = session_to_finish();
    let wrong_role = sign(
        &device,
        &signed_message(SESSION_SIG_PREFIX, Role::Daemon, &daemon.transcript),
    );
    let f = frame::handshake_frame(FrameType::Finish, &lp_concat(&[&wrong_role]));
    assert_eq!(err(daemon.finish(&f)), Error::BadSignature);
    // Reflecting HELLO_REPLY's frame as FINISH.
    let (daemon, _, reply, _) = session_to_finish();
    assert_eq!(err(daemon.finish(&reply)), Error::UnexpectedFrame(0x02));
    // Trailing bytes.
    let (daemon, finish, _, _) = session_to_finish();
    let mut b = body_of(&finish);
    b.push(0);
    let f = frame::handshake_frame(FrameType::Finish, &b);
    assert_eq!(err(daemon.finish(&f)), Error::TrailingBytes);
    // Short signature field.
    let (daemon, _, _, _) = session_to_finish();
    let f = frame::handshake_frame(FrameType::Finish, &lp_concat(&[&[0u8; 63]]));
    assert_eq!(
        err(daemon.finish(&f)),
        Error::FieldLength {
            field: "sig_k",
            len: 63
        }
    );
    // ABORT.
    let (daemon, _, _, _) = session_to_finish();
    assert_eq!(err(daemon.finish(&frame::abort())), Error::Aborted);
}

#[test]
fn ready_refusals() {
    let setup = || {
        let (identity, device, page, hello) = session_parts();
        let (daemon, reply) = parse_session_hello(&hello)
            .unwrap()
            .respond(&identity, Some(&pub_bytes(&device)))
            .unwrap();
        let (await_ready, finish) = page.on_reply(&reply).unwrap();
        (await_ready, daemon.finish(&finish).unwrap())
    };
    // A response instead of ready.
    let (w, mut d) = setup();
    let resp = Response {
        request_id: 1,
        status: 200,
        headers: vec![],
        body: vec![],
    };
    let f = d
        .sealer
        .seal_fragment(&resp.encode().unwrap(), true)
        .unwrap();
    assert_eq!(
        err(w.on_ready(&f)),
        Error::BadMessage("expected ready control")
    );
    // Other control JSON.
    let (w, mut d) = setup();
    let f = d
        .sealer
        .seal_fragment(&message::encode_control(br#"{"type":"ready","x":1}"#), true)
        .unwrap();
    assert_eq!(
        err(w.on_ready(&f)),
        Error::BadMessage("expected ready control")
    );
    // A non-final fragment.
    let (w, mut d) = setup();
    let f = d
        .sealer
        .seal_fragment(&message::encode_control(READY_JSON), false)
        .unwrap();
    assert_eq!(
        err(w.on_ready(&f)),
        Error::BadMessage("ready must be one frame")
    );
    // A second handshake frame instead.
    let (w, _) = setup();
    let f = frame::handshake_frame(FrameType::HelloReply, b"");
    assert_eq!(err(w.on_ready(&f)), Error::UnexpectedFrame(0x02));
    // Ready at the wrong counter.
    let (w, mut d) = setup();
    let _ = d.ready_frame().unwrap();
    let f = d.ready_frame().unwrap();
    assert_eq!(
        err(w.on_ready(&f)),
        Error::Counter {
            expected: 0,
            got: 1
        }
    );
}

#[test]
fn high_s_signature_accepted() {
    let (daemon, finish, _, device) = session_to_finish();
    let _ = finish;
    let msg = signed_message(SESSION_SIG_PREFIX, Role::Device, &daemon.transcript);
    let sig: Signature = device.sign(&msg);
    let (r, s) = sig.split_scalars();
    let high_s: NonZeroScalar = -s;
    let high = Signature::from_scalars(r.to_bytes(), high_s.to_bytes()).unwrap();
    let high_bytes: [u8; 64] = high.to_bytes().into();
    assert_ne!(high_bytes, <[u8; 64]>::from(sig.to_bytes()));
    let f = frame::handshake_frame(FrameType::Finish, &lp_concat(&[&high_bytes]));
    assert!(daemon.finish(&f).is_ok());
}

fn pairing_reply(
    secret_daemon: &[u8],
    hash_len: usize,
) -> (PagePairing, Result<(DaemonPairing, Vec<u8>)>) {
    let (page, hello) = PagePairing::start(&key(2), NAME, SECRET).unwrap();
    let h = parse_pairing_hello(&hello).unwrap();
    let hash = device_id(h.device_pub());
    (page, h.respond(&key(1), secret_daemon, &hash[..hash_len]))
}

#[test]
fn pairing_daemon_refusals() {
    assert_eq!(err(pairing_reply(b"7K3M9QXR2TVW", 32).1), Error::BadSecret);
    assert_eq!(err(pairing_reply(b"7k3m9qxr2tvw4", 32).1), Error::BadSecret);
    assert_eq!(err(pairing_reply(b"7K3M9QXR2TVWU", 32).1), Error::BadSecret);
    assert_eq!(err(pairing_reply(SECRET, 3).1), Error::BadDeviceKeyHash);
    assert!(pairing_reply(SECRET, 4).1.is_ok());
    // Hash of another device.
    let (_, hello) = PagePairing::start(&key(2), NAME, SECRET).unwrap();
    let h = parse_pairing_hello(&hello).unwrap();
    let other = device_id(&pub_bytes(&key(8)));
    assert_eq!(
        err(h.respond(&key(1), SECRET, &other[..8])),
        Error::DeviceKeyHashMismatch
    );
    // device_pub must be a point.
    let e_k = point_bytes(&eph(1).public_key());
    let f = frame::handshake_frame(
        FrameType::Hello,
        &lp_concat(&[&[MODE_PAIRING], &e_k, &[0; 32], &[4u8; 65], NAME.as_bytes()]),
    );
    assert_eq!(err(parse_hello(&f)), Error::BadPoint("device_pub"));
    // Device names.
    let dp = pub_bytes(&key(2));
    for name in [&b""[..], &[b'a'; 61][..], &[0xC3][..]] {
        let f = frame::handshake_frame(
            FrameType::Hello,
            &lp_concat(&[&[MODE_PAIRING], &e_k, &[0; 32], &dp, name]),
        );
        assert_eq!(err(parse_hello(&f)), Error::BadDeviceName);
    }
    let f = frame::handshake_frame(
        FrameType::Hello,
        &lp_concat(&[&[MODE_PAIRING], &e_k, &[0; 32], &dp, &[b'a'; 60]]),
    );
    assert!(parse_hello(&f).is_ok());
    assert_eq!(
        err(PagePairing::start(&key(2), "", SECRET)),
        Error::BadDeviceName
    );
    assert_eq!(
        err(PagePairing::start(&key(2), NAME, b"short")),
        Error::BadSecret
    );
}

#[test]
fn pairing_page_refusals() {
    // Daemon used a different secret: the page's MAC check fails.
    let (page, r) = pairing_reply(b"7K3M9QXR2TVW5", 32);
    let (_, reply) = r.unwrap();
    assert_eq!(err(page.on_reply(&reply)), Error::BadMac);
    // Every flipped body byte is refused.
    for i in frame::HEADER_LEN.. {
        let (page, r) = pairing_reply(SECRET, 32);
        let (_, reply) = r.unwrap();
        if i >= reply.len() {
            break;
        }
        let mut bad = reply.clone();
        bad[i] ^= 0x80;
        assert!(page.on_reply(&bad).is_err(), "byte {i}");
    }
    // Valid MAC but sig_d by a different key than identity_pub claims.
    let (page, r) = pairing_reply(SECRET, 32);
    let (daemon, reply) = r.unwrap();
    let mut r = Reader::new(&reply[frame::HEADER_LEN..]);
    let fields: Vec<Vec<u8>> = (0..6).map(|_| r.lp().unwrap().to_vec()).collect();
    let forged = sign(
        &key(9),
        &signed_message(PAIRING_SIG_PREFIX, Role::Daemon, &daemon.transcript),
    );
    let f = frame::handshake_frame(
        FrameType::HelloReply,
        &lp_concat(&[
            &fields[0], &fields[1], &fields[2], &fields[3], &fields[4], &forged,
        ]),
    );
    assert_eq!(err(page.on_reply(&f)), Error::BadSignature);
    // Session prefix instead of pairing prefix.
    let (page, r) = pairing_reply(SECRET, 32);
    let (daemon, reply) = r.unwrap();
    let mut r = Reader::new(&reply[frame::HEADER_LEN..]);
    let fields: Vec<Vec<u8>> = (0..6).map(|_| r.lp().unwrap().to_vec()).collect();
    let wrong = sign(
        &key(1),
        &signed_message(SESSION_SIG_PREFIX, Role::Daemon, &daemon.transcript),
    );
    let f = frame::handshake_frame(
        FrameType::HelloReply,
        &lp_concat(&[
            &fields[0], &fields[1], &fields[2], &fields[3], &fields[4], &wrong,
        ]),
    );
    assert_eq!(err(page.on_reply(&f)), Error::BadSignature);
}

#[test]
fn pairing_finish_refusals() {
    let run = || {
        let (page, r) = pairing_reply(SECRET, 32);
        let (daemon, reply) = r.unwrap();
        let (_, finish) = page.on_reply(&reply).unwrap();
        (daemon, finish)
    };
    // Tampered MAC.
    let (daemon, finish) = run();
    let mut bad = finish.clone();
    bad[frame::HEADER_LEN + 4] ^= 1;
    assert_eq!(err(daemon.finish(&bad)), Error::BadMac);
    // Tampered signature.
    let (daemon, finish) = run();
    let mut bad = finish.clone();
    let last = bad.len() - 1;
    bad[last] ^= 1;
    assert_eq!(err(daemon.finish(&bad)), Error::BadSignature);
    // Reflected mac_d instead of mac_k.
    let (daemon, finish) = run();
    let mac_d = hmac(
        &ConfirmKeys::derive(&daemon.th, &daemon.ikm).daemon,
        &daemon.th,
    );
    let mut r = Reader::new(&finish[frame::HEADER_LEN..]);
    let _ = r.lp().unwrap();
    let sig_k = r.lp().unwrap().to_vec();
    let f = frame::handshake_frame(FrameType::Finish, &lp_concat(&[&mac_d, &sig_k]));
    assert_eq!(err(daemon.finish(&f)), Error::BadMac);
    // Missing MAC (a session FINISH).
    let (daemon, finish) = run();
    let mut r = Reader::new(&finish[frame::HEADER_LEN..]);
    let _ = r.lp().unwrap();
    let sig_k = r.lp().unwrap().to_vec();
    let f = frame::handshake_frame(FrameType::Finish, &lp_concat(&[&sig_k]));
    assert!(daemon.finish(&f).is_err());
    // Trailing.
    let (daemon, finish) = run();
    let mut b = body_of(&finish);
    b.push(9);
    let f = frame::handshake_frame(FrameType::Finish, &b);
    assert_eq!(err(daemon.finish(&f)), Error::TrailingBytes);
}

#[test]
fn modes_do_not_cross() {
    // Session keys and pairing keys from the same ECDH differ (secret in
    // the ikm, different transcripts): a sanity check on the derivation.
    let th = [1u8; 32];
    let a = DataKeys::derive(&th, &[2u8; 32]);
    let b = DataKeys::derive(&th, &pairing_ikm(&[2u8; 32], SECRET));
    assert_ne!(a.d2p, b.d2p);
    assert_ne!(a.d2p, a.p2d);
}

#[test]
fn deterministic_with_fixed_randomness() {
    let identity = key(1);
    let device = key(2);
    let run = || {
        let (page, hello) =
            PageSession::start_with(&device, &pub_bytes(&identity), eph(1), [3; 32]).unwrap();
        let (_, reply) = parse_session_hello(&hello)
            .unwrap()
            .respond_with(&identity, Some(&pub_bytes(&device)), eph(2), [4; 32])
            .unwrap();
        let (_, finish) = page.on_reply(&reply).unwrap();
        (hello, reply, finish)
    };
    assert_eq!(run(), run());
}
