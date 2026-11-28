//! Test vectors (SPEC §9). This test recomputes `vectors/v1.json` from
//! fixed inputs and fails if the file differs, so the vectors cannot drift
//! from the code. Regenerate on purpose with
//! `UPDATE_BRIDGE_VECTORS=1 cargo test -p aidaemon-bridge-proto --test vectors`.

use aidaemon_bridge_proto::frame::{self, FrameType, Sealer};
use aidaemon_bridge_proto::handshake::{
    self, ConfirmKeys, DataKeys, PAIRING_SIG_PREFIX, PagePairing, PageSession, Role,
    SESSION_SIG_PREFIX, ecdh, hkdf_extract, hmac, paired_json, pairing_transcript,
    parse_pairing_hello, parse_session_hello, session_transcript, signed_message,
};
use aidaemon_bridge_proto::ids::{
    daemon_id, device_id, parse_ecdh_point, point_bytes, route_id, sha256, verifying_key_bytes,
};
use aidaemon_bridge_proto::login::{login_message, sign_login, verify_login};
use aidaemon_bridge_proto::message::{Request, Response, encode_control};
use p256::ecdsa::{Signature, SigningKey};
use p256::{NonZeroScalar, SecretKey};
use serde_json::{Value, json};

const SECRET: &str = "7K3M9QXR2TVW4";
const NAMEPLATE: &str = "4F7Q";
const DEVICE_NAME: &str = "Mia\u{2019}s Kindle";

fn seed(label: &str) -> [u8; 32] {
    sha256(format!("aidaemon-bridge-v1 test vector {label}").as_bytes())
}

fn signing(label: &str) -> ([u8; 32], SigningKey) {
    let d = seed(label);
    (d, SigningKey::from_slice(&d).expect("scalar in range"))
}

fn ephemeral(label: &str) -> ([u8; 32], SecretKey) {
    let d = seed(label);
    (d, SecretKey::from_slice(&d).expect("scalar in range"))
}

fn h(bytes: impl AsRef<[u8]>) -> Value {
    Value::String(hex::encode(bytes))
}

fn high_s(sig: &[u8]) -> [u8; 64] {
    let sig = Signature::from_slice(sig).unwrap();
    let (r, s) = sig.split_scalars();
    let neg: NonZeroScalar = -s;
    Signature::from_scalars(r.to_bytes(), neg.to_bytes())
        .unwrap()
        .to_bytes()
        .into()
}

fn lp_fields(frame_bytes: &[u8]) -> Vec<Vec<u8>> {
    let mut body = &frame_bytes[frame::HEADER_LEN..];
    let mut out = Vec::new();
    while !body.is_empty() {
        let len = u32::from_be_bytes(body[..4].try_into().unwrap()) as usize;
        out.push(body[4..4 + len].to_vec());
        body = &body[4 + len..];
    }
    out
}

fn data(
    direction: &str,
    counter: u64,
    final_fragment: bool,
    plaintext: &[u8],
    frame: &[u8],
) -> Value {
    json!({
        "direction": direction,
        "counter": counter,
        "final": final_fragment,
        "plaintext": h(plaintext),
        "frame": h(frame),
    })
}

fn request() -> Request {
    Request {
        request_id: 0x0102_0304,
        method: "POST".into(),
        path_and_query: "/client/v1/sessions/synthetic-1/messages?wait=1".into(),
        headers: vec![
            ("authorization".into(), "Bearer synthetic-token".into()),
            ("content-type".into(), "application/json".into()),
            ("x-aidaemon-request-id".into(), "req-1".into()),
        ],
        body: br#"{"text":"hello from the e-ink reader"}"#.to_vec(),
    }
}

fn response() -> Response {
    Response {
        request_id: 0x0102_0304,
        status: 201,
        headers: vec![("content-type".into(), "application/json".into())],
        body: br#"{"ok":true}"#.to_vec(),
    }
}

fn request_json(r: &Request) -> Value {
    json!({
        "request_id": r.request_id,
        "method": r.method,
        "path_and_query": r.path_and_query,
        "headers": r.headers.iter().map(|(n, v)| json!([n, v])).collect::<Vec<_>>(),
        "body": h(&r.body),
        "encoded": h(r.encode().unwrap()),
    })
}

fn response_json(r: &Response) -> Value {
    json!({
        "request_id": r.request_id,
        "status": r.status,
        "headers": r.headers.iter().map(|(n, v)| json!([n, v])).collect::<Vec<_>>(),
        "body": h(&r.body),
        "encoded": h(r.encode().unwrap()),
    })
}

fn build() -> Value {
    let (identity_d, identity) = signing("identity key");
    let (device_d, device) = signing("device key");
    let (login_d, login) = signing("login key");
    let identity_pub = verifying_key_bytes(identity.verifying_key());
    let device_pub = verifying_key_bytes(device.verifying_key());
    let login_pub = verifying_key_bytes(login.verifying_key());
    let ids = json!({
        "daemon_id": h(daemon_id(&identity_pub)),
        "device_id": h(device_id(&device_pub)),
        "route_id": h(route_id(&login_pub)),
    });

    let login_nonce = seed("login nonce");
    let login_sig = sign_login(&login, &login_nonce);
    assert!(verify_login(&login_pub, &login_nonce, &login_sig));

    let req = request();
    let resp = response();
    let req_bytes = req.encode().unwrap();
    let resp_bytes = resp.encode().unwrap();

    // ---- Session (mode 1).
    let (sk_d, sk) = ephemeral("session page ephemeral");
    let (sd_d, sd) = ephemeral("session daemon ephemeral");
    let n_k: [u8; 32] = seed("session page nonce");
    let n_d: [u8; 32] = seed("session daemon nonce");
    let e_k = point_bytes(&sk.public_key());
    let e_d = point_bytes(&sd.public_key());
    let (page, hello) = PageSession::start_with(&device, &identity_pub, sk.clone(), n_k).unwrap();
    let (daemon, reply) = parse_session_hello(&hello)
        .unwrap()
        .respond_with(&identity, Some(&device_pub), sd.clone(), n_d)
        .unwrap();
    let (await_ready, finish) = page.on_reply(&reply).unwrap();
    let mut daemon_ch = daemon.finish(&finish).unwrap();
    let ready_message = encode_control(handshake::READY_JSON);
    let ready = daemon_ch.ready_frame().unwrap();
    let mut page_ch = await_ready.on_ready(&ready).unwrap();
    let t = session_transcript(&identity_pub, &device_pub, &e_k, &e_d, &n_k, &n_d);
    let th = sha256(&t);
    assert_eq!(th, daemon_ch.transcript_hash);
    let shared = ecdh(&sk, &parse_ecdh_point(&e_d).unwrap());
    assert_eq!(shared, ecdh(&sd, &parse_ecdh_point(&e_k).unwrap()));
    let keys = DataKeys::derive(&th, &shared);
    let m_d = signed_message(SESSION_SIG_PREFIX, Role::Daemon, &t);
    let m_k = signed_message(SESSION_SIG_PREFIX, Role::Device, &t);
    let sig_d = lp_fields(&reply)[2].clone();
    let sig_k = lp_fields(&finish)[0].clone();
    assert_eq!(sig_d, handshake::sign(&identity, &m_d));
    let high = high_s(&sig_d);
    handshake::verify(&identity_pub, &m_d, &high).unwrap();

    let p2d = page_ch.sealer.seal_fragment(&req_bytes, true).unwrap();
    assert_eq!(daemon_ch.opener.open(&p2d).unwrap().unwrap(), req_bytes);
    let d2p = daemon_ch.sealer.seal_fragment(&resp_bytes, true).unwrap();
    assert_eq!(page_ch.opener.open(&d2p).unwrap().unwrap(), resp_bytes);
    // Two fragments of one response at counters 2 and 3.
    let (a, b) = resp_bytes.split_at(7);
    let frag1 = daemon_ch.sealer.seal_fragment(a, false).unwrap();
    let frag2 = daemon_ch.sealer.seal_fragment(b, true).unwrap();
    assert_eq!(page_ch.opener.open(&frag1).unwrap(), None);
    assert_eq!(page_ch.opener.open(&frag2).unwrap().unwrap(), resp_bytes);

    let session = json!({
        "page_ephemeral_private": h(sk_d),
        "E_k": h(e_k),
        "daemon_ephemeral_private": h(sd_d),
        "E_d": h(e_d),
        "N_k": h(n_k),
        "N_d": h(n_d),
        "hello": h(&hello),
        "transcript": h(&t),
        "transcript_hash": h(th),
        "daemon_signed_message": h(&m_d),
        "device_signed_message": h(&m_k),
        "sig_d": h(&sig_d),
        "sig_d_high_s": h(high),
        "sig_k": h(&sig_k),
        "hello_reply": h(&reply),
        "finish": h(&finish),
        "ecdh": h(shared),
        "prk": h(hkdf_extract(&th, &shared)),
        "key_d2p": h(keys.d2p),
        "key_p2d": h(keys.p2d),
        "ready_json": std::str::from_utf8(handshake::READY_JSON).unwrap(),
        "data": [
            data("d2p", 0, true, &ready_message, &ready),
            data("p2d", 0, true, &req_bytes, &p2d),
            data("d2p", 1, true, &resp_bytes, &d2p),
            data("d2p", 2, false, a, &frag1),
            data("d2p", 3, true, b, &frag2),
        ],
    });

    // ---- Pairing (mode 2).
    let (pk_d, pk) = ephemeral("pairing page ephemeral");
    let (pd_d, pd) = ephemeral("pairing daemon ephemeral");
    let pn_k: [u8; 32] = seed("pairing page nonce");
    let pn_d: [u8; 32] = seed("pairing daemon nonce");
    let pairing_id: [u8; 16] = seed("pairing id")[..16].try_into().unwrap();
    let pe_k = point_bytes(&pk.public_key());
    let pe_d = point_bytes(&pd.public_key());
    let device_key_hash = &device_id(&device_pub)[..8];
    let (ppage, phello) =
        PagePairing::start_with(&device, DEVICE_NAME, SECRET.as_bytes(), pk.clone(), pn_k).unwrap();
    let (pdaemon, preply) = parse_pairing_hello(&phello)
        .unwrap()
        .respond_with(
            &identity,
            SECRET.as_bytes(),
            device_key_hash,
            pd.clone(),
            pn_d,
            pairing_id,
        )
        .unwrap();
    let (mut ppaired, pfinish) = ppage.on_reply(&preply).unwrap();
    let mut dpaired = pdaemon.finish(&pfinish).unwrap();
    let pt = pairing_transcript(
        &device_pub,
        DEVICE_NAME.as_bytes(),
        &pe_k,
        &pn_k,
        &identity_pub,
        &pe_d,
        &pn_d,
        &pairing_id,
    );
    let pth = sha256(&pt);
    assert_eq!(pth, dpaired.channel.transcript_hash);
    let pshared = ecdh(&pk, &parse_ecdh_point(&pe_d).unwrap());
    let mut ikm = pshared.to_vec();
    ikm.extend_from_slice(SECRET.as_bytes());
    let confirm = ConfirmKeys::derive(&pth, &ikm);
    let pkeys = DataKeys::derive(&pth, &ikm);
    let mac_d = hmac(&confirm.daemon, &pth);
    let mac_k = hmac(&confirm.device, &pth);
    let pm_d = signed_message(PAIRING_SIG_PREFIX, Role::Daemon, &pt);
    let pm_k = signed_message(PAIRING_SIG_PREFIX, Role::Device, &pt);
    let reply_fields = lp_fields(&preply);
    assert_eq!(reply_fields[4], mac_d);
    let finish_fields = lp_fields(&pfinish);
    assert_eq!(finish_fields[0], mac_k);
    let psig_d = reply_fields[5].clone();
    let psig_k = finish_fields[1].clone();
    let route = route_id(&login_pub);
    let control_json = paired_json(&daemon_id(&identity_pub), &route);
    let control = dpaired.paired_control(&route);
    assert_eq!(control, encode_control(&control_json));
    let pd2p = dpaired
        .channel
        .sealer
        .seal_fragment(&control, true)
        .unwrap();
    assert_eq!(
        ppaired.channel.opener.open(&pd2p).unwrap().unwrap(),
        control
    );
    let pp2d = ppaired
        .channel
        .sealer
        .seal_fragment(&req_bytes, true)
        .unwrap();
    assert_eq!(
        dpaired.channel.opener.open(&pp2d).unwrap().unwrap(),
        req_bytes
    );

    let pairing = json!({
        "secret": SECRET,
        "nameplate": NAMEPLATE,
        "device_name": DEVICE_NAME,
        "device_name_utf8": h(DEVICE_NAME.as_bytes()),
        "device_key_hash": h(device_key_hash),
        "pairing_id": h(pairing_id),
        "page_ephemeral_private": h(pk_d),
        "E_k": h(pe_k),
        "daemon_ephemeral_private": h(pd_d),
        "E_d": h(pe_d),
        "N_k": h(pn_k),
        "N_d": h(pn_d),
        "hello": h(&phello),
        "transcript": h(&pt),
        "transcript_hash": h(pth),
        "ecdh": h(pshared),
        "ikm": h(&ikm),
        "prk": h(hkdf_extract(&pth, &ikm)),
        "confirm_daemon": h(confirm.daemon),
        "confirm_device": h(confirm.device),
        "mac_d": h(mac_d),
        "mac_k": h(mac_k),
        "daemon_signed_message": h(&pm_d),
        "device_signed_message": h(&pm_k),
        "sig_d": h(&psig_d),
        "sig_k": h(&psig_k),
        "hello_reply": h(&preply),
        "finish": h(&pfinish),
        "key_d2p": h(pkeys.d2p),
        "key_p2d": h(pkeys.p2d),
        "paired_json": String::from_utf8(control_json.clone()).unwrap(),
        "data": [
            data("d2p", 0, true, &control, &pd2p),
            data("p2d", 0, true, &req_bytes, &pp2d),
        ],
    });

    json!({
        "spec": "aidaemon bridge protocol v1 (SPEC.md)",
        "notes": [
            "All byte strings are lowercase hex. Private scalars are SHA-256 of 'aidaemon-bridge-v1 test vector <label>'.",
            "Signatures are Rust's deterministic RFC 6979 ECDSA P-256/SHA-256 (r || s); WebCrypto signatures are randomized, so JavaScript verifies these instead of reproducing them. sig_d_high_s is sig_d with s replaced by n - s and must also verify.",
            "hello, transcript, ecdh, prk, keys, MACs and DATA frames are fully deterministic.",
            "device_key_hash is the leading 8 bytes of device_id, as a typed code would give.",
            "session.data[0] is the daemon's ready control (d2p counter 0); the page sends no DATA before opening it.",
            "abort is the ABORT frame."
        ],
        "keys": {
            "identity_private": h(identity_d),
            "identity_pub": h(identity_pub),
            "device_private": h(device_d),
            "device_pub": h(device_pub),
            "login_private": h(login_d),
            "login_pub": h(login_pub),
        },
        "ids": ids,
        "login": {
            "nonce": h(login_nonce),
            "message": h(login_message(&login_nonce)),
            "sig": h(login_sig),
        },
        "abort": h(frame::abort()),
        "session": session,
        "pairing": pairing,
        "messages": {
            "request": request_json(&req),
            "response": response_json(&resp),
            "control": {
                "json": String::from_utf8(control_json.clone()).unwrap(),
                "encoded": h(encode_control(&control_json)),
            },
        },
    })
}

#[test]
fn vectors_match_file() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("vectors/v1.json");
    let built = build();
    if std::env::var_os("UPDATE_BRIDGE_VECTORS").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut text = serde_json::to_string_pretty(&built).unwrap();
        text.push('\n');
        std::fs::write(&path, text).unwrap();
        return;
    }
    let text = std::fs::read_to_string(&path)
        .expect("vectors/v1.json missing; run with UPDATE_BRIDGE_VECTORS=1");
    let on_disk: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(on_disk, built, "vectors/v1.json is stale");
}

#[test]
fn file_frames_decode() {
    // The vector frames obey the frame rules (a cheap cross-check).
    let v = build();
    for mode in ["session", "pairing"] {
        for (field, kind) in [
            ("hello", FrameType::Hello),
            ("hello_reply", FrameType::HelloReply),
            ("finish", FrameType::Finish),
        ] {
            let bytes = hex::decode(v[mode][field].as_str().unwrap()).unwrap();
            frame::decode_handshake(&bytes, kind).unwrap();
        }
    }
    let key: [u8; 32] = hex::decode(v["session"]["key_p2d"].as_str().unwrap())
        .unwrap()
        .try_into()
        .unwrap();
    let mut s = Sealer::new(&key);
    let pt = hex::decode(v["session"]["data"][1]["plaintext"].as_str().unwrap()).unwrap();
    assert_eq!(
        h(s.seal_fragment(&pt, true).unwrap()),
        v["session"]["data"][1]["frame"]
    );
}

#[test]
fn webcrypto_signature_verifies_in_rust() {
    // Made once by Node's WebCrypto (randomized ECDSA, P1363 r || s) with
    // the vector device key over session.device_signed_message.
    const WEBCRYPTO_SIG_K: &str = "199aa0769b4c134fb6b0afc0fd7ee736e3ecb7e7304eed033c20fe43a0eec67aabf09bd955f86107a200db24f214e551c2f9bcbee4485e5e347bd0fcb674c7ce";
    let v = build();
    let device_pub = hex::decode(v["keys"]["device_pub"].as_str().unwrap()).unwrap();
    let message = hex::decode(v["session"]["device_signed_message"].as_str().unwrap()).unwrap();
    let sig = hex::decode(WEBCRYPTO_SIG_K).unwrap();
    handshake::verify(&device_pub, &message, &sig).unwrap();
    let mut bad = sig.clone();
    bad[10] ^= 1;
    assert!(handshake::verify(&device_pub, &message, &bad).is_err());
}
