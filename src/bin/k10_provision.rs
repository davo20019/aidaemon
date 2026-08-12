use std::io::{Read, Write};
use std::time::{Duration, Instant};

use aidaemon::nodes::simulator::NodeSimulator;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 8 {
        eprintln!("Usage: k10_provision <serial-port> <gateway-url> <offer-id> <offer-secret> <wifi-ssid> <wifi-password> <device-label> [--wifi <ssid> <password>]... [--gateway-fallback <https-origin>]...");
        std::process::exit(2);
    }
    let mut wifi_networks = vec![serde_json::json!({"ssid": args[5], "password": args[6]})];
    let mut gateways = vec![args[2].trim_end_matches('/').to_string()];
    let mut index = 8;
    while index < args.len() {
        match args[index].as_str() {
            "--wifi" => {
                anyhow::ensure!(index + 2 < args.len(), "--wifi requires <ssid> <password>");
                anyhow::ensure!(
                    wifi_networks.len() < 4,
                    "the K10 supports at most four trusted Wi-Fi networks"
                );
                wifi_networks.push(
                    serde_json::json!({"ssid": args[index + 1], "password": args[index + 2]}),
                );
                index += 3;
            }
            "--gateway-fallback" => {
                anyhow::ensure!(
                    index + 1 < args.len(),
                    "--gateway-fallback requires <https-origin>"
                );
                anyhow::ensure!(
                    gateways.len() < 3,
                    "the K10 supports at most three Gateway origins"
                );
                let origin = args[index + 1].trim_end_matches('/').to_string();
                anyhow::ensure!(
                    origin.starts_with("https://")
                        && !origin[8..]
                            .chars()
                            .any(|character| matches!(character, '/' | '@' | '?' | '#')),
                    "Gateway fallbacks must be HTTPS origins"
                );
                anyhow::ensure!(
                    !gateways.contains(&origin),
                    "Gateway origins must be unique"
                );
                gateways.push(origin);
                index += 2;
            }
            option => anyhow::bail!("unknown provisioning option: {option}"),
        }
    }
    let mut port = serialport::new(&args[1], 115_200)
        .timeout(Duration::from_millis(250))
        .open()?;
    let ready_deadline = Instant::now() + Duration::from_secs(20);
    let mut received = Vec::new();
    let mut buffer = [0_u8; 256];
    while Instant::now() < ready_deadline {
        match port.read(&mut buffer) {
            Ok(count) => {
                received.extend_from_slice(&buffer[..count]);
                if received
                    .windows(b"PROVISIONING_READY".len())
                    .any(|window| window == b"PROVISIONING_READY")
                {
                    break;
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::TimedOut => {}
            Err(error) => return Err(error.into()),
        }
    }
    anyhow::ensure!(
        received
            .windows(b"PROVISIONING_READY".len())
            .any(|window| window == b"PROVISIONING_READY"),
        "K10 did not announce provisioning readiness"
    );

    let mut simulator = NodeSimulator::new(&args[2])?;
    let enrollment = simulator.enroll_as(&args[3], &args[4], "k10").await?;
    let packet = serde_json::json!({
        "command": "provision",
        "gateway": args[2],
        "gateways": gateways,
        "node_id": enrollment.node_id,
        "credential_id": enrollment.credential_id,
        "private_key": simulator.identity.private_key_base64url(),
        "wifi_networks": wifi_networks,
        "label": args[7],
    });
    let packet = serde_json::to_string(&packet)?;
    anyhow::ensure!(
        packet.len() <= 1800,
        "provisioning packet exceeds Device limit"
    );
    for chunk in packet.as_bytes().chunks(64) {
        port.write_all(chunk)?;
        port.flush()?;
        std::thread::sleep(Duration::from_millis(10));
    }
    port.write_all(b"\n")?;
    port.flush()?;
    let deadline = Instant::now() + Duration::from_secs(12);
    received.clear();
    while Instant::now() < deadline {
        match port.read(&mut buffer) {
            Ok(count) => {
                received.extend_from_slice(&buffer[..count]);
                if received
                    .windows(b"PROVISIONED".len())
                    .any(|window| window == b"PROVISIONED")
                {
                    println!(
                        "K10 provisioning completed for Node {}.",
                        enrollment.node_id
                    );
                    return Ok(());
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::TimedOut => {}
            Err(error) => return Err(error.into()),
        }
    }
    if let Ok(text) = std::str::from_utf8(&received) {
        if let Some(reason) = text
            .lines()
            .find_map(|line| line.trim().strip_prefix("PROVISIONING_REJECTED:"))
        {
            anyhow::bail!("K10 rejected provisioning: {reason}");
        }
    }
    anyhow::bail!("K10 did not confirm provisioning before the deadline")
}
