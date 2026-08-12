use std::io::{Read, Write};
use std::time::{Duration, Instant};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 || !(args.len() - 2).is_multiple_of(2) {
        eprintln!("Usage: k10_wifi_update <serial-port> <wifi-ssid> <wifi-password> [<wifi-ssid> <wifi-password>]...");
        std::process::exit(2);
    }
    let pairs = args[2..].chunks(2).collect::<Vec<_>>();
    anyhow::ensure!(
        pairs.len() <= 4,
        "the K10 supports at most four trusted Wi-Fi networks"
    );
    let networks = pairs
        .into_iter()
        .map(|pair| serde_json::json!({"ssid": pair[0], "password": pair[1]}))
        .collect::<Vec<_>>();
    let packet = serde_json::to_string(&serde_json::json!({
        "command": "wifi_update",
        "wifi_networks": networks,
    }))?;
    anyhow::ensure!(
        packet.len() <= 1200,
        "Wi-Fi update packet exceeds Device limit"
    );

    let mut port = serialport::new(&args[1], 115_200)
        .timeout(Duration::from_millis(250))
        .open()?;
    for chunk in packet.as_bytes().chunks(64) {
        port.write_all(chunk)?;
        port.flush()?;
        std::thread::sleep(Duration::from_millis(10));
    }
    port.write_all(b"\n")?;
    port.flush()?;

    let deadline = Instant::now() + Duration::from_secs(8);
    let mut received = Vec::new();
    let mut buffer = [0_u8; 256];
    while Instant::now() < deadline {
        match port.read(&mut buffer) {
            Ok(count) => {
                received.extend_from_slice(&buffer[..count]);
                if received
                    .windows(b"WIFI_UPDATED".len())
                    .any(|window| window == b"WIFI_UPDATED")
                {
                    println!("K10 trusted Wi-Fi networks updated; the Device is restarting.");
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
            .find_map(|line| line.trim().strip_prefix("WIFI_UPDATE_REJECTED:"))
        {
            anyhow::bail!("K10 rejected the Wi-Fi update: {reason}");
        }
    }
    anyhow::bail!("K10 did not confirm the Wi-Fi update before the deadline")
}
