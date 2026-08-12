use aidaemon::nodes::simulator::NodeSimulator;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: node_simulator <gateway-url> <offer-id> <offer-secret> [message]");
        std::process::exit(2);
    }
    let mut simulator = NodeSimulator::new(&args[1])?;
    let enrollment = simulator.enroll(&args[2], &args[3]).await?;
    println!(
        "Enrolled Node {} with policy {}",
        enrollment.node_id, enrollment.policy_profile
    );
    simulator.open_session().await?;
    println!("Authenticated Node Session opened");
    let events = simulator
        .text_turn(
            args.get(4)
                .map(String::as_str)
                .unwrap_or("Hello from the Node simulator"),
        )
        .await?;
    for event in events {
        println!("{} {} {}", event.cursor, event.event_type, event.payload);
    }
    Ok(())
}
