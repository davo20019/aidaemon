use assert_cmd::Command;

pub fn aidaemon_bin() -> Command {
    #[allow(deprecated)]
    {
        Command::cargo_bin("aidaemon").expect("aidaemon test binary should build")
    }
}
