mod common;

#[test]
fn help_flag_exits_successfully() {
    common::aidaemon_bin().arg("--help").assert().success();
}

#[test]
fn version_flag_exits_successfully() {
    common::aidaemon_bin().arg("--version").assert().success();
}
