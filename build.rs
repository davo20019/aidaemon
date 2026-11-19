// Apple targets link clang's runtime library (libclang_rt.osx.a).
//
// ONNX Runtime's CoreML code (in the static library ort-sys links for the
// local embedding model) checks the OS version with Objective-C @available,
// which calls ___isPlatformVersionAtLeast. Rust links with -nodefaultlibs, so
// clang's runtime is left out, and Rust's own copy of that function in std is
// dropped by the release profile's fat LTO because no Rust code calls it: the
// release build then fails to link ("Undefined symbols ...
// ___isPlatformVersionAtLeast"). -fapple-link-rtlib makes clang link its
// runtime anyway. rustc-link-arg applies to every binary, test and example of
// this package.
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var("CARGO_CFG_TARGET_VENDOR").as_deref() == Ok("apple") {
        println!("cargo:rustc-link-arg=-fapple-link-rtlib");
    }
}
