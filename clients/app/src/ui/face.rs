//! The typeface. Apple's devices and the desktop app draw with the
//! system's font (San Francisco), which reads best there. Elsewhere in a
//! browser the system's sans is whatever the device has (Roboto on
//! Android, Segoe on Windows): there the app brings Inter, so it reads
//! the same on every screen. Five weights, cut to Latin, about 115 KB
//! (`assets/fonts`, SIL Open Font License in `OFL.txt`).

#[cfg(target_arch = "wasm32")]
use nsh_core::prelude::*;

#[cfg(target_arch = "wasm32")]
const FACES: [&[u8]; 5] = [
    include_bytes!("../../assets/fonts/Inter-400Regular.woff2"),
    include_bytes!("../../assets/fonts/Inter-400Regular_Italic.woff2"),
    include_bytes!("../../assets/fonts/Inter-500Medium.woff2"),
    include_bytes!("../../assets/fonts/Inter-600SemiBold.woff2"),
    include_bytes!("../../assets/fonts/Inter-700Bold.woff2"),
];

/// Brings Inter where the system's font is not Apple's: registers its
/// faces, then makes it the app's sans. Until then (a moment) text is in
/// the system's.
pub fn load() {
    #[cfg(target_arch = "wasm32")]
    if !crate::link::web::apple_device() {
        spawn(async {
            let mut family = None;
            for face in FACES {
                if let Ok(name) = register_font(face.to_vec()).await {
                    family = Some(name);
                }
            }
            if let Some(family) = family {
                let mut tokens = (*tokens()).clone();
                tokens.set_font(nsh_core::element::FONT_SANS, family);
                set_tokens(tokens);
            }
        });
    }
}
