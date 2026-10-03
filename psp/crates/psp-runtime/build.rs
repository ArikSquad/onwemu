#[cfg(feature = "desktop")]
#[path = "build_desktop.rs"]
mod desktop;

fn main() {
    #[cfg(feature = "desktop")]
    desktop::run();
}
