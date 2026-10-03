fn main() {
    if let Err(error) = psp_runtime::run_cli() {
        eprintln!("{error:#}");
        std::process::exit(1);
    }
}
