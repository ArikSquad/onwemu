# gbsml

A clean-room safe-Rust emulator for the original Nintendo Game Boy (DMG).

## Run

```sh
cargo run --release --features desktop --bin gbsml -- game.gb
```

Controls: arrows, `Z` (A), `X` (B), Enter (Start), Backspace (Select), Escape (quit).

For serial-output test ROMs:

```sh
cargo run --release --bin gbsml-headless -- tests/roms/cpu_instrs.gb
```

# ROM notice

Commercial ROM images are copyrighted, we don't support obtaining them in any other way than legally.

