# psp-rs

PSP emulator made in Rust. 

This workspace is licensed under the GNU Affero General Public License version 3. See [`LICENSE`](LICENSE).

## Build and validate

```bash
cargo xtask validate
```

Building requires Rust 1.88 or newer. The graphical Linux frontend uses SDL3 GPU with the Vulkan backend and
Wayland by default. You can download all the other deps like this:

```bash
sudo apt install build-essential cmake pkg-config libwayland-dev libxkbcommon-dev \
  wayland-protocols libvulkan-dev libpipewire-0.3-dev libasound2-dev libpulse-dev
```

Graphical scale selects both the internal GPU render target and the host presentation size, from 1x to 4x:

```bash
cargo run --release -p psp-rs -- run game.chd --scale 2
```

Inspect a user-owned ISO or CHD image with:

```bash
cargo run -p psp-rs -- inspect game.chd
```

## Tests

Run the full workspace checks with:

```bash
cargo xtask validate
cargo test --workspace --all-targets
```

## Statement on Agentic Development

I've used different LLMs to orchestrate the creation of this emulator, I've been mostly creating this at classes which doesn't leave
much room for writing lots of code. Recently I've been making some of these projects just so I can test out some certain games on desktop
without committing lots of time into creating these projects themselves when studying cybersecurity and other topics. If I don't get bored
of this, I'll probably go through everything myself to clean things up. This is also kind of a disclaimer that there might be a lot of issues 
as most code is not written by a human. 
