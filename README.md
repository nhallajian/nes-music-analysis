# nes-music-analysis

## Prerequisites

You need [Rust](https://www.rust-lang.org/tools/install) and [Python](https://www.python.org/downloads/) installed.

## Setup

- Clone the repo with `git clone --recurse-submodules https://github.com/nhallajian/nes-music-analysis`.

- Change into the repository directory (e.g. `cd nes-music-analysis`)

- Change into the `posemir` submodule and run `cargo build --release` to build it. This also builds the `posemir_cli` command line tool which we invoke from Python to run the pattern analysis.
