# Fuzz Corpus

Seeds replayed by `zig build fuzz-corpus` (CI, every PR) and loaded by every fuzz campaign.
See [docs/fuzzing.md](../../docs/fuzzing.md).

| Dir | Targets |
|---|---|
| `ipc-stream/` | `ipc-stream`, `ipc-stream-zero-copy`, `compute` |
| `ipc-file/` | `ipc-file` |
| `array-layout/` | `array-layout` |
| `c-abi/` | `c-abi` |
| `regex/` | `regex` (hand-made: flag byte, u16 pattern length, pattern, text) |

File names:

- `at-*`: copied from apache/arrow-testing (Apache-2.0, see `NOTICE` in each dir) by
  `tools/fuzz_import_seeds.py`.
- `gen-*`: written by `zig build fuzz-gen-seeds` (zarrow writers, plain / lz4 / zstd).
- `regress-<hash>.bin`: reproducers of fixed fuzz findings; they must keep returning an error
  (or decoding) without a crash, leak or allocation blow-up.
- anything else: small hand-made seeds.

Keep the whole corpus under about 2 MB; prefer small inputs.
