# h3-metal

Native MiniMax-H3 inference for Apple Silicon. The project is being built as a
sequence of working vertical slices: deterministic host/model metadata first,
then portable Metal block parity, prompt encoding, one real denoising step, and
finally prompt-to-video before the conditioning modes are added.

Current milestone: M5 prompt-to-video full-quality validation and M5 Max bring-up.

```sh
make
make test
./h3 --info -d MiniMax-H3
./h3 -d MiniMax-H3 -p "A red fox walking through snow" \
  --width 512 --height 512 --frames 22 -o outputs/fox.mp4
```

`make test` runs the deterministic host suite and, when the ignored MLX fixture
is installed under `misc/fixtures/`, compiles the Metal source at runtime and
checks a complete toy H3 block against named MLX outputs. Runtime compilation is
intentional: it follows Iris and does not require Xcode's optional offline Metal
toolchain. The test covers both an F32 diagnosis path and the production BF16
storage path; wide BF16 matrix products and SDPA use cached MPSGraph graphs, with
direct Metal correctness fallbacks. `make parity` runs only those Metal/MLX
checks.

The CLI follows Iris-style conventions. `--show` has a callback-based terminal
renderer for Kitty/Ghostty and iTerm2/WezTerm/Konsole. Prompt-to-video is now a
working native vertical slice: a corrected 512x512x22 run produces a coherent
fox walking through a snowy pine forest, semantically matching the independent
MLX render. The default sampler follows current SGLang serving: 50 shifted sigma
points (49 Euler forwards) with independent video/audio schedules. `--steps 20`
is useful for quicker development renders.

The released checkpoint stores DiT QKV rows interleaved per attention head.
Native Metal consumes that layout directly in the fused QK-normalization/RoPE
kernel, avoiding a checkpoint transpose and extra RAM. The earlier identity
interpretation was the cause of the noisy diagnostic outputs.

Generated audio, longer timelines, first/last-frame conditioning, and ordered
image/video/audio references remain subsequent incremental milestones.

The native baseline targets the original `FL2VA/` and `Ref2VA/` checkpoint
trees. Model phases are loaded and released separately so the 33B transformer,
Qwen encoder, and decoders never have to coexist in unified memory.
