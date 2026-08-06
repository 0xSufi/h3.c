# h3-metal

Native MiniMax-H3 inference for Apple Silicon. The project is being built as a
sequence of working vertical slices: deterministic host/model metadata first,
then portable Metal block parity, prompt encoding, prompt-to-video/audio, and
first/last-frame conditioning before full ordered references are added.

Current milestone: M7 FL2VA prompt/first/last-frame generation is working;
ordered Ref2VA image/video/audio references and H3-specific performance work
follow incrementally.

```sh
make
make test
./h3 --info -d MiniMax-H3
./h3 -d MiniMax-H3 -p "A red fox walking through snow" \
  --width 512 --height 512 --frames 22 -o outputs/fox.mp4
./h3 -d MiniMax-H3 -p "The fox keeps walking" \
  --width 512 --height 512 --frames 22 --first-frame fox.png \
  --last-frame fox-later.png -o outputs/fox-anchored.mp4
```

`make test` runs the deterministic host suite and, when the ignored MLX fixture
is installed under `misc/fixtures/`, compiles the Metal source at runtime and
checks a complete toy H3 block against named MLX outputs. Runtime compilation is
intentional: it follows Iris and does not require Xcode's optional offline Metal
toolchain. The test covers both an F32 diagnosis path and the production BF16
storage path; wide BF16 matrix products and SDPA use cached MPSGraph graphs, with
direct Metal correctness fallbacks. `make parity` runs only those Metal/MLX
checks.

FFmpeg must be available on `PATH` for image inputs and when writing an MP4
(`H3_FFMPEG` may select an explicit executable). Generated RGB24 and
32 kHz stereo F32 PCM are fed through concurrent pipes; no intermediate
uncompressed media file is created.

The CLI follows Iris-style conventions. `--show` has a callback-based terminal
renderer for Kitty/Ghostty and iTerm2/WezTerm/Konsole. Prompt-to-video is now a
working native vertical slice: a corrected 512x512x22 run produces a coherent
fox walking through a snowy pine forest, semantically matching the independent
MLX render. The default sampler follows current SGLang serving: 50 shifted sigma
points (49 Euler forwards) with independent video/audio schedules. `--steps 20`
is useful for quicker development renders. The full native 768x768, 50-point
M5-Max validation also produces a clean photorealistic fox rather than noise.

The released checkpoint stores DiT QKV rows interleaved per attention head.
Native Metal consumes that layout directly in the fused QK-normalization/RoPE
kernel, avoiding a checkpoint transpose and extra RAM. The earlier identity
interpretation was the cause of the noisy diagnostic outputs.

The public generation path decodes the joint audio latent with a streamed native
BigVGAN/AudioVAE and writes synchronized H.264 plus 32 kHz stereo AAC. The native
waveform agrees with the corrected MLX oracle to relative L2 `6.94e-5`.
`--first-frame`, `--last-frame`, and their combination use the released visual
VAE encoder, Qwen3-VL vision tower and three-deepstack multimodal presentation,
0.999 condition augmentation, and fixed condition rows in the native DiT. The
first image is stretched to the target canvas; the last image is aspect-cover
scaled and center cropped, matching the reference implementation. Full ordered
image/video/audio references remain the next incremental milestone.

The native baseline targets the original `FL2VA/` and `Ref2VA/` checkpoint
trees. Model phases are loaded and released separately so the 33B transformer,
Qwen encoder, and decoders never have to coexist in unified memory.
