# h3-metal

Native MiniMax-H3 inference for Apple Silicon. The project is being built as a
sequence of working vertical slices: deterministic host/model metadata first,
then portable Metal block parity, prompt encoding, prompt-to-video/audio, and
first/last-frame conditioning and then ordered references.

Current milestone: M8 Ref2VA ordered image/video/audio reference support is complete;
H3-specific Metal performance and memory work follows incrementally.

```sh
make
make test
./h3 --info -d MiniMax-H3
./h3 -d MiniMax-H3 -p "A red fox walking through snow" \
  --width 512 --height 512 --frames 22 -o outputs/fox.mp4
./h3 -d MiniMax-H3 -p "The fox keeps walking" \
  --width 512 --height 512 --frames 22 --first-frame fox.png \
  --last-frame fox-later.png -o outputs/fox-anchored.mp4
./h3 -d MiniMax-H3 -p "Use the animal and setting in the reference" \
  --width 512 --height 512 --frames 22 --ref-image fox.png \
  -o outputs/fox-reference.mp4
./h3 -d MiniMax-H3 -p "Continue the motion in this clip" \
  --width 512 --height 512 --frames 22 --ref-silent-video fox.mp4 \
  -o outputs/fox-video-reference.mp4
./h3 -d MiniMax-H3 -p "Use the animal and the music" \
  --width 512 --height 512 --frames 22 \
  --ref-image fox.png --ref-audio music.wav \
  -o outputs/fox-image-audio-reference.mp4
./h3 -d MiniMax-H3 -p "Continue this audiovisual scene" \
  --width 512 --height 512 --frames 56 --ref-video fox-with-audio.mp4 \
  -o outputs/fox-video-audio-reference.mp4
./h3 -d MiniMax-H3 -p "Continue the clip with this replacement soundtrack" \
  --width 512 --height 512 --frames 56 \
  --ref-video-audio silent-fox.mp4 replacement.wav \
  -o outputs/fox-replaced-audio-reference.mp4
./h3 --profile -d MiniMax-H3 -p "A fox walks through snow" \
  --width 512 --height 512 --frames 22 --steps 20 \
  -o outputs/profile.mp4
```

`make test` runs the deterministic host suite and, when the ignored MLX fixture
is installed under `misc/fixtures/`, compiles the Metal source at runtime and
checks a complete toy H3 block against named MLX outputs. Runtime compilation is
intentional: it follows Iris and does not require Xcode's optional offline Metal
toolchain. The test covers both an F32 diagnosis path and the production BF16
storage path; wide BF16 matrix products and SDPA use cached MPSGraph graphs, with
direct Metal correctness fallbacks. `make parity` runs only those Metal/MLX
checks.

FFmpeg and FFprobe must be available on `PATH` for media inputs and MP4 output
(`H3_FFMPEG` and `H3_FFPROBE` may select explicit executables). Generated RGB24 and
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
scaled and center cropped, matching the reference implementation. `--ref-image`
selects the distinct Ref2VA transformer, preserves ordered `<Picture N>`
presentation, and uses the released down-only aspect-preserving reference canvas.
`--ref-silent-video` additionally performs bounded 24 fps decoding, the visual
VAE's causal `ceil(T/4)` compression, two-frame Qwen sampling, and timestamped
`<Video N>` presentation. `--ref-video` preserves an embedded soundtrack,
`--ref-video-audio VIDEO AUDIO` supplies an explicit replacement, and
`--ref-audio` appends an ordered standalone clip. Reference audio is decoded as
32 kHz stereo F32, encoded by the native AudioVAE posterior-mean path, mixed as
0.999 clean latent plus 0.001 seeded noise, pinned to the audio condition
timestep 1.0, and packed as width-32 rows on the same rotary timeline as visual
references. Audio inputs are 2-15 seconds, at most three are
accepted, their total decoded duration is capped at 15 seconds, and a standalone
audio reference must be combined with an image or video reference.

The native audio encoder matches the corrected MLX oracle at relative L2
`3.59e-6` on a real two-second stereo fixture. The correction is important: the
original MLX reshape interleaved left/right samples, whereas the official
PyTorch/SGLang path folds intact stereo channels into the batch dimension. On
the 128 GB M5 Max, clean end-to-end image+audio and embedded-video+audio renders
completed in 74.58 and 76.99 seconds respectively, each with about a 40.1 GB
peak physical footprint and zero swaps.

`--profile` reports each Metal-backed phase separately: wall time, CPU-side
command encoding, complete commit-to-fence wait, root-command GPU timestamps,
peak live tensor storage, cumulative allocation, and dispatch counts. The wait
measurement is the complete command turnaround; the root GPU timestamp alone
can omit child buffers scheduled internally by MPSGraph and is labeled
accordingly.

The native baseline targets the original `FL2VA/` and `Ref2VA/` checkpoint
trees. Model phases are loaded and released separately so the 33B transformer,
Qwen encoder, and decoders never have to coexist in unified memory.
