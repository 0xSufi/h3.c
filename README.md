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
./h3 --profile -d MiniMax-H3 -p "A surfer riding a blue ocean wave" \
  --width 512 --height 512 --frames 22 --steps 20 --core-reuse 4 \
  -o outputs/fast-surfer.mp4
./h3 --profile -d MiniMax-H3 -p "A red fox walking through snow" \
  --width 512 --height 512 --frames 22 --steps 20 \
  --layers 45 --reuse 2 --token-reduction -o outputs/fast-fox.mp4
./h3 --profile -d MiniMax-H3 -p "A red fox walking through snow" \
  --width 512 --height 512 --render-width 384 --render-height 384 \
  --frames 22 --steps 20 --reuse 3 -o outputs/fast-scaled-fox.mp4
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
renderer for Kitty/Ghostty and iTerm2/WezTerm/Konsole. `--frames-dir DIR`
writes callback frames as PPM files; combine it with `-o ''` to inspect frames
on a machine without FFmpeg while retaining the same generation path.
Prompt-to-video is now a
working native vertical slice: a corrected 512x512x22 run produces a coherent
fox walking through a snowy pine forest, semantically matching the independent
MLX render. The default sampler follows current SGLang serving: 50 shifted sigma
points (49 Euler forwards) with independent video/audio schedules. `--steps 20`
is useful for quicker development renders. The full native 768x768, 50-point
M5-Max validation also produces a clean photorealistic fox rather than noise.
`--reuse 2` evaluates roughly half the denoiser forwards and is the validated
fast-quality setting. `--reuse 3` is the aggressive setting; it evaluates
roughly one third. On the common 20-point serving grid, the aggressive path
uses the quality-tuned six-forward placement `0,3,6,10,14,18` instead of the
seven forwards selected by a uniform interval. Full 22-frame sweeps preserved
clean prompt semantics on both surfer and fox prompts; the survivor was chosen
from nine nearby schedules, including several that looked clean but changed
the requested subject. Reuse extrapolates skipped video and audio velocities
on their independent sigma grids. The default `--reuse 1` remains the close
reference path. `--layers 45` independently drops the five least-active DiT
residual blocks, selected from their actual AdaLN gates, and is the validated
fast-quality setting; `--layers 40` is more aggressive. The exact default is
`--layers 50`. Unused block weights and schedule tensors are not retained, so
the setting reduces both transformer time and unified-memory use.
`--core-reuse 4` is a stronger fast-quality mode: it reuses the previous full
transformer-core residual while still refreshing the patch projection and
timestep-dependent final head at every denoiser step. `--core-reuse 6` is the
validated aggressive limit; values above 6 lose subject fidelity. Core reuse
and whole-velocity `--reuse` are intentionally mutually exclusive.
`--token-reduction` is an independent aggressive DiT mode. After block 3 it
pairs adjacent horizontal target-video tokens while leaving text, audio,
conditions, and reference tokens exact. The complete full-resolution state is
kept as a bypass. During the first ten noisy evaluations it restores before
block 40; subsequent detail-forming evaluations restore before block 30. Each
token returns as its original value plus the update learned by its pair, so
within-pair detail is not discarded.
The pooling kernel writes only true-pair baselines into a dense tail of the
already allocated attention scratch buffer; odd-width singleton tokens need no
baseline. The full bypass uses the oversized QKV tail when it fits, with a
guarded dedicated fallback only for reference-heavy layouts. Common text-only
canvases therefore add no activation arena at any token-grid width. Pooling
also snapshots both source tokens while their BF16 values are already in
registers, avoiding a separate full-hidden blit and redundant source read.
At the restore boundary, the first full-resolution attention AdaLN is fused
into expansion: a 10.5 KiB threadgroup row avoids a global residual reread while
still writing the exact bypass needed by the following residual branch.
On a thermal-balanced 512x512x22, 19-forward IT M5 Max A/B this reduced denoise
time from 39.13 to 28.06 seconds (28.3%). Final video/audio latent relative L2
was 5.56%/15.14%. First/middle/last fox frames retained one clean muzzle,
coherent legs, and sharp fur; an independent surfer remained consistent with
one rider and board through the wave spray. It changes composition and is
therefore opt-in rather than the close-reference default.
`H3_TOKEN_REDUCTION_BLOCKS` can override the later `4:30` interval;
`H3_TOKEN_REDUCTION_EARLY=STEPS:END` overrides the early schedule and `0`
disables it. `H3_DISABLE_TOKEN_REDUCTION=1` provides an in-context exact oracle.
Token reduction composes cleanly with the validated `--layers 45 --reuse 2`
settings: on the same 512 benchmark it reduced that profile from 16.69 to
12.60 seconds (24.5% marginal), and independent fox and surfer renders stayed
coherent. Do not combine it with both `--layers 40` and `--reuse 3`; that
6.47-second experiment produced chromatic ringing and ghosted limbs despite
acceptable latent norms.
`--render-width` and `--render-height` run the model and VAE on a lower
same-aspect internal canvas, then high-quality vImage-scale RGB frames to the
requested output size before callbacks, terminal display, and encoding. This is an
explicit quality/speed tradeoff: a measured 384-to-512 prompt render reduced
M5 DiT time by 33% and video-VAE time by 18% while retaining a clean,
recognizable photorealistic result. Both values must be multiples of 32; the
exact output canvas remains the default.
For square 512 output, 384 is the fast-quality point and 320 is the validated
aggressive point. The latter produced a coherent walking fox and repeated at
8.02 seconds of DiT versus about 15.82 seconds natively. A 256 test softened
the face and legs and was slightly slower overall than 320, so it is allowed
but not recommended.
The video VAE automatically chooses a 256-320 pixel spatial tile from the
requested canvas geometry, minimizing repeated overlap work while keeping peak
storage bounded. `H3_VAE_TILE_PIXELS=256` restores the original conservative
tile plan for close-reference diagnosis.
On M5-class GPUs, persistent transformer weights are mapped directly from their
safetensor shards instead of copied into anonymous shared buffers. This keeps
the 37 GiB model file-backed/reclaimable and slightly improves total transformer
time; M3 uses the faster copied-buffer path. `H3_ZERO_COPY_WEIGHTS=0` disables
the M5 selection for diagnostics.
The streamed Qwen text encoder preallocates a small ring of future layer
buffers and fills them on eight I/O workers while Metal executes the current
layer. The default ring depth is two layers on M3/older hardware and three on
M5, where the target machine has 128 GiB. `H3_QWEN_PREFETCH=0` restores the
single-layer synchronous reference path; values 1-8 select the worker count,
and `H3_QWEN_PREFETCH_DEPTH=1` through `6` overrides the ring depth.
An M5-only native BF16 Metal 4/TensorOps linear path is available with
`H3_NAX=1`. It is guarded at runtime and falls back to the unchanged portable
library if TensorOps compilation is unavailable. The path passes the complete
50-block MLX fixture, but remains opt-in: exact-shape microbenchmarks favor its
128-row tile while full DiT runs currently favor MPSGraph scheduling. This
keeps a working NAX integration available for later quantized/fused kernels
without making a benchmark regression the default.
`H3_NAX=mlp` selects a more specialized Metal 4 path: paired FC1 gate/up
TensorOps tiles apply SwiGLU in threadgroup memory and write only the
14,336-wide activated intermediate, then FC2 also stays on TensorOps.
`H3_DISABLE_NAX_MLP=1` keeps the MPSGraph MLP in a context created this way for
same-process A/B testing. The path is deliberately opt-in because scheduling
depends on the OS GPU stack: the primary macOS 26.5.2 M5 Max gained 1.3-2.0%
in isolated real-weight MLP runs but lost about 1-3% in a complete 50-block forward,
while an otherwise identical macOS 26.5 M5 Max gained 1.4% in a same-context
forward A/B. The resulting 50-block velocities were close (1.9% video and 2.4%
audio relative L2), but not byte-identical.
The narrow DiT audio/video output heads convert their small released F32
weights to BF16 once and use the Iris-derived 16x16 tiled linear directly on
BF16 activations. At the production 320-render geometry, isolated paired-head
measurements are 2.30x faster on M3 Max and 1.83x faster on M5 Max, with
relative L2 `8.64e-4`; the absolute M5 saving is about 0.6 ms per evaluated
step. Full fox and surfer sequences remained clean and measured 29.9/38.4 dB
against the F32-head renders. `H3_DIT_F32_FINAL=1` restores the close-reference
head and its extra activation buffers.
The F32 `96->5376` video and `32->5376` audio patch projections use a dedicated
16x16 cooperative tile, retaining F32 weights, activations and accumulation.
Paired production-shape measurements are 1.77x faster on M3 and 1.62-1.78x
on M5; the complete generated RGB stream is byte-identical to the scalar path.
`H3_SCALAR_PATCH=1` selects that scalar diagnostic path.
The DiT core is split into two ordered Metal command buffers so GPU execution
of the first part overlaps CPU encoding of the second. Thermal-balanced ABBA
measurements select a 60%-depth split on M5 (30/50, 27/45, and 24/40), with
roughly 0.5-1.8% wins; M3 automatically splits only the validated 30/50 case,
which measured 1.2% faster, because 24/40 regressed there. The operation order
and generated bytes are unchanged. `H3_DIT_COMMAND_BLOCKS=0` restores one
command buffer; values 1-50 override the split for further tuning.
MPSGraph tensor-data wrappers for immutable DiT weights and biases are retained
with their resident buffers. This avoids rebuilding the same binding metadata
for every block and denoiser evaluation without copying tensor storage; measured
ABBA gains were 1.6% on M3 Max and 0.4-1.1% on M5 Max. Activation wrappers stay
transient because retaining them regressed the M5. The outputs remain
byte-identical, and `H3_DISABLE_GRAPH_DATA_CACHE=1` restores transient wrappers
for all tensors.
On M3/older hardware, the four MPSGraph segments in each DiT block also reuse
one `MPSCommandBuffer` wrapper for their shared underlying Metal command buffer.
Repeated thermal-balanced runs measured 1.0-1.6% faster on M3 Max; M5 measured
neutral, so it retains fresh wrappers. `H3_REUSE_MPS_COMMAND=0` or `1` overrides
the automatic selection. Results are byte-identical.
On M5, the serving Euler sampler keeps its patch-packed F32 latents and cached
BF16 velocities in Metal buffers. Each selected denoiser refresh is completed
before the next is encoded, avoiding MPSGraph back-pressure while removing all
intermediate latent/velocity readbacks and repacking. Two warm eight-run A/B
sequences measured small 0.1% and 0.3% gains with byte-identical final latents;
the path also saves roughly 16 bytes of transient host state per video-latent
element (about 136 MB at the 768p shape). M3 and older GPUs retain the CPU
sampler by default. `H3_CPU_SAMPLER=1` restores it on M5;
`H3_GPU_SAMPLER=1` selects the GPU-state path explicitly, and
`H3_GPU_SAMPLER_WINDOW=0` enables the slower unbounded encode-ahead diagnostic.

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

The DiT fast path evaluates each BF16 `fc1 -> SwiGLU -> fc2` block as one cached
graph, avoiding separate graph boundaries and persistent intermediate tensors.
Set `H3_DISABLE_FUSED_MLP=1` to retain the close-reference operation boundaries
for numerical diagnosis.

The native baseline targets the original `FL2VA/` and `Ref2VA/` checkpoint
trees. Model phases are loaded and released separately so the 33B transformer,
Qwen encoder, and decoders never have to coexist in unified memory.
