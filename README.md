# h3-metal

Native MiniMax-H3 inference for Apple Silicon. The project is being built as a
sequence of working vertical slices: deterministic host/model metadata first,
then portable Metal block parity, prompt encoding, one real denoising step, and
finally prompt-to-video before the conditioning modes are added.

Current milestone: M2 portable Metal block baseline.

```sh
make
make test
./h3 --info -d MiniMax-H3
```

`make test` runs the deterministic host suite and, when the ignored MLX fixture
is installed under `misc/fixtures/`, compiles the Metal source at runtime and
checks a complete toy H3 block against named MLX outputs. Runtime compilation is
intentional: it follows Iris and does not require Xcode's optional offline Metal
toolchain. `make parity` runs only that Metal/MLX check.

The eventual CLI follows Iris-style conventions:

```sh
./h3 -d MiniMax-H3 -p "a cat walking through Rome" -o cat.mp4
```

The native baseline targets the original `FL2VA/` and `Ref2VA/` checkpoint
trees. Model phases are loaded and released separately so the 33B transformer,
Qwen encoder, and decoders never have to coexist in unified memory.
