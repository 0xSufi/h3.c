# ComfyUI nodes for h3

`tools/comfyui_h3` is a ComfyUI custom-node package that drives the `h3`
binary: prompt (plus optional first/last-frame images) in, MP4 with the
generated audio out, previewed in the graph and returned as an IMAGE batch
for further nodes.

Install (Linux, the engine built with `make linux` and the MiniMax-H3
snapshot present):

```sh
ln -s /path/to/h3.c/tools/comfyui_h3 ~/ComfyUI/custom_nodes/comfyui_h3
H3_DIR=/path/to/h3.c H3_MODEL_DIR=/path/to/h3.c/MiniMax-H3 \
    python main.py --listen 0.0.0.0 --port 8188
```

The node's `preset` picks the engine configuration: exact BF16, the FP8
paths (`H3_CUDA_FP8=1 H3_CUDA_SDPA_FP8=1`), and the README's validated
`--layers 45 --reuse 2` / `--token-reduction` schedule controls; any other
`h3` flag goes in `extra_args`. Each run is one `h3` process (model load is
about 30 s on a DGX Spark), so queue several prompts to amortize nothing —
the interactive `h3` session keeps the model resident if that matters.
`example_workflow.json` is a one-node starting graph.
