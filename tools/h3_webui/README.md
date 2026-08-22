# h3 web UI

`tools/h3_webui` is a single-page web UI for the engine: prompt plus
optional first/last-frame images (FL2VA) and — when the Ref2VA checkpoint
is installed under the model directory — ordered reference images, video,
and audio; results appear in a gallery with the generated soundtrack.
One `h3` process runs at a time from a queue, with live phase/denoise
progress, ETA, and cancel. Needs Python 3 with aiohttp; everything else is
the standard library.

```sh
H3_DIR=~/h3.c H3_MODEL_DIR=~/h3.c/MiniMax-H3 python3 tools/h3_webui/server.py
# http://host:7860   (H3_WEBUI_PORT overrides; data in ~/h3-webui)
```

The speed presets map to the engine's validated knobs (exact BF16, FP8,
`--layers 45 --reuse 2`, `--token-reduction`); anything else goes in the
"Extra h3 flags" box. The server is unauthenticated and executes local
h3 with caller-supplied flags: serve it on trusted networks only.
