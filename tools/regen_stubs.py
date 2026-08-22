#!/usr/bin/env python3
"""Regenerate h3_cuda_stubs.cu: stubs for h3_gpu.h entry points not yet
implemented by the CUDA backend. Implemented names are recorded in
.cuda_implemented (one per line). Usage:

    python3 tools/regen_stubs.py h3_gpu_silu_f32 h3_gpu_rms_norm_f32 ...

marks those functions as implemented and rewrites h3_cuda_stubs.cu.
Run from the repository root."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / ".cuda_implemented"
STUBS = ROOT / "h3_cuda_stubs.cu"

# Functions implemented in h3_gpu_cuda.cu itself (context/tensors/copies).
BUILTIN = set("""
h3_gpu_create h3_gpu_free h3_gpu_is_m5 h3_gpu_has_nax_mlp
h3_gpu_has_int8_mlp h3_gpu_tensor_new_f32 h3_gpu_tensor_new_bf16
h3_gpu_tensor_new_i8 h3_gpu_tensor_from_f32 h3_gpu_tensor_from_bf16
h3_gpu_tensor_from_u32 h3_gpu_tensor_load_bf16 h3_gpu_tensor_load_f32
h3_gpu_tensor_read_file_bf16 h3_gpu_tensor_stream_file_bf16
h3_gpu_tensor_free h3_gpu_tensor_elements h3_gpu_tensor_dtype
h3_gpu_tensor_read_f32 h3_gpu_tensor_read_f32_range h3_gpu_tensor_read_bf16
h3_gpu_tensor_write_f32 h3_gpu_tensor_write_f32_range
h3_gpu_tensor_write_bf16 h3_gpu_tensor_write_bf16_range
h3_gpu_begin h3_gpu_continue h3_gpu_submit h3_gpu_error h3_gpu_get_stats
h3_gpu_profile_set_label h3_gpu_profile_mark
h3_gpu_cast_f32_to_bf16 h3_gpu_cast_bf16_to_f32
h3_gpu_copy_bf16 h3_gpu_copy_f32
""".split())

header = (ROOT / "h3_gpu.h").read_text()
src = re.sub(r"/\*.*?\*/", "", header, flags=re.S)
src = re.sub(r"//.*", "", src)
decls = re.findall(r"int\s+(h3_gpu_\w+)\s*\(([^;]*?)\)\s*;", src, flags=re.S)
known = {name for name, _ in decls}

implemented = set(STATE.read_text().split()) if STATE.exists() else set()
for name in sys.argv[1:]:
    if name not in known:
        sys.exit(f"unknown h3_gpu.h function: {name}")
    if name in BUILTIN:
        sys.exit(f"{name} is implemented in h3_gpu_cuda.cu, not a stub")
    implemented.add(name)
STATE.write_text("\n".join(sorted(implemented)) + "\n")

lines = [
    "/* Temporary stubs for h3_gpu.h entry points not yet ported to",
    " * CUDA. Each milestone deletes the functions it implements from",
    " * this file. Stubs fail cleanly so unimplemented paths are loud.",
    " * Regenerate with tools/regen_stubs.py.",
    " */",
    '/* h3_gpu.h is a C API: give the declarations C linkage so gcc-compiled',
    ' * model-layer objects link against these nvcc-compiled definitions. */',
    'extern "C" {',
    '#include "h3_gpu.h"',
    '}',
    '#include "h3_cuda_internal.h"',
    "",
]
count = 0
for name, args in decls:
    if name in BUILTIN or name in implemented:
        continue
    count += 1
    args = " ".join(args.split())
    match = re.search(r"h3_gpu\s*\*\s*(\w+)", args)
    gpu = match.group(1) if match else "NULL"
    lines.append(f"int {name}({args}) {{")
    lines.append(f"    return h3_cuda_fail((struct h3_gpu *){gpu},")
    lines.append(f'                        "{name} is not implemented in the CUDA backend");')
    lines.append("}")
    lines.append("")
STUBS.write_text("\n".join(lines))
print(f"{count} stubs remain, {len(implemented)} implemented")
