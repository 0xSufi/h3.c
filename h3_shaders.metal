#include <metal_stdlib>
using namespace metal;

inline float h3_bf16_to_f32(ushort value) {
    return as_type<float>(uint(value) << 16);
}

inline ushort h3_f32_to_bf16(float value) {
    uint bits = as_type<uint>(value);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return ushort(bits >> 16);
}

struct linear_args {
    uint rows;
    uint input_dim;
    uint output_dim;
    uint has_bias;
};

kernel void h3_linear_f32(device const float *input [[buffer(0)]],
                          device const float *weight [[buffer(1)]],
                          device const float *bias [[buffer(2)]],
                          device float *output [[buffer(3)]],
                          constant linear_args &args [[buffer(4)]],
                          uint2 gid [[thread_position_in_grid]]) {
    uint column = gid.x;
    uint row = gid.y;
    if (row >= args.rows || column >= args.output_dim) return;
    float sum = args.has_bias ? bias[column] : 0.0f;
    device const float *x = input + row * args.input_dim;
    device const float *w = weight + column * args.input_dim;
    for (uint k = 0; k < args.input_dim; k++) sum = fma(x[k], w[k], sum);
    output[row * args.output_dim + column] = sum;
}

kernel void h3_silu_f32(device const float *input [[buffer(0)]],
                        device float *output [[buffer(1)]],
                        constant uint &count [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    float value = input[gid];
    output[gid] = value / (1.0f + exp(-value));
}

kernel void h3_cast_f32_to_bf16(device const float *input [[buffer(0)]],
                                device ushort *output [[buffer(1)]],
                                constant uint &count [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid < count) output[gid] = h3_f32_to_bf16(input[gid]);
}

kernel void h3_cast_bf16_to_f32(device const ushort *input [[buffer(0)]],
                                device float *output [[buffer(1)]],
                                constant uint &count [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid < count) output[gid] = h3_bf16_to_f32(input[gid]);
}

struct norm_args {
    uint rows;
    uint width;
    float epsilon;
};

struct qkv_args {
    uint sequence;
    uint heads;
    uint head_dim;
    uint rope_half;
    uint grouped;
    float epsilon;
};

kernel void h3_rms_norm_f32(device const float *input [[buffer(0)]],
                            device const float *weight [[buffer(1)]],
                            device float *output [[buffer(2)]],
                            constant norm_args &args [[buffer(3)]],
                            uint3 group [[threadgroup_position_in_grid]],
                            uint3 thread_position [[thread_position_in_threadgroup]],
                            uint3 threadgroup_size [[threads_per_threadgroup]]) {
    uint row = group.x;
    uint tid = thread_position.x;
    uint threads = threadgroup_size.x;
    if (row >= args.rows) return;
    threadgroup float reductions[256];
    device const float *x = input + row * args.width;
    float local_sum = 0.0f;
    for (uint k = tid; k < args.width; k += threads)
        local_sum = fma(x[k], x[k], local_sum);
    reductions[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse = rsqrt(reductions[0] / float(args.width) + args.epsilon);
    for (uint column = tid; column < args.width; column += threads)
        output[row * args.width + column] = x[column] * inverse * weight[column];
}

struct scale_add_args { uint rows; uint width; };

kernel void h3_scale_add_f32(device const float *residual [[buffer(0)]],
                             device const float *branch [[buffer(1)]],
                             device const float *scale [[buffer(2)]],
                             device float *output [[buffer(3)]],
                             constant scale_add_args &args [[buffer(4)]],
                             uint2 gid [[thread_position_in_grid]]) {
    uint column = gid.x;
    uint row = gid.y;
    if (row >= args.rows || column >= args.width) return;
    uint index = row * args.width + column;
    output[index] = residual[index] + branch[index] * scale[column];
}

kernel void h3_layer_norm_f32(device const float *input [[buffer(0)]],
                              device const float *weight [[buffer(1)]],
                              device const float *bias [[buffer(2)]],
                              device float *output [[buffer(3)]],
                              constant norm_args &args [[buffer(4)]],
                              uint3 group [[threadgroup_position_in_grid]],
                              uint3 thread_position [[thread_position_in_threadgroup]],
                              uint3 threadgroup_size [[threads_per_threadgroup]]) {
    uint row = group.x;
    uint tid = thread_position.x;
    uint threads = threadgroup_size.x;
    if (row >= args.rows) return;
    threadgroup float reductions[256];
    device const float *x = input + row * args.width;
    float local = 0.0f;
    for (uint k = tid; k < args.width; k += threads) local += x[k];
    reductions[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = reductions[0] / float(args.width);
    local = 0.0f;
    for (uint k = tid; k < args.width; k += threads) {
        float centered = x[k] - mean;
        local = fma(centered, centered, local);
    }
    reductions[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse = rsqrt(reductions[0] / float(args.width) + args.epsilon);
    for (uint column = tid; column < args.width; column += threads)
        output[row * args.width + column] =
            (x[column] - mean) * inverse * weight[column] + bias[column];
}

kernel void h3_video_qkv_rope_f32(
                            device const float *qkv [[buffer(0)]],
                            device const float *rope_cos [[buffer(1)]],
                            device const float *rope_sin [[buffer(2)]],
                            device float *query [[buffer(3)]],
                            device float *key [[buffer(4)]],
                            device float *value [[buffer(5)]],
                            constant qkv_args &args [[buffer(6)]],
                            uint3 gid [[thread_position_in_grid]]) {
    uint dimension = gid.x;
    uint head = gid.y;
    uint row = gid.z;
    if (dimension >= args.head_dim || head >= args.heads || row >= args.sequence) return;
    uint base = (row * args.heads + head) * args.head_dim * 3;
    float q_sum = 0.0f, k_sum = 0.0f;
    for (uint d = 0; d < args.head_dim; d++) {
        float q = qkv[base + d];
        float k = qkv[base + args.head_dim + d];
        q_sum = fma(q, q, q_sum);
        k_sum = fma(k, k, k_sum);
    }
    float qi = rsqrt(q_sum / float(args.head_dim) + args.epsilon);
    float ki = rsqrt(k_sum / float(args.head_dim) + args.epsilon);
    float q0 = qkv[base + dimension] * qi;
    float k0 = qkv[base + args.head_dim + dimension] * ki;
    if (dimension < args.rope_half) {
        uint pair = dimension + args.rope_half;
        float q1 = qkv[base + pair] * qi;
        float k1 = qkv[base + args.head_dim + pair] * ki;
        float c = rope_cos[row * args.rope_half + dimension];
        float s = rope_sin[row * args.rope_half + dimension];
        q0 = q0 * c - q1 * s;
        k0 = k0 * c - k1 * s;
    } else if (dimension < args.rope_half * 2) {
        uint pair = dimension - args.rope_half;
        float q1 = qkv[base + pair] * qi;
        float k1 = qkv[base + args.head_dim + pair] * ki;
        float c = rope_cos[row * args.rope_half + pair];
        float s = rope_sin[row * args.rope_half + pair];
        q0 = q0 * c + q1 * s;
        k0 = k0 * c + k1 * s;
    }
    uint output_index = (row * args.heads + head) * args.head_dim + dimension;
    query[output_index] = q0;
    key[output_index] = k0;
    value[output_index] = qkv[base + args.head_dim * 2 + dimension];
}

struct adaln_args {
    uint rows;
    uint width;
    uint slots;
    uint shift_slot;
    uint scale_slot;
    float epsilon;
};

kernel void h3_adaln_f32(device const float *input [[buffer(0)]],
                         device const float *weight [[buffer(1)]],
                         device const float *modulation [[buffer(2)]],
                         device const uint *row_map [[buffer(3)]],
                         device float *output [[buffer(4)]],
                         constant adaln_args &args [[buffer(5)]],
                         uint2 gid [[thread_position_in_grid]]) {
    uint column = gid.x;
    uint row = gid.y;
    if (row >= args.rows || column >= args.width) return;
    device const float *x = input + row * args.width;
    float sum = 0.0f;
    for (uint k = 0; k < args.width; k++) sum = fma(x[k], x[k], sum);
    float normalized = x[column] * rsqrt(sum / float(args.width) + args.epsilon) * weight[column];
    uint modulation_row = row_map[row];
    uint base = modulation_row * args.slots * args.width;
    float shift = modulation[base + args.shift_slot * args.width + column];
    float scale = modulation[base + args.scale_slot * args.width + column];
    output[row * args.width + column] = normalized * (1.0f + scale) + shift;
}

struct gate_args {
    uint rows;
    uint width;
    uint slots;
    uint gate_slot;
};

kernel void h3_gate_f32(device const float *residual [[buffer(0)]],
                        device const float *branch [[buffer(1)]],
                        device const float *modulation [[buffer(2)]],
                        device const uint *row_map [[buffer(3)]],
                        device float *output [[buffer(4)]],
                        constant gate_args &args [[buffer(5)]],
                        uint2 gid [[thread_position_in_grid]]) {
    uint column = gid.x;
    uint row = gid.y;
    if (row >= args.rows || column >= args.width) return;
    uint base = row_map[row] * args.slots * args.width;
    float gate = modulation[base + args.gate_slot * args.width + column];
    uint index = row * args.width + column;
    output[index] = residual[index] + branch[index] * gate;
}

kernel void h3_qkv_rope_f32(device const float *qkv [[buffer(0)]],
                            device const float *q_weight [[buffer(1)]],
                            device const float *k_weight [[buffer(2)]],
                            device const float *rope_cos [[buffer(3)]],
                            device const float *rope_sin [[buffer(4)]],
                            device float *query [[buffer(5)]],
                            device float *key [[buffer(6)]],
                            device float *value [[buffer(7)]],
                            constant qkv_args &args [[buffer(8)]],
                            uint3 gid [[thread_position_in_grid]]) {
    uint dimension = gid.x;
    uint head = gid.y;
    uint row = gid.z;
    if (dimension >= args.head_dim || head >= args.heads || row >= args.sequence) return;
    uint inner = args.heads * args.head_dim;
    uint row_base = row * inner * 3;
    uint q_base = row_base + head * args.head_dim;
    uint k_base = q_base + inner;
    uint v_base = q_base + inner * 2;
    if (args.grouped) {
        q_base = row_base + head * args.head_dim * 3;
        k_base = q_base + args.head_dim;
        v_base = k_base + args.head_dim;
    }
    float q_sum = 0.0f;
    float k_sum = 0.0f;
    for (uint d = 0; d < args.head_dim; d++) {
        float q = qkv[q_base + d];
        float k = qkv[k_base + d];
        q_sum = fma(q, q, q_sum);
        k_sum = fma(k, k, k_sum);
    }
    float q_inverse = rsqrt(q_sum / float(args.head_dim) + args.epsilon);
    float k_inverse = rsqrt(k_sum / float(args.head_dim) + args.epsilon);
    float q0 = qkv[q_base + dimension] * q_inverse * q_weight[dimension];
    float k0 = qkv[k_base + dimension] * k_inverse * k_weight[dimension];
    if (dimension < args.rope_half) {
        uint pair = dimension + args.rope_half;
        float q1 = qkv[q_base + pair] * q_inverse * q_weight[pair];
        float k1 = qkv[k_base + pair] * k_inverse * k_weight[pair];
        float c = rope_cos[row * args.rope_half + dimension];
        float s = rope_sin[row * args.rope_half + dimension];
        q0 = q0 * c - q1 * s;
        k0 = k0 * c - k1 * s;
    } else if (dimension < args.rope_half * 2) {
        uint pair = dimension - args.rope_half;
        float q1 = qkv[q_base + pair] * q_inverse * q_weight[pair];
        float k1 = qkv[k_base + pair] * k_inverse * k_weight[pair];
        float c = rope_cos[row * args.rope_half + pair];
        float s = rope_sin[row * args.rope_half + pair];
        q0 = q0 * c + q1 * s;
        k0 = k0 * c + k1 * s;
    }
    uint output_index = (row * args.heads + head) * args.head_dim + dimension;
    query[output_index] = q0;
    key[output_index] = k0;
    value[output_index] = qkv[v_base + dimension];
}

struct swiglu_args {
    uint rows;
    uint width;
};

kernel void h3_swiglu_f32(device const float *fused [[buffer(0)]],
                          device float *output [[buffer(1)]],
                          constant swiglu_args &args [[buffer(2)]],
                          uint2 gid [[thread_position_in_grid]]) {
    uint column = gid.x;
    uint row = gid.y;
    if (row >= args.rows || column >= args.width) return;
    uint base = row * args.width * 2;
    float gate = fused[base + column];
    float up = fused[base + args.width + column];
    output[row * args.width + column] = gate / (1.0f + exp(-gate)) * up;
}

struct weight_norm_args {
    uint outer;
    uint inner;
};

kernel void h3_weight_norm_f32(device const float *vector [[buffer(0)]],
                               device const float *magnitude [[buffer(1)]],
                               device float *output [[buffer(2)]],
                               constant weight_norm_args &args [[buffer(3)]],
                               uint row [[thread_position_in_grid]]) {
    if (row >= args.outer) return;
    uint base = row * args.inner;
    float square_sum = 0.0f;
    for (uint index = 0; index < args.inner; index++) {
        float value = vector[base + index];
        square_sum = fma(value, value, square_sum);
    }
    float scale = magnitude[row] * rsqrt(square_sum);
    for (uint index = 0; index < args.inner; index++)
        output[base + index] = vector[base + index] * scale;
}

struct add_scaled_args {
    uint elements;
    float left_scale;
    float right_scale;
};

kernel void h3_add_scaled_f32(device const float *left [[buffer(0)]],
                              device const float *right [[buffer(1)]],
                              device float *output [[buffer(2)]],
                              constant add_scaled_args &args [[buffer(3)]],
                              uint index [[thread_position_in_grid]]) {
    if (index < args.elements)
        output[index] = left[index] * args.left_scale +
                        right[index] * args.right_scale;
}

struct audio_activation_args {
    uint batch;
    uint length;
    uint channels;
};

/* BigVGAN's fixed 2x upsample -> SnakeBeta -> 2x low-pass downsample is
 * fused at the original rate. This avoids materializing the doubled waveform
 * for each of the decoder's 127 alias-free activations. */
kernel void h3_alias_free_snake_f32(
        device const float *input [[buffer(0)]],
        device const float *alpha_log [[buffer(1)]],
        device const float *beta_log [[buffer(2)]],
        device const float *upsample_filter [[buffer(3)]],
        device const float *downsample_filter [[buffer(4)]],
        device float *output [[buffer(5)]],
        constant audio_activation_args &args [[buffer(6)]],
        uint3 gid [[thread_position_in_grid]]) {
    uint channel = gid.x;
    uint time = gid.y;
    uint batch = gid.z;
    if (channel >= args.channels || time >= args.length ||
        batch >= args.batch) return;
    float alpha = exp(alpha_log[channel]);
    float beta = exp(beta_log[channel]);
    float result = 0.0f;
    for (int down_k = 0; down_k < 12; down_k++) {
        int up_time = int(time * 2) + down_k - 5;
        up_time = clamp(up_time, 0, int(args.length * 2) - 1);
        int raw_time = up_time + 15;
        float upsampled = 0.0f;
        for (int up_k = 0; up_k < 12; up_k++) {
            int numerator = raw_time - up_k;
            if (numerator < 0 || (numerator & 1)) continue;
            int padded_time = numerator / 2;
            int source_time = clamp(padded_time - 5, 0,
                                    int(args.length) - 1);
            uint source = (batch * args.length + uint(source_time)) *
                          args.channels + channel;
            upsampled = fma(input[source], 2.0f * upsample_filter[up_k],
                            upsampled);
        }
        float sine = sin(alpha * upsampled);
        float activated = upsampled + sine * sine / (beta + 1e-9f);
        result = fma(activated, downsample_filter[down_k], result);
    }
    uint destination = (batch * args.length + time) * args.channels + channel;
    output[destination] = result;
}

struct clip_args {
    uint elements;
    float minimum;
    float maximum;
};

kernel void h3_clip_f32(device const float *input [[buffer(0)]],
                        device float *output [[buffer(1)]],
                        constant clip_args &args [[buffer(2)]],
                        uint index [[thread_position_in_grid]]) {
    if (index < args.elements)
        output[index] = clamp(input[index], args.minimum, args.maximum);
}

kernel void h3_linear_bf16(device const ushort *input [[buffer(0)]],
                           device const ushort *weight [[buffer(1)]],
                           device const ushort *bias [[buffer(2)]],
                           device ushort *output [[buffer(3)]],
                           constant linear_args &args [[buffer(4)]],
                           uint2 tid [[thread_position_in_threadgroup]],
                           uint2 group [[threadgroup_position_in_grid]]) {
    /* Adapted from Iris's short-sequence Qwen path. Each 16x16 weight/input
     * tile is loaded once per threadgroup instead of once per output scalar. */
    threadgroup float input_tile[16][16];
    threadgroup float weight_tile[16][16];
    uint row = group.y * 16 + tid.y;
    uint column = group.x * 16 + tid.x;
    float sum = args.has_bias && column < args.output_dim ?
        h3_bf16_to_f32(bias[column]) : 0.0f;
    uint tile_count = (args.input_dim + 15) / 16;
    for (uint tile = 0; tile < tile_count; tile++) {
        uint input_k = tile * 16 + tid.x;
        input_tile[tid.y][tid.x] =
            row < args.rows && input_k < args.input_dim ?
            h3_bf16_to_f32(input[row * args.input_dim + input_k]) : 0.0f;
        uint weight_k = tile * 16 + tid.y;
        weight_tile[tid.y][tid.x] =
            column < args.output_dim && weight_k < args.input_dim ?
            h3_bf16_to_f32(weight[column * args.input_dim + weight_k]) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 16; k++) {
            sum = fma(input_tile[tid.y][k], weight_tile[k][tid.x], sum);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < args.rows && column < args.output_dim) {
        output[row * args.output_dim + column] = h3_f32_to_bf16(sum);
    }
}

kernel void h3_silu_bf16(device const ushort *input [[buffer(0)]],
                         device ushort *output [[buffer(1)]],
                         constant uint &count [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    float value = h3_bf16_to_f32(input[gid]);
    output[gid] = h3_f32_to_bf16(value / (1.0f + exp(-value)));
}

kernel void h3_rms_norm_bf16(device const ushort *input [[buffer(0)]],
                             device const ushort *weight [[buffer(1)]],
                             device ushort *output [[buffer(2)]],
                             constant norm_args &args [[buffer(3)]],
                             uint3 group [[threadgroup_position_in_grid]],
                             uint3 thread_position [[thread_position_in_threadgroup]],
                             uint3 threadgroup_size [[threads_per_threadgroup]]) {
    uint row = group.x;
    uint tid = thread_position.x;
    uint threads = threadgroup_size.x;
    if (row >= args.rows) return;
    threadgroup float reductions[256];
    device const ushort *x = input + row * args.width;
    float local_sum = 0.0f;
    for (uint k = tid; k < args.width; k += threads) {
        float value = h3_bf16_to_f32(x[k]);
        local_sum = fma(value, value, local_sum);
    }
    reductions[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse = rsqrt(reductions[0] / float(args.width) + args.epsilon);
    for (uint column = tid; column < args.width; column += threads) {
        float normalized = h3_bf16_to_f32(x[column]) * inverse;
        output[row * args.width + column] =
            h3_f32_to_bf16(normalized * h3_bf16_to_f32(weight[column]));
    }
}

kernel void h3_adaln_bf16(device const ushort *input [[buffer(0)]],
                          device const ushort *weight [[buffer(1)]],
                          device const ushort *modulation [[buffer(2)]],
                          device const uint *row_map [[buffer(3)]],
                          device ushort *output [[buffer(4)]],
                          constant adaln_args &args [[buffer(5)]],
                          uint3 group [[threadgroup_position_in_grid]],
                          uint3 thread_position [[thread_position_in_threadgroup]],
                          uint3 threadgroup_size [[threads_per_threadgroup]]) {
    uint row = group.x;
    uint tid = thread_position.x;
    uint threads = threadgroup_size.x;
    if (row >= args.rows) return;
    threadgroup float reductions[256];
    device const ushort *x = input + row * args.width;
    float local_sum = 0.0f;
    for (uint k = tid; k < args.width; k += threads) {
        float value = h3_bf16_to_f32(x[k]);
        local_sum = fma(value, value, local_sum);
    }
    reductions[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse = rsqrt(reductions[0] / float(args.width) + args.epsilon);
    uint base = row_map[row] * args.slots * args.width;
    for (uint column = tid; column < args.width; column += threads) {
        float normalized = h3_bf16_to_f32(x[column]) * inverse *
            h3_bf16_to_f32(weight[column]);
        float shift = h3_bf16_to_f32(
            modulation[base + args.shift_slot * args.width + column]);
        float scale = h3_bf16_to_f32(
            modulation[base + args.scale_slot * args.width + column]);
        output[row * args.width + column] =
            h3_f32_to_bf16(normalized * (1.0f + scale) + shift);
    }
}

kernel void h3_gate_bf16(device const ushort *residual [[buffer(0)]],
                         device const ushort *branch [[buffer(1)]],
                         device const ushort *modulation [[buffer(2)]],
                         device const uint *row_map [[buffer(3)]],
                         device ushort *output [[buffer(4)]],
                         constant gate_args &args [[buffer(5)]],
                         uint2 gid [[thread_position_in_grid]]) {
    uint column = gid.x;
    uint row = gid.y;
    if (row >= args.rows || column >= args.width) return;
    uint base = row_map[row] * args.slots * args.width;
    float gate = h3_bf16_to_f32(modulation[base + args.gate_slot * args.width + column]);
    uint index = row * args.width + column;
    float value = h3_bf16_to_f32(residual[index]) +
                  h3_bf16_to_f32(branch[index]) * gate;
    output[index] = h3_f32_to_bf16(value);
}

kernel void h3_qkv_rope_bf16(device const ushort *qkv [[buffer(0)]],
                             device const ushort *q_weight [[buffer(1)]],
                             device const ushort *k_weight [[buffer(2)]],
                             device const ushort *rope_cos [[buffer(3)]],
                             device const ushort *rope_sin [[buffer(4)]],
                             device ushort *query [[buffer(5)]],
                             device ushort *key [[buffer(6)]],
                             device ushort *value [[buffer(7)]],
                             constant qkv_args &args [[buffer(8)]],
                             uint3 gid [[thread_position_in_grid]]) {
    uint dimension = gid.x;
    uint head = gid.y;
    uint row = gid.z;
    if (dimension >= args.head_dim || head >= args.heads || row >= args.sequence) return;
    uint inner = args.heads * args.head_dim;
    uint row_base = row * inner * 3;
    uint q_base = row_base + head * args.head_dim;
    uint k_base = q_base + inner;
    uint v_base = q_base + inner * 2;
    if (args.grouped) {
        q_base = row_base + head * args.head_dim * 3;
        k_base = q_base + args.head_dim;
        v_base = k_base + args.head_dim;
    }
    float q_sum = 0.0f;
    float k_sum = 0.0f;
    for (uint d = 0; d < args.head_dim; d++) {
        float q = h3_bf16_to_f32(qkv[q_base + d]);
        float k = h3_bf16_to_f32(qkv[k_base + d]);
        q_sum = fma(q, q, q_sum);
        k_sum = fma(k, k, k_sum);
    }
    float q_inverse = rsqrt(q_sum / float(args.head_dim) + args.epsilon);
    float k_inverse = rsqrt(k_sum / float(args.head_dim) + args.epsilon);
    float q0 = h3_bf16_to_f32(qkv[q_base + dimension]) * q_inverse *
               h3_bf16_to_f32(q_weight[dimension]);
    float k0 = h3_bf16_to_f32(qkv[k_base + dimension]) * k_inverse *
               h3_bf16_to_f32(k_weight[dimension]);
    if (dimension < args.rope_half) {
        uint pair = dimension + args.rope_half;
        float q1 = h3_bf16_to_f32(qkv[q_base + pair]) * q_inverse *
                   h3_bf16_to_f32(q_weight[pair]);
        float k1 = h3_bf16_to_f32(qkv[k_base + pair]) * k_inverse *
                   h3_bf16_to_f32(k_weight[pair]);
        float c = h3_bf16_to_f32(rope_cos[row * args.rope_half + dimension]);
        float s = h3_bf16_to_f32(rope_sin[row * args.rope_half + dimension]);
        q0 = q0 * c - q1 * s;
        k0 = k0 * c - k1 * s;
    } else if (dimension < args.rope_half * 2) {
        uint pair = dimension - args.rope_half;
        float q1 = h3_bf16_to_f32(qkv[q_base + pair]) * q_inverse *
                   h3_bf16_to_f32(q_weight[pair]);
        float k1 = h3_bf16_to_f32(qkv[k_base + pair]) * k_inverse *
                   h3_bf16_to_f32(k_weight[pair]);
        float c = h3_bf16_to_f32(rope_cos[row * args.rope_half + pair]);
        float s = h3_bf16_to_f32(rope_sin[row * args.rope_half + pair]);
        q0 = q0 * c + q1 * s;
        k0 = k0 * c + k1 * s;
    }
    uint output_index = (row * args.heads + head) * args.head_dim + dimension;
    query[output_index] = h3_f32_to_bf16(q0);
    key[output_index] = h3_f32_to_bf16(k0);
    value[output_index] = qkv[v_base + dimension];
}

kernel void h3_swiglu_bf16(device const ushort *fused [[buffer(0)]],
                           device ushort *output [[buffer(1)]],
                           constant swiglu_args &args [[buffer(2)]],
                           uint2 gid [[thread_position_in_grid]]) {
    uint column = gid.x;
    uint row = gid.y;
    if (row >= args.rows || column >= args.width) return;
    uint base = row * args.width * 2;
    float gate = h3_bf16_to_f32(fused[base + column]);
    float up = h3_bf16_to_f32(fused[base + args.width + column]);
    output[row * args.width + column] =
        h3_f32_to_bf16(gate / (1.0f + exp(-gate)) * up);
}

struct embedding_args {
    uint tokens;
    uint vocab_size;
    uint width;
};

kernel void h3_embedding_bf16(device const ushort *weight [[buffer(0)]],
                              device const uint *token_ids [[buffer(1)]],
                              device ushort *output [[buffer(2)]],
                              constant embedding_args &args [[buffer(3)]],
                              uint2 gid [[thread_position_in_grid]]) {
    uint column = gid.x;
    uint token = gid.y;
    if (token >= args.tokens || column >= args.width) return;
    uint identifier = token_ids[token];
    output[token * args.width + column] = identifier < args.vocab_size ?
        weight[identifier * args.width + column] : ushort(0);
}

struct text_rope_args {
    uint sequence;
    uint query_heads;
    uint kv_heads;
    uint head_dim;
    float epsilon;
};

kernel void h3_text_qk_rope_bf16(
        device const ushort *query_input [[buffer(0)]],
        device const ushort *key_input [[buffer(1)]],
        device const ushort *q_weight [[buffer(2)]],
        device const ushort *k_weight [[buffer(3)]],
        device const ushort *rope_cos [[buffer(4)]],
        device const ushort *rope_sin [[buffer(5)]],
        device ushort *query_output [[buffer(6)]],
        device ushort *key_output [[buffer(7)]],
        constant text_rope_args &args [[buffer(8)]],
        uint3 gid [[thread_position_in_grid]]) {
    uint dimension = gid.x;
    uint head = gid.y;
    uint row = gid.z;
    if (dimension >= args.head_dim || head >= args.query_heads || row >= args.sequence) return;
    uint half_dim = args.head_dim / 2;
    uint pair = dimension < half_dim ? dimension + half_dim : dimension - half_dim;
    float c = h3_bf16_to_f32(rope_cos[row * half_dim + (dimension % half_dim)]);
    float s = h3_bf16_to_f32(rope_sin[row * half_dim + (dimension % half_dim)]);

    uint q_base = (row * args.query_heads + head) * args.head_dim;
    float q_sum = 0.0f;
    for (uint d = 0; d < args.head_dim; d++) {
        float value = h3_bf16_to_f32(query_input[q_base + d]);
        q_sum = fma(value, value, q_sum);
    }
    float q_inverse = rsqrt(q_sum / float(args.head_dim) + args.epsilon);
    float q0 = h3_bf16_to_f32(query_input[q_base + dimension]) * q_inverse *
               h3_bf16_to_f32(q_weight[dimension]);
    float q1 = h3_bf16_to_f32(query_input[q_base + pair]) * q_inverse *
               h3_bf16_to_f32(q_weight[pair]);
    float q_rotated = dimension < half_dim ? q0 * c - q1 * s : q0 * c + q1 * s;
    query_output[q_base + dimension] = h3_f32_to_bf16(q_rotated);

    if (head < args.kv_heads) {
        uint k_base = (row * args.kv_heads + head) * args.head_dim;
        float k_sum = 0.0f;
        for (uint d = 0; d < args.head_dim; d++) {
            float value = h3_bf16_to_f32(key_input[k_base + d]);
            k_sum = fma(value, value, k_sum);
        }
        float k_inverse = rsqrt(k_sum / float(args.head_dim) + args.epsilon);
        float k0 = h3_bf16_to_f32(key_input[k_base + dimension]) * k_inverse *
                   h3_bf16_to_f32(k_weight[dimension]);
        float k1 = h3_bf16_to_f32(key_input[k_base + pair]) * k_inverse *
                   h3_bf16_to_f32(k_weight[pair]);
        float k_rotated = dimension < half_dim ? k0 * c - k1 * s : k0 * c + k1 * s;
        key_output[k_base + dimension] = h3_f32_to_bf16(k_rotated);
    }
}

struct gqa_args {
    uint sequence;
    uint query_heads;
    uint kv_heads;
    uint head_dim;
    float scale;
};

struct head_norm_args {
    uint sequence;
    uint heads;
    uint head_dim;
    float epsilon;
};

/* Adapted from Iris's Qwen3 BF16 per-head norm: a single thread owns a
 * complete head, making in-place normalization race-free. */
kernel void h3_head_rms_norm_bf16(
        device ushort *tensor [[buffer(0)]],
        device const ushort *weight [[buffer(1)]],
        constant head_norm_args &args [[buffer(2)]],
        uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.x;
    uint head = gid.y;
    if (row >= args.sequence || head >= args.heads) return;
    uint base = (row * args.heads + head) * args.head_dim;
    float sum = 0.0f;
    for (uint d = 0; d < args.head_dim; d++) {
        float value = h3_bf16_to_f32(tensor[base + d]);
        sum = fma(value, value, sum);
    }
    float inverse = rsqrt(sum / float(args.head_dim) + args.epsilon);
    for (uint d = 0; d < args.head_dim; d++) {
        float value = h3_bf16_to_f32(tensor[base + d]);
        tensor[base + d] = h3_f32_to_bf16(
            value * inverse * h3_bf16_to_f32(weight[d]));
    }
}

struct text_rope_inplace_args {
    uint sequence;
    uint query_heads;
    uint kv_heads;
    uint head_dim;
};

/* Iris-style text RoPE. F32 tables avoid compounding table quantization while
 * Q/K remain BF16 at the operation boundary. */
kernel void h3_rope_text_bf16(
        device ushort *query [[buffer(0)]],
        device ushort *key [[buffer(1)]],
        device const float *rope_cos [[buffer(2)]],
        device const float *rope_sin [[buffer(3)]],
        constant text_rope_inplace_args &args [[buffer(4)]],
        uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.x;
    uint head = gid.y;
    if (row >= args.sequence) return;
    uint half_dim = args.head_dim / 2;
    if (head < args.query_heads) {
        uint base = (row * args.query_heads + head) * args.head_dim;
        for (uint d = 0; d < half_dim; d++) {
            float first = h3_bf16_to_f32(query[base + d]);
            float second = h3_bf16_to_f32(query[base + half_dim + d]);
            float c = rope_cos[row * half_dim + d];
            float s = rope_sin[row * half_dim + d];
            query[base + d] = h3_f32_to_bf16(first * c - second * s);
            query[base + half_dim + d] = h3_f32_to_bf16(second * c + first * s);
        }
    }
    if (head < args.kv_heads) {
        uint base = (row * args.kv_heads + head) * args.head_dim;
        for (uint d = 0; d < half_dim; d++) {
            float first = h3_bf16_to_f32(key[base + d]);
            float second = h3_bf16_to_f32(key[base + half_dim + d]);
            float c = rope_cos[row * half_dim + d];
            float s = rope_sin[row * half_dim + d];
            key[base + d] = h3_f32_to_bf16(first * c - second * s);
            key[base + half_dim + d] = h3_f32_to_bf16(second * c + first * s);
        }
    }
}

kernel void h3_gqa_causal_bf16(
        device const ushort *query [[buffer(0)]],
        device const ushort *key [[buffer(1)]],
        device const ushort *value [[buffer(2)]],
        device ushort *output [[buffer(3)]],
        constant gqa_args &args [[buffer(4)]],
        threadgroup float *scores [[threadgroup(0)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint3 thread_position [[thread_position_in_threadgroup]],
        uint3 threadgroup_size [[threads_per_threadgroup]]) {
    uint tid = thread_position.x;
    uint threads = threadgroup_size.x;
    uint query_row = group.x;
    uint query_head = group.y;
    if (query_row >= args.sequence || query_head >= args.query_heads) return;
    uint kv_head = query_head / (args.query_heads / args.kv_heads);
    uint q_base = (query_row * args.query_heads + query_head) * args.head_dim;
    uint key_count = query_row + 1;
    threadgroup float reductions[128];
    threadgroup float shared_query[128];

    for (uint d = tid; d < args.head_dim; d += threads) {
        shared_query[d] = h3_bf16_to_f32(query[q_base + d]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_max = -INFINITY;
    for (uint key_row = tid; key_row < key_count; key_row += threads) {
        uint k_base = (key_row * args.kv_heads + kv_head) * args.head_dim;
        float dot = 0.0f;
        for (uint d = 0; d < args.head_dim; d++) {
            dot = fma(shared_query[d], h3_bf16_to_f32(key[k_base + d]), dot);
        }
        float score = dot * args.scale;
        scores[key_row] = score;
        local_max = max(local_max, score);
    }
    reductions[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] = max(reductions[tid], reductions[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float maximum = reductions[0];
    float local_sum = 0.0f;
    for (uint key_row = tid; key_row < key_count; key_row += threads) {
        float probability = exp(scores[key_row] - maximum);
        scores[key_row] = probability;
        local_sum += probability;
    }
    reductions[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse_sum = 1.0f / reductions[0];
    for (uint d = tid; d < args.head_dim; d += threads) {
        float sum = 0.0f;
        for (uint key_row = 0; key_row < key_count; key_row++) {
            uint v_index = (key_row * args.kv_heads + kv_head) * args.head_dim + d;
            sum = fma(scores[key_row] * inverse_sum,
                      h3_bf16_to_f32(value[v_index]), sum);
        }
        output[q_base + d] = h3_f32_to_bf16(sum);
    }
}

kernel void h3_add_bf16(device const ushort *left [[buffer(0)]],
                         device const ushort *right [[buffer(1)]],
                         device ushort *output [[buffer(2)]],
                         constant uint &count [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    output[gid] = h3_f32_to_bf16(h3_bf16_to_f32(left[gid]) +
                                  h3_bf16_to_f32(right[gid]));
}

kernel void h3_silu_mul_bf16(device const ushort *gate [[buffer(0)]],
                              device const ushort *up [[buffer(1)]],
                              device ushort *output [[buffer(2)]],
                              constant uint &count [[buffer(3)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    float value = h3_bf16_to_f32(gate[gid]);
    float other = h3_bf16_to_f32(up[gid]);
    output[gid] = h3_f32_to_bf16(value / (1.0f + exp(-value)) * other);
}
