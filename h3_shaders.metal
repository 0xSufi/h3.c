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

struct norm_args {
    uint rows;
    uint width;
    float epsilon;
};

kernel void h3_rms_norm_f32(device const float *input [[buffer(0)]],
                            device const float *weight [[buffer(1)]],
                            device float *output [[buffer(2)]],
                            constant norm_args &args [[buffer(3)]],
                            uint2 gid [[thread_position_in_grid]]) {
    uint column = gid.x;
    uint row = gid.y;
    if (row >= args.rows || column >= args.width) return;
    device const float *x = input + row * args.width;
    float sum = 0.0f;
    for (uint k = 0; k < args.width; k++) sum = fma(x[k], x[k], sum);
    float inverse = rsqrt(sum / float(args.width) + args.epsilon);
    output[row * args.width + column] = x[column] * inverse * weight[column];
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

struct qkv_args {
    uint sequence;
    uint heads;
    uint head_dim;
    uint rope_half;
    float epsilon;
};

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
    uint base = row * inner * 3 + head * args.head_dim;
    float q_sum = 0.0f;
    float k_sum = 0.0f;
    for (uint d = 0; d < args.head_dim; d++) {
        float q = qkv[base + d];
        float k = qkv[base + inner + d];
        q_sum = fma(q, q, q_sum);
        k_sum = fma(k, k, k_sum);
    }
    float q_inverse = rsqrt(q_sum / float(args.head_dim) + args.epsilon);
    float k_inverse = rsqrt(k_sum / float(args.head_dim) + args.epsilon);
    float q0 = qkv[base + dimension] * q_inverse * q_weight[dimension];
    float k0 = qkv[base + inner + dimension] * k_inverse * k_weight[dimension];
    if (dimension < args.rope_half) {
        uint pair = dimension + args.rope_half;
        float q1 = qkv[base + pair] * q_inverse * q_weight[pair];
        float k1 = qkv[base + inner + pair] * k_inverse * k_weight[pair];
        float c = rope_cos[row * args.rope_half + dimension];
        float s = rope_sin[row * args.rope_half + dimension];
        q0 = q0 * c - q1 * s;
        k0 = k0 * c - k1 * s;
    } else if (dimension < args.rope_half * 2) {
        uint pair = dimension - args.rope_half;
        float q1 = qkv[base + pair] * q_inverse * q_weight[pair];
        float k1 = qkv[base + inner + pair] * k_inverse * k_weight[pair];
        float c = rope_cos[row * args.rope_half + pair];
        float s = rope_sin[row * args.rope_half + pair];
        q0 = q0 * c + q1 * s;
        k0 = k0 * c + k1 * s;
    }
    uint output_index = (row * args.heads + head) * args.head_dim + dimension;
    query[output_index] = q0;
    key[output_index] = k0;
    value[output_index] = qkv[base + inner * 2 + dimension];
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

kernel void h3_linear_bf16(device const ushort *input [[buffer(0)]],
                           device const ushort *weight [[buffer(1)]],
                           device const ushort *bias [[buffer(2)]],
                           device ushort *output [[buffer(3)]],
                           constant linear_args &args [[buffer(4)]],
                           uint2 gid [[thread_position_in_grid]]) {
    uint column = gid.x;
    uint row = gid.y;
    if (row >= args.rows || column >= args.output_dim) return;
    float sum = args.has_bias ? h3_bf16_to_f32(bias[column]) : 0.0f;
    device const ushort *x = input + row * args.input_dim;
    device const ushort *w = weight + column * args.input_dim;
    for (uint k = 0; k < args.input_dim; k++) {
        sum = fma(h3_bf16_to_f32(x[k]), h3_bf16_to_f32(w[k]), sum);
    }
    output[row * args.output_dim + column] = h3_f32_to_bf16(sum);
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
                             uint2 gid [[thread_position_in_grid]]) {
    uint column = gid.x;
    uint row = gid.y;
    if (row >= args.rows || column >= args.width) return;
    device const ushort *x = input + row * args.width;
    float sum = 0.0f;
    for (uint k = 0; k < args.width; k++) {
        float value = h3_bf16_to_f32(x[k]);
        sum = fma(value, value, sum);
    }
    float value = h3_bf16_to_f32(x[column]);
    float normalized = value * rsqrt(sum / float(args.width) + args.epsilon);
    output[row * args.width + column] =
        h3_f32_to_bf16(normalized * h3_bf16_to_f32(weight[column]));
}

kernel void h3_adaln_bf16(device const ushort *input [[buffer(0)]],
                          device const ushort *weight [[buffer(1)]],
                          device const ushort *modulation [[buffer(2)]],
                          device const uint *row_map [[buffer(3)]],
                          device ushort *output [[buffer(4)]],
                          constant adaln_args &args [[buffer(5)]],
                          uint2 gid [[thread_position_in_grid]]) {
    uint column = gid.x;
    uint row = gid.y;
    if (row >= args.rows || column >= args.width) return;
    device const ushort *x = input + row * args.width;
    float sum = 0.0f;
    for (uint k = 0; k < args.width; k++) {
        float value = h3_bf16_to_f32(x[k]);
        sum = fma(value, value, sum);
    }
    float normalized = h3_bf16_to_f32(x[column]) *
        rsqrt(sum / float(args.width) + args.epsilon) *
        h3_bf16_to_f32(weight[column]);
    uint base = row_map[row] * args.slots * args.width;
    float shift = h3_bf16_to_f32(modulation[base + args.shift_slot * args.width + column]);
    float scale = h3_bf16_to_f32(modulation[base + args.scale_slot * args.width + column]);
    output[row * args.width + column] = h3_f32_to_bf16(normalized * (1.0f + scale) + shift);
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
    uint base = row * inner * 3 + head * args.head_dim;
    float q_sum = 0.0f;
    float k_sum = 0.0f;
    for (uint d = 0; d < args.head_dim; d++) {
        float q = h3_bf16_to_f32(qkv[base + d]);
        float k = h3_bf16_to_f32(qkv[base + inner + d]);
        q_sum = fma(q, q, q_sum);
        k_sum = fma(k, k, k_sum);
    }
    float q_inverse = rsqrt(q_sum / float(args.head_dim) + args.epsilon);
    float k_inverse = rsqrt(k_sum / float(args.head_dim) + args.epsilon);
    float q0 = h3_bf16_to_f32(qkv[base + dimension]) * q_inverse *
               h3_bf16_to_f32(q_weight[dimension]);
    float k0 = h3_bf16_to_f32(qkv[base + inner + dimension]) * k_inverse *
               h3_bf16_to_f32(k_weight[dimension]);
    if (dimension < args.rope_half) {
        uint pair = dimension + args.rope_half;
        float q1 = h3_bf16_to_f32(qkv[base + pair]) * q_inverse *
                   h3_bf16_to_f32(q_weight[pair]);
        float k1 = h3_bf16_to_f32(qkv[base + inner + pair]) * k_inverse *
                   h3_bf16_to_f32(k_weight[pair]);
        float c = h3_bf16_to_f32(rope_cos[row * args.rope_half + dimension]);
        float s = h3_bf16_to_f32(rope_sin[row * args.rope_half + dimension]);
        q0 = q0 * c - q1 * s;
        k0 = k0 * c - k1 * s;
    } else if (dimension < args.rope_half * 2) {
        uint pair = dimension - args.rope_half;
        float q1 = h3_bf16_to_f32(qkv[base + pair]) * q_inverse *
                   h3_bf16_to_f32(q_weight[pair]);
        float k1 = h3_bf16_to_f32(qkv[base + inner + pair]) * k_inverse *
                   h3_bf16_to_f32(k_weight[pair]);
        float c = h3_bf16_to_f32(rope_cos[row * args.rope_half + pair]);
        float s = h3_bf16_to_f32(rope_sin[row * args.rope_half + pair]);
        q0 = q0 * c + q1 * s;
        k0 = k0 * c + k1 * s;
    }
    uint output_index = (row * args.heads + head) * args.head_dim + dimension;
    query[output_index] = h3_f32_to_bf16(q0);
    key[output_index] = h3_f32_to_bf16(k0);
    value[output_index] = qkv[base + inner * 2 + dimension];
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
