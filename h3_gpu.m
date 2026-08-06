#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "h3_gpu.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

@interface H3Tensor : NSObject
@property(nonatomic, strong) id<MTLBuffer> buffer;
@property(nonatomic) size_t elements;
@end
@implementation H3Tensor
@end

@interface H3SDPA : NSObject
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphTensor *query;
@property(nonatomic, strong) MPSGraphTensor *key;
@property(nonatomic, strong) MPSGraphTensor *value;
@property(nonatomic, strong) MPSGraphTensor *output;
@property(nonatomic, strong) NSArray<NSNumber *> *shape;
@end
@implementation H3SDPA
@end

@interface H3GPU : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) id<MTLLibrary> library;
@property(nonatomic, strong) id<MTLCommandBuffer> command;
@property(nonatomic, strong) NSDictionary<NSString *, id<MTLComputePipelineState>> *pipelines;
@property(nonatomic, strong) NSMutableDictionary<NSString *, H3SDPA *> *sdpaCache;
@property(nonatomic, copy) NSString *lastError;
@end
@implementation H3GPU
@end

static H3GPU *GPU(h3_gpu *gpu) {
    return (__bridge H3GPU *)gpu;
}

static H3Tensor *TENSOR(const h3_gpu_tensor *tensor) {
    return (__bridge H3Tensor *)(void *)tensor;
}

static void h3_gpu_set_error(H3GPU *gpu, NSString *format, ...) {
    va_list arguments;
    va_start(arguments, format);
    gpu.lastError = [[NSString alloc] initWithFormat:format arguments:arguments];
    va_end(arguments);
}

static int h3_gpu_require_command(H3GPU *gpu) {
    if (!gpu || !gpu.command) {
        if (gpu) h3_gpu_set_error(gpu, @"h3_gpu_begin() was not called");
        return 0;
    }
    return 1;
}

static int h3_gpu_require_elements(H3GPU *gpu, const h3_gpu_tensor *tensor,
                                   size_t elements, NSString *label) {
    if (!tensor || TENSOR(tensor).elements < elements) {
        h3_gpu_set_error(gpu, @"%@ tensor is absent or too small", label);
        return 0;
    }
    return 1;
}

static id<MTLComputePipelineState> h3_gpu_pipeline(H3GPU *gpu, NSString *name) {
    id<MTLComputePipelineState> pipeline = gpu.pipelines[name];
    if (!pipeline) h3_gpu_set_error(gpu, @"missing Metal pipeline %@", name);
    return pipeline;
}

static int h3_gpu_dispatch_1d(H3GPU *gpu, NSString *name, uint32_t count,
                              void (^bindings)(id<MTLComputeCommandEncoder>)) {
    if (!h3_gpu_require_command(gpu)) return 0;
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(gpu, name);
    if (!pipeline) return 0;
    id<MTLComputeCommandEncoder> encoder = [gpu.command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    bindings(encoder);
    NSUInteger width = MIN((NSUInteger)256, pipeline.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    [encoder endEncoding];
    return 1;
}

static int h3_gpu_dispatch_2d(H3GPU *gpu, NSString *name, uint32_t width,
                              uint32_t height,
                              void (^bindings)(id<MTLComputeCommandEncoder>)) {
    if (!h3_gpu_require_command(gpu)) return 0;
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(gpu, name);
    if (!pipeline) return 0;
    id<MTLComputeCommandEncoder> encoder = [gpu.command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    bindings(encoder);
    NSUInteger x = 16;
    NSUInteger y = MIN((NSUInteger)16, pipeline.maxTotalThreadsPerThreadgroup / x);
    [encoder dispatchThreads:MTLSizeMake(width, height, 1)
       threadsPerThreadgroup:MTLSizeMake(x, y, 1)];
    [encoder endEncoding];
    return 1;
}

static int h3_gpu_dispatch_3d(H3GPU *gpu, NSString *name, MTLSize grid,
                              void (^bindings)(id<MTLComputeCommandEncoder>)) {
    if (!h3_gpu_require_command(gpu)) return 0;
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(gpu, name);
    if (!pipeline) return 0;
    id<MTLComputeCommandEncoder> encoder = [gpu.command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    bindings(encoder);
    [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(8, 4, 1)];
    [encoder endEncoding];
    return 1;
}

h3_gpu *h3_gpu_create(const char *shader_source_path,
                      char *error, size_t error_size) {
    @autoreleasepool {
        H3GPU *gpu = [[H3GPU alloc] init];
        gpu.device = MTLCreateSystemDefaultDevice();
        gpu.queue = [gpu.device newCommandQueue];
        gpu.sdpaCache = [NSMutableDictionary dictionary];
        if (!gpu.device || !gpu.queue) {
            if (error && error_size) snprintf(error, error_size, "cannot initialize Metal");
            return NULL;
        }
        const char *source_path = shader_source_path ? shader_source_path :
                                                       "h3_shaders.metal";
        NSString *path = [NSString stringWithUTF8String:source_path];
        NSError *libraryError = nil;
        NSString *source = [NSString stringWithContentsOfFile:path
                                                     encoding:NSUTF8StringEncoding
                                                        error:&libraryError];
        if (source) {
            MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
            options.mathMode = MTLMathModeSafe;
            gpu.library = [gpu.device newLibraryWithSource:source
                                                   options:options
                                                     error:&libraryError];
        }
        if (!gpu.library) {
            if (error && error_size) {
                const char *description = libraryError.localizedDescription.UTF8String;
                snprintf(error, error_size, "cannot compile %s: %s",
                         source_path, description ? description : "unknown error");
            }
            return NULL;
        }
        NSArray<NSString *> *names = @[
            @"h3_linear_f32", @"h3_silu_f32", @"h3_rms_norm_f32",
            @"h3_adaln_f32", @"h3_gate_f32", @"h3_qkv_rope_f32",
            @"h3_swiglu_f32"
        ];
        NSMutableDictionary *pipelines = [NSMutableDictionary dictionary];
        for (NSString *name in names) {
            id<MTLFunction> function = [gpu.library newFunctionWithName:name];
            NSError *pipelineError = nil;
            id<MTLComputePipelineState> pipeline =
                function ? [gpu.device newComputePipelineStateWithFunction:function
                                                                       error:&pipelineError] : nil;
            if (!pipeline) {
                if (error && error_size) {
                    const char *description = pipelineError.localizedDescription.UTF8String;
                    snprintf(error, error_size, "cannot build %s: %s", name.UTF8String,
                             description ? description : "function missing");
                }
                return NULL;
            }
            pipelines[name] = pipeline;
        }
        gpu.pipelines = pipelines;
        return (__bridge_retained h3_gpu *)gpu;
    }
}

void h3_gpu_free(h3_gpu *gpu) {
    if (!gpu) return;
    H3GPU *object = CFBridgingRelease(gpu);
    object.command = nil;
}

static h3_gpu_tensor *h3_gpu_tensor_new(h3_gpu *opaque, const void *values,
                                        size_t elements, size_t item_size) {
    H3GPU *gpu = GPU(opaque);
    if (!gpu || elements > SIZE_MAX / item_size) return NULL;
    size_t bytes = elements * item_size;
    H3Tensor *tensor = [[H3Tensor alloc] init];
    tensor.elements = elements;
    tensor.buffer = [gpu.device newBufferWithLength:MAX(bytes, (size_t)1)
                                            options:MTLResourceStorageModeShared];
    if (!tensor.buffer) {
        h3_gpu_set_error(gpu, @"cannot allocate %zu-byte Metal buffer", bytes);
        return NULL;
    }
    if (values && bytes) memcpy(tensor.buffer.contents, values, bytes);
    return (__bridge_retained h3_gpu_tensor *)tensor;
}

h3_gpu_tensor *h3_gpu_tensor_new_f32(h3_gpu *gpu, size_t elements) {
    return h3_gpu_tensor_new(gpu, NULL, elements, sizeof(float));
}

h3_gpu_tensor *h3_gpu_tensor_from_f32(h3_gpu *gpu, const float *values,
                                      size_t elements) {
    return h3_gpu_tensor_new(gpu, values, elements, sizeof(float));
}

h3_gpu_tensor *h3_gpu_tensor_from_u32(h3_gpu *gpu, const uint32_t *values,
                                      size_t elements) {
    return h3_gpu_tensor_new(gpu, values, elements, sizeof(uint32_t));
}

void h3_gpu_tensor_free(h3_gpu_tensor *tensor) {
    if (!tensor) return;
    H3Tensor *object = CFBridgingRelease(tensor);
    object.buffer = nil;
}

size_t h3_gpu_tensor_elements(const h3_gpu_tensor *tensor) {
    return tensor ? TENSOR(tensor).elements : 0;
}

int h3_gpu_tensor_read_f32(const h3_gpu_tensor *tensor, float *values,
                           size_t elements) {
    if (!tensor || !values || elements > TENSOR(tensor).elements) return 0;
    memcpy(values, TENSOR(tensor).buffer.contents, elements * sizeof(float));
    return 1;
}

int h3_gpu_begin(h3_gpu *opaque) {
    H3GPU *gpu = GPU(opaque);
    if (!gpu || gpu.command) return 0;
    gpu.lastError = nil;
    gpu.command = [gpu.queue commandBuffer];
    if (!gpu.command) {
        h3_gpu_set_error(gpu, @"cannot create Metal command buffer");
        return 0;
    }
    return 1;
}

int h3_gpu_submit(h3_gpu *opaque) {
    H3GPU *gpu = GPU(opaque);
    if (!gpu || !gpu.command) return 0;
    id<MTLCommandBuffer> command = gpu.command;
    gpu.command = nil;
    [command commit];
    [command waitUntilCompleted];
    if (command.status == MTLCommandBufferStatusError) {
        h3_gpu_set_error(gpu, @"Metal command failed: %@", command.error.localizedDescription);
        return 0;
    }
    return 1;
}

const char *h3_gpu_error(const h3_gpu *opaque) {
    H3GPU *gpu = GPU((h3_gpu *)(void *)opaque);
    const char *message = gpu.lastError.UTF8String;
    return message ? message : "unknown Metal error";
}

typedef struct { uint32_t rows, input_dim, output_dim, has_bias; } linear_args;
typedef struct { uint32_t rows, width; float epsilon; } norm_args;
typedef struct {
    uint32_t rows, width, slots, shift_slot, scale_slot;
    float epsilon;
} adaln_args;
typedef struct { uint32_t rows, width, slots, gate_slot; } gate_args;
typedef struct { uint32_t sequence, heads, head_dim, rope_half; float epsilon; } qkv_args;
typedef struct { uint32_t rows, width; } swiglu_args;

int h3_gpu_linear_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t rows,
                      uint32_t input_dim, uint32_t output_dim) {
    H3GPU *gpu = GPU(opaque);
    size_t input_count = (size_t)rows * input_dim;
    size_t weight_count = (size_t)output_dim * input_dim;
    size_t output_count = (size_t)rows * output_dim;
    if (!h3_gpu_require_elements(gpu, input, input_count, @"linear input") ||
        !h3_gpu_require_elements(gpu, weight, weight_count, @"linear weight") ||
        !h3_gpu_require_elements(gpu, output, output_count, @"linear output") ||
        (bias && !h3_gpu_require_elements(gpu, bias, output_dim, @"linear bias"))) return 0;
    linear_args args = {rows, input_dim, output_dim, bias ? 1u : 0u};
    const h3_gpu_tensor *bias_buffer = bias ? bias : input;
    return h3_gpu_dispatch_2d(gpu, @"h3_linear_f32", output_dim, rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(bias_buffer).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:3];
            [encoder setBytes:&args length:sizeof(args) atIndex:4];
        });
}

int h3_gpu_silu_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t elements) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_elements(gpu, input, elements, @"SiLU input") ||
        !h3_gpu_require_elements(gpu, output, elements, @"SiLU output")) return 0;
    return h3_gpu_dispatch_1d(gpu, @"h3_silu_f32", elements,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&elements length:sizeof(elements) atIndex:2];
        });
}

int h3_gpu_rms_norm_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                        const h3_gpu_tensor *input,
                        const h3_gpu_tensor *weight, uint32_t rows,
                        uint32_t width, float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)rows * width;
    if (!h3_gpu_require_elements(gpu, input, count, @"RMSNorm input") ||
        !h3_gpu_require_elements(gpu, weight, width, @"RMSNorm weight") ||
        !h3_gpu_require_elements(gpu, output, count, @"RMSNorm output")) return 0;
    norm_args args = {rows, width, epsilon};
    return h3_gpu_dispatch_2d(gpu, @"h3_rms_norm_f32", width, rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:2];
            [encoder setBytes:&args length:sizeof(args) atIndex:3];
        });
}

int h3_gpu_adaln_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input,
                     const h3_gpu_tensor *norm_weight,
                     const h3_gpu_tensor *modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t shift_slot,
                     uint32_t scale_slot, float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)rows * width;
    if (!h3_gpu_require_elements(gpu, input, count, @"AdaLN input") ||
        !h3_gpu_require_elements(gpu, norm_weight, width, @"AdaLN norm") ||
        !h3_gpu_require_elements(gpu, row_map, rows, @"AdaLN row map") ||
        !h3_gpu_require_elements(gpu, output, count, @"AdaLN output") ||
        !modulation || shift_slot >= slots || scale_slot >= slots) return 0;
    adaln_args args = {rows, width, slots, shift_slot, scale_slot, epsilon};
    return h3_gpu_dispatch_2d(gpu, @"h3_adaln_f32", width, rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(norm_weight).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(modulation).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(row_map).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:4];
            [encoder setBytes:&args length:sizeof(args) atIndex:5];
        });
}

int h3_gpu_gate_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                    const h3_gpu_tensor *residual,
                    const h3_gpu_tensor *branch,
                    const h3_gpu_tensor *modulation,
                    const h3_gpu_tensor *row_map, uint32_t rows,
                    uint32_t width, uint32_t slots, uint32_t gate_slot) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)rows * width;
    if (!h3_gpu_require_elements(gpu, residual, count, @"gate residual") ||
        !h3_gpu_require_elements(gpu, branch, count, @"gate branch") ||
        !h3_gpu_require_elements(gpu, row_map, rows, @"gate row map") ||
        !h3_gpu_require_elements(gpu, output, count, @"gate output") ||
        !modulation || gate_slot >= slots) return 0;
    gate_args args = {rows, width, slots, gate_slot};
    return h3_gpu_dispatch_2d(gpu, @"h3_gate_f32", width, rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(residual).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(branch).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(modulation).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(row_map).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:4];
            [encoder setBytes:&args length:sizeof(args) atIndex:5];
        });
}

int h3_gpu_qkv_rope_f32(h3_gpu *opaque, h3_gpu_tensor *query,
                        h3_gpu_tensor *key, h3_gpu_tensor *value,
                        const h3_gpu_tensor *qkv,
                        const h3_gpu_tensor *q_norm,
                        const h3_gpu_tensor *k_norm,
                        const h3_gpu_tensor *rope_cos,
                        const h3_gpu_tensor *rope_sin, uint32_t sequence,
                        uint32_t heads, uint32_t head_dim,
                        uint32_t rope_half, float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t inner = (size_t)heads * head_dim;
    size_t count = (size_t)sequence * inner;
    size_t rope_count = (size_t)sequence * rope_half;
    if (!h3_gpu_require_elements(gpu, qkv, count * 3, @"QKV input") ||
        !h3_gpu_require_elements(gpu, q_norm, head_dim, @"Q norm") ||
        !h3_gpu_require_elements(gpu, k_norm, head_dim, @"K norm") ||
        !h3_gpu_require_elements(gpu, rope_cos, rope_count, @"RoPE cosine") ||
        !h3_gpu_require_elements(gpu, rope_sin, rope_count, @"RoPE sine") ||
        !h3_gpu_require_elements(gpu, query, count, @"query") ||
        !h3_gpu_require_elements(gpu, key, count, @"key") ||
        !h3_gpu_require_elements(gpu, value, count, @"value") ||
        rope_half * 2 > head_dim) return 0;
    qkv_args args = {sequence, heads, head_dim, rope_half, epsilon};
    return h3_gpu_dispatch_3d(gpu, @"h3_qkv_rope_f32",
        MTLSizeMake(head_dim, heads, sequence),
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(qkv).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(q_norm).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(k_norm).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(rope_cos).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(rope_sin).buffer offset:0 atIndex:4];
            [encoder setBuffer:TENSOR(query).buffer offset:0 atIndex:5];
            [encoder setBuffer:TENSOR(key).buffer offset:0 atIndex:6];
            [encoder setBuffer:TENSOR(value).buffer offset:0 atIndex:7];
            [encoder setBytes:&args length:sizeof(args) atIndex:8];
        });
}

static H3SDPA *h3_gpu_sdpa_graph(H3GPU *gpu, uint32_t sequence,
                                 uint32_t heads, uint32_t head_dim, float scale) {
    NSString *cacheKey = [NSString stringWithFormat:@"%u:%u:%u:%.9g",
                          sequence, heads, head_dim, scale];
    H3SDPA *cached = gpu.sdpaCache[cacheKey];
    if (cached) return cached;
    MPSGraph *graph = [[MPSGraph alloc] init];
    NSArray<NSNumber *> *shape = @[@1, @(sequence), @(heads), @(head_dim)];
    MPSGraphTensor *q = [graph placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor *k = [graph placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor *v = [graph placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor *qt = [graph transposeTensor:q dimension:1 withDimension:2 name:nil];
    MPSGraphTensor *kt = [graph transposeTensor:k dimension:1 withDimension:2 name:nil];
    MPSGraphTensor *vt = [graph transposeTensor:v dimension:1 withDimension:2 name:nil];
    if (![graph respondsToSelector:@selector(scaledDotProductAttentionWithQueryTensor:keyTensor:valueTensor:scale:name:)]) {
        h3_gpu_set_error(gpu, @"native MPSGraph SDPA is unavailable");
        return nil;
    }
    MPSGraphTensor *attention = [graph scaledDotProductAttentionWithQueryTensor:qt
                                                                      keyTensor:kt
                                                                    valueTensor:vt
                                                                         scale:scale
                                                                           name:nil];
    H3SDPA *result = [[H3SDPA alloc] init];
    result.graph = graph;
    result.query = q;
    result.key = k;
    result.value = v;
    result.output = [graph transposeTensor:attention dimension:1 withDimension:2 name:nil];
    result.shape = shape;
    gpu.sdpaCache[cacheKey] = result;
    return result;
}

int h3_gpu_sdpa_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                    const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                    const h3_gpu_tensor *value, uint32_t sequence,
                    uint32_t heads, uint32_t head_dim, float scale) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)sequence * heads * head_dim;
    if (!h3_gpu_require_command(gpu) ||
        !h3_gpu_require_elements(gpu, query, count, @"SDPA query") ||
        !h3_gpu_require_elements(gpu, key, count, @"SDPA key") ||
        !h3_gpu_require_elements(gpu, value, count, @"SDPA value") ||
        !h3_gpu_require_elements(gpu, output, count, @"SDPA output")) return 0;
    H3SDPA *cache = h3_gpu_sdpa_graph(gpu, sequence, heads, head_dim, scale);
    if (!cache) return 0;
    MPSCommandBuffer *command = [MPSCommandBuffer commandBufferWithCommandBuffer:gpu.command];
    MPSGraphTensorData *(^data)(const h3_gpu_tensor *) = ^MPSGraphTensorData *(const h3_gpu_tensor *tensor) {
        return [[MPSGraphTensorData alloc] initWithMTLBuffer:TENSOR(tensor).buffer
                                                     shape:cache.shape
                                                  dataType:MPSDataTypeFloat32];
    };
    NSDictionary *feeds = @{
        cache.query: data(query), cache.key: data(key), cache.value: data(value)
    };
    NSDictionary *results = @{cache.output: data(output)};
    @try {
        [cache.graph encodeToCommandBuffer:command feeds:feeds targetOperations:nil
                         resultsDictionary:results executionDescriptor:nil];
    } @catch (NSException *exception) {
        h3_gpu_set_error(gpu, @"MPSGraph SDPA failed: %@", exception.reason);
        return 0;
    }
    gpu.command = command.rootCommandBuffer;
    return 1;
}

int h3_gpu_swiglu_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                      const h3_gpu_tensor *fused, uint32_t rows,
                      uint32_t width) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_elements(gpu, fused, (size_t)rows * width * 2, @"SwiGLU input") ||
        !h3_gpu_require_elements(gpu, output, (size_t)rows * width, @"SwiGLU output")) return 0;
    swiglu_args args = {rows, width};
    return h3_gpu_dispatch_2d(gpu, @"h3_swiglu_f32", width, rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(fused).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
        });
}
