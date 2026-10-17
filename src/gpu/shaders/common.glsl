// Shared declarations for the GPU backend's compute shaders (gpu-plan.md §1).
// Buffers are passed as device addresses in push constants, so no shader uses
// descriptor sets.

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(buffer_reference, std430, buffer_reference_align = 4) buffer F32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) buffer V4 { vec4 v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer U32 { uint v[]; };
layout(buffer_reference, std430, buffer_reference_align = 2) buffer U16 { uint16_t v[]; };
layout(buffer_reference, std430, buffer_reference_align = 2) buffer F16 { float16_t v[]; };
layout(buffer_reference, std430, buffer_reference_align = 8) buffer U2 { uvec2 v[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) buffer U4 { uvec4 v[]; };

// Q6_K as uploaded (gpu/model.rs, `repack_q6k`): a tensor's 192-byte ql+qh
// parts of every block first, then a 20-byte (scales, d, padding) part per
// block, so the bulk loads are 16-byte aligned.
#define Q6K_QS_BYTES 192u
#define Q6K_SC_BYTES 20u

// Byte `i` of a little-endian word array.
#define BYTE_OF(word, i) (((word) >> (8u * (i))) & 0xFFu)

// Sign-extends the low 8 bits.
int sext8(uint b) { return int(b << 24) >> 24; }

// Q4_K scale and min of sub-block j (0..8) from the 12 scale bytes held in three
// words (llama.cpp's get_scale_min_k4).
void q4k_scale_min(uint j, uvec3 s, out float sc, out float mn) {
    uint b[12];
    for (uint i = 0; i < 12; i++) b[i] = BYTE_OF(s[i / 4], i % 4);
    if (j < 4) {
        sc = float(b[j] & 63u);
        mn = float(b[j + 4] & 63u);
    } else {
        sc = float((b[j + 4] & 0xFu) | ((b[j - 4] >> 6) << 4));
        mn = float((b[j + 4] >> 4) | ((b[j] >> 6) << 4));
    }
}

shared float red_[64];

// Sum over the workgroup; every invocation gets the result. Needs
// red_[gl_NumSubgroups] and a uniform call site.
float workgroup_sum(float v) {
    v = subgroupAdd(v);
    if (subgroupElect()) red_[gl_SubgroupID] = v;
    barrier();
    float s = 0.0;
    for (uint i = 0; i < gl_NumSubgroups; i++) s += red_[i];
    barrier();
    return s;
}

float workgroup_max(float v) {
    v = subgroupMax(v);
    if (subgroupElect()) red_[gl_SubgroupID] = v;
    barrier();
    float s = -1.0 / 0.0;
    for (uint i = 0; i < gl_NumSubgroups; i++) s = max(s, red_[i]);
    barrier();
    return s;
}
