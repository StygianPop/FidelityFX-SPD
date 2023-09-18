#pragma once

////////////////////////////////////////////////////////////////////////////////
// USAGE:
//
// Include this header in your source file. Provide implementations for the
// declared functions below (stopping at "Begin SPD implementation"). Several of
// these functions accept a template parameter, which you'd specialize for the
// types needed. For example, to specialize SpdLoadSourceImage, you might
// implement it as the following:
//
// template <>
// float4 SpdLoadSourceImage<float4>(uint2 uv, uint slice)
// { /* Sample your image or load a texel depending */ }
//
// You'll need 16x16 float4 LDS storage for each component. For example:
//     float4 spd_lds[16][16];
//
// This LDS storage interacts with the SpdLoadIntermediate and
// SpdStoreIntermediate functions.
//
// You'll also need an additional uint LDS counter to broadcast group status
// after an interlocked add of the global counter.
//
// Compilation of a shader including this header requires SM 6.0 or greater,
// and HLSL edition 2021 or later.
//
// To integrate this shader on the GPU, only this file is needed. On the host,
// the instructions in ffx_spd.h under "INTEGRATION SUMMARY FOR CPU" remain
// unchanged except that this file no longer provides the SpdSetup function,
// previously guarded by the now non-existent A_CPU macro definition.
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// MODIFICATIONS:
//
// This header differs from the provided ffx_spd.h and ffx_a.h files in several
// ways:
//
// 1. All GLSL compat code was removed.
// 2. Some helper functions were replaced by an intrinsic for readability.
// 3. The reduced type was replaced by a template parameter.
// 4. Preprocessor usage was removed with the exception of SPD_LINEAR_SAMPLER.
// 5. All CPU code was removed (along with the A_CPU definition).
// 6. Some functions like ABfiM and ABfi/ABfe were renamed for readability.
// 7. Wave intrinsics (i.e. SM 6.0) are presumed available.
// 8. SpdLoad was renamed to SpdLoad_Mip6 to indicate that MIP index 6 should be read.
// 9. SpdResetAtomicCounter is only called by one thread in the final thread group.
// 10. SpdStore's mip parameter is now the actual mip index (instead of off-by-one)
//     for the mip being written.
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// User-provided functions

// If SPD_LINEAR_SAMPLER is defined, this function should perform a single
// sample on the source image. Otherwise, this function is expected to load a
// single texel.
template <typename T>
T SpdLoadSourceImage(int2 texel, uint slice);

// Load a texel from a given slice in the mip level 6 (0-indexed).
template <typename T>
T SpdLoad_Mip6(int2 texel, uint slice);

// Store a texel in a given mip slice. Note that for mip == 6, a globallycoherent
// descriptor reference is needed. Note that the mip parameter is different from
// the original SPD code, which had off-by-one indexing. The mip value passed here
// is the mip we are writing to, not the mip we are currently downsampling.
template <typename T>
void SpdStore(int2 texel, T value, uint mip, uint slice);

// Increment the global atomic counter associated with a given slice.
// The original value should be stored to LDS to broadcast across the group.
void SpdIncreaseAtomicCounter(uint slice);

// Reset the global atomic counter for a given slice to zero.
void SpdResetAtomicCounter(uint slice);

// Return the original value of the atomic counter stored in LDS.
// NOTE: This abstraction is a bit sketchy, but this value returns whatever the
// original counter value is after invoking SpdIncreaseAtomicCounter.
uint SpdGetAtomicCounter();

// Load a 4-vector from LDS. Specialize this for type float4 or min16float4
// depending on reduction width.
template <typename T>
T SpdLoadIntermediate(uint x, uint y);

// Store a 4-vector to LDS. Specialize this for type float4 or min16float4
// depending on reduction width.
template <typename T>
void SpdStoreIntermediate(uint x, uint y, T value);

// Reduce four values into one. Commonly implemented as:
//     0.25 * (v0 + v1 + v2 + v3).
template <typename T>
T SpdReduce4(T v0, T v1, T v2, T v3);
////////////////////////////////////////////////////////////////////////////////

// Begin SPD implementation

// Insert count bits from ins into src.
uint bitfield_insert(uint src, uint ins, uint count)
{
    uint mask = (1u << count) - 1;
    return (ins & mask) | (src & (~mask));
}

// Extract count bits at an offset from src.
uint bitfield_extract(uint src, uint offset, uint count)
{
    uint mask = (1u << count) - 1;
    return (src >> offset) & mask;
}

// More complex remap 64x1 to 8x8 which is necessary for 2D wave reductions.
//  543210
//  ======
//  .xx..x
//  y..yy.
// Details,
//  LANE TO 8x8 MAPPING
//  ===================
//  00 01 08 09 10 11 18 19 
//  02 03 0a 0b 12 13 1a 1b
//  04 05 0c 0d 14 15 1c 1d
//  06 07 0e 0f 16 17 1e 1f 
//  20 21 28 29 30 31 38 39 
//  22 23 2a 2b 32 33 3a 3b
//  24 25 2c 2d 34 35 3c 3d
//  26 27 2e 2f 36 37 3e 3f 
uint2 ARmpRed8x8(uint a)
{
    return uint2(
        bitfield_insert(bitfield_extract(a, 2u, 3u), a, 1u),
        bitfield_insert(bitfield_extract(a, 3u, 3u), bitfield_extract(a, 1u, 2u), 2u));
}

// Only last active workgroup should proceed
bool SpdExitWorkgroup(uint numWorkGroups, uint localInvocationIndex, uint slice) 
{
    // global atomic counter
    if (localInvocationIndex == 0)
    {
        SpdIncreaseAtomicCounter(slice);
    }
    GroupMemoryBarrierWithGroupSync();
    return (SpdGetAtomicCounter() != (numWorkGroups - 1));
}

template <typename T>
T SpdReduceQuad(T v)
{
    uint quad = WaveGetLaneIndex() & (~0x3);
    T v0 = v;
    T v1 = WaveReadLaneAt(v, quad | 1);
    T v2 = WaveReadLaneAt(v, quad | 2);
    T v3 = WaveReadLaneAt(v, quad | 3);
    return SpdReduce4<T>(v0, v1, v2, v3);
}

template <typename T>
T SpdReduceLoad4(uint2 i0, uint2 i1, uint2 i2, uint2 i3, uint slice)
{
    T v0 = SpdLoad_Mip6<T>(int2(i0), slice);
    T v1 = SpdLoad_Mip6<T>(int2(i1), slice);
    T v2 = SpdLoad_Mip6<T>(int2(i2), slice);
    T v3 = SpdLoad_Mip6<T>(int2(i3), slice);
    return SpdReduce4<T>(v0, v1, v2, v3);
}

template <typename T>
T SpdReduceLoad4(uint2 base, uint slice)
{
    return SpdReduceLoad4<T>(
        uint2(base + uint2(0, 0)),
        uint2(base + uint2(0, 1)), 
        uint2(base + uint2(1, 0)), 
        uint2(base + uint2(1, 1)),
        slice);
}

template <typename T>
T SpdReduceLoadSourceImage4(uint2 i0, uint2 i1, uint2 i2, uint2 i3, uint slice)
{
    T v0 = SpdLoadSourceImage<T>(int2(i0), slice);
    T v1 = SpdLoadSourceImage<T>(int2(i1), slice);
    T v2 = SpdLoadSourceImage<T>(int2(i2), slice);
    T v3 = SpdLoadSourceImage<T>(int2(i3), slice);
    return SpdReduce4<T>(v0, v1, v2, v3);
}

template <typename T>
T SpdReduceLoadSourceImage(uint2 base, uint slice)
{
#ifdef SPD_LINEAR_SAMPLER
    return SpdLoadSourceImage<T>(int2(base), slice);
#else
    return SpdReduceLoadSourceImage4<T>(
        uint2(base + uint2(0, 0)),
        uint2(base + uint2(0, 1)), 
        uint2(base + uint2(1, 0)), 
        uint2(base + uint2(1, 1)),
        slice);
#endif
}

template <typename T>
void SpdDownsampleMips_0_1(
    uint x,
    uint y,
    uint2 workGroupID,
    uint localInvocationIndex,
    uint mip,
    uint slice)
{
    T v[4];

    int2 tex = int2(workGroupID.xy * 64) + int2(x * 2, y * 2);
    int2 pix = int2(workGroupID.xy * 32) + int2(x, y);
    v[0] = SpdReduceLoadSourceImage<T>(tex, slice);
    SpdStore<T>(pix, v[0], 1, slice);

    tex = int2(workGroupID.xy * 64) + int2(x * 2 + 32, y * 2);
    pix = int2(workGroupID.xy * 32) + int2(x + 16, y);
    v[1] = SpdReduceLoadSourceImage<T>(tex, slice);
    SpdStore<T>(pix, v[1], 1, slice);
    
    tex = int2(workGroupID.xy * 64) + int2(x * 2, y * 2 + 32);
    pix = int2(workGroupID.xy * 32) + int2(x, y + 16);
    v[2] = SpdReduceLoadSourceImage<T>(tex, slice);
    SpdStore<T>(pix, v[2], 1, slice);
    
    tex = int2(workGroupID.xy * 64) + int2(x * 2 + 32, y * 2 + 32);
    pix = int2(workGroupID.xy * 32) + int2(x + 16, y + 16);
    v[3] = SpdReduceLoadSourceImage<T>(tex, slice);
    SpdStore<T>(pix, v[3], 1, slice);

    if (mip <= 1)
        return;

    v[0] = SpdReduceQuad<T>(v[0]);
    v[1] = SpdReduceQuad<T>(v[1]);
    v[2] = SpdReduceQuad<T>(v[2]);
    v[3] = SpdReduceQuad<T>(v[3]);

    if ((localInvocationIndex % 4) == 0)
    {
        SpdStore<T>(int2(workGroupID.xy * 16) + 
            int2(x/2, y/2), v[0], 2, slice);
        SpdStoreIntermediate<T>(
            x/2, y/2, v[0]);

        SpdStore<T>(int2(workGroupID.xy * 16) + 
            int2(x/2 + 8, y/2), v[1], 2, slice);
        SpdStoreIntermediate<T>(
            x/2 + 8, y/2, v[1]);

        SpdStore<T>(int2(workGroupID.xy * 16) + 
            int2(x/2, y/2 + 8), v[2], 2, slice);
        SpdStoreIntermediate<T>(
            x/2, y/2 + 8, v[2]);

        SpdStore<T>(int2(workGroupID.xy * 16) + 
            int2(x/2 + 8, y/2 + 8), v[3], 2, slice);
        SpdStoreIntermediate<T>(
            x/2 + 8, y/2 + 8, v[3]);
    }
}

template <typename T>
void SpdDownsampleMip_2(uint x, uint y, uint2 workGroupID, uint localInvocationIndex, uint mip, uint slice)
{
    T v = SpdLoadIntermediate<T>(x, y);
    v = SpdReduceQuad<T>(v);
    // quad index 0 stores result
    if (localInvocationIndex % 4 == 0)
    {
        SpdStore<T>(int2(workGroupID.xy * 8) + int2(x/2, y/2), v, mip, slice);
        SpdStoreIntermediate<T>(x + (y/2) % 2, y, v);
    }
}

template <typename T>
void SpdDownsampleMip_3(uint x, uint y, uint2 workGroupID, uint localInvocationIndex, uint mip, uint slice)
{
    if (localInvocationIndex < 64)
    {
        T v = SpdLoadIntermediate<T>(x * 2 + y % 2,y * 2);
        v = SpdReduceQuad<T>(v);
        // quad index 0 stores result
        if (localInvocationIndex % 4 == 0)
        {   
            SpdStore<T>(int2(workGroupID.xy * 4) + int2(x/2, y/2), v, mip, slice);
            SpdStoreIntermediate<T>(x * 2 + y/2, y * 2, v);
        }
    }
}

template <typename T>
void SpdDownsampleMip_4(uint x, uint y, uint2 workGroupID, uint localInvocationIndex, uint mip, uint slice)
{
    if (localInvocationIndex < 16)
    {
        T v = SpdLoadIntermediate<T>(x * 4 + y,y * 4);
        v = SpdReduceQuad<T>(v);
        // quad index 0 stores result
        if (localInvocationIndex % 4 == 0)
        {   
            SpdStore<T>(int2(workGroupID.xy * 2) + int2(x/2, y/2), v, mip, slice);
            SpdStoreIntermediate<T>(x / 2 + y, 0, v);
        }
    }
}

template <typename T>
void SpdDownsampleMip_5(uint2 workGroupID, uint localInvocationIndex, uint mip, uint slice)
{
    if (localInvocationIndex < 4)
    {
        T v = SpdLoadIntermediate<T>(localInvocationIndex,0);
        v = SpdReduceQuad<T>(v);
        // quad index 0 stores result
        if (localInvocationIndex % 4 == 0)
        {   
            SpdStore<T>(int2(workGroupID.xy), v, mip, slice);
        }
    }
}

template <typename T>
void SpdDownsampleMips_6_7(uint x, uint y, uint mips, uint slice)
{
    int2 tex = int2(x * 4 + 0, y * 4 + 0);
    int2 pix = int2(x * 2 + 0, y * 2 + 0);
    T v0 = SpdReduceLoad4<T>(tex, slice);
    SpdStore<T>(pix, v0, 7, slice);

    tex = int2(x * 4 + 2, y * 4 + 0);
    pix = int2(x * 2 + 1, y * 2 + 0);
    T v1 = SpdReduceLoad4<T>(tex, slice);
    SpdStore<T>(pix, v1, 7, slice);

    tex = int2(x * 4 + 0, y * 4 + 2);
    pix = int2(x * 2 + 0, y * 2 + 1);
    T v2 = SpdReduceLoad4<T>(tex, slice);
    SpdStore<T>(pix, v2, 7, slice);

    tex = int2(x * 4 + 2, y * 4 + 2);
    pix = int2(x * 2 + 1, y * 2 + 1);
    T v3 = SpdReduceLoad4<T>(tex, slice);
    SpdStore<T>(pix, v3, 7, slice);

    if (mips <= 7) return;
    // no barrier needed, working on values only from the same thread

    T v = SpdReduce4<T>(v0, v1, v2, v3);
    SpdStore<T>(int2(x, y), v, 8, slice);
    SpdStoreIntermediate<T>(x, y, v);
}

template <typename T>
void SpdDownsampleNextFour(
    uint x,
    uint y,
    uint2 workGroupID,
    uint localInvocationIndex,
    uint baseMip,
    uint mips,
    uint slice)
{
    if (mips <= baseMip) return;
    GroupMemoryBarrierWithGroupSync();
    SpdDownsampleMip_2<T>(x, y, workGroupID, localInvocationIndex, baseMip + 1, slice);

    if (mips <= baseMip + 1) return;
    GroupMemoryBarrierWithGroupSync();
    SpdDownsampleMip_3<T>(x, y, workGroupID, localInvocationIndex, baseMip + 2, slice);

    if (mips <= baseMip + 2) return;
    GroupMemoryBarrierWithGroupSync();
    SpdDownsampleMip_4<T>(x, y, workGroupID, localInvocationIndex, baseMip + 3, slice);

    if (mips <= baseMip + 3) return;
    GroupMemoryBarrierWithGroupSync();
    SpdDownsampleMip_5<T>(workGroupID, localInvocationIndex, baseMip + 4, slice);
}

template <typename T>
void SpdDownsample(
    uint2 workGroupID,
    uint localInvocationIndex,
    uint mips,
    uint numWorkGroups,
    uint slice)
{
    uint2 sub_xy = ARmpRed8x8(localInvocationIndex % 64);
    uint x = sub_xy.x + 8 * ((localInvocationIndex >> 6) % 2);
    uint y = sub_xy.y + 8 * ((localInvocationIndex >> 7));
    SpdDownsampleMips_0_1<T>(x, y, workGroupID, localInvocationIndex, mips, slice);

    SpdDownsampleNextFour<T>(x, y, workGroupID, localInvocationIndex, 2, mips, slice);

    if (mips <= 6) return;

    if (SpdExitWorkgroup(numWorkGroups, localInvocationIndex, slice)) return;

    if (localInvocationIndex == 0)
    {
        SpdResetAtomicCounter(slice);
    }

    // After mip 6 there is only a single workgroup left that downsamples the remaining up to 64x64 texels.
    SpdDownsampleMips_6_7<T>(x, y, mips, slice);

    SpdDownsampleNextFour<T>(x, y, uint2(0, 0), localInvocationIndex, 8, mips, slice);
}

template <typename T>
void SpdDownsample(
    uint2 workGroupID,
    uint localInvocationIndex,
    uint mips,
    uint numWorkGroups,
    uint slice,
    uint2 workGroupOffset)
{
    SpdDownsample<T>(workGroupID + workGroupOffset, localInvocationIndex, mips, numWorkGroups, slice);
}
