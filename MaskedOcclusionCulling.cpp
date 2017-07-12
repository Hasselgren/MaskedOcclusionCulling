/*
 * Copyright 2017 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http ://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and limitations under the License.
 */
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include "MaskedOcclusionCulling.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Compiler specific functions & SIMD math
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "CompilerSpecificUtilities.inl"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Utility functions (not directly related to the algorithm/rasterizer)
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void MaskedOcclusionCulling::TransformVertices(const float *mtx, const float *inVtx, float *xfVtx, unsigned int nVtx, const VertexLayout &vtxLayout)
{
	// This function pretty slow, about 10-20% slower than if the vertices are stored in aligned SOA form.
	if (nVtx == 0)
		return;

	// Load matrix and swizzle out the z component. For post-multiplication (OGL), the matrix is assumed to be column 
	// major, with one column per SSE register. For pre-multiplication (DX), the matrix is assumed to be row major.
	vec4f mtxCol0 = vec4_loadu_ps(mtx);
	vec4f mtxCol1 = vec4_loadu_ps(mtx + 4);
	vec4f mtxCol2 = vec4_loadu_ps(mtx + 8);
	vec4f mtxCol3 = vec4_loadu_ps(mtx + 12);

	int stride = vtxLayout.mStride;
	const char *vPtr = (const char *)inVtx;
	float *outPtr = xfVtx;

	// Iterate through all vertices and transform
	for (unsigned int vtx = 0; vtx < nVtx; ++vtx)
	{
		vec4f xVal = vec4_load1_ps((float*)(vPtr));
		vec4f yVal = vec4_load1_ps((float*)(vPtr + vtxLayout.mOffsetY));
		vec4f zVal = vec4_load1_ps((float*)(vPtr + vtxLayout.mOffsetZ));

		vec4f xform = (mtxCol0 * xVal) + (mtxCol1 * yVal) + (mtxCol2 * zVal) + mtxCol3;
		vec4_storeu_ps(outPtr, xform);
		vPtr += stride;
		outPtr += 4;
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Typedefs
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef MaskedOcclusionCulling::pfnAlignedAlloc pfnAlignedAlloc;
typedef MaskedOcclusionCulling::pfnAlignedFree  pfnAlignedFree;
typedef MaskedOcclusionCulling::VertexLayout    VertexLayout;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Common SSE2/SSE4.1 defines
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define SIMD_LANES             4
#define TILE_HEIGHT_SHIFT      2

#define SIMD_LANE_IDX vec4_setr_epi32(0, 1, 2, 3)

#define SIMD_SUB_TILE_COL_OFFSET vec4_setr_epi32(0, SUB_TILE_WIDTH, SUB_TILE_WIDTH * 2, SUB_TILE_WIDTH * 3)
#define SIMD_SUB_TILE_ROW_OFFSET vec4_setzero_epi32()
#define SIMD_SUB_TILE_COL_OFFSET_F vec4_setr_ps(0, SUB_TILE_WIDTH, SUB_TILE_WIDTH * 2, SUB_TILE_WIDTH * 3)
#define SIMD_SUB_TILE_ROW_OFFSET_F vec4_setzero_ps()

#define SIMD_LANE_YCOORD_I vec4_setr_epi32(128, 384, 640, 896)
#define SIMD_LANE_YCOORD_F vec4_setr_ps(128.0f, 384.0f, 640.0f, 896.0f)

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Common SSE2/SSE4.1 functions
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef vec4f vecNf;
typedef vec4i vecNi;

#define mw_f32 m128_f32
#define mw_i32 m128i_i32

#define vecN_set1_ps vec4_set1_ps
#define vecN_setzero_ps vec4_setzero_ps
#define vecN_andnot_ps vec4_andnot_ps
#define vecN_min_ps vec4_min_ps
#define vecN_max_ps vec4_max_ps
#define vecN_movemask_ps vec4_movemask_ps
#define vecN_cmpge_ps(a,b) vec4_cmpge_ps(a, b)
#define vecN_cmpgt_ps(a,b) vec4_cmpgt_ps(a, b)
#define vecN_cmpeq_ps(a,b) vec4_cmpeq_ps(a, b)
#define vecN_fmadd_ps(a,b,c) (((a)*(b)) + (c))
#define vecN_fmsub_ps(a,b,c) (((a)*(b)) - (c))
#define vecN_shuffle_ps vec4_shuffle_ps
#define vecN_insertf32x4_ps(a,b,c) (b)

#define vecN_set1_epi32 vec4_set1_epi32
#define vecN_setzero_epi32 vec4_setzero_epi32
#define vecN_andnot_epi32 vec4_andnot_epi32
#define vecN_subs_epu16 vec4_subs_epu16
#define vecN_cmpeq_epi32 vec4_cmpeq_epi32
#define vecN_cmpgt_epi32 vec4_cmpgt_epi32
#define vecN_srai_epi32 vec4_srai_epi32
#define vecN_srli_epi32 vec4_srli_epi32
#define vecN_slli_epi32 vec4_slli_epi32
#define vecN_abs_epi32 vec4_abs_epi32

#define vecN_cvtps_epi32 vec4_cvtps_epi32
#define vecN_cvttps_epi32 vec4_cvttps_epi32
#define vecN_cvtepi32_ps vec4_cvtepi32_ps

#define vec4x_fmadd_ps vecN_fmadd_ps
#define vec4x_max_epi32 vecN_max_epi32
#define vec4x_min_epi32 vecN_min_epi32

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Specialized SSE input assembly function for general vertex gather 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

FORCE_INLINE void GatherVertices(vec4f *vtxX, vec4f *vtxY, vec4f *vtxW, const float *inVtx, const unsigned int *inTrisPtr, int numLanes, const VertexLayout &vtxLayout)
{
	for (int lane = 0; lane < numLanes; lane++)
	{
		for (int i = 0; i < 3; i++)
		{
			char *vPtrX = (char *)inVtx + inTrisPtr[lane * 3 + i] * vtxLayout.mStride;
			char *vPtrY = vPtrX + vtxLayout.mOffsetY;
			char *vPtrW = vPtrX + vtxLayout.mOffsetW;

			vtxX[i].m128_f32[lane] = *((float*)vPtrX);
			vtxY[i].m128_f32[lane] = *((float*)vPtrY);
			vtxW[i].m128_f32[lane] = *((float*)vPtrW);
		}
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// SSE4.1 version
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace MaskedOcclusionCullingSSE41
{
	static FORCE_INLINE vec4i vecN_mullo_epi32(const vec4i &a, const vec4i &b) { return vec4_mullo_epi32(a, b); }
	static FORCE_INLINE vec4i vecN_min_epi32(const vec4i &a, const vec4i &b) { return vec4_min_epi32(a, b); }
	static FORCE_INLINE vec4i vecN_max_epi32(const vec4i &a, const vec4i &b) { return vec4_max_epi32(a, b); }
	static FORCE_INLINE vec4f vecN_blendv_ps(const vec4f &a, const vec4f &b, const vec4f &c) { return vec4_blendv_ps(a, b, c); }
	static FORCE_INLINE int vecN_testz_epi32(const vec4i &a, const vec4i &b) { return vec4_testz_epi32(a, b); }
	static FORCE_INLINE vec4f vec4x_dp4_ps(const vec4f &a, const vec4f &b) { return vec4_dp_ps(a, b, 0xFF); }
	static FORCE_INLINE vec4f vecN_floor_ps(const vec4f &a) { return vec4_round_ps(a, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC); }
	static FORCE_INLINE vec4f vecN_ceil_ps(const vec4f &a) { return vec4_round_ps(a, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);	}
	static FORCE_INLINE vec4i vecN_transpose_epi8(const vec4i &a)
	{
		const vec4i shuff = vec4_setr_epi8(0x0, 0x4, 0x8, 0xC, 0x1, 0x5, 0x9, 0xD, 0x2, 0x6, 0xA, 0xE, 0x3, 0x7, 0xB, 0xF);
		return vec4_shuffle_epi8(a, shuff);
	}
	static FORCE_INLINE vec4i vecN_sllv_ones(const vec4i &ishift)
	{
		vec4i shift = vec4_min_epi32(ishift, vec4_set1_epi32(32));

		// Uses lookup tables and shuffle_epi8 to perform sllv_epi32(~0, shift)
		const vec4i byteShiftLUT = vec4_setr_epi8((char)0xFF, (char)0xFE, (char)0xFC, (char)0xF8, (char)0xF0, (char)0xE0, (char)0xC0, (char)0x80, 0, 0, 0, 0, 0, 0, 0, 0);
		const vec4i byteShiftOffset = vec4_setr_epi8(0, 8, 16, 24, 0, 8, 16, 24, 0, 8, 16, 24, 0, 8, 16, 24);
		const vec4i byteShiftShuffle = vec4_setr_epi8(0x0, 0x0, 0x0, 0x0, 0x4, 0x4, 0x4, 0x4, 0x8, 0x8, 0x8, 0x8, 0xC, 0xC, 0xC, 0xC);

		vec4i byteShift = vec4_shuffle_epi8(shift, byteShiftShuffle);
		byteShift = vec4_min_epi8(vec4_subs_epu8(byteShift, byteShiftOffset), vec4_set1_epi8(8));
		vec4i retMask = vec4_shuffle_epi8(byteShiftLUT, byteShift);

		return retMask;
	}

	static MaskedOcclusionCulling::Implementation gInstructionSet = MaskedOcclusionCulling::SSE41;

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Include common algorithm implementation (general, SIMD independent code)
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	#include "MaskedOcclusionCullingCommon.inl"

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Utility function to create a new object using the allocator callbacks
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	static bool DetectSSE41()
	{
		static bool initialized = false;
		static bool SSE41Supported = false;

		int cpui[4];
		if (!initialized)
		{
			initialized = true;

			int nIds;
			__cpuidex(cpui, 0, 0);
			nIds = cpui[0];

			if (nIds >= 1)
			{
				// Test SSE4.1 support
				__cpuidex(cpui, 1, 0);
				SSE41Supported = (cpui[2] & 0x080000) == 0x080000;
			}
		}
		return SSE41Supported;
	}

	MaskedOcclusionCulling *CreateMaskedOcclusionCulling(pfnAlignedAlloc memAlloc, pfnAlignedFree memFree)
	{
		if (!DetectSSE41())
			return nullptr;

		MaskedOcclusionCullingPrivate *object = (MaskedOcclusionCullingPrivate *)memAlloc(32, sizeof(MaskedOcclusionCullingPrivate));
		new (object) MaskedOcclusionCullingPrivate(memAlloc, memFree);
		return object;
	}
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// SSE2 version
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace MaskedOcclusionCullingSSE2
{
	static FORCE_INLINE vec4i vecN_mullo_epi32(const vec4i &a, const vec4i &b)
	{ 
		// Do products for even / odd lanes & merge the result
		vec4i even = vec4_mul_epu32(a, b) & vec4_setr_epi32(~0, 0, ~0, 0);
		vec4i odd = vec4_slli_epi64(vec4_mul_epu32(vec4_srli_epi64(a, 32), vec4_srli_epi64(b, 32)), 32);
		return even | odd;
	}
	static FORCE_INLINE vec4i vecN_min_epi32(const vec4i &a, const vec4i &b)
	{ 
		vec4i cond = vec4_cmpgt_epi32(a, b);
		return vec4_andnot_epi32(cond, a) | (cond & b);
	}
	static FORCE_INLINE vec4i vecN_max_epi32(const vec4i &a, const vec4i &b)
	{ 
		vec4i cond = vec4_cmpgt_epi32(b, a);
		return vec4_andnot_epi32(cond, a) | (cond & b);
	}
	static FORCE_INLINE int vecN_testz_epi32(const vec4i &a, const vec4i &b) 
	{ 
		return vec4_movemask_epi8(vec4_cmpeq_epi8((a & b), vec4_setzero_epi32())) == 0xFFFF;
	}
	static FORCE_INLINE vec4f vecN_blendv_ps(const vec4f &a, const vec4f &b, const vec4f &c)
	{	
		vec4f cond = simd_cast<vec4f>(vec4_srai_epi32(simd_cast<vec4i>(c), 31));
		return vec4_andnot_ps(cond, a) | (cond & b);
	}
	static FORCE_INLINE vec4f vec4x_dp4_ps(const vec4f &a, const vec4f &b)
	{ 
		// Product and two shuffle/adds pairs (similar to hadd_ps)
		vec4f prod = a * b;
		vec4f dp = prod + vec4_shuffle_ps(prod, prod, _MM_SHUFFLE(2, 3, 0, 1));
		dp = dp + vec4_shuffle_ps(dp, dp, _MM_SHUFFLE(0, 1, 2, 3));
		return dp;
	}
	static FORCE_INLINE vec4f vecN_floor_ps(const vec4f &a)
	{ 
		int originalMode = _MM_GET_ROUNDING_MODE();
		_MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
		vec4f rounded = vec4_cvtepi32_ps(vec4_cvtps_epi32(a));
		_MM_SET_ROUNDING_MODE(originalMode);
		return rounded;
	}
	static FORCE_INLINE vec4f vecN_ceil_ps(const vec4f &a) 
	{ 
		int originalMode = _MM_GET_ROUNDING_MODE();
		_MM_SET_ROUNDING_MODE(_MM_ROUND_UP);
		vec4f rounded = vec4_cvtepi32_ps(vec4_cvtps_epi32(a));
		_MM_SET_ROUNDING_MODE(originalMode);
		return rounded;
	}
	static FORCE_INLINE vec4i vecN_transpose_epi8(const vec4i &a)
	{
		// Perform transpose through two 16->8 bit pack and byte shifts
		vec4i res = a;
		const vec4i mask = vec4_setr_epi8(~0, 0, ~0, 0, ~0, 0, ~0, 0, ~0, 0, ~0, 0, ~0, 0, ~0, 0);
		res = vec4_packus_epi16(res & mask, vec4_srli_epi16(res, 8));
		res = vec4_packus_epi16(res & mask, vec4_srli_epi16(res, 8));
		return res;
	}
	static FORCE_INLINE vec4i vecN_sllv_ones(const vec4i &ishift)
	{
		vec4i shift = vecN_min_epi32(ishift, vec4_set1_epi32(32));
		
		// Uses scalar approach to perform sllv_epi32(~0, shift)
		static const unsigned int maskLUT[33] = {
			~0U << 0, ~0U << 1, ~0U << 2 ,  ~0U << 3, ~0U << 4, ~0U << 5, ~0U << 6 , ~0U << 7, ~0U << 8, ~0U << 9, ~0U << 10 , ~0U << 11, ~0U << 12, ~0U << 13, ~0U << 14 , ~0U << 15,
			~0U << 16, ~0U << 17, ~0U << 18 , ~0U << 19, ~0U << 20, ~0U << 21, ~0U << 22 , ~0U << 23, ~0U << 24, ~0U << 25, ~0U << 26 , ~0U << 27, ~0U << 28, ~0U << 29, ~0U << 30 , ~0U << 31,
			0U };

		vec4i retMask;
		retMask.m128i_u32[0] = maskLUT[shift.m128i_u32[0]];
		retMask.m128i_u32[1] = maskLUT[shift.m128i_u32[1]];
		retMask.m128i_u32[2] = maskLUT[shift.m128i_u32[2]];
		retMask.m128i_u32[3] = maskLUT[shift.m128i_u32[3]];
		return retMask;
	}

	static MaskedOcclusionCulling::Implementation gInstructionSet = MaskedOcclusionCulling::SSE2;

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Include common algorithm implementation (general, SIMD independent code)
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	#include "MaskedOcclusionCullingCommon.inl"

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Utility function to create a new object using the allocator callbacks
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	MaskedOcclusionCulling *CreateMaskedOcclusionCulling(pfnAlignedAlloc memAlloc, pfnAlignedFree memFree)
	{
		MaskedOcclusionCullingPrivate *object = (MaskedOcclusionCullingPrivate *)memAlloc(32, sizeof(MaskedOcclusionCullingPrivate));
		new (object) MaskedOcclusionCullingPrivate(memAlloc, memFree);
		return object;
	}
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Object construction and allocation
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
namespace MaskedOcclusionCullingAVX512
{
	extern MaskedOcclusionCulling *CreateMaskedOcclusionCulling(pfnAlignedAlloc memAlloc, pfnAlignedFree memFree);
}

namespace MaskedOcclusionCullingAVX2
{
	extern MaskedOcclusionCulling *CreateMaskedOcclusionCulling(pfnAlignedAlloc memAlloc, pfnAlignedFree memFree);
}

MaskedOcclusionCulling *MaskedOcclusionCulling::Create()
{
	return Create(aligned_alloc, aligned_free);
}

MaskedOcclusionCulling *MaskedOcclusionCulling::Create(pfnAlignedAlloc memAlloc, pfnAlignedFree memFree)
{
	MaskedOcclusionCulling *object = nullptr;

	// Return best supported version
	if (object == nullptr)
		object = MaskedOcclusionCullingAVX512::CreateMaskedOcclusionCulling(memAlloc, memFree);	// Use AVX512 version
	if (object == nullptr)
		object = MaskedOcclusionCullingAVX2::CreateMaskedOcclusionCulling(memAlloc, memFree);	// Use AVX2 version
	if (object == nullptr)
		object = MaskedOcclusionCullingSSE41::CreateMaskedOcclusionCulling(memAlloc, memFree);	// Use SSE4.1 version
	if (object == nullptr)
		object = MaskedOcclusionCullingSSE2::CreateMaskedOcclusionCulling(memAlloc, memFree);	// Use SSE2 (slow) version

	return object;
}

void MaskedOcclusionCulling::Destroy(MaskedOcclusionCulling *moc)
{
	pfnAlignedFree alignedFreeCallback = moc->mAlignedFreeCallback;
	moc->~MaskedOcclusionCulling();
	alignedFreeCallback(moc);
}
