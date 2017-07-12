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

#include <string.h>
#include <assert.h>
#include <float.h>
#include "MaskedOcclusionCulling.h"

// Make sure compiler features AVX-512 intrinsics
#if (defined(__INTEL_COMPILER) && __INTEL_COMPILER >= 1600) || (defined(__GNUC__) && __GNUC__ >= 5)

#ifndef __AVX2__
	#error For best performance, MaskedOcclusionCullingAVX512.cpp should be compiled with /arch:AVX2
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Compiler specific functions & SIMD math
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define __AVX512__
#include "CompilerSpecificUtilities.inl"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// AVX specific defines and constants
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define SIMD_LANES             16
#define TILE_HEIGHT_SHIFT      4

#define SIMD_LANE_IDX vec16_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)

#define SIMD_SUB_TILE_COL_OFFSET vec16_setr_epi32(0, SUB_TILE_WIDTH, SUB_TILE_WIDTH * 2, SUB_TILE_WIDTH * 3, 0, SUB_TILE_WIDTH, SUB_TILE_WIDTH * 2, SUB_TILE_WIDTH * 3, 0, SUB_TILE_WIDTH, SUB_TILE_WIDTH * 2, SUB_TILE_WIDTH * 3, 0, SUB_TILE_WIDTH, SUB_TILE_WIDTH * 2, SUB_TILE_WIDTH * 3)
#define SIMD_SUB_TILE_ROW_OFFSET vec16_setr_epi32(0, 0, 0, 0, SUB_TILE_HEIGHT, SUB_TILE_HEIGHT, SUB_TILE_HEIGHT, SUB_TILE_HEIGHT, SUB_TILE_HEIGHT * 2, SUB_TILE_HEIGHT * 2, SUB_TILE_HEIGHT * 2, SUB_TILE_HEIGHT * 2, SUB_TILE_HEIGHT * 3, SUB_TILE_HEIGHT * 3, SUB_TILE_HEIGHT * 3, SUB_TILE_HEIGHT * 3)
#define SIMD_SUB_TILE_COL_OFFSET_F vec16_setr_ps(0, SUB_TILE_WIDTH, SUB_TILE_WIDTH * 2, SUB_TILE_WIDTH * 3, 0, SUB_TILE_WIDTH, SUB_TILE_WIDTH * 2, SUB_TILE_WIDTH * 3, 0, SUB_TILE_WIDTH, SUB_TILE_WIDTH * 2, SUB_TILE_WIDTH * 3, 0, SUB_TILE_WIDTH, SUB_TILE_WIDTH * 2, SUB_TILE_WIDTH * 3)
#define SIMD_SUB_TILE_ROW_OFFSET_F vec16_setr_ps(0, 0, 0, 0, SUB_TILE_HEIGHT, SUB_TILE_HEIGHT, SUB_TILE_HEIGHT, SUB_TILE_HEIGHT, SUB_TILE_HEIGHT * 2, SUB_TILE_HEIGHT * 2, SUB_TILE_HEIGHT * 2, SUB_TILE_HEIGHT * 2, SUB_TILE_HEIGHT * 3, SUB_TILE_HEIGHT * 3, SUB_TILE_HEIGHT * 3, SUB_TILE_HEIGHT * 3)

//#define SIMD_SHUFFLE_SCANLINE_TO_SUBTILES vec16_set_epi8(0xF, 0xB, 0x7, 0x3, 0xE, 0xA, 0x6, 0x2, 0xD, 0x9, 0x5, 0x1, 0xC, 0x8, 0x4, 0x0, 0xF, 0xB, 0x7, 0x3, 0xE, 0xA, 0x6, 0x2, 0xD, 0x9, 0x5, 0x1, 0xC, 0x8, 0x4, 0x0, 0xF, 0xB, 0x7, 0x3, 0xE, 0xA, 0x6, 0x2, 0xD, 0x9, 0x5, 0x1, 0xC, 0x8, 0x4, 0x0, 0xF, 0xB, 0x7, 0x3, 0xE, 0xA, 0x6, 0x2, 0xD, 0x9, 0x5, 0x1, 0xC, 0x8, 0x4, 0x0)
#define SIMD_SHUFFLE_SCANLINE_TO_SUBTILES vec16_set_epi32(0x0F0B0703, 0x0E0A0602, 0x0D090501, 0x0C080400, 0x0F0B0703, 0x0E0A0602, 0x0D090501, 0x0C080400, 0x0F0B0703, 0x0E0A0602, 0x0D090501, 0x0C080400, 0x0F0B0703, 0x0E0A0602, 0x0D090501, 0x0C080400)

#define SIMD_LANE_YCOORD_I vec16_setr_epi32(128, 384, 640, 896, 1152, 1408, 1664, 1920, 2176, 2432, 2688, 2944, 3200, 3456, 3712, 3968)
#define SIMD_LANE_YCOORD_F vec16_setr_ps(128.0f, 384.0f, 640.0f, 896.0f, 1152.0f, 1408.0f, 1664.0f, 1920.0f, 2176.0f, 2432.0f, 2688.0f, 2944.0f, 3200.0f, 3456.0f, 3712.0f, 3968.0f)

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// AVX specific typedefs and functions
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef vec16f vecNf;
typedef vec16i vecNi;

#define mw_f32 m512_f32
#define mw_i32 m512i_i32

#define vecN_set1_ps vec16_set1_ps
#define vecN_setzero_ps vec16_setzero_ps
#define vecN_andnot_ps vec16_andnot_ps
#define vecN_fmadd_ps vec16_fmadd_ps
#define vecN_fmsub_ps vec16_fmsub_ps
#define vecN_min_ps vec16_min_ps
#define vecN_max_ps vec16_max_ps
#define vecN_shuffle_ps vec16_shuffle_ps

#define vecN_set1_epi32 vec16_set1_epi32
#define vecN_setzero_epi32 vec16_setzero_epi32
#define vecN_insertf32x4_ps vec16_insertf32x4_ps
#define vecN_andnot_epi32 vec16_andnot_epi32
#define vecN_min_epi32 vec16_min_epi32
#define vecN_max_epi32 vec16_max_epi32
#define vecN_subs_epu16 vec16_subs_epu16
#define vecN_mullo_epi32 vec16_mullo_epi32
#define vecN_srai_epi32 vec16_srai_epi32
#define vecN_srli_epi32 vec16_srli_epi32
#define vecN_slli_epi32 vec16_slli_epi32
#define vecN_sllv_ones(x) vec16_sllv_epi32(SIMD_BITS_ONE, x)
#define vecN_transpose_epi8(x) vec16_shuffle_epi8(x, SIMD_SHUFFLE_SCANLINE_TO_SUBTILES)
#define vecN_abs_epi32 vec16_abs_epi32

#define vecN_cvtps_epi32 vec16_cvtps_epi32
#define vecN_cvttps_epi32 vec16_cvttps_epi32
#define vecN_cvtepi32_ps vec16_cvtepi32_ps

#define vec4x_dp4_ps(a, b) vec4_dp_ps(a, b, 0xFF)
#define vec4x_fmadd_ps vec4_fmadd_ps
#define vec4x_max_epi32 vec4_max_epi32
#define vec4x_min_epi32 vec4_min_epi32

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Specialized AVX input assembly function for general vertex gather 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef MaskedOcclusionCulling::VertexLayout VertexLayout;

static FORCE_INLINE void GatherVertices(vec16f *vtxX, vec16f *vtxY, vec16f *vtxW, const float *inVtx, const unsigned int *inTrisPtr, int numLanes, const VertexLayout &vtxLayout)
{
	assert(numLanes >= 1);

	const vec16i SIMD_TRI_IDX_OFFSET = vec16_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45);
	static const vec16i SIMD_LANE_MASK[17] = {
		vec16_setr_epi32( 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0),
		vec16_setr_epi32(~0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0),
		vec16_setr_epi32(~0, ~0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0),
		vec16_setr_epi32(~0, ~0, ~0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0),
		vec16_setr_epi32(~0, ~0, ~0, ~0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0),
		vec16_setr_epi32(~0, ~0, ~0, ~0, ~0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0),
		vec16_setr_epi32(~0, ~0, ~0, ~0, ~0, ~0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0),
		vec16_setr_epi32(~0, ~0, ~0, ~0, ~0, ~0, ~0,  0,  0,  0,  0,  0,  0,  0,  0,  0),
		vec16_setr_epi32(~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0,  0,  0,  0,  0,  0,  0,  0,  0),
		vec16_setr_epi32(~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0,  0,  0,  0,  0,  0,  0,  0),
		vec16_setr_epi32(~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0,  0,  0,  0,  0,  0,  0),
		vec16_setr_epi32(~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0,  0,  0,  0,  0,  0),
		vec16_setr_epi32(~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0,  0,  0,  0,  0),
		vec16_setr_epi32(~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0,  0,  0,  0),
		vec16_setr_epi32(~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0,  0,  0),
		vec16_setr_epi32(~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0,  0),
		vec16_setr_epi32(~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0)
	};

	// Compute per-lane index list offset that guards against out of bounds memory accesses
	vec16i safeTriIdxOffset = SIMD_TRI_IDX_OFFSET & SIMD_LANE_MASK[numLanes];

	// Fetch triangle indices. 
	vec16i vtxIdx[3];
	vtxIdx[0] = vec16_mullo_epi32(vec16_i32gather_epi32(safeTriIdxOffset, (const int*)inTrisPtr + 0, 4), vec16_set1_epi32(vtxLayout.mStride));
	vtxIdx[1] = vec16_mullo_epi32(vec16_i32gather_epi32(safeTriIdxOffset, (const int*)inTrisPtr + 1, 4), vec16_set1_epi32(vtxLayout.mStride));
	vtxIdx[2] = vec16_mullo_epi32(vec16_i32gather_epi32(safeTriIdxOffset, (const int*)inTrisPtr + 2, 4), vec16_set1_epi32(vtxLayout.mStride));

	char *vPtr = (char *)inVtx;

	// Fetch triangle vertices
	for (int i = 0; i < 3; i++)
	{
		vtxX[i] = vec16_i32gather_ps(vtxIdx[i], (float *)vPtr, 1);
		vtxY[i] = vec16_i32gather_ps(vtxIdx[i], (float *)(vPtr + vtxLayout.mOffsetY), 1);
		vtxW[i] = vec16_i32gather_ps(vtxIdx[i], (float *)(vPtr + vtxLayout.mOffsetW), 1);
	}
}

namespace MaskedOcclusionCullingAVX512
{
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Poorly implemented functions. TODO: fix common (maskedOcclusionCullingCommon.inl) code to improve perf
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	static FORCE_INLINE vec16f vecN_floor_ps(vec16f x)
	{
		return vec16_roundscale_ps(x, 1); // 1 = floor
	}

	static FORCE_INLINE vec16f vecN_ceil_ps(vec16f x)
	{
		return vec16_roundscale_ps(x, 2); // 2 = ceil
	}

	static FORCE_INLINE vec16i vecN_cmpeq_epi32(vec16i a, vec16i b)
	{
		__mmask16 mask = vec16_cmpeq_epi32_mask(a, b);
		return vec16_mask_mov_epi32(vec16_setzero_epi32(), mask, vec16_set1_epi32(~0));
	}

	static FORCE_INLINE vec16i vecN_cmpgt_epi32(vec16i a, vec16i b)
	{
		__mmask16 mask = vec16_cmpgt_epi32_mask(a, b);
		return vec16_mask_mov_epi32(vec16_setzero_epi32(), mask, vec16_set1_epi32(~0));
	}

	static FORCE_INLINE bool vecN_testz_epi32(vec16i a, vec16i b)
	{
		__mmask16 mask = vec16_cmpeq_epi32_mask(a & b, vec16_setzero_epi32());
		return mask == 0xFFFF;
	}

	static FORCE_INLINE vec16f vecN_cmpge_ps(vec16f a, vec16f b)
	{
		__mmask16 mask = vec16_cmp_ps_mask(a, b, _CMP_GE_OQ);
		return simd_cast<vec16f>(vec16_mask_mov_epi32(vec16_setzero_epi32(), mask, vec16_set1_epi32(~0)));
	}

	static FORCE_INLINE vec16f vecN_cmpgt_ps(vec16f a, vec16f b)
	{
		__mmask16 mask = vec16_cmp_ps_mask(a, b, _CMP_GT_OQ);
		return simd_cast<vec16f>(vec16_mask_mov_epi32(vec16_setzero_epi32(), mask, vec16_set1_epi32(~0)));
	}

	static FORCE_INLINE vec16f vecN_cmpeq_ps(vec16f a, vec16f b)
	{
		__mmask16 mask = vec16_cmp_ps_mask(a, b, _CMP_EQ_OQ);
		return simd_cast<vec16f>(vec16_mask_mov_epi32(vec16_setzero_epi32(), mask, vec16_set1_epi32(~0)));
	}

	static FORCE_INLINE __mmask16 vecN_movemask_ps(vec16f a)
	{
		__mmask16 mask = vec16_cmp_epi32_mask(simd_cast<vec16i>(a) & vec16_set1_epi32(0x80000000), vec16_setzero_epi32(), 4);	// a & 0x8000000 != 0
		return mask;
	}

	static FORCE_INLINE vec16f vecN_blendv_ps(const vec16f &a, const vec16f &b, const vec16f &c)
	{
		__mmask16 mask = vecN_movemask_ps(c);
		return vec16_mask_mov_ps(a, mask, b);
	} 

	static MaskedOcclusionCulling::Implementation gInstructionSet = MaskedOcclusionCulling::AVX512;

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Include common algorithm implementation (general, SIMD independent code)
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	#include "MaskedOcclusionCullingCommon.inl"

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Utility function to create a new object using the allocator callbacks
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	typedef MaskedOcclusionCulling::pfnAlignedAlloc            pfnAlignedAlloc;
	typedef MaskedOcclusionCulling::pfnAlignedFree             pfnAlignedFree;

	static bool DetectAVX512()
	{
		static bool initialized = false;
		static bool AVX512Support = false;

		int cpui[4];
		if (!initialized)
		{
			initialized = true;
			AVX512Support = false;

			int nIds, nExIds;
			__cpuidex(cpui, 0, 0);
			nIds = cpui[0];
			__cpuidex(cpui, 0x80000000, 0);
			nExIds = cpui[0];

			if (nIds >= 7 && nExIds >= (int)0x80000001)
			{
				AVX512Support = true;

				// Check support for bit counter instructions (lzcnt)
				__cpuidex(cpui, 0x80000001, 0);
				if ((cpui[2] & 0x20) != 0x20)
					AVX512Support = false;

				// Check masks for misc instructions (FMA)
				static const unsigned int FMA_MOVBE_OSXSAVE_MASK = (1 << 12) | (1 << 22) | (1 << 27);
				__cpuidex(cpui, 1, 0);
				if ((cpui[2] & FMA_MOVBE_OSXSAVE_MASK) != FMA_MOVBE_OSXSAVE_MASK)
					AVX512Support = false;
				
				// Check XCR0 register to ensure that all registers are enabled (by OS)
				static const unsigned int XCR0_MASK = (1 << 7) | (1 << 6) | (1 << 5) | (1 << 2) | (1 << 1); // OPMASK | ZMM0-15 | ZMM16-31 | XMM | YMM
				if (AVX512Support && (_xgetbv(0) & XCR0_MASK) != XCR0_MASK)
					AVX512Support = false;

				// Detect AVX2 & AVX512 instruction sets
				static const unsigned int AVX2_FLAGS = (1 << 3) | (1 << 5) | (1 << 8); // BMI1 (bit manipulation) | BMI2 (bit manipulation)| AVX2
				static const unsigned int AVX512_FLAGS = AVX2_FLAGS | (1 << 16) | (1 << 17) | (1 << 28) | (1 << 30) | (1 << 31); // AVX512F | AVX512DQ | AVX512CD | AVX512BW | AVX512VL
				__cpuidex(cpui, 7, 0);
				if ((cpui[1] & AVX512_FLAGS) != AVX512_FLAGS)
					AVX512Support = false;
			}
		}
		return AVX512Support;
	}

	MaskedOcclusionCulling *CreateMaskedOcclusionCulling(pfnAlignedAlloc memAlloc, pfnAlignedFree memFree)
	{
		if (!DetectAVX512())
			return nullptr;

		MaskedOcclusionCullingPrivate *object = (MaskedOcclusionCullingPrivate *)memAlloc(64, sizeof(MaskedOcclusionCullingPrivate));
		new (object) MaskedOcclusionCullingPrivate(memAlloc, memFree);
		return object;
	}
};

#else

namespace MaskedOcclusionCullingAVX512
{
	typedef MaskedOcclusionCulling::pfnAlignedAlloc            pfnAlignedAlloc;
	typedef MaskedOcclusionCulling::pfnAlignedFree             pfnAlignedFree;

	MaskedOcclusionCulling *CreateMaskedOcclusionCulling(pfnAlignedAlloc memAlloc, pfnAlignedFree memFree)
	{
		return nullptr;
	}
};

#endif
