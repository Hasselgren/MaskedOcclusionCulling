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
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include "MaskedOcclusionCulling.h"

#ifndef __AVX2__
	#error For best performance, MaskedOcclusionCullingAVX512.cpp should be compiled with /arch:AVX2
#endif

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER) && !defined(__clang__) && _MSC_VER < 1900
	// If you remove/comment this error line, the code will compile & use the SSE41 version instead. 
	#error Older versions than visual studio 2015 not supported due to compiler bug(s)
#endif

#if defined(__INTEL_COMPILER) || defined(__clang__) || defined(__GNUC__) || _MSC_VER >= 1900 // Make sure compiler features AVX2 & bug workaround for visual studio

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Compiler specific functions & SIMD math
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "CompilerSpecificUtilities.inl"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// AVX specific defines and constants
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define SIMD_LANES             8
#define TILE_HEIGHT_SHIFT      3

#define SIMD_LANE_IDX vec8_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)

#define SIMD_SUB_TILE_COL_OFFSET vec8_setr_epi32(0, SUB_TILE_WIDTH, SUB_TILE_WIDTH * 2, SUB_TILE_WIDTH * 3, 0, SUB_TILE_WIDTH, SUB_TILE_WIDTH * 2, SUB_TILE_WIDTH * 3)
#define SIMD_SUB_TILE_ROW_OFFSET vec8_setr_epi32(0, 0, 0, 0, SUB_TILE_HEIGHT, SUB_TILE_HEIGHT, SUB_TILE_HEIGHT, SUB_TILE_HEIGHT)
#define SIMD_SUB_TILE_COL_OFFSET_F vec8_setr_ps(0, SUB_TILE_WIDTH, SUB_TILE_WIDTH * 2, SUB_TILE_WIDTH * 3, 0, SUB_TILE_WIDTH, SUB_TILE_WIDTH * 2, SUB_TILE_WIDTH * 3)
#define SIMD_SUB_TILE_ROW_OFFSET_F vec8_setr_ps(0, 0, 0, 0, SUB_TILE_HEIGHT, SUB_TILE_HEIGHT, SUB_TILE_HEIGHT, SUB_TILE_HEIGHT)

#define SIMD_SHUFFLE_SCANLINE_TO_SUBTILES vec8_setr_epi8( 0x0, 0x4, 0x8, 0xC, 0x1, 0x5, 0x9, 0xD, 0x2, 0x6, 0xA, 0xE, 0x3, 0x7, 0xB, 0xF, 0x0, 0x4, 0x8, 0xC, 0x1, 0x5, 0x9, 0xD, 0x2, 0x6, 0xA, 0xE, 0x3, 0x7, 0xB, 0xF)

#define SIMD_LANE_YCOORD_I vec8_setr_epi32(128, 384, 640, 896, 1152, 1408, 1664, 1920)
#define SIMD_LANE_YCOORD_F vec8_setr_ps(128.0f, 384.0f, 640.0f, 896.0f, 1152.0f, 1408.0f, 1664.0f, 1920.0f)

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// AVX specific typedefs and functions
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef vec8f vecNf;
typedef vec8i vecNi;

#define mw_f32 m256_f32
#define mw_i32 m256i_i32

#define vecN_set1_ps vec8_set1_ps
#define vecN_setzero_ps vec8_setzero_ps
#define vecN_andnot_ps vec8_andnot_ps
#define vecN_fmadd_ps vec8_fmadd_ps
#define vecN_fmsub_ps vec8_fmsub_ps
#define vecN_min_ps vec8_min_ps
#define vecN_max_ps vec8_max_ps
#define vecN_movemask_ps vec8_movemask_ps
#define vecN_blendv_ps vec8_blendv_ps
#define vecN_cmpge_ps(a,b) vec8_cmp_ps(a, b, _CMP_GE_OQ)
#define vecN_cmpgt_ps(a,b) vec8_cmp_ps(a, b, _CMP_GT_OQ)
#define vecN_cmpeq_ps(a,b) vec8_cmp_ps(a, b, _CMP_EQ_OQ)
#define vecN_floor_ps(x) vec8_round_ps(x, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)
#define vecN_ceil_ps(x) vec8_round_ps(x, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC)
#define vecN_shuffle_ps vec8_shuffle_ps
#define vecN_insertf32x4_ps vec8_insertf32x4_ps

#define vecN_set1_epi32 vec8_set1_epi32
#define vecN_setzero_epi32 vec8_setzero_epi32
#define vecN_andnot_epi32 vec8_andnot_epi32
#define vecN_min_epi32 vec8_min_epi32
#define vecN_max_epi32 vec8_max_epi32
#define vecN_subs_epu16 vec8_subs_epu16
#define vecN_mullo_epi32 vec8_mullo_epi32
#define vecN_cmpeq_epi32 vec8_cmpeq_epi32
#define vecN_testz_epi32 vec8_testz_epi32
#define vecN_cmpgt_epi32 vec8_cmpgt_epi32
#define vecN_srai_epi32 vec8_srai_epi32
#define vecN_srli_epi32 vec8_srli_epi32
#define vecN_slli_epi32 vec8_slli_epi32
#define vecN_sllv_ones(x) vec8_sllv_epi32(SIMD_BITS_ONE, x)
#define vecN_transpose_epi8(x) vec8_shuffle_epi8(x, SIMD_SHUFFLE_SCANLINE_TO_SUBTILES)
#define vecN_abs_epi32 vec8_abs_epi32

#define vecN_cvtps_epi32 vec8_cvtps_epi32
#define vecN_cvttps_epi32 vec8_cvttps_epi32
#define vecN_cvtepi32_ps vec8_cvtepi32_ps

#define vec4x_dp4_ps(a, b) vec4_dp_ps(a, b, 0xFF)
#define vec4x_fmadd_ps vec4_fmadd_ps
#define vec4x_max_epi32 vec4_max_epi32
#define vec4x_min_epi32 vec4_min_epi32

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Specialized AVX input assembly function for general vertex gather 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef MaskedOcclusionCulling::VertexLayout VertexLayout;

static FORCE_INLINE void GatherVertices(vec8f *vtxX, vec8f *vtxY, vec8f *vtxW, const float *inVtx, const unsigned int *inTrisPtr, int numLanes, const VertexLayout &vtxLayout)
{
	assert(numLanes >= 1);

	const vec8i SIMD_TRI_IDX_OFFSET = vec8_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
	static const vec8i SIMD_LANE_MASK[9] = {
		vec8_setr_epi32( 0,  0,  0,  0,  0,  0,  0,  0),
		vec8_setr_epi32(~0,  0,  0,  0,  0,  0,  0,  0),
		vec8_setr_epi32(~0, ~0,  0,  0,  0,  0,  0,  0),
		vec8_setr_epi32(~0, ~0, ~0,  0,  0,  0,  0,  0),
		vec8_setr_epi32(~0, ~0, ~0, ~0,  0,  0,  0,  0),
		vec8_setr_epi32(~0, ~0, ~0, ~0, ~0,  0,  0,  0),
		vec8_setr_epi32(~0, ~0, ~0, ~0, ~0, ~0,  0,  0),
		vec8_setr_epi32(~0, ~0, ~0, ~0, ~0, ~0, ~0,  0),
		vec8_setr_epi32(~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0)
	};

	// Compute per-lane index list offset that guards against out of bounds memory accesses
	vec8i safeTriIdxOffset = SIMD_TRI_IDX_OFFSET & SIMD_LANE_MASK[numLanes];

	// Fetch triangle indices. 
	vec8i vtxIdx[3];
	vtxIdx[0] = vec8_mullo_epi32(vec8_i32gather_epi32((const int*)inTrisPtr + 0, safeTriIdxOffset, 4), vec8_set1_epi32(vtxLayout.mStride));
	vtxIdx[1] = vec8_mullo_epi32(vec8_i32gather_epi32((const int*)inTrisPtr + 1, safeTriIdxOffset, 4), vec8_set1_epi32(vtxLayout.mStride));
	vtxIdx[2] = vec8_mullo_epi32(vec8_i32gather_epi32((const int*)inTrisPtr + 2, safeTriIdxOffset, 4), vec8_set1_epi32(vtxLayout.mStride));

	char *vPtr = (char *)inVtx;

	// Fetch triangle vertices
	for (int i = 0; i < 3; i++)
	{
		vtxX[i] = vec8_i32gather_ps((float *)vPtr, vtxIdx[i], 1);
		vtxY[i] = vec8_i32gather_ps((float *)(vPtr + vtxLayout.mOffsetY), vtxIdx[i], 1);
		vtxW[i] = vec8_i32gather_ps((float *)(vPtr + vtxLayout.mOffsetW), vtxIdx[i], 1);
	}
}

namespace MaskedOcclusionCullingAVX2
{
	static MaskedOcclusionCulling::Implementation gInstructionSet = MaskedOcclusionCulling::AVX2;

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Include common algorithm implementation (general, SIMD independent code)
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	#include "MaskedOcclusionCullingCommon.inl"

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Utility function to create a new object using the allocator callbacks
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	typedef MaskedOcclusionCulling::pfnAlignedAlloc            pfnAlignedAlloc;
	typedef MaskedOcclusionCulling::pfnAlignedFree             pfnAlignedFree;

	static bool DetectAVX2()
	{
		static bool initialized = false;
		static bool AVX2Support = false;

		int cpui[4];
		if (!initialized)
		{
			initialized = true;
			AVX2Support = false;

			int nIds, nExIds;
			__cpuidex(cpui, 0, 0);
			nIds = cpui[0];
			__cpuidex(cpui, 0x80000000, 0);
			nExIds = cpui[0];

			if (nIds >= 7 && nExIds >= (int)0x80000001)
			{
				AVX2Support = true;

				// Check support for bit counter instructions (lzcnt)
				__cpuidex(cpui, 0x80000001, 0);
				if ((cpui[2] & 0x20) != 0x20)
					AVX2Support = false;

				// Check masks for misc instructions (FMA)
				static const unsigned int FMA_MOVBE_OSXSAVE_MASK = (1 << 12) | (1 << 22) | (1 << 27);
				__cpuidex(cpui, 1, 0);
				if ((cpui[2] & FMA_MOVBE_OSXSAVE_MASK) != FMA_MOVBE_OSXSAVE_MASK)
					AVX2Support = false;

				// Check XCR0 register to ensure that all registers are enabled (by OS)
				static const unsigned int XCR0_MASK = (1 << 2) | (1 << 1); // XMM | YMM
				if (AVX2Support && (_xgetbv(0) & XCR0_MASK) != XCR0_MASK)
					AVX2Support = false;

				// Detect AVX2 & AVX512 instruction sets
				static const unsigned int AVX2_FLAGS = (1 << 3) | (1 << 5) | (1 << 8); // BMI1 (bit manipulation) | BMI2 (bit manipulation) | AVX2
				__cpuidex(cpui, 7, 0);
				if ((cpui[1] & AVX2_FLAGS) != AVX2_FLAGS)
					AVX2Support = false;
			}
		}
		return AVX2Support;
	}

	MaskedOcclusionCulling *CreateMaskedOcclusionCulling(pfnAlignedAlloc memAlloc, pfnAlignedFree memFree)
	{
		if (!DetectAVX2())
			return nullptr;
		
		MaskedOcclusionCullingPrivate *object = (MaskedOcclusionCullingPrivate *)memAlloc(32, sizeof(MaskedOcclusionCullingPrivate));
		new (object) MaskedOcclusionCullingPrivate(memAlloc, memFree);
		return object;
	}
};

#else

namespace MaskedOcclusionCullingAVX2
{
	typedef MaskedOcclusionCulling::pfnAlignedAlloc            pfnAlignedAlloc;
	typedef MaskedOcclusionCulling::pfnAlignedFree             pfnAlignedFree;

	MaskedOcclusionCulling *CreateMaskedOcclusionCulling(pfnAlignedAlloc memAlloc, pfnAlignedFree memFree)
	{
		return nullptr;
	}
};

#endif
