////////////////////////////////////////////////////////////////////////////////
// Copyright 2017 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License.  You may obtain a copy
// of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
// License for the specific language governing permissions and limitations
// under the License.
////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Common defines and constants
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define SIMD_ALL_LANES_MASK    ((1 << SIMD_LANES) - 1)

// Tile dimensions are 32xN pixels. These values are not tweakable and the code must also be modified
// to support different tile sizes as it is tightly coupled with the SSE/AVX register size
#define TILE_WIDTH_SHIFT       5
#define TILE_WIDTH             (1 << TILE_WIDTH_SHIFT)
#define TILE_HEIGHT            (1 << TILE_HEIGHT_SHIFT)

// Sub-tiles (used for updating the masked HiZ buffer) are 8x4 tiles, so there are 4x2 sub-tiles in a tile for AVX2
#define SUB_TILE_WIDTH          8
#define SUB_TILE_HEIGHT         4

// The number of fixed point bits used to represent vertex coordinates / edge slopes.
#if PRECISE_COVERAGE != 0
	#define FP_BITS             8
	#define FP_HALF_PIXEL       (1 << (FP_BITS - 1))
	#define FP_INV              (1.0f / (float)(1 << FP_BITS))
#else
	// Note that too low precision, without precise coverage, may cause overshoots / false coverage during rasterization.
	// This is configured for 14 bits for AVX512 and 16 bits for SSE. Max tile slope delta is roughly
	// (screenWidth + 2*(GUARD_BAND_PIXEL_SIZE + 1)) * (2^FP_BITS * (TILE_HEIGHT + GUARD_BAND_PIXEL_SIZE + 1))
	// and must fit in 31 bits. With this config, max image resolution (width) is ~3272, so stay well clear of this limit.
	#define FP_BITS             (19 - TILE_HEIGHT_SHIFT)
#endif

// Tile dimensions in fixed point coordinates
#define FP_TILE_HEIGHT_SHIFT    (FP_BITS + TILE_HEIGHT_SHIFT)
#define FP_TILE_HEIGHT          (1 << FP_TILE_HEIGHT_SHIFT)

// Maximum number of triangles that may be generated during clipping. We process SIMD_LANES triangles at a time and
// clip against 5 planes, so the max should be 5*8 = 40 (we immediately draw the first clipped triangle).
// This number must be a power of two.
#define MAX_CLIPPED             64
#define MAX_CLIPPED_WRAP        (MAX_CLIPPED - 1)

// Size of guard band in pixels. Clipping doesn't seem to be very expensive so we use a small guard band
// to improve rasterization performance. It's not recommended to set the guard band to zero, as this may
// cause leakage along the screen border due to precision/rounding.
#define GUARD_BAND_PIXEL_SIZE   1.0f

// We classify triangles as big if the bounding box is wider than this given threshold and use a tighter
// but slightly more expensive traversal algorithm. This improves performance greatly for sliver triangles
#define BIG_TRIANGLE            3

// Only gather statistics if enabled.
#if ENABLE_STATS != 0
	#define STATS_ADD(var, val)     (var) += (val)
#else
	#define STATS_ADD(var, val)
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// SIMD common defines (constant values)
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define SIMD_BITS_ONE       _mmw_set1_epi32(~0)
#define SIMD_BITS_ZERO      _mmw_setzero_epi32()
#define SIMD_TILE_WIDTH     _mmw_set1_epi32(TILE_WIDTH)

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Vertex fetch utility function, need to be in global namespace due to template specialization. This function is used for the optimized
// case of (X,Y,Z,W) packed vertices, specified by the VertexLayout struct.
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<int N> FORCE_INLINE void VtxFetch4(__mw *v, const unsigned int *inTrisPtr, int triVtx, const float *inVtx, int numLanes)
{
	// Fetch 4 vectors (matching 1 sse part of the SIMD register), and continue to the next
	const int ssePart = (SIMD_LANES / 4) - N;
	for (int k = 0; k < 4; k++)
	{
		int lane = 4 * ssePart + k;
		if (numLanes > lane)
			v[k] = _mmw_insertf32x4_ps(v[k], _mm_loadu_ps(&inVtx[inTrisPtr[lane * 3 + triVtx] << 2]), ssePart);
	}
	VtxFetch4<N - 1>(v, inTrisPtr, triVtx, inVtx, numLanes);
}

template<> FORCE_INLINE void VtxFetch4<0>(__mw *v, const unsigned int *inTrisPtr, int triVtx, const float *inVtx, int numLanes)
{
	// Workaround for unused parameter warning
	UNUSED_PARAMETER(v); UNUSED_PARAMETER(inTrisPtr); UNUSED_PARAMETER(triVtx); UNUSED_PARAMETER(inVtx); UNUSED_PARAMETER(numLanes);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Private class containing the implementation
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class MaskedOcclusionCullingPrivate : public MaskedOcclusionCulling
{
public:

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Helper structs
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	struct ZTile
	{
		__mw        mZMin[2];
		__mwi       mMask;
	};

	struct Interpolant
	{
		__mw        mVal0;
		__mw        mDx;
		__mw        mDy;

		FORCE_INLINE __mw interpolate(__mw pixelX, __mw pixelY)
		{
			return _mmw_fmadd_ps(mDy, pixelY, _mmw_fmadd_ps(mDx, pixelX, mVal0));
		}
	};

	struct TextureInterpolants
	{
		// Interpolants for (u/z, v/z) and (1/z)
		Interpolant zInterpolant;
		Interpolant uInterpolant;
		Interpolant vInterpolant;

		// Constants for perspective corrected derivatives
		__mw        uDerivConsts[3];
		__mw        vDerivConsts[3];
	};

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Member variables
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	__mw            mHalfWidth;
	__mw            mHalfHeight;
	__mw            mCenterX;
	__mw            mCenterY;
	__m128          mCSFrustumPlanes[5];
	__m128          mIHalfSize;
	__m128          mICenter;
	__m128i         mIScreenSize;

	float           mNearDist;
	int             mWidth;
	int             mHeight;
	int             mTilesWidth;
	int             mTilesHeight;

	ZTile           *mMaskedHiZBuffer;
	ScissorRect     mFullscreenScissor;
#if QUERY_DEBUG_BUFFER != 0
	__mwi           *mQueryDebugBuffer;
#endif

	// Asmjit Runtime
	JitRuntime      mRuntime;

	// Asmjit callout function(s)
	typedef int(*ASM_RasterizeTriangleBatchFN)(MaskedOcclusionCullingPrivate *, __mwi*, __mwi*, __mw*, __mw*, __mw*, __mw*, __mw*, int, ScissorRect *);

	ASM_RasterizeTriangleBatchFN mASMRasterizeTriangleBatch;

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Constructors and state handling
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	/*
	 * Constructor, initializes the object to a default state.
	 */
	MaskedOcclusionCullingPrivate(pfnAlignedAlloc alignedAlloc, pfnAlignedFree alignedFree) : mFullscreenScissor(0, 0, 0, 0)
	{
#if QUERY_DEBUG_BUFFER != 0
		mQueryDebugBuffer = nullptr;
#endif
		mMaskedHiZBuffer = nullptr;
		mAlignedAllocCallback = alignedAlloc;
		mAlignedFreeCallback = alignedFree;

		SetNearClipPlane(0.0f);
		mCSFrustumPlanes[0] = _mm_setr_ps(0.0f, 0.0f, 1.0f, 0.0f);
		mCSFrustumPlanes[1] = _mm_setr_ps(1.0f, 0.0f, 1.0f, 0.0f);
		mCSFrustumPlanes[2] = _mm_setr_ps(-1.0f, 0.0f, 1.0f, 0.0f);
		mCSFrustumPlanes[3] = _mm_setr_ps(0.0f, 1.0f, 1.0f, 0.0f);
		mCSFrustumPlanes[4] = _mm_setr_ps(0.0f, -1.0f, 1.0f, 0.0f);

		memset(&mStats, 0, sizeof(OcclusionCullingStatistics));

		SetResolution(0, 0);

		GenerateASM();
	}

#if 0
	struct ASM_RegAllocator
	{
		X86Compiler &mcc;

		ASM_RegAllocator(X86Compiler &cc) : mcc(cc)
		{
		}

		void wipe()
		{
		}

		void printStats(const char *name)
		{
		}

		void printLive()
		{
		}

		X86Gp newI32(const char *name, int offset = -1)
		{
			if (offset >= 0)
				return mcc.newI32("%s[%d]", name, offset);
			return mcc.newI32(name);
		}

		X86Gp newI64(const char *name, int offset = -1)
		{
			if (offset >= 0)
				return mcc.newI64("%s[%d]", name, offset);
			return mcc.newI64(name);
		}

		X86Xmm newXmm(const char *name, int offset = -1)
		{
			if (offset >= 0)
				return mcc.newXmm("%s[%d]", name, offset);
			return mcc.newXmm(name);
		}

		X86Ymm newYmm(const char *name, int offset = -1)
		{
			if (offset >= 0)
				return mcc.newYmm("%s[%d]", name, offset);
			return mcc.newYmm(name);
		}

		X86Mem newStack(unsigned int size, unsigned int align, const char *name)
		{
			return mcc.newStack(size, align, name);
		}

		void free(const X86Gp &reg)
		{
		}

		void free(const X86Xmm &reg)
		{
		}

		void free(const X86Ymm &reg)
		{
		}

		void InitializeStack() {}
		void CleanupStack() {}
	}; 
#else
	struct ASM_RegAllocator
	{

		struct Register
		{
			bool         mUsed;
			std::string  mName;
		};

		struct StackEntry
		{
			unsigned int mOffset;
			std::string  mName;

			StackEntry(unsigned int offset, std::string name) : mOffset(offset), mName(name) {}
		};

		X86Compiler                  &mcc;

		Register                     mGpRegs[16];
		Register                     mYmmRegs[16];

		std::vector<StackEntry>      mStack;
		unsigned int                 mStackPtr;
		static const unsigned int    STACK_PROLOGUE = 8;

		ASM_RegAllocator(X86Compiler &cc) : mcc(cc)
		{
			wipeRegs();
			mStackPtr = STACK_PROLOGUE; // Allocate prologue
		}

		void wipeRegs()
		{
			for (int i = 0; i < 16; ++i)
				mGpRegs[i].mUsed = false;
			for (int i = 0; i < 16; ++i)
				mYmmRegs[i].mUsed = false;
		}

		void printStats(const char *name)
		{
			int gpUsed = 0, ymmUsed = 0;
			for (int i = 0; i < 16; ++i)
				if (mGpRegs[i].mUsed) gpUsed++;
			for (int i = 0; i < 16; ++i)
				if (mYmmRegs[i].mUsed) ymmUsed++;
			printf("%32s: %2d/15 GP, %2d/16 Ymm\n", name, gpUsed, ymmUsed);
		}

		void printLive()
		{
			printf("Live registers:\n");
			printf("---------------\n");
			printf("GP:\n");
			for (int i = 0; i < 16; ++i)
				if (mGpRegs[i].mUsed)
					printf("    r%2d: %s\n", i, mGpRegs[i].mName.c_str());
			printf("YMM:\n");
			for (int i = 0; i < 16; ++i)
				if (mYmmRegs[i].mUsed)
					printf("    r%2d: %s\n", i, mYmmRegs[i].mName.c_str());
		}

		int alignPtr(int ptr, int align) 
		{
			return (ptr + align - 1) - (ptr + align - 1) % align;
		}

		std::string copyName(const char *name, int offset)
		{
			char buf[512];
			if (offset >= 0)
				sprintf(buf, "%s[%d]", name, offset);
			else
				sprintf(buf, "%s", name);
			return std::string(buf);
		}

		X86Gp getI64(const X86Gp &reg, const char *name, int offset = -1)
		{
			assert(!mGpRegs[reg.getId()].mUsed); // Register was already taken :(
			mGpRegs[reg.getId()].mUsed = true;
			mGpRegs[reg.getId()].mName = copyName(name, offset);
			return x86OpData.gpq[reg.getId()].r64();
		}

		X86Gp getI32(const X86Gp &reg, const char *name, int offset = -1)
		{
			return getI64(reg, name, offset).r32();
		}

		X86Gp newI64(const char *name, int offset = -1)
		{
			for (int i = 0; i < 16; ++i)
			{
				if (i != X86Gp::kIdSp && i != X86Gp::kIdCx && !mGpRegs[i].mUsed)
				{
					mGpRegs[i].mUsed = true;
					mGpRegs[i].mName = copyName(name, offset);
					return x86OpData.gpq[i].r64();
				}
			}

			if (!mGpRegs[X86Gp::kIdCx].mUsed)
			{
				mGpRegs[X86Gp::kIdCx].mUsed = true;
				mGpRegs[X86Gp::kIdCx].mName = copyName(name, offset);
				return x86OpData.gpq[X86Gp::kIdCx].r64();
			}

			// Uh oh, ran out of registers. We don't do automatic spilling
			printLive();
			assert(false);
		}

		X86Gp newI32(const char *name, int offset = -1)
		{
			return newI64(name, offset).r32();
		}

		X86Ymm newYmm(const char *name, int offset = -1)
		{
			for (int i = 0; i < 16; ++i)
			{
				if (!mYmmRegs[i].mUsed)
				{
					assert(x86OpData.ymm[i].ymm().getId() == i);
					mYmmRegs[i].mUsed = true;
					mYmmRegs[i].mName = copyName(name, offset);
					return x86OpData.ymm[i].ymm();
				}
			}
			printLive();
			assert(false); // Uh oh, ran out of registers. We don't do automatic spilling
			return x86OpData.ymm[0].ymm();
		}

		X86Xmm newXmm(const char *name, int offset = -1)
		{
			return newYmm(name, offset).xmm();
		}

		void free(const X86Gp &reg)
		{
			assert(reg.getId() >= X86Gp::kIdAx && reg.getId() <= X86Gp::kIdR15 && reg.getId() != X86Gp::kIdSp);
			mGpRegs[reg.getId()].mUsed = false;
			mGpRegs[reg.getId()].mName = "";
		}

		void free(const X86Xmm &reg)
		{
			assert(reg.getId() >= 0 && reg.getId() <= 15);
			mYmmRegs[reg.getId()].mUsed = false;
			mYmmRegs[reg.getId()].mName = "";
		}

		void free(const X86Ymm &reg)
		{
			assert(reg.getId() >= 0 && reg.getId() <= 15);
			mYmmRegs[reg.getId()].mUsed = false;
			mYmmRegs[reg.getId()].mName = "";
		}

		X86Mem newStack(unsigned int size, unsigned int align, const char *name)
		{
			assert(align <= 64);
			int insertPos = alignPtr(mStackPtr, align); // Make sure element is aligned
			mStack.push_back(StackEntry(insertPos, name));
			mStackPtr = insertPos + size;
			return X86Mem(x86::rsp, insertPos);
		}

		void InitializeStack()
		{
			int stackSize = alignPtr(mStackPtr, 64);
			X86Gp espStore = newI64("esp_store");
			mcc.mov(espStore, x86::rsp);                   // Save rsp so we can restore it later
			mcc.and_(x86::rsp, ~63);                       // Make sure stack pointer is 64 byte aligned
			mcc.sub(x86::rsp, stackSize);                  // Allocate all required space on the stack
			mcc.mov(x86::qword_ptr(x86::rsp), espStore);   // Save
			free(espStore);
		}

		void CleanupStack()
		{
			mcc.mov(x86::rsp, x86::qword_ptr(x86::rsp));   // Restore old stack pointer
		}

	};
#endif

	struct ASM_RasterizerContext
	{
		// Compile time constants
		X86Mem simd_i_0;
		X86Mem simd_i_1;
		X86Mem simd_i_not_0;
		X86Mem simd_i_pad_w_mask;
		X86Mem simd_i_pad_h_mask;
		X86Mem simd_i_tile_width;
		X86Mem simd_i_tile_height;
		X86Mem simd_i_lane_idx;
		X86Mem simd_i_shuffle_scanlines_to_subtiles;

		X86Mem simd_f_neg_0;
		X86Mem simd_f_1_0;
		X86Mem simd_f_2_0;
		X86Mem simd_f_flt_max;
		X86Mem simd_f_shl_fp_bits;
		X86Mem simd_f_tile_width;
		X86Mem simd_f_tile_height;
		X86Mem simd_f_sub_tile_width;
		X86Mem simd_f_sub_tile_height;
		X86Mem simd_f_sub_tile_col_offset;
		X86Mem simd_f_sub_tile_row_offset;
		X86Mem simd_f_guardband;

		// Runtime constants (pointer buffer resolution etc)
		X86Mem simd_i_res_tiles_width;
		X86Mem simd_f_horizontal_slope_delta;
		X86Mem simd_f_neg_horizontal_slope_delta;

		X86Mem i64_scanline_step;
		X86Mem i64_msoc_buffer_ptr;

		// Runtime variables (stuff that doesn't fit in registers)
		X86Mem Mem_zMin, Mem_zMax;
		X86Mem Mem_zPlaneOffset, Mem_zPixelDx, Mem_zPixelDy;

		X86Mem Mem_bbTileSizeX, Mem_bbTileSizeY;
		X86Mem Mem_bbBottomIdx, Mem_bbMidIdx, Mem_bbTopIdx;

		X86Mem Mem_eventStart[3], Mem_slopeFP[3], Mem_slopeTileDelta[3];
		X86Mem Mem_triSlopeTileDelta[3];

		X86Mem Mem_zTriMin, Mem_zTriMax;
		X86Mem Mem_zTriTileDx, Mem_zTriTileDy;

		X86Mem Mem_clipVtxBuffer;

		X86Mem Mem_midVtxLeft, Mem_tileMidRowAddr;

		template<class T> X86Mem ASM_newYmmConst(X86Compiler &cc, T val)
		{
			Data256 data;
			memcpy(&data, &val, sizeof(T));
			return cc.newYmmConst(kConstScopeGlobal, data);
		}

		ASM_RasterizerContext(X86Compiler &cc, ASM_RegAllocator &ra)
		{
			// Create SIMD integer constants
			simd_i_0 = cc.newYmmConst(kConstScopeGlobal, Data256::fromI32(0));
			simd_i_1 = cc.newYmmConst(kConstScopeGlobal, Data256::fromI32(1));
			simd_i_not_0 = cc.newYmmConst(kConstScopeGlobal, Data256::fromI32(~0));
			simd_i_pad_w_mask = cc.newYmmConst(kConstScopeGlobal, Data256::fromI32(~(TILE_WIDTH - 1)));
			simd_i_pad_h_mask = cc.newYmmConst(kConstScopeGlobal, Data256::fromI32(~(TILE_HEIGHT - 1)));
			simd_i_tile_width = cc.newYmmConst(kConstScopeGlobal, Data256::fromI32(TILE_WIDTH));
			simd_i_tile_height = cc.newYmmConst(kConstScopeGlobal, Data256::fromI32(TILE_HEIGHT));
			simd_i_lane_idx = cc.newYmmConst(kConstScopeGlobal, Data256::fromI32(0, 1, 2, 3, 4, 5, 6, 7));
			simd_i_shuffle_scanlines_to_subtiles = cc.newYmmConst(kConstScopeGlobal, Data256::fromI8(0x0, 0x4, 0x8, 0xC, 0x1, 0x5, 0x9, 0xD, 0x2, 0x6, 0xA, 0xE, 0x3, 0x7, 0xB, 0xF, 0x0, 0x4, 0x8, 0xC, 0x1, 0x5, 0x9, 0xD, 0x2, 0x6, 0xA, 0xE, 0x3, 0x7, 0xB, 0xF));

			// Create SIMD fp32 constants
			simd_f_neg_0 = cc.newYmmConst(kConstScopeGlobal, Data256::fromF32(-0.0f));
			simd_f_1_0 = cc.newYmmConst(kConstScopeGlobal, Data256::fromF32(1.0f));
			simd_f_2_0 = cc.newYmmConst(kConstScopeGlobal, Data256::fromF32(2.0f));
			simd_f_flt_max = cc.newYmmConst(kConstScopeGlobal, Data256::fromF32(FLT_MAX));
			simd_f_shl_fp_bits = cc.newYmmConst(kConstScopeGlobal, Data256::fromF32((float)(1 << FP_BITS)));
			simd_f_tile_width = cc.newYmmConst(kConstScopeGlobal, Data256::fromF32((float)TILE_WIDTH));
			simd_f_tile_height = cc.newYmmConst(kConstScopeGlobal, Data256::fromF32((float)TILE_HEIGHT));
			simd_f_sub_tile_width = cc.newYmmConst(kConstScopeGlobal, Data256::fromF32((float)SUB_TILE_WIDTH));
			simd_f_sub_tile_height = cc.newYmmConst(kConstScopeGlobal, Data256::fromF32((float)SUB_TILE_HEIGHT));
			simd_f_sub_tile_col_offset = ASM_newYmmConst(cc, SIMD_SUB_TILE_COL_OFFSET_F);
			simd_f_sub_tile_row_offset = ASM_newYmmConst(cc, SIMD_SUB_TILE_ROW_OFFSET_F);
			simd_f_guardband = cc.newYmmConst(kConstScopeGlobal, Data256::fromF32(2.0f*(GUARD_BAND_PIXEL_SIZE + 1.0f)));

			// Allocate stack memory for runtime constants
			simd_i_res_tiles_width = ra.newStack(sizeof(__mw), sizeof(__mw), "simd_i_res_tiles_width");
			simd_f_horizontal_slope_delta = ra.newStack(sizeof(__mw), sizeof(__mw), "simd_f_horizontal_slope_delta");
			simd_f_neg_horizontal_slope_delta = ra.newStack(sizeof(__mw), sizeof(__mw), "simd_f_neg_horizontal_slope_delta");

			i64_scanline_step = ra.newStack(sizeof(void*), sizeof(void*), "i64_scanline_step");
			i64_msoc_buffer_ptr = ra.newStack(sizeof(void*), sizeof(void*), "i64_msoc_buffer_ptr");

			// Allocate memory for local variables
			Mem_zMin = ra.newStack(sizeof(__mw), sizeof(__mw), "Mem_zMin");
			Mem_zMax = ra.newStack(sizeof(__mw), sizeof(__mw), "Mem_zMax");
			Mem_zPlaneOffset = ra.newStack(sizeof(__mw), sizeof(__mw), "Mem_zPlaneOffset");
			Mem_zPixelDx = ra.newStack(sizeof(__mw), sizeof(__mw), "Mem_zPixelDx");
			Mem_zPixelDy = ra.newStack(sizeof(__mw), sizeof(__mw), "Mem_zPixelDy");

			Mem_bbTileSizeX = ra.newStack(sizeof(__mw), sizeof(__mw), "Mem_bbTileSizeX");
			Mem_bbTileSizeY = ra.newStack(sizeof(__mw), sizeof(__mw), "Mem_bbTileSizeY");
			Mem_bbBottomIdx = ra.newStack(sizeof(__mw), sizeof(__mw), "Mem_bbBottomIdx");
			Mem_bbMidIdx = ra.newStack(sizeof(__mw), sizeof(__mw), "Mem_bbMidIdx");
			Mem_bbTopIdx = ra.newStack(sizeof(__mw), sizeof(__mw), "Mem_bbTopIdx");

			for (int i = 0; i < 3; ++i)
				Mem_eventStart[i] = ra.newStack(sizeof(__mw), sizeof(__mw), "Mem_eventStart[i]");
			for (int i = 0; i < 3; ++i)
				Mem_slopeFP[i] = ra.newStack(sizeof(__mw), sizeof(__mw), "Mem_slopeFP[i]");
			for (int i = 0; i < 3; ++i)
				Mem_slopeTileDelta[i] = ra.newStack(sizeof(__mw), sizeof(__mw), "Mem_slopeTileDelta[i]");
			for (int i = 0; i < 3; ++i)
				Mem_triSlopeTileDelta[i] = ra.newStack(sizeof(__mw), sizeof(__mw), "Mem_triSlopeTileDelta[i]");

			Mem_zTriMin = ra.newStack(sizeof(__mw), sizeof(__mw), "Mem_zTriMin");
			Mem_zTriMax = ra.newStack(sizeof(__mw), sizeof(__mw), "Mem_zTriMax");
			Mem_zTriTileDx = ra.newStack(sizeof(__mw), sizeof(__mw), "Mem_zTriTileDx");
			Mem_zTriTileDy = ra.newStack(sizeof(__mw), sizeof(__mw), "Mem_zTriTileDy");

			Mem_clipVtxBuffer = ra.newStack(sizeof(__m128)*MAX_CLIPPED * 3, sizeof(__m128), "Mem_clipVtxBuffer");

			Mem_midVtxLeft = ra.newStack(sizeof(int), sizeof(int), "Mem_midVtxLeft");
			Mem_tileMidRowAddr = ra.newStack(sizeof(void*), sizeof(void*), "Mem_tileMidRowAddr");
		}
	};

	void ASM_set1(X86Compiler &cc, X86Ymm &dst, const X86Gp &src)
	{
		cc.vmovd(dst.xmm(), src);
		cc.vpshufd(dst.xmm(), dst.xmm(), 0);
		cc.vinsertf128(dst, dst, dst.xmm(), 1);
	}

	void ASM_set1(X86Compiler &cc, X86Ymm &dst, const X86Mem &src)
	{
		cc.vmovd(dst.xmm(), src);
		cc.vpshufd(dst.xmm(), dst.xmm(), 0);
		cc.vinsertf128(dst, dst, dst.xmm(), 1);
	}

	void ASM_zerobits(X86Compiler &cc, X86Ymm &reg)
	{
		cc.vpxor(reg, reg, reg);
	}

	void ASM_UpdateTileQuick(
		X86Compiler &cc,
		ASM_RegAllocator &ra,
		const ASM_RasterizerContext &context,
		const X86Gp &tileAddr,
		const X86Ymm &coverage, 
		const X86Ymm &zTri)
	{
#ifdef _DEBUG
		X86Gp _debug_name = ra.newI32("ASM_UpdateTileQuick");
		cc.mov(_debug_name, 0xDEADC0DE);
		ra.free(_debug_name);
#endif
		// Registers / temporaries
		X86Mem zMinPtr[2] = { x86::yword_ptr(tileAddr, offsetof(ZTile, mZMin[0])), x86::yword_ptr(tileAddr, offsetof(ZTile, mZMin[1])) };
		X86Mem zMaskPtr = x86::yword_ptr(tileAddr, offsetof(ZTile, mMask));

		X86Ymm zMin0 = ra.newYmm("zMin0");
		cc.vmovaps(zMin0, zMinPtr[0]);
		X86Ymm zMin1 = ra.newYmm("zMin1");
		cc.vmovaps(zMin1, zMinPtr[1]);

		// Mask out all subtiles failing the depth test (don't update these subtiles)
		X86Ymm sub_t_0_sign = ra.newYmm("sub_t_0_sign");
		cc.vsubps(sub_t_0_sign, zTri, zMin0);                                      // sub_t_0_sign = zTri - tile.zMin[0]
		cc.vpsrad(sub_t_0_sign, sub_t_0_sign, 31);                                 // sub_t_0_sign >>= 31;
		X86Ymm deadLane = ra.newYmm("deadLane");
		cc.vpcmpeqd(deadLane, coverage, context.simd_i_0);                         // deadLane = coverage == 0 ? ~0 : 0
		cc.vpor(deadLane, deadLane, sub_t_0_sign);                                 // deadlane |= sub_t_0_sign
		X86Ymm maskedCoverage = ra.newYmm("maskedCoverage");
		cc.vpandn(maskedCoverage, sub_t_0_sign, coverage);                         // maskedCoverage = ~sub_t_0_sign & coverage
		ra.free(sub_t_0_sign);
		ra.free(coverage);

		// Use distance heuristic to discard layer 1 if incoming triangle is significantly nearer to observer
		// than the buffer contents. See Section 3.2 in "Masked Software Occlusion Culling"
		X86Ymm &diff_t_0_1 = ra.newYmm("diff_t_0_1");
		cc.vaddps(diff_t_0_1, zTri, zMin0);                                        // diff_t_0_1 = zTri + tile.zMin[0]
		X86Ymm fullyCoveredLane = ra.newYmm("fullyCoveredLane");
		cc.vpcmpeqd(fullyCoveredLane, maskedCoverage, context.simd_i_not_0);       // maskedCoverage == ~0
		cc.vfmsub231ps(diff_t_0_1, zMin1, context.simd_f_2_0);                     // diff_t_0_1 = tile.zMin1*2.0f - diff_t_0_1
		cc.vpsrad(diff_t_0_1, diff_t_0_1, 31);                                     // diff_t_0_1 >>=  31
		ra.free(fullyCoveredLane);
		cc.vpor(diff_t_0_1, diff_t_0_1, fullyCoveredLane);                         // diff_t_0_1 |= fullyCoveredLane
		X86Ymm discardLayerMask = ra.newYmm("discardLayerMask");
		ra.free(diff_t_0_1);
		cc.vpandn(discardLayerMask, deadLane, diff_t_0_1);                         // discardLayerMask = ~deadlane & diff_t_0_1

		// Update the mask with incoming triangle coverage
		X86Ymm zMask = ra.newYmm("zMask");
		cc.vpandn(zMask, discardLayerMask, zMaskPtr);                              // zMask = ~discardLayerMask & tile.zMask
		ra.free(maskedCoverage);
		cc.vpor(zMask, maskedCoverage, zMask);                                     // zMask |= maskedCoverage
		
		// Update the zMask
		X86Ymm zMaskFull = ra.newYmm("zMaskFull");
		cc.vpcmpeqd(zMaskFull, zMask, context.simd_i_not_0);                       // zMaskFull = zMask == ~0 ? ~0 : 0
		cc.vpandn(zMask, zMaskFull, zMask);                                        // zMask = ~zMaskFull & zMask
		ra.free(zMask);
		cc.vmovaps(zMaskPtr, zMask);

		// Compute new value for zMin[1]. This has one of four outcomes: zMin[1] = min(zMin[1], zTriv),  zMin[1] = zTriv,
		// zMin[1] = FLT_MAX or unchanged, depending on if the layer is updated, discarded, fully covered, or not updated
		X86Ymm opA = ra.newYmm("opA");
		ra.free(deadLane);
		cc.vblendvps(opA, zTri, zMin1, deadLane);                                  // opA = deadLane & 0x80000000 ? zMin1 : zTri
		X86Ymm opB = ra.newYmm("opB");
		ra.free(zTri);
		ra.free(discardLayerMask);
		cc.vblendvps(opB, zMin1, zTri, discardLayerMask);                          // opB = discardLayerMask & 0x80000000 ? zTri : zMin1
		ra.free(opA);
		ra.free(opB);
		X86Ymm z1tMin = ra.newYmm("z1tMin");
		cc.vminps(z1tMin, opA, opB);                                               // z1tMin = min(opA, opB)
		cc.vblendvps(zMin1, z1tMin, context.simd_f_flt_max, zMaskFull);            // zMin1 = maskFull & 0x80000000 ? z1tMin : zMin1

		// Propagate zMin[1] back to zMin[0] if tile was fully covered, and update the mask
		ra.free(z1tMin);
		ra.free(zMaskFull);
		cc.vblendvps(zMin0, zMin0, z1tMin, zMaskFull);                             // zMin0 = zMaskFull & 0x80000000 ? z1tMin : zMin0

		// Write data back to tile
		ra.free(zMin1);
		cc.vmovaps(zMinPtr[1], zMin1);
		ra.free(zMin0);
		cc.vmovaps(zMinPtr[0], zMin0);
	}

	void ASM_ComputeBoundingBox(
		X86Compiler &cc,
		ASM_RegAllocator &ra,
		const ASM_RasterizerContext &context,
		X86Ymm &bbPixelMinX,
		X86Ymm &bbPixelMinY,
		X86Ymm &bbPixelMaxX,
		X86Ymm &bbPixelMaxY,
		const X86Mem *pVtxX,
		const X86Mem *pVtxY,
		const X86Gp &scissorPtr)
	{
#ifdef _DEBUG
		X86Gp _debug_name = ra.newI32("ASM_ComputeBoundingBox");
		cc.mov(_debug_name, 0xDEADC0DE);
		ra.free(_debug_name);
#endif

		X86Ymm pVtxX0 = ra.newYmm("pVtxX0");
		cc.vmovaps(pVtxX0, pVtxX[0]);                                              // pVtxX0 = pVtxX[0]

		X86Ymm pVtxY0 = ra.newYmm("pVtxY0");
		cc.vmovaps(pVtxY0, pVtxY[0]);                                              // pVtxY0 = pVtxY[0]

		// Find Min/Max vertices
		cc.vminps(bbPixelMinX, pVtxX0, pVtxX[1]);                                  // bbPixelMinX = min(pVtxX[0], pVtxX[1])
		cc.vminps(bbPixelMinX, bbPixelMinX, pVtxX[2]);                             // bbPixelMinX = min(bbPixelMinX, pVtxX[2])
		cc.vcvttps2dq(bbPixelMinX, bbPixelMinX);                                   // bbPixelMinX = cvttps_epi32(bbPixelMinX)

		cc.vminps(bbPixelMinY, pVtxY0, pVtxY[1]);                                  // bbPixelMinY = min(pVtxY[0], pVtxY[1])
		cc.vminps(bbPixelMinY, bbPixelMinY, pVtxY[2]);                             // bbPixelMinY = min(bbPixelMinY, pVtxY[1])
		cc.vcvttps2dq(bbPixelMinY, bbPixelMinY);                                   // bbPixelMinY = cvttps_epi32(bbPixelMinY)

		ra.free(pVtxX0);
		cc.vmaxps(bbPixelMaxX, pVtxX0, pVtxX[1]);                                  // bbPixelMaxX = min(pVtxX[0], pVtxX[1])
		cc.vmaxps(bbPixelMaxX, bbPixelMaxX, pVtxX[2]);                             // bbPixelMaxX = min(bbPixelMaxX, pVtxX[1])
		cc.vcvttps2dq(bbPixelMaxX, bbPixelMaxX);                                   // bbPixelMaxX = cvttps_epi32(bbPixelMaxX)

		ra.free(pVtxY0);
		cc.vmaxps(bbPixelMaxY, pVtxY0, pVtxY[1]);                                  // bbPixelMaxY = min(pVtxY[0], pVtxY[1])
		cc.vmaxps(bbPixelMaxY, bbPixelMaxY, pVtxY[2]);                             // bbPixelMaxY = min(bbPixelMaxY, pVtxY[1])
		cc.vcvttps2dq(bbPixelMaxY, bbPixelMaxY);                                   // bbPixelMaxY = cvttps_epi32(bbPixelMaxY)

		// Clamp to tile boundaries
		cc.vpand(bbPixelMinX, bbPixelMinX, context.simd_i_pad_w_mask);             // bbPixelMinX &= ~(TILE_WIDTH - 1)
		cc.vpand(bbPixelMinY, bbPixelMinY, context.simd_i_pad_h_mask);             // bbPixelMinY &= ~(TILE_HEIGHT - 1)
		cc.vpaddd(bbPixelMaxX, bbPixelMaxX, context.simd_i_tile_width);            // bbPixelMaxX += TILE_WIDTH
		cc.vpand(bbPixelMaxX, bbPixelMaxX, context.simd_i_pad_w_mask);             // bbPixelMaxX &= ~(TILE_WIDTH - 1)
		cc.vpaddd(bbPixelMaxY, bbPixelMaxY, context.simd_i_tile_height);           // bbPixelMaxY += TILE_HEIGHT
		cc.vpand(bbPixelMaxY, bbPixelMaxY, context.simd_i_pad_h_mask);             // bbPixelMaxY &= ~(TILE_HEIGHT - 1)

		// Clip to scissor
		X86Ymm scissorMinX = ra.newYmm("scissorMinX");
		X86Ymm scissorMinY = ra.newYmm("scissorMinY");
		X86Ymm scissorMaxX = ra.newYmm("scissorMaxX");
		X86Ymm scissorMaxY = ra.newYmm("scissorMaxY");
		ASM_set1(cc, scissorMinX, x86::dword_ptr(scissorPtr, offsetof(ScissorRect, mMinX)));
		ASM_set1(cc, scissorMinY, x86::dword_ptr(scissorPtr, offsetof(ScissorRect, mMinY)));
		ASM_set1(cc, scissorMaxX, x86::dword_ptr(scissorPtr, offsetof(ScissorRect, mMaxX)));
		ASM_set1(cc, scissorMaxY, x86::dword_ptr(scissorPtr, offsetof(ScissorRect, mMaxY)));
		ra.free(scissorMinX);
		cc.vpmaxsd(bbPixelMinX, bbPixelMinX, scissorMinX);                         // bbminX = max(bbminX, scissor->mMinX)
		ra.free(scissorMinY);
		cc.vpmaxsd(bbPixelMinY, bbPixelMinY, scissorMinY);                         // bbminY = max(bbminY, scissor->mMinY)
		ra.free(scissorMaxX);
		cc.vpminsd(bbPixelMaxX, bbPixelMaxX, scissorMaxX);                         // bbmaxX = max(bbmaxX, scissor->mMaxX)
		ra.free(scissorMaxY);
		cc.vpminsd(bbPixelMaxY, bbPixelMaxY, scissorMaxY);                         // bbmaxY = max(bbmaxY, scissor->mMaxY)
		ra.free(scissorPtr);
	}

	void ASM_InterpolationSetup(
		X86Compiler &cc,
		ASM_RegAllocator &ra,
		const ASM_RasterizerContext &context,
		const X86Mem *pVtxX,
		const X86Mem *pVtxY, 
		const X86Mem *pVtxA, 
		X86Ymm &pVtxX0,
		X86Ymm &pVtxY0,
		X86Ymm &pVtxA0,
		X86Ymm &aPixelDx,
		X86Ymm &aPixelDy)
	{
#ifdef _DEBUG
		X86Gp _debug_name = ra.newI32("ASM_InterpolationSetup");
		cc.mov(_debug_name, 0xDEADC0DE);
		ra.free(_debug_name);
#endif

		X86Ymm one = ra.newYmm("one");
		cc.vmovaps(one, context.simd_f_1_0);

		// Setup a(x,y) = a0 + dx*x + dy*y screen space plane equation
		X86Ymm x1 = ra.newYmm("x1");
		cc.vsubps(x1, pVtxX0, pVtxX[1]);                                           // x1 = pVtxX[0] - pVtxX[1]
		X86Ymm x2 = ra.newYmm("x2");
		cc.vsubps(x2, pVtxX0, pVtxX[2]);                                           // x2 = pVtxX[0] - pVtxX[2]
		X86Ymm y1 = ra.newYmm("y1");
		cc.vsubps(y1, pVtxY0, pVtxY[1]);                                           // y1 = pVtxY[0] - pVtxY[1]
		X86Ymm y2 = ra.newYmm("y2");
		cc.vsubps(y2, pVtxY0, pVtxY[2]);                                           // y2 = pVtxY[0] - pVtxY[2]
		
		X86Ymm d = ra.newYmm("d");
		cc.vmulps(d, y1, x2);                                                      // d = y1 * x2
		cc.vfmsub231ps(d, x1, y2);                                                 // d = y1 * y2 - d
		ra.free(one);
		cc.vdivps(d, one, d);                                                      // d = 1.0f / d

		X86Ymm a1 = ra.newYmm("a1");
		cc.vsubps(a1, pVtxA0, pVtxA[1]);                                           // a1 = pVtxA[0] - pVtxA[1]
		X86Ymm a2 = ra.newYmm("a2");
		cc.vsubps(a2, pVtxA0, pVtxA[2]);                                           // a2 = pVtxA[0] - pVtxA[2]

		aPixelDx = ra.newYmm("aPixelDx");
		ra.free(y1);
		cc.vmulps(aPixelDx, y1, a2);                                               // aPixelDx = y1 * a2
		ra.free(y2);
		cc.vfmsub231ps(aPixelDx, a1, y2);                                          // aPixelDx = a1 * y2 - aPixelDx
		cc.vmulps(aPixelDx, aPixelDx, d);                                          // aPixelDx *= d

		aPixelDy = ra.newYmm("aPixelDy");
		ra.free(a1);
		ra.free(x2);
		cc.vmulps(aPixelDy, a1, x2);                                               // aPixelDy = a1 * x2
		ra.free(a2);
		ra.free(x1);
		cc.vfmsub231ps(aPixelDy, x1, a2);                                          // aPixelDy = x1 * a2 - aPixelDy
		ra.free(d);
		cc.vmulps(aPixelDy, aPixelDy, d);                                          // aPixelDy *= d

		//cc.vmulps(aPixel0, aPixelDx, pVtxX0);                                      // aPixel0 = aPixelDx * pVtxX[0]
		//cc.vfmadd231ps(aPixel0, aPixelDy, pVtxY0);                                 // aPixel0 = aPixelDy * pVtxY[0] + aPixel0
		//cc.vsubps(aPixel0, pVtxA0, aPixel0);                                       // aPixel0 = pVtxA[0] - aPixel0
	}

	void ASM_SortVertices(
		X86Compiler &cc,
		ASM_RegAllocator &ra,
		const ASM_RasterizerContext &context,
		X86Mem *pVtxXMem,
		X86Mem *pVtxYMem)
	{
#ifdef _DEBUG
		X86Gp _debug_name = ra.newI32("ASM_SortVertices");
		cc.mov(_debug_name, 0xDEADC0DE);
		ra.free(_debug_name);
#endif

		X86Ymm pVtxX[3], pVtxY[3];
		for (int i = 0; i < 3; ++i)
		{
			pVtxX[i] = ra.newYmm("pVtxXReg", i);
			cc.vmovaps(pVtxX[i], pVtxXMem[i]);
			pVtxY[i] = ra.newYmm("pVtxYReg", i);
			cc.vmovaps(pVtxY[i], pVtxYMem[i]);
		}

		X86Ymm C_zerobits = ra.newYmm("ASM_SortVertices::C_zerobits");
		cc.vpxor(C_zerobits, x86::ymm0, x86::ymm0);                                // amjit bug workaround: cc.vpxor(C_zerobits, C_zerobits, C_zerobits) breaks register allocator

		// Rotate the triangle in the winding order until v0 is the vertex with lowest Y value
		for (int i = 0; i < 2; i++)
		{
			X86Ymm ey1 = ra.newYmm("ey1");
			X86Ymm ey2 = ra.newYmm("ey2");
			cc.vsubps(ey1, pVtxY[1], pVtxY[0]);                                    // ey1 = pVtxY[1] - pVtxY[0]
			cc.vsubps(ey2, pVtxY[2], pVtxY[0]);                                    // ey2 = pVtxY[2] - pVtxY[0]
			X86Ymm swapMask = ra.newYmm("swapMask");
			cc.vpcmpeqd(swapMask, ey2, C_zerobits);                                // swapMask = ey == 0 ? ~0 : 0
			ra.free(ey1);
			cc.vpor(swapMask, swapMask, ey1);                                      // swapMask |= ey1
			ra.free(ey2);
			cc.vpor(swapMask, swapMask, ey2);                                      // swapMask |= ey2

			X86Ymm sX = ra.newYmm("sX");
			cc.vblendvps(sX, pVtxX[2], pVtxX[0], swapMask);                        // sX = blendv_ps(pVtxX[2], pVtxX[0], swapMask)
			cc.vblendvps(pVtxX[0], pVtxX[0], pVtxX[1], swapMask);                  // pVtxX[0] = blendv_ps(pVtxX[0], pVtxX[1], swapMask)
			cc.vblendvps(pVtxX[1], pVtxX[1], pVtxX[2], swapMask);                  // pVtxX[1] = blendv_ps(pVtxX[1], pVtxX[2], swapMask)
			ra.free(sX);
			cc.vmovaps(pVtxX[2], sX);                                              // pVtxX[2] = sX

			X86Ymm sY = ra.newYmm("sY");
			cc.vblendvps(sY, pVtxY[2], pVtxY[0], swapMask);                        // sY = blendv_ps(pVtxY[2], pVtxY[0], swapMask)
			cc.vblendvps(pVtxY[0], pVtxY[0], pVtxY[1], swapMask);                  // pVtxY[0] = blendv_ps(pVtxY[0], pVtxY[1], swapMask)
			ra.free(swapMask);
			cc.vblendvps(pVtxY[1], pVtxY[1], pVtxY[2], swapMask);                  // pVtxY[1] = blendv_ps(pVtxY[1], pVtxY[2], swapMask)
			ra.free(sY);
			cc.vmovaps(pVtxY[2], sX);                                              // pVtxY[2] = sY
		}
		ra.free(C_zerobits);

		for (int i = 0; i < 3; ++i)
		{
			ra.free(pVtxX[i]);
			cc.vmovaps(pVtxXMem[i], pVtxX[i]);
			ra.free(pVtxY[i]);
			cc.vmovaps(pVtxYMem[i], pVtxY[i]);
		}
	}

	void ASM_TraverseTile(
		X86Compiler &cc,
		ASM_RegAllocator &ra,
		int TEST_Z,
		int nLeftEvents,
		int nRightEvents,
		const Label &L_VisibleExit,
		const ASM_RasterizerContext &context,
		const X86Gp &tileAddr,
		X86Ymm *left,
		X86Ymm *right,
		const X86Ymm &z0)
	{
#ifdef _DEBUG
		X86Gp _debug_name = ra.newI32("ASM_TraverseTile");
		cc.mov(_debug_name, 0xDEADC0DE);
		ra.free(_debug_name);
#endif

		// Labels
		Label L_TileCompleted = cc.newLabel();

		//if (TEST_Z)
		//	STATS_ADD(mStats.mOccludees.mNumTilesTraversed, 1);
		//else
		//	STATS_ADD(mStats.mOccluders.mNumTilesTraversed, 1);

		// Perform a coarse test to quickly discard occluded tiles
#if QUICK_MASK != 0
		// Only use the reference layer (layer 0) to cull as it is always conservative
		X86Ymm zMinBuf = ra.newYmm("zMinBuf");
		cc.vmovaps(zMinBuf, x86::yword_ptr(tileAddr, offsetof(ZTile, mZMin[0])));               // zMinBuf = mMaskedHiZBuffer[tileIdx].mZMin[0]
#else
		// TODO
#endif

		X86Ymm C_onebits = ra.newYmm("ASM_TraverseTile::C_onebits");
		cc.vpcmpeqd(C_onebits, C_onebits, C_onebits);

		X86Ymm zDistTriMin = ra.newYmm("zDistTriMin");
		if (!TEST_Z)
			ra.free(zMinBuf);
		cc.vsubps(zDistTriMin, zMinBuf, context.Mem_zTriMax);                                   // zDistTriMin = zTriMax - zMinBuf
		X86Gp zDistMask = ra.newI32("zDistMask");
		ra.free(zDistTriMin);
		cc.vmovmskps(zDistMask, zDistTriMin);                                                   // zDistMask = movemask_ps(zDistTriMin)
		ra.free(zDistMask);
		cc.cmp(zDistMask, 0);                                                                   // if (zDistMask)
		cc.je(L_TileCompleted);                                                                 // {
		{
			// Compute coverage mask for entire 32xN using shift operations
			X86Ymm rastMask32x1 = ra.newYmm("rastMask32x1");
			cc.vpsllvd(rastMask32x1, C_onebits, left[0]);                                       // rastMask32x1 = ~0 >> left[0]
			for (int i = 1; i < nLeftEvents; ++i)
			{
				X86Ymm tmpReg = ra.newYmm("tmpReg");
				cc.vpsllvd(tmpReg, C_onebits, left[i]);                                         // tmpReg = ~0 >> left[i]
				ra.free(tmpReg);
				cc.vpand(rastMask32x1, rastMask32x1, tmpReg);                                   // rastMask32x1 &= tmpReg
			}
			for (int i = 0; i < nRightEvents; ++i)
			{
				X86Ymm tmpReg = ra.newYmm("tmpReg");
				cc.vpsllvd(tmpReg, C_onebits, right[i]);                                        // tmpReg = ~0 >> right[i]
				ra.free(tmpReg);
				cc.vpandn(rastMask32x1, tmpReg, rastMask32x1);                                  // rastMask32x1 = ~tmpReg & rastMask32x1
			}
			
			// Swizzle rasterization mask to 8x4 subtiles
			X86Ymm rastMask8x4 = ra.newYmm("rastMask8x4");
			ra.free(rastMask32x1);
			cc.vpshufb(rastMask8x4, rastMask32x1, context.simd_i_shuffle_scanlines_to_subtiles); // rastMask8x4 = shuffle_epi8(rastMask32x1, shuffleScanlinesToSubtiles)

			ra.free(C_onebits);

			// Perform conservative texture lookup for alpha tested triangles
			//if (TEXTURE_COORDINATES)
			//	rastMask8x4 = TextureAlphaTest(tileIdx, rastMask8x4, dist0, texInterpolants, texture);

			if (TEST_Z) // TestTriangles
			{
				
				X86Ymm deadLane = ra.newYmm("deadLane");
				ra.free(rastMask8x4);
				cc.vpcmpeqd(deadLane, rastMask8x4, context.simd_i_0);                           // deadLane = rastMask8x4 == 0 ? ~0 : 0
				X86Ymm zSubtileMax = ra.newYmm("zSubtileMax");
				cc.vminps(zSubtileMax, z0, context.Mem_zTriMax);                                // zSubtileMax = min(z0, zTriMax)
				X86Ymm zPass = ra.newYmm("zPass");
				ra.free(zSubtileMax);
				ra.free(zMinBuf);
				cc.vcmpps(zPass, zSubtileMax, zMinBuf, _CMP_GE_OQ);                             // zPass = zSubtileMax >= zMinBuf ? ~0 : 0
				ra.free(deadLane);
				cc.vpandn(zPass, deadLane, zPass);                                              // zPass = ~deadLane & zPass
#if QUERY_DEBUG_BUFFER != 0
				//	__mwi debugVal = _mmw_blendv_epi32(_mmw_set1_epi32(0), _mmw_blendv_epi32(_mmw_set1_epi32(1), _mmw_set1_epi32(2), zPass), _mmw_not_epi32(deadLane));
				//	mQueryDebugBuffer[tileIdx] = debugVal;
#endif
				ra.free(zPass);
				cc.vptest(zPass, zPass);                                                        // if (zPass != 0)
				cc.jnz(L_VisibleExit);                                                          //     goto L_VisibleExit

			}
			else // RenderTriangles
			{
				// Compute zmin for rasterized triangle in the current tile
				X86Ymm zTriSubtile = ra.newYmm("zTriSubtile");
				cc.vmaxps(zTriSubtile, z0, context.Mem_zTriMin);                                // zTriSubtile = max(z0, zTriMin)

				// Perform update heuristic
#if QUICK_MASK != 0
				ASM_UpdateTileQuick(cc, ra, context, tileAddr, rastMask8x4, zTriSubtile);
#else
#endif
				// note rastMask8x4 and zTriSubtile are free'd inside the ASM_UpdateTile* functions
				// ra.free(zTriSubtile);
				// ra.free(rastMask8x4);
			}
		}
		cc.bind(L_TileCompleted);                                                               // }

	}

	void ASM_TraverseScanline(
		X86Compiler &cc,
		ASM_RegAllocator &ra,
		int TEST_Z,
		int leftEvent,
		int rightEvent,
		int nLeftEvents,
		int nRightEvents,
		const Label &L_VisibleExit,
		const ASM_RasterizerContext &context,
		const X86Gp *leftOffset,
		const X86Gp &rightOffset, 
		const X86Gp &tileRowAddr,
		const X86Ymm *events,
		const X86Ymm &z0)
	{
#ifdef _DEBUG
		X86Gp _debug_name = ra.newI32("ASM_TraverseScanline");
		cc.mov(_debug_name, 0xDEADC0DE);
		ra.free(_debug_name);
#endif
		Label L_ScanlineLoopStart = cc.newLabel();
		Label L_ScanlineLoopExit = cc.newLabel();

		// Floor edge events to integer pixel coordinates (shift out fixed point bits)
		X86Ymm vLeftOffsetZ0 = ra.newYmm("vLeftOffset/z0");
		if (leftOffset != nullptr)
			ASM_set1(cc, vLeftOffsetZ0, *leftOffset);
		else
			cc.vxorps(vLeftOffsetZ0, vLeftOffsetZ0, vLeftOffsetZ0);                    // vLeftOffset = 0
		X86Ymm eventOffset = ra.newYmm("eventOffset");
		cc.vpslld(eventOffset, vLeftOffsetZ0, TILE_WIDTH_SHIFT);                       // eventOffset = vLeftOffset >> TILE_WIDTH_SHIFT

		X86Ymm C_zerobits = ra.newYmm("ASM_TraverseScanline::C_zerobits");
		cc.vpxor(C_zerobits, x86::ymm0, x86::ymm0);                                    // amjit bug workaround: cc.vpxor(C_zerobits, C_zerobits, C_zerobits) breaks register allocator

		X86Ymm right[2], left[2];
		for (int i = 0; i < nLeftEvents; ++i)
		{
			left[i] = ra.newYmm("left", i);
			cc.vpsrad(left[i], events[leftEvent - i], FP_BITS);                        // left[i] = events[leftEvent-i] >> FP_BITS
			cc.vpsubd(left[i], left[i], eventOffset);                                  // left[i] -= eventOffset
			cc.vpmaxsd(left[i], left[i], C_zerobits);                                  // left[i] = max(left[i], 0)
		}

		for (int i = 0; i < nRightEvents; ++i)
		{
			right[i] = ra.newYmm("right", i);
			cc.vpsrad(right[i], events[rightEvent + i], FP_BITS);                      // right[i] = events[rightEvent+i] >> FP_BITS
			cc.vpsubd(right[i], right[i], eventOffset);                                // right[i] -= eventOffset
			cc.vpmaxsd(right[i], right[i], C_zerobits);                                // right[i] = max(right[i], 0)
		}
		ra.free(eventOffset);
		ra.free(C_zerobits);

		cc.vcvtdq2ps(vLeftOffsetZ0, vLeftOffsetZ0);                                    // vLeftOffsetZ0 = cvtepi32_ps(leftOffset);
		cc.vfmadd132ps(vLeftOffsetZ0, z0, context.Mem_zTriTileDx);                     // z0 = zTriTileDx*leftOffset + z0

		// tileAddr = mMaskedHiZBuffer + sizeof(HiZTile)*(iLeftOffset + iTileIdx)
		X86Gp tileAddr = ra.newI64("tileAddr");
		if (leftOffset != nullptr)
		{
			X86Gp offset = ra.newI64("offset");
			cc.lea(offset, X86Mem(*leftOffset, *leftOffset, 1, 0));                     // offset = leftOffset * 3
			cc.shl(offset, (TILE_HEIGHT_SHIFT + 2));                                    // offset *= sizeof(__mw)
			ra.free(offset);
			cc.lea(tileAddr, X86Mem(tileRowAddr, offset, 0, 0));                        // tileAddr = iTileRowAddr + offset
		}
		else
			cc.mov(tileAddr, tileRowAddr);                                              // tileAddr = tileRowAddr

		// tileAddrEnd = mMaskedHiZBuffer + sizeof(HiZTile)*(iRightOffset + iTileIdx)
		X86Gp offset = ra.newI64("offset");
		cc.lea(offset, X86Mem(rightOffset, rightOffset, 1, 0));                         // offset = rightOffset * 3
		cc.shl(offset, (TILE_HEIGHT_SHIFT + 2));                                        // offset *= sizeof(__mw)
		ra.free(offset);
		X86Gp tileAddrEnd = ra.newI64("tileAddrEnd");
		cc.lea(tileAddrEnd, X86Mem(tileRowAddr, offset, 0, 0));                         // tileAddrEnd = tileRowAddr + offset

		// Traverse first tile (must alawys be >= 1 tiles)
		ASM_TraverseTile(cc, ra, TEST_Z, nLeftEvents, nRightEvents, L_VisibleExit, context, tileAddr, left, right, vLeftOffsetZ0);

		cc.add(tileAddr, sizeof(ZTile));                                               // tileAddr += sizeof(ZTILE)
		cc.bind(L_ScanlineLoopStart);                                                  // while(true) {
		{
			cc.cmp(tileAddr, tileAddrEnd);                                             // if (tileAddr >= tileAddrEnd)
			cc.jge(L_ScanlineLoopExit);                                                //     break;

			// Update all interpolants / events
			cc.vaddps(vLeftOffsetZ0, vLeftOffsetZ0, context.Mem_zTriTileDx);           // z0 += zTriTileDx
			for (int i = 0; i < nLeftEvents; ++i)
				cc.vpsubusw(left[i], left[i], context.simd_i_tile_width);              // left[i] = max(0, left[i] - SIMD_TILE_WIDTH)
			for (int i = 0; i < nRightEvents; ++i)
				cc.vpsubusw(right[i], right[i], context.simd_i_tile_width);            // right[i] = max(0, right[i] - SIMD_TILE_WIDTH)

			// Traverse next tile
			ASM_TraverseTile(cc, ra, TEST_Z, nLeftEvents, nRightEvents, L_VisibleExit, context, tileAddr, left, right, vLeftOffsetZ0);

			cc.add(tileAddr, sizeof(ZTile));                                           // tileAddr += sizeof(ZTILE)
			cc.jmp(L_ScanlineLoopStart);
		}
		cc.bind(L_ScanlineLoopExit);                                                   // }
		for (int i = 0; i < nLeftEvents; ++i)
			ra.free(left[i]);
		for (int i = 0; i < nRightEvents; ++i)
			ra.free(right[i]);
		ra.free(tileAddr);
		ra.free(tileAddrEnd);
		ra.free(vLeftOffsetZ0);
	}

	void ASM_RasterizeTriangleSegment(
		X86Compiler &cc,
		ASM_RegAllocator &ra,
		int TEST_Z,
		int TIGHT_TRAVERSAL,
		int UPDATE_EVENTS_LAST_ITERATION,
		int leftEvent, 
		int rightEvent,
		const Label &L_VisibleExit,
		const ASM_RasterizerContext &context,
		X86Gp &tileRowAddr,
		const X86Gp &tileStopAddr,
		const X86Gp &bbWidth,
		X86Gp *tileEvent,
		X86Ymm *triEvent,
		X86Ymm &z0)
	{
#ifdef _DEBUG
		X86Gp _debug_name = ra.newI32("ASM_RasterizeTriangleSegment");
		cc.mov(_debug_name, 0xDEADC0DE);
		ra.free(_debug_name);
#endif

		Label L_LoopStart = cc.newLabel(), L_LoopExit = cc.newLabel();

		assert(leftEvent >= 0 && rightEvent >= 0 && leftEvent <= 2 && rightEvent <= 2);

		cc.cmp(tileRowAddr, tileStopAddr);
		cc.jge(L_LoopExit);                                                            // while(tileRowIdx < tileStopIdx) 
		cc.bind(L_LoopStart);
		{
			if (TIGHT_TRAVERSAL)
			{

				// start = max(0, min(bbWidth - 1, tileEvent[leftEvent] >> (TILE_WIDTH_SHIFT + FP_BITS)));
				X86Gp start = ra.newI32("start");
				X86Gp bbWidth_1 = ra.newI32("bbWidth_1");
				X86Gp zeroTmp = ra.newI32("zeroTmp");

				cc.lea(bbWidth_1, X86Mem(bbWidth, -1));
				cc.xor_(zeroTmp, zeroTmp);

				cc.mov(start, tileEvent[leftEvent]);
				cc.sar(start, (TILE_WIDTH_SHIFT + FP_BITS));                           // start = tileEvent[leftEvent] >> (TILE_WIDTH_SHIFT + FP_BITS)
				cc.cmp(start, bbWidth_1);
				cc.cmovg(start, bbWidth_1);                                            // start = min(start, bbWidth - 1)
				cc.cmp(start, 0);
				cc.cmovl(start, zeroTmp);                                              // start = max(start, 0)
				ra.free(zeroTmp);
				ra.free(bbWidth_1);

				// end = min(bbWidth, ((int)tileEvent[rightEvent] >> (TILE_WIDTH_SHIFT + FP_BITS)));
				X86Gp end = ra.newI32("end");
				
				cc.mov(end, tileEvent[rightEvent]);
				cc.sar(end, TILE_WIDTH_SHIFT + FP_BITS);                               // end = tileEvent[rightEvent] >> (TILE_WIDTH_SHIFT + FP_BITS)
				cc.cmp(end, bbWidth);
				cc.cmovg(end, bbWidth);                                                // end = min(start, bbWidth)

				// Traverse the scanline and update the masked hierarchical z buffer
				ASM_TraverseScanline(cc, ra, TEST_Z, leftEvent, rightEvent, 1, 1, L_VisibleExit, context, &start, end, tileRowAddr, triEvent, z0);
				
				ra.free(end);
				ra.free(start);
			}
			else
			{
				// Traverse the scanline and update the masked hierarchical z buffer
				ASM_TraverseScanline(cc, ra, TEST_Z, leftEvent, rightEvent, 1, 1, L_VisibleExit, context, nullptr, bbWidth, tileRowAddr, triEvent, z0);
			}
			
			// move to the next scanline of tiles, update edge events and interpolate z
			cc.add(tileRowAddr, context.i64_scanline_step);                            // tileRowIdx += mTilesWidth;
			
			if (!UPDATE_EVENTS_LAST_ITERATION)
			{
				cc.cmp(tileRowAddr, tileStopAddr);
				cc.jge(L_LoopExit);                                                    // if(tileRowIdx >= tileStopIdx) break;
			}

			cc.vaddps(z0, z0, context.Mem_zTriTileDy);                                 // z0 += zTriTileDy
#if PRECISE_COVERAGE != 0
#else
			cc.vpaddd(triEvent[leftEvent], triEvent[leftEvent], context.Mem_triSlopeTileDelta[leftEvent]);
			cc.vpaddd(triEvent[rightEvent], triEvent[rightEvent], context.Mem_triSlopeTileDelta[rightEvent]);
#endif
			if (TIGHT_TRAVERSAL)
			{
				cc.add(tileEvent[leftEvent], context.Mem_slopeTileDelta[leftEvent]);   // startEvent += startTileDelta
				cc.add(tileEvent[rightEvent], context.Mem_slopeTileDelta[rightEvent]); // endEvent += endTileDelta
			}

			if (UPDATE_EVENTS_LAST_ITERATION)
			{
				cc.cmp(tileRowAddr, tileStopAddr);
				cc.jge(L_LoopExit);                                                    // if(tileRowIdx >= tileStopIdx) break;
			}

			cc.jmp(L_LoopStart);
		}
		cc.bind(L_LoopExit);
	}

	void ASM_RasterizeTriangle(
		X86Compiler &cc,
		ASM_RegAllocator &ra,
		int TEST_Z,
		int TIGHT_TRAVERSAL,
		int MID_VTX_RIGHT,
		const Label &L_VisibleExit,
		const ASM_RasterizerContext &context,
		X86Gp &triIdx,
		X86Gp &bbWidth, 
		X86Gp &tileRowAddr,
		X86Gp &tileEndRowAddr,
		X86Ymm &z0) 
	{
#ifdef _DEBUG
		X86Gp _debug_name = ra.newI32("ASM_RasterizeTriangle");
		cc.mov(_debug_name, 0xDEADC0DE);
		ra.free(_debug_name);
#endif

		Label L_FlatTriangle = cc.newLabel(), L_TriangleExit = cc.newLabel();

		//if (TEST_Z)
		//	STATS_ADD(mStats.mOccludees.mNumRasterizedTriangles, 1);
		//else
		//	STATS_ADD(mStats.mOccluders.mNumRasterizedTriangles, 1);

#if PRECISE_COVERAGE != 0
#else
		X86Ymm triEvent[3];
		for (int i = 0; i < 3; ++i)
		{
			// Setup edge events for first batch of SIMD_LANES scanlines
			X86Ymm eStart = ra.newYmm("eStart");
			ASM_set1(cc, eStart, context.Mem_eventStart[i]);                                              // eStart = _mmw_set1_epi32(eventStart[i].m_i32[triIdx])
			triEvent[i] = ra.newYmm("triEvent", i);
			ASM_set1(cc, triEvent[i], context.Mem_slopeFP[i]);                                            // triEvent[i] = _mmw_set1_epi32(slope[i].m_i32[triIdx])
			cc.vpmulld(triEvent[i], triEvent[i], context.simd_i_lane_idx);                                // triEvent[i] *= SIMD_LANE_IDX
			ra.free(eStart);
			cc.vpaddd(triEvent[i], triEvent[i], eStart);                                                  // triEvent[i] += eStart
		}
#endif
		X86Gp tileEvent[3];

		// For big triangles track start & end tile for each scanline and only traverse the valid region
		if (TIGHT_TRAVERSAL)
		{
			for (int i = 0; i < 3; ++i)
			{
				bool rightEdge = i == 0 || (i == 1 && MID_VTX_RIGHT);
#if PRECISE_COVERAGE != 0
				// Compute conservative bounds for the edge events over a 32xN tile
				startEvent = simd_i32(eventStart[2])[triIdx] + min(0, startDelta);
				endEvent = simd_i32(eventStart[0])[triIdx] + max(0, endDelta) + (TILE_WIDTH << FP_BITS);
				if (MID_VTX_RIGHT)
					topEvent = simd_i32(eventStart[1])[triIdx] + max(0, topDelta) + (TILE_WIDTH << FP_BITS);
				else
					topEvent = simd_i32(eventStart[1])[triIdx] + min(0, topDelta);
#endif

				tileEvent[i] = ra.newI32("tileEvent", i);
				cc.mov(tileEvent[i], context.Mem_eventStart[i]);                                         // tileEvent[i] = eventStart[i].m_i32[triIdx]
				
				X86Gp tmp0i32 = ra.newI32("tmp0i32");
				cc.xor_(tmp0i32, tmp0i32);
				cc.cmp(context.Mem_slopeTileDelta[i], tmp0i32);
				if (rightEdge)
				{
					cc.cmovge(tmp0i32, context.Mem_slopeTileDelta[i]);
					cc.add(tmp0i32, TILE_WIDTH << FP_BITS);                                              // slopeTileDelta = max(0, tileEventDelta[i]) + (TILE_WIDTH << FP_BITS)
				}
				else
				{
					cc.cmovle(tmp0i32, context.Mem_slopeTileDelta[i]);                                   // slopeTileDelta = min(0, tileEventDelta[i])
				}
				ra.free(tmp0i32);
				cc.add(tileEvent[i], tmp0i32);                                                           // tileEvent[i] += slopeTileDelta
			}
		}

		cc.cmp(tileRowAddr, context.Mem_tileMidRowAddr);                                                  // if (tileRowIdx > tileMidRowIdx)
		cc.jg(L_FlatTriangle);                                                                            //      goto L_FlatTriangle
		{
			X86Gp tileStopAddr = ra.newI64("tileStopIdx");	
			cc.mov(tileStopAddr, context.Mem_tileMidRowAddr);
			cc.cmp(tileStopAddr, tileEndRowAddr);
			cc.cmovg(tileStopAddr, tileEndRowAddr);                                                       // tileStopAddr = min(tileEndRowIdx, tileMidRowIdx)
			
			// Traverse bottom triangle segment
			ASM_RasterizeTriangleSegment(cc, ra, TEST_Z, TIGHT_TRAVERSAL, 1, 2, 0, L_VisibleExit, context, tileRowAddr, tileStopAddr, bbWidth, tileEvent, triEvent, z0);
			ra.free(tileStopAddr);

			// Traverse middle scanline
			cc.cmp(tileRowAddr, tileEndRowAddr);                                                          // if (tileRowAddr >= tileEndRowAddr)
			cc.jge(L_TriangleExit);                                                                       //     goto L_TriangleExit
			{
				if (TIGHT_TRAVERSAL)
				{
					// start = max(0, min(bbWidth - 1, tileEvent[2] >> (TILE_WIDTH_SHIFT + FP_BITS)));
					X86Gp start = ra.newI32("start");
					X86Gp bbWidth_1 = ra.newI32("bbWidth_1");
					X86Gp zeroTmp = ra.newI32("zeroTmp");

					cc.lea(bbWidth_1, X86Mem(bbWidth, -1));
					cc.xor_(zeroTmp, zeroTmp);

					cc.mov(start, tileEvent[2]);
					cc.sar(start, (TILE_WIDTH_SHIFT + FP_BITS));                                          // start = tileEvent[2] >> (TILE_WIDTH_SHIFT + FP_BITS)
					cc.cmp(start, bbWidth_1);
					ra.free(bbWidth_1);
					cc.cmovg(start, bbWidth_1);                                                           // start = min(start, bbWidth - 1)
					cc.cmp(start, 0);
					ra.free(zeroTmp);
					cc.cmovl(start, zeroTmp);                                                             // start = max(start, 0)

					// end = min(bbWidth, ((int)endEvent >> (TILE_WIDTH_SHIFT + FP_BITS)));
					X86Gp end = ra.newI32("end");
					cc.mov(end, tileEvent[0]);
					cc.sar(end, TILE_WIDTH_SHIFT + FP_BITS);                                              // end = tileEvent[0] >> (TILE_WIDTH_SHIFT + FP_BITS)
					cc.cmp(end, bbWidth);
					cc.cmovg(end, bbWidth);                                                               // end = min(start, bbWidth)

					// Traverse the scanline and update the masked hierarchical z buffer
					if (MID_VTX_RIGHT)
						ASM_TraverseScanline(cc, ra, TEST_Z, 2, 0, 1, 2, L_VisibleExit, context, &start, end, tileRowAddr, triEvent, z0);
					else
						ASM_TraverseScanline(cc, ra, TEST_Z, 2, 0, 2, 1, L_VisibleExit, context, &start, end, tileRowAddr, triEvent, z0);
					ra.free(start);
					ra.free(end);
				}
				else
				{
					// Traverse the scanline and update the masked hierarchical z buffer
					if (MID_VTX_RIGHT)
						ASM_TraverseScanline(cc, ra, TEST_Z, 2, 0, 1, 2, L_VisibleExit, context, nullptr, bbWidth, tileRowAddr, triEvent, z0);
					else
						ASM_TraverseScanline(cc, ra, TEST_Z, 2, 0, 2, 0, L_VisibleExit, context, nullptr, bbWidth, tileRowAddr, triEvent, z0);
				}

				// move to the next scanline of tiles, update edge events and interpolate z
				cc.add(tileRowAddr, context.i64_scanline_step);                                          // tileRowAddr += mTilesWidth;
			}

			// Traverse top segment
			cc.cmp(tileRowAddr, tileEndRowAddr);                                                         // if (tileRowAddr >= tileEndRowAddr)
			cc.jge(L_TriangleExit);                                                                      //     goto L_TriangleExit
			{
				// Defered update, moving down 1 scanline from the middle scanline
				cc.vaddps(z0, z0, context.Mem_zTriTileDy);                                               // z0 += tileY
#if PRECISE_COVERAGE != 0
#else
				cc.vpaddd(triEvent[MID_VTX_RIGHT + 1], triEvent[MID_VTX_RIGHT + 1], context.Mem_triSlopeTileDelta[MID_VTX_RIGHT + 1]);
				cc.vpaddd(triEvent[MID_VTX_RIGHT + 0], triEvent[MID_VTX_RIGHT + 0], context.Mem_triSlopeTileDelta[MID_VTX_RIGHT + 0]);
#endif
				if (TIGHT_TRAVERSAL)
				{
					cc.add(tileEvent[MID_VTX_RIGHT + 1], context.Mem_slopeTileDelta[MID_VTX_RIGHT + 1]);              // startEvent += startDelta
					cc.add(tileEvent[MID_VTX_RIGHT + 0], context.Mem_slopeTileDelta[MID_VTX_RIGHT + 0]);              // endEvent += endDelta
				}

				// Traverse top triangle segment
				ASM_RasterizeTriangleSegment(cc, ra, TEST_Z, TIGHT_TRAVERSAL, 0, MID_VTX_RIGHT + 1, MID_VTX_RIGHT + 0, L_VisibleExit, context, tileRowAddr, tileEndRowAddr, bbWidth, tileEvent, triEvent, z0);
			}

			cc.jmp(L_TriangleExit);                                                                       // }
		}
		cc.bind(L_FlatTriangle);                                                                          // else
		{
			cc.cmp(tileRowAddr, tileEndRowAddr);                                                          // if (tileRowAddr < tileEndRowAddr)
			cc.jg(L_TriangleExit);                                                                        // {
			{
				// Traverse top triangle segment
				ASM_RasterizeTriangleSegment(cc, ra, TEST_Z, TIGHT_TRAVERSAL, 0, MID_VTX_RIGHT + 1, MID_VTX_RIGHT + 0, L_VisibleExit, context, tileRowAddr, tileEndRowAddr, bbWidth, tileEvent, triEvent, z0);
			}                                                                                             // }
		}
		cc.bind(L_TriangleExit);                                                                          // }
		
		for (int i = 0; i < 3; ++i)
		{
			ra.free(triEvent[i]);
			if (TIGHT_TRAVERSAL)
				ra.free(tileEvent[i]);
		}
	}

	void ASM_RasterizeTriangleBatch(
		X86Compiler &cc,
		ASM_RegAllocator &ra,
		int TEST_Z,
		X86Gp &msocPtr,
		X86Gp &ipVtxXPtr,
		X86Gp &ipVtxYPtr,
		X86Gp &pVtxXPtr,
		X86Gp &pVtxYPtr,
		X86Gp &pVtxZPtr,
		X86Gp &pVtxUPtr,
		X86Gp &pVtxVPtr,
		X86Gp &triMask,
		const X86Gp &scissorPtr)
	{
		ra.free(ipVtxXPtr);
		ra.free(ipVtxYPtr);
		ra.free(pVtxUPtr);
		ra.free(pVtxVPtr);

		Label L_VisibleExit = cc.newLabel();
		Label L_ViewCulledExit = cc.newLabel();
		Label L_Exit = cc.newLabel();

		Label L_TriLoopStart = cc.newLabel();
		Label L_SmallTriangle = cc.newLabel();
		Label L_EndSmallTriangle = cc.newLabel();
		Label L_TriLoopExit = cc.newLabel();

		ASM_RasterizerContext context(cc, ra);

		ra.InitializeStack();

		//////////////////////////////////////////////////////////////////////////////
		// Set up run-time constants
		//////////////////////////////////////////////////////////////////////////////

		{
			// Set up i_res_tiles_width
			X86Gp i64_tiles_width = ra.newI64("i64_tiles_width"); 
			X86Gp i64_scanline_step = ra.newI64("i64_scanline_step");
			cc.xor_(i64_tiles_width, i64_tiles_width);
			cc.mov(i64_tiles_width.r32(), x86::dword_ptr(msocPtr, offsetof(MaskedOcclusionCullingPrivate, mTilesWidth)));
			cc.mov(i64_scanline_step, i64_tiles_width);
			cc.imul(i64_scanline_step, sizeof(ZTile));
			ra.free(i64_scanline_step);
			cc.mov(context.i64_scanline_step, i64_scanline_step);

			// Set up buffer pointer
			X86Gp i64_msoc_buffer_ptr = ra.newI64("i64_msoc_buffer_ptr");
			cc.mov(i64_msoc_buffer_ptr, x86::qword_ptr(msocPtr, offsetof(MaskedOcclusionCullingPrivate, mMaskedHiZBuffer)));
			ra.free(i64_msoc_buffer_ptr);
			cc.mov(context.i64_msoc_buffer_ptr, i64_msoc_buffer_ptr);

			// Set up simd_i_res_tiles_width
			X86Ymm tmp = ra.newYmm("tmp");
			ASM_set1(cc, tmp, i64_tiles_width.r32());
			ra.free(i64_tiles_width);
			cc.vmovdqa(context.simd_i_res_tiles_width, tmp);

			// Set up C_horizontalSlopeDelta and C_negHorizontalSlopeDelta
			X86Gp i32_width = ra.newI32("i32_width");
			ra.free(msocPtr);
			cc.mov(i32_width, x86::dword_ptr(msocPtr, offsetof(MaskedOcclusionCullingPrivate, mWidth)));
			ASM_set1(cc, tmp, i32_width);
			ra.free(i32_width);
			cc.vcvtdq2ps(tmp, tmp);
			cc.vaddps(tmp, tmp, context.simd_f_guardband);
			cc.vmovdqa(context.simd_f_horizontal_slope_delta, tmp);

			cc.vxorps(tmp, tmp, context.simd_f_neg_0);
			ra.free(tmp);
			cc.vmovdqa(context.simd_f_neg_horizontal_slope_delta, tmp);
		}

		//////////////////////////////////////////////////////////////////////////////
		// Set up pointers to vtx data
		//////////////////////////////////////////////////////////////////////////////

		X86Mem pVtxX[3], pVtxY[3], pVtxZ[3];
		for (int i = 0; i < 3; ++i)
		{
			pVtxX[i] = x86::yword_ptr(pVtxXPtr, i * sizeof(__mw));
			pVtxY[i] = x86::yword_ptr(pVtxYPtr, i * sizeof(__mw));
			pVtxZ[i] = x86::yword_ptr(pVtxZPtr, i * sizeof(__mw));
		}

		//////////////////////////////////////////////////////////////////////////////
		// Compute bounding box and clamp to tile coordinates
		//////////////////////////////////////////////////////////////////////////////

		X86Ymm bbPixelMinX = ra.newYmm("bbPixelMinX");
		X86Ymm bbPixelMinY = ra.newYmm("bbPixelMinY");

		X86Ymm bbTileMinX = ra.newYmm("bbTileMinX");
		X86Ymm bbTileMinY = ra.newYmm("bbTileMinY");
		X86Ymm bbTileMaxY = ra.newYmm("bbTileMaxY");

		{
			X86Ymm bbPixelMaxX = ra.newYmm("bbPixelMaxX");
			X86Ymm bbPixelMaxY = ra.newYmm("bbPixelMaxY");

			X86Ymm bbTileMaxX = ra.newYmm("bbTileMaxX");

			ASM_ComputeBoundingBox(cc, ra, context, bbPixelMinX, bbPixelMinY, bbPixelMaxX, bbPixelMaxY, pVtxX, pVtxY, scissorPtr);

			// Clamp bounding box to tiles (it's already padded in computeBoundingBox)
			cc.vpsrad(bbTileMinX, bbPixelMinX, TILE_WIDTH_SHIFT);               // bbTileMinX = bbPixelMinX >> TILE_WIDTH_SHIFT
			cc.vpsrad(bbTileMinY, bbPixelMinY, TILE_HEIGHT_SHIFT);              // bbTileMinY = bbPixelMinY >> TILE_HEIGHT_SHIFT
			ra.free(bbPixelMaxX);
			cc.vpsrad(bbTileMaxX, bbPixelMaxX, TILE_WIDTH_SHIFT);               // bbTileMaxX = bbPixelMaxX >> TILE_WIDTH_SHIFT
			ra.free(bbPixelMaxY);
			cc.vpsrad(bbTileMaxY, bbPixelMaxY, TILE_HEIGHT_SHIFT);              // bbTileMaxY = bbPixelMaxY >> TILE_HEIGHT_SHIFT
			X86Ymm bbTileSizeX = ra.newYmm("bbTileSizeX");
			ra.free(bbTileMaxX);
			cc.vpsubd(bbTileSizeX, bbTileMaxX, bbTileMinX);                     // bbTileSizeX = bbTileMaxX - bbTileMinX
			X86Ymm bbTileSizeY = ra.newYmm("bbTileSizeY");
			cc.vpsubd(bbTileSizeY, bbTileMaxY, bbTileMinY);                     // bbTileSizeY = bbTileMaxY - bbTileMinY

			// Cull triangles with zero bounding box
			X86Ymm bbX_1 = ra.newYmm("bbX_1");
			X86Ymm bbY_1 = ra.newYmm("bbY_1");
			X86Ymm bbAreaSign = ra.newYmm("bbAreaSign");
			X86Gp bbAreaMask = ra.newI32("bbAreaMask");

			cc.vpsubd(bbY_1, bbTileSizeY, context.simd_i_1);                    // bbY_1 = bbTileSizeY - 1
			cc.vpsubd(bbX_1, bbTileSizeX, context.simd_i_1);                    // bbX_1 = bbTileSizeX - 1
			ra.free(bbX_1);
			ra.free(bbY_1);
			cc.vpor(bbAreaSign, bbX_1, bbY_1);                                  // bbAreaSign = bbX_1 | bbY_1
			ra.free(bbAreaSign);
			cc.vmovmskps(bbAreaMask, bbAreaSign);                               // bbAreaMask = movemask_ps(bbAreaSign)

			cc.not_(bbAreaMask);                                                // bbAreaMask = ~bbAreaMask
			ra.free(bbAreaMask);
			cc.and_(triMask, bbAreaMask);                                       // triMask &= bbAreaMask
			cc.jz(L_ViewCulledExit);


			//////////////////////////////////////////////////////////////////////////////
			// Store bounding box top/bottom indices used later during traversal
			//////////////////////////////////////////////////////////////////////////////

			X86Ymm bbBottomIdx = ra.newYmm("bbBottomIdx");
			X86Ymm bbTopIdx = ra.newYmm("bbTopIdx");

			cc.vpmulld(bbBottomIdx, bbTileMinY, context.simd_i_res_tiles_width); // bbBottomIdx = bbTileMinY * simd_i_res_tiles_width
			cc.vpaddd(bbBottomIdx, bbBottomIdx, bbTileMinX);                     // bbBottomIdx += bbTileMinX

			cc.vpaddd(bbTopIdx, bbTileMinY, bbTileSizeY);                        // bbTopIdx = bbTileMinY + bbTileSizeY
			cc.vpmulld(bbTopIdx, bbTopIdx, context.simd_i_res_tiles_width);      // bbTopIdx = bbTopIdx * simd_i_res_tiles_width
			cc.vpaddd(bbTopIdx, bbTopIdx, bbTileMinX);                           // bbTopIdx += bbTileMinX

			// Move indices to stack so we can access individual lanes
			ra.free(bbBottomIdx);
			cc.vmovaps(context.Mem_bbBottomIdx, bbBottomIdx);
			ra.free(bbTopIdx);
			cc.vmovaps(context.Mem_bbTopIdx, bbTopIdx);
			ra.free(bbTileSizeX);
			cc.vmovaps(context.Mem_bbTileSizeX, bbTileSizeX);
			ra.free(bbTileSizeY);
			cc.vmovaps(context.Mem_bbTileSizeY, bbTileSizeY);

		}

		//////////////////////////////////////////////////////////////////////////////
		// Set up screen space depth plane
		//////////////////////////////////////////////////////////////////////////////

		{
			X86Ymm is_pVtxX0 = ra.newYmm("IS_pVtxX0");
			cc.vmovaps(is_pVtxX0, pVtxX[0]);
			
			X86Ymm is_pVtxY0 = ra.newYmm("IS_pVtxY0");
			cc.vmovaps(is_pVtxY0, pVtxY[0]);
			
			X86Ymm is_pVtxZ0 = ra.newYmm("IS_pVtxZ0");
			cc.vmovaps(is_pVtxZ0, pVtxZ[0]);

			X86Ymm zPixelDx, zPixelDy;
			ASM_InterpolationSetup(cc, ra, context, pVtxX, pVtxY, pVtxZ, is_pVtxX0, is_pVtxY0, is_pVtxZ0, zPixelDx, zPixelDy);

			// Compute z value at min corner of bounding box. Offset to make sure z is conservative for all 8x4 subtiles
			X86Ymm zPlaneOffset = ra.newYmm("zPlaneOffset/bbMinXV0");
			cc.vcvtdq2ps(zPlaneOffset, bbPixelMinX);                            // bbMinXV0 = cvtepi32_ps(bbPixelMinX)
			X86Ymm bbMinYV0 = ra.newYmm("bbMinYV0");
			cc.vcvtdq2ps(bbMinYV0, bbPixelMinY);                                // bbMinYV0 = cvtepi32_ps(bbPixelMinY)
			ra.free(is_pVtxX0);
			cc.vsubps(zPlaneOffset, zPlaneOffset, is_pVtxX0);                   // bbMinXV0 -= pVtxX[0]
			ra.free(is_pVtxY0);
			cc.vsubps(bbMinYV0, bbMinYV0, is_pVtxY0);                           // bbMinYV0 -= pVtxY[0]

			cc.vfmadd213ps(bbMinYV0, zPixelDy, is_pVtxZ0);                      // bbMinYV0 = bbMinYV0*zPixelDy + pVtxZ[0]
			ra.free(bbMinYV0);
			cc.vfmadd132ps(zPlaneOffset, bbMinYV0, zPixelDx);                   // zPlaneOffset = bbMinXV0*zPixelDx + bbMinYV0

			X86Ymm zSubTileDx = ra.newYmm("zSubTileDx");
			X86Ymm zSubTileDy = ra.newYmm("zSubTileDy");
			cc.vmulps(zSubTileDx, zPixelDx, context.simd_f_sub_tile_width);     // zSubTileDx = zPixelDx * SUB_TILE_WIDTH
			cc.vmulps(zSubTileDy, zPixelDy, context.simd_f_sub_tile_height);    // zSubTileDy = zPixelDy * SUB_TILE_HEIGHT

			if (TEST_Z)
			{
				X86Ymm C_zerobits = ra.newYmm("ASM_RasterizeTriangleBatch::C_zerobits");
				cc.vpxor(C_zerobits, x86::ymm0, x86::ymm0);                     // amjit bug workaround: cc.vpxor(C_zerobits, C_zerobits, C_zerobits) breaks register allocator

				cc.vmaxps(zSubTileDx, zSubTileDx, C_zerobits);                  // zSubTileDx = max(0, zSubTileDx)
				ra.free(C_zerobits);
				cc.vmaxps(zSubTileDy, zSubTileDy, C_zerobits);                  // zSubTileDy = max(0, zSubTileDy)
				ra.free(zSubTileDy);
				cc.vaddps(zPlaneOffset, zPlaneOffset, zSubTileDx);              // zPlaneOffset += zSubTileDx
				ra.free(zSubTileDx);
				cc.vaddps(zPlaneOffset, zPlaneOffset, zSubTileDy);              // zPlaneOffset += zSubTileDy
			}
			else
			{
				X86Ymm C_zerobits = ra.newYmm("ASM_RasterizeTriangleBatch::C_zerobits");
				cc.vpxor(C_zerobits, x86::ymm0, x86::ymm0);                     // amjit bug workaround: cc.vpxor(C_zerobits, C_zerobits, C_zerobits) breaks register allocator

				cc.vminps(zSubTileDx, zSubTileDx, C_zerobits);                  // zSubTileDx = min(0, zSubTileDx)
				ra.free(C_zerobits);
				cc.vminps(zSubTileDy, zSubTileDy, C_zerobits);                  // zSubTileDy = min(0, zSubTileDy)
				ra.free(zSubTileDy);
				cc.vaddps(zPlaneOffset, zPlaneOffset, zSubTileDx);              // zPlaneOffset += zSubTileDx
				ra.free(zSubTileDx);
				cc.vaddps(zPlaneOffset, zPlaneOffset, zSubTileDy);              // zPlaneOffset += zSubTileDy
			}

			// Compute Zmin and Zmax for the triangle (used to narrow the range for difficult tiles)
			X86Ymm zMin = ra.newYmm("zMin");
			X86Ymm zMax = ra.newYmm("zMax");
			cc.vminps(zMin, is_pVtxZ0, pVtxZ[1]);                               // zMin = min(pVtxZ[0], pVtxZ[1])
			cc.vminps(zMin, zMin, pVtxZ[2]);                                    // zMin = min(zMin, pVtxZ[2])
			ra.free(is_pVtxZ0);
			cc.vmaxps(zMax, is_pVtxZ0, pVtxZ[1]);                               // zMax = min(pVtxZ[0], pVtxZ[1])
			cc.vmaxps(zMax, zMax, pVtxZ[2]);                                    // zMax = min(zMax, pVtxZ[1])

			// Write to stack so we can access simd lanes later
			ra.free(zPixelDx);
			cc.vmovaps(context.Mem_zPixelDx, zPixelDx);
			ra.free(zPixelDy);
			cc.vmovaps(context.Mem_zPixelDy, zPixelDy);
			ra.free(zPlaneOffset);
			cc.vmovaps(context.Mem_zPlaneOffset, zPlaneOffset);
			ra.free(zMin);
			cc.vmovaps(context.Mem_zMin, zMin);
			ra.free(zMax);
			cc.vmovaps(context.Mem_zMax, zMax);

		}

		//////////////////////////////////////////////////////////////////////////////
		// Set up texture (u, v) interpolation
		//////////////////////////////////////////////////////////////////////////////

		//__mw uPixelDx, uPixelDy, uPixel0, vPixelDx, vPixelDy, vPixel0, uDerivConsts[3], vDerivConsts[3];
		//if (TEXTURE_COORDINATES)
		//{
		//	InterpolationSetup(pVtxX, pVtxY, pVtxU, uPixelDx, uPixelDy, uPixel0);
		//	InterpolationSetup(pVtxX, pVtxY, pVtxV, vPixelDx, vPixelDy, vPixel0);

		//	uDerivConsts[0] = _mmw_fmsub_ps(uPixelDx, zPixelDy, _mmw_mul_ps(uPixelDy, zPixelDx));
		//	uDerivConsts[1] = _mmw_fmsub_ps(uPixelDx, zPixel0, _mmw_mul_ps(uPixel0, zPixelDx));
		//	uDerivConsts[2] = _mmw_fmsub_ps(uPixelDy, zPixel0, _mmw_mul_ps(uPixel0, zPixelDy));

		//	vDerivConsts[0] = _mmw_fmsub_ps(vPixelDx, zPixelDy, _mmw_mul_ps(vPixelDy, zPixelDx));
		//	vDerivConsts[1] = _mmw_fmsub_ps(vPixelDx, zPixel0, _mmw_mul_ps(vPixel0, zPixelDx));
		//	vDerivConsts[2] = _mmw_fmsub_ps(vPixelDy, zPixel0, _mmw_mul_ps(vPixel0, zPixelDy));
		//}

		//////////////////////////////////////////////////////////////////////////////
		// Sort vertices (v0 has lowest Y, and the rest is in winding order) 
		//////////////////////////////////////////////////////////////////////////////

#if PRECISE_COVERAGE != 0

#else // PRECISE_COVERAGE

		ASM_SortVertices(cc, ra, context, pVtxX, pVtxY);

		//////////////////////////////////////////////////////////////////////////////
		// Compute edges, also find the middle vertex and compute tile
		//////////////////////////////////////////////////////////////////////////////
		

		{

			X86Ymm e_pVtxX1 = ra.newYmm("E_pVtxX1");
			X86Ymm e_pVtxY1 = ra.newYmm("E_pVtxY1");
			X86Ymm e_pVtxX2 = ra.newYmm("E_pVtxX2");
			X86Ymm e_pVtxY2 = ra.newYmm("E_pVtxY2");
			cc.vmovaps(e_pVtxX1, pVtxX[1]);
			cc.vmovaps(e_pVtxY1, pVtxY[1]);
			cc.vmovaps(e_pVtxX2, pVtxX[2]);
			cc.vmovaps(e_pVtxY2, pVtxY[2]);

			// Compute edges
			X86Ymm edgeX[3] = { ra.newYmm("edgeX[0]"), ra.newYmm("edgeX[1]"), ra.newYmm("edgeX[2]") };
			X86Ymm edgeY[3] = { ra.newYmm("edgeY[0]"), ra.newYmm("edgeY[1]"), ra.newYmm("edgeY[2]") };

			cc.vsubps(edgeX[0], e_pVtxX1, pVtxX[0]);                            // edgeX[0] = pVtxX[1] - pVtxX[0]
			cc.vsubps(edgeX[1], e_pVtxX2, e_pVtxX1);                            // edgeX[1] = pVtxX[2] - pVtxX[1]
			ra.free(e_pVtxX2);
			cc.vsubps(edgeX[2], e_pVtxX2, pVtxX[0]);                            // edgeX[2] = pVtxX[2] - pVtxX[0]
			cc.vsubps(edgeY[0], e_pVtxY1, pVtxY[0]);                            // edgeY[0] = pVtxY[1] - pVtxY[0]
			cc.vsubps(edgeY[1], e_pVtxY2, e_pVtxY1);                            // edgeY[1] = pVtxY[2] - pVtxY[1]
			ra.free(e_pVtxY2);
			cc.vsubps(edgeY[2], e_pVtxY2, pVtxY[0]);                            // edgeY[2] = pVtxY[2] - pVtxY[0]

			// Classify if the middle vertex is on the left or right and compute its position

			X86Gp midVtxLeft = ra.newI32("midVtxLeft");
			cc.vmovmskps(midVtxLeft, edgeY[1]);                                 // midVtxLeft = movemask_ps(edgeY[i])
			ra.free(midVtxLeft);
			cc.mov(context.Mem_midVtxLeft, midVtxLeft);                         // Mem_midVtxLeft = midVtxLeft

			X86Ymm midPixelX = ra.newYmm("midPixelX");
			ra.free(e_pVtxX1);
			cc.vblendvps(midPixelX, e_pVtxX1, pVtxX[2], edgeY[1]);              // midPixelX = blendv_ps(pVtxX[1], pVtxX[2], edgeY[1])
			X86Ymm midPixelY = ra.newYmm("midPixelY");
			ra.free(e_pVtxY1);
			cc.vblendvps(midPixelY, e_pVtxY1, pVtxY[2], edgeY[1]);              // midPixelY = blendv_ps(pVtxY[1], pVtxY[2], edgeY[1])
			X86Ymm C_zerobits = ra.newYmm("ASM_RasterizeTriangleBatch::C_zerobits");
			cc.vpxor(C_zerobits, x86::ymm0, x86::ymm0);                         // amjit bug workaround: cc.vpxor(C_zerobits, C_zerobits, C_zerobits) breaks register allocator

			X86Ymm midTileY = ra.newYmm("midTileY");
			cc.vcvttps2dq(midTileY, midPixelY);                                 // midTileY = cvttps_epi32(midPixelY)
			cc.vpmaxsd(midTileY, midTileY, C_zerobits);                         // midTileY = max(midTileY, 0)
			cc.vpsrad(midTileY, midTileY, TILE_HEIGHT_SHIFT);                   // midtileY >>= TILE_HEIGHT_SHIFT

			ra.free(bbTileMaxY);
			X86Ymm bbMidTileY = ra.newYmm("bbMidTileY");
			cc.vpminsd(bbMidTileY, bbTileMaxY, midTileY);                       // bbMidTileY = min(bbTileMaxY, midTileY)
			ra.free(bbTileMinY);
			cc.vpmaxsd(bbMidTileY, bbMidTileY, bbTileMinY);                     // bbMidTileY = max(bbMidTileY, bbTileMinY)

			//////////////////////////////////////////////////////////////////////////////
			// Edge slope setup - Note we do not conform to DX/GL rasterization rules
			//////////////////////////////////////////////////////////////////////////////

			// Compute floating point slopes
			X86Ymm slope[3];
			for (int i = 0; i < 3; ++i)
			{
				ra.free(edgeX[i]);
				slope[i] = ra.newYmm("slope[0]");
				cc.vdivps(slope[i], edgeX[i], edgeY[i]);                        // slope[i] = edgeX[i] / edgeY[i]
			}
			ra.free(edgeY[2]);

			// Modify slope of horizontal edges to make sure they mask out pixels above/below the edge. The slope is set to screen
			// width to mask out all pixels above or below the horizontal edge. We must also add a small bias to acount for that
			// vertices may end up off screen due to clipping. We're assuming that the round off error is no bigger than 1.0
			ra.free(edgeY[0]);
			X86Ymm cnd0 = ra.newYmm("cnd0");
			cc.vcmpps(cnd0, edgeY[0], C_zerobits, _CMP_EQ_OQ);
			ra.free(cnd0);
			cc.vblendvps(slope[0], slope[0], context.simd_f_horizontal_slope_delta, cnd0);

			ra.free(C_zerobits);
			X86Ymm cnd1 = ra.newYmm("cnd1");
			cc.vcmpps(cnd1, edgeY[1], C_zerobits, _CMP_EQ_OQ);
			ra.free(cnd1);
			cc.vblendvps(slope[1], slope[1], context.simd_f_neg_horizontal_slope_delta, cnd1);

			// Convert floaing point slopes to fixed point
			X86Ymm slopeFP[3];
			for (int i = 0; i < 3; ++i)
			{
				ra.free(slope[i]);
				slopeFP[i] = ra.newYmm("slopeFP", i);
				cc.vmulps(slopeFP[i], slope[i], context.simd_f_shl_fp_bits);    // slopeFP[i] = slope[i] * (1 << FP_BITS)
				cc.vcvtps2dq(slopeFP[i], slopeFP[i]);                           // slopeFP[i] = cvtps_epi32(slopeFP[i])
			}

			X86Ymm C_onebits = ra.newYmm("ASM_TraverseTile::C_onebits");
			cc.vpcmpeqd(C_onebits, C_onebits, C_onebits);

			// Fan out edge slopes to avoid (rare) cracks at vertices. We increase right facing slopes
			// by 1 LSB, which results in overshooting vertices slightly, increasing triangle coverage.
			// e0 is always right facing, e1 depends on if the middle vertex is on the left or right
			X86Ymm signY1 = ra.newYmm("signY1");
			ra.free(C_onebits);
			cc.vxorps(signY1, edgeY[1], C_onebits);                             // signY1 = ~edgeY[1]
			cc.vpaddd(slopeFP[0], slopeFP[0], context.simd_i_1);                // slopeFP[0] += 1;
			ra.free(edgeY[1]);
			cc.vpsrld(signY1, signY1, 31);                                      // signY1 >>= 31
			ra.free(signY1);
			cc.vpaddd(slopeFP[1], slopeFP[1], signY1);                          // slopeFP[1] += signY1

			// Compute slope deltas for an SIMD_LANES scanline step (tile height)
			for (int i = 0; i < 3; ++i)
			{
				X86Ymm slopeTileDelta = ra.newYmm("slopeTileDelta");
				cc.vpslld(slopeTileDelta, slopeFP[i], TILE_HEIGHT_SHIFT);       // slopeTileDelta[i] = slopeFP[i] << TILE_HEIGHT_SHIFT
				cc.vmovaps(context.Mem_slopeFP[i], slopeFP[i]);
				ra.free(slopeTileDelta);
				cc.vmovaps(context.Mem_slopeTileDelta[i], slopeTileDelta);
			}

			// Compute edge events for the bottom of the bounding box, or for the middle tile in case of
			// the edge originating from the middle vertex.
			X86Ymm xDiffi[2] = { ra.newYmm("xDiffi[0]"), ra.newYmm("xDiffi[1]") };
			X86Ymm yDiffi[2] = { ra.newYmm("yDiffi[0]"), ra.newYmm("yDiffi[1]") };

			cc.vcvttps2dq(xDiffi[0], pVtxX[0]);                                 // xDiffi[0] = cvttps_epi32(pVtxX[0])
			cc.vpsubd(xDiffi[0], xDiffi[0], bbPixelMinX);                       // xDiffi[0] -= bbPixelMinX
			cc.vpslld(xDiffi[0], xDiffi[0], FP_BITS);                           // xDiffi[0] <<= FP_BITS

			ra.free(midPixelX);
			cc.vcvttps2dq(xDiffi[1], midPixelX);                                // xDiffi[1] = cvttps_epi32(midPixelX)
			ra.free(bbPixelMinX);
			cc.vpsubd(xDiffi[1], xDiffi[1], bbPixelMinX);                       // xDiffi[1] -= bbPixelMinX
			cc.vpslld(xDiffi[1], xDiffi[1], FP_BITS);                           // xDiffi[1] <<= FP_BITS

			cc.vcvttps2dq(yDiffi[0], pVtxY[0]);                                 // yDiffi[0] = cvttps_epi32(pVtxY[0])
			ra.free(bbPixelMinY);
			cc.vpsubd(yDiffi[0], yDiffi[0], bbPixelMinY);                       // yDiffi[0] -= bbPixelMinY

			X86Ymm midTilePixelY = ra.newYmm("midTilePixelX");
			ra.free(bbMidTileY);
			cc.vpslld(midTilePixelY, bbMidTileY, TILE_HEIGHT_SHIFT);            // midTilePixelY = bbMidTileY << TILE_HEIGHT_SHIFT
			ra.free(midPixelY);
			cc.vcvttps2dq(yDiffi[1], midPixelY);                                // yDiffi[1] = cvttps_epi32(midPixelY)
			ra.free(midTilePixelY);
			cc.vpsubd(yDiffi[1], yDiffi[1], midTilePixelY);                     // yDiffi[1] -= midTilePixelY

			for (int i = 0; i < 3; ++i)
			{
				ra.free(slopeFP[i]);
				X86Ymm eventStart = ra.newYmm("eventStart", i);
				cc.vpmulld(eventStart, slopeFP[i], yDiffi[i & 1]);              // eventStart = slopeFP[i] * yDiffi[i & 1]
				cc.vpsubd(eventStart, xDiffi[i & 1], eventStart);               // eventStart = xDiffi[i & 1] - eventStart
				ra.free(eventStart);
				cc.vmovaps(context.Mem_eventStart[i], eventStart);
			}
			ra.free(xDiffi[0]);
			ra.free(xDiffi[1]);
			ra.free(yDiffi[0]);
			ra.free(yDiffi[1]);
#endif

			//////////////////////////////////////////////////////////////////////////////
			// Split bounding box into bottom - middle - top region.
			//////////////////////////////////////////////////////////////////////////////

			X86Ymm bbMidIdx = ra.newYmm("bbMidIdx");

			ra.free(midTileY);
			cc.vpmulld(bbMidIdx, midTileY, context.simd_i_res_tiles_width);     // bbMidIdx = midTileY * simd_i_res_tiles_width
			ra.free(bbTileMinX);
			cc.vpaddd(bbMidIdx, bbMidIdx, bbTileMinX);                          // bbMidIdx += bbTileMinX

			// Move indices to stack so we can access individual lanes
			ra.free(bbMidIdx);
			cc.vmovaps(context.Mem_bbMidIdx, bbMidIdx);
		}

		ra.free(pVtxXPtr);
		ra.free(pVtxYPtr);
		ra.free(pVtxZPtr);

		//////////////////////////////////////////////////////////////////////////////
		// Loop over non-culled triangle and change SIMD axis to per-pixel
		//////////////////////////////////////////////////////////////////////////////
		cc.bind(L_TriLoopStart);
		cc.cmp(triMask, 0);
		cc.je(L_TriLoopExit);                                                               // while (triMask)
		{
			X86Gp triIdx = ra.getI32(x86::ecx, "triIdx");                                   // need to put this in CX register to allor shr instruction (below)
			X86Gp triMidVtxLeft = ra.newI32("triMidVtxLeft");

			cc.bsf(triIdx, triMask);                                                        // _BitScanForward
			X86Gp tmp = ra.newI32("tmp");
			cc.lea(tmp, X86Mem(triMask, -1));                                               // tmp = triMask - 1
			ra.free(tmp);
			cc.and_(triMask, tmp);                                                          // triMask &= (triMask - 1)

			cc.mov(triMidVtxLeft, context.Mem_midVtxLeft);                                  // triMidVtxLeft = Mem_midVtxLeft
			cc.shr(triMidVtxLeft, triIdx);                                                  // triMidVtxLeft >>= triIdx
			cc.and_(triMidVtxLeft, 1);                                                      // triMidVtxLeft &= 1

			// Make sure to update all pointers point to current triangle idx
			context.Mem_zMin.setIndex(triIdx, 2);
			context.Mem_zMax.setIndex(triIdx, 2);
			context.Mem_zPlaneOffset.setIndex(triIdx, 2);
			context.Mem_zPixelDx.setIndex(triIdx, 2);
			context.Mem_zPixelDy.setIndex(triIdx, 2);
			context.Mem_bbTileSizeX.setIndex(triIdx, 2);
			context.Mem_bbTileSizeY.setIndex(triIdx, 2);
			context.Mem_bbBottomIdx.setIndex(triIdx, 2);
			context.Mem_bbMidIdx.setIndex(triIdx, 2);
			context.Mem_bbTopIdx.setIndex(triIdx, 2);
			for (int i = 0; i < 3; ++i)
			{
#if PRECISE_COVERAGE != 0
#endif
				context.Mem_eventStart[i].setIndex(triIdx, 2);
				context.Mem_slopeFP[i].setIndex(triIdx, 2);
				context.Mem_slopeTileDelta[i].setIndex(triIdx, 2);
			}


			// Get Triangle Zmin zMax
			X86Ymm zTriMin = ra.newYmm("zTriMin");
			X86Ymm zTriMax = ra.newYmm("zTriMax");
			ASM_set1(cc, zTriMin, context.Mem_zMin);                                       // zTriMin = zMin.m_f32[triIdx]
			ASM_set1(cc, zTriMax, context.Mem_zMax);                                       // zTriMax = zMax.m_f32[triIdx]
			ra.free(zTriMin);
			cc.vmovaps(context.Mem_zTriMin, zTriMin);                                      // constant per triangle, spill to stack
			ra.free(zTriMax);
			cc.vmovaps(context.Mem_zTriMax, zTriMax);                                      // constant per triangle, spill to stack

			// Setup Zmin value for first set of 8x4 subtiles
			X86Ymm zTri0 = ra.newYmm("zTri0");
			X86Ymm zTriPixelDx = ra.newYmm("zTriPixelDx");
			X86Ymm zTriPixelDy = ra.newYmm("zTriPixelDy");
			ASM_set1(cc, zTri0, context.Mem_zPlaneOffset);                                  // zTri0 = zPlaneOffset.m_f32[triIdx]
			ASM_set1(cc, zTriPixelDx, context.Mem_zPixelDx);                                // zTriPixelDx = zPixelDx.m_f32[triIdx]
			ASM_set1(cc, zTriPixelDy, context.Mem_zPixelDy);                                // zTriPixelDy = zPixelDy.m_f32[triIdx]
			
			cc.vfmadd231ps(zTri0, zTriPixelDx, context.simd_f_sub_tile_col_offset);         // zTri0 += zTriPixelDx * simd_f_sub_tile_col_offset
			cc.vfmadd231ps(zTri0, zTriPixelDy, context.simd_f_sub_tile_row_offset);         // zTri0 += zTriPixelDy * simd_f_sub_tile_row_offset

			X86Ymm zTriTileDx = ra.newYmm("zTriTileDx");
			X86Ymm zTriTileDy = ra.newYmm("zTriTileDy");
			ra.free(zTriPixelDx);
			cc.vmulps(zTriTileDx, zTriPixelDx, context.simd_f_tile_width);                  // zTriTileDx = zTriPixelDx * TILE_WIDTH
			ra.free(zTriPixelDy);
			cc.vmulps(zTriTileDy, zTriPixelDy, context.simd_f_tile_height);                 // zTriTileDy = zTriPixelDy * TILE_HEIGHT
			ra.free(zTriTileDx);
			cc.vmovaps(context.Mem_zTriTileDx, zTriTileDx);                                 // constant per triangle, spill to stack
			ra.free(zTriTileDy);
			cc.vmovaps(context.Mem_zTriTileDy, zTriTileDy);                                 // constant per triangle, spill to stack

			// Set up tiled versions of triangle slopes
			for (int i = 0; i < 3; ++i)
			{
				X86Ymm triSlopeTileDelta = ra.newYmm("triSlopeTileDelta");
				ASM_set1(cc, triSlopeTileDelta, context.Mem_slopeTileDelta[i]);             // triSlopeTileDelta = _mmw_set1_epi32(slopeTileDelta[i].m_i32[triIdx]);
				ra.free(triSlopeTileDelta);
				cc.vmovaps(context.Mem_triSlopeTileDelta[i], triSlopeTileDelta);            // constant per triangle, so spill to stack
			}

			// Get dimension of bounding box bottom, mid & top segments
			X86Gp tmpAddr = ra.newI64("tmpAddr");
			cc.xor_(tmpAddr, tmpAddr);                                                      // tmpAddr = 0
			cc.mov(tmpAddr.r32(), context.Mem_bbMidIdx);                                    // tmpAddr = (int64_t)bbMidIdx.m_i32[triIdx]
			cc.imul(tmpAddr, sizeof(ZTile));                                                // tmpAddr *= sizeof(ZTile)
			cc.add(tmpAddr, context.i64_msoc_buffer_ptr);                                   // tmpAddr += i64_msoc_buffer_ptr
			ra.free(tmpAddr);
			cc.mov(context.Mem_tileMidRowAddr, tmpAddr);                                    // tileMidRowAddr = tmpAddr (spill to stack)

			X86Gp tileEndRowAddr = ra.newI64("tileEndRowAddr");
			cc.xor_(tileEndRowAddr, tileEndRowAddr);                                        // tileEndRowAddr = 0
			cc.mov(tileEndRowAddr.r32(), context.Mem_bbTopIdx);                             // tileEndRowAddr = (int64_t)bbTopIdx.m_i32[triIdx]
			cc.imul(tileEndRowAddr, sizeof(ZTile));                                         // tileEndRowAddr *= sizeof(ZTile)
			cc.add(tileEndRowAddr, context.i64_msoc_buffer_ptr);                            // tileEndRowAddr += i64_msoc_buffer_ptr

			X86Gp tileRowAddr = ra.newI64("tileRowAddr");
			cc.xor_(tileRowAddr, tileRowAddr);                                              // tileRowAddr = 0
			cc.mov(tileRowAddr.r32(), context.Mem_bbBottomIdx);                             // tileRowAddr = (int64_t)bbBottomIdx.m_i32[triIdx]
			cc.imul(tileRowAddr, sizeof(ZTile));                                            // tileRowAddr *= sizeof(ZTile)
			cc.add(tileRowAddr, context.i64_msoc_buffer_ptr);                               // tileRowAddr += i64_msoc_buffer_ptr

			X86Gp bbWidth = ra.newI32("bbWidth");
			X86Gp bbHeight = ra.newI32("bbHeight");

			cc.mov(bbWidth, context.Mem_bbTileSizeX);                                        // bbWidth = bbTileSizeX.m_i32[triIdx]
			cc.mov(bbHeight, context.Mem_bbTileSizeY);                                       // bbHeight = bbTileSizeY.m_i32[triIdx]

			//// Setup texture (u,v) interpolation parameters, TODO: Simdify
			//TextureInterpolants texInterpolants;
			//if (TEXTURE_COORDINATES)
			//{
			//	texInterpolants.zInterpolant.mDx = _mmw_set1_ps(simd_f32(zPixelDx)[triIdx]);
			//	texInterpolants.zInterpolant.mDy = _mmw_set1_ps(simd_f32(zPixelDy)[triIdx]);
			//	texInterpolants.zInterpolant.mVal0 = _mmw_set1_ps(simd_f32(zPixel0)[triIdx]);
			//	texInterpolants.uInterpolant.mDx = _mmw_set1_ps(simd_f32(uPixelDx)[triIdx]);
			//	texInterpolants.uInterpolant.mDy = _mmw_set1_ps(simd_f32(uPixelDy)[triIdx]);
			//	texInterpolants.uInterpolant.mVal0 = _mmw_set1_ps(simd_f32(uPixel0)[triIdx]);
			//	texInterpolants.vInterpolant.mDx = _mmw_set1_ps(simd_f32(vPixelDx)[triIdx]);
			//	texInterpolants.vInterpolant.mDy = _mmw_set1_ps(simd_f32(vPixelDy)[triIdx]);
			//	texInterpolants.vInterpolant.mVal0 = _mmw_set1_ps(simd_f32(vPixel0)[triIdx]);

			//	texInterpolants.uDerivConsts[0] = _mmw_set1_ps(simd_f32(uDerivConsts[0])[triIdx]);
			//	texInterpolants.uDerivConsts[1] = _mmw_set1_ps(simd_f32(uDerivConsts[1])[triIdx]);
			//	texInterpolants.uDerivConsts[2] = _mmw_set1_ps(simd_f32(uDerivConsts[2])[triIdx]);

			//	texInterpolants.vDerivConsts[0] = _mmw_set1_ps(simd_f32(vDerivConsts[0])[triIdx]);
			//	texInterpolants.vDerivConsts[1] = _mmw_set1_ps(simd_f32(vDerivConsts[1])[triIdx]);
			//	texInterpolants.vDerivConsts[2] = _mmw_set1_ps(simd_f32(vDerivConsts[2])[triIdx]);
			//}

			cc.cmp(bbWidth, BIG_TRIANGLE);
			cc.jle(L_SmallTriangle);
			ra.free(bbHeight);
			cc.cmp(bbHeight, BIG_TRIANGLE);
			cc.jle(L_SmallTriangle);                                                        // if (bbWidth > BIG_TRIANGLE && bbHeight > BIG_TRIANGLE)
			ra.free(triMidVtxLeft);
			{
				// For big triangles we use a more expensive but tighter traversal algorithm
//#if PRECISE_COVERAGE != 0
//#else
				Label L_else = cc.newLabel(); 
				Label L_exit = cc.newLabel();
				cc.cmp(triMidVtxLeft, 0);
				cc.je(L_else);  // Mid vtx is on the left
				{
					ASM_RasterizeTriangle(cc, ra, TEST_Z, 1, 0, L_VisibleExit, context, triIdx, bbWidth, tileRowAddr, tileEndRowAddr, zTri0);
					cc.jmp(L_exit);
				}
				cc.bind(L_else); // Mid vtx is on the right
				{
					ASM_RasterizeTriangle(cc, ra, TEST_Z, 1, 1, L_VisibleExit, context, triIdx, bbWidth, tileRowAddr, tileEndRowAddr, zTri0);
				}
				cc.bind(L_exit);
//#endif
				cc.jmp(L_EndSmallTriangle);
			}
			cc.bind(L_SmallTriangle);                                                       // else
			{
//#if PRECISE_COVERAGE != 0
//#else
				Label L_else = cc.newLabel();
				Label L_exit = cc.newLabel();
				cc.cmp(triMidVtxLeft, 0);
				cc.jne(L_else);  // Mid vtx is on the left
				{
					ASM_RasterizeTriangle(cc, ra, TEST_Z, 0, 0, L_VisibleExit, context, triIdx, bbWidth, tileRowAddr, tileEndRowAddr, zTri0);
					cc.jmp(L_exit);
				}
				cc.bind(L_else); // Mid vtx is on the right
				{
					ASM_RasterizeTriangle(cc, ra, TEST_Z, 0, 1, L_VisibleExit, context, triIdx, bbWidth, tileRowAddr, tileEndRowAddr, zTri0);
				}
				cc.bind(L_exit);
//#endif
			}
			cc.bind(L_EndSmallTriangle);

			ra.free(tileEndRowAddr);
			ra.free(bbWidth);
			ra.free(tileRowAddr);
			ra.free(zTri0);
			ra.free(triIdx);
			cc.jmp(L_TriLoopStart);
		}
		cc.bind(L_TriLoopExit);
		ra.free(triMask);

		// Went through the whole loop, means everytrhing was occluded (if we're testing)
		cc.mov(x86::rax, TEST_Z ? OCCLUDED : VISIBLE);
		cc.jmp(L_Exit);

		// Everything was view culled
		cc.bind(L_ViewCulledExit);
		cc.mov(x86::rax, VIEW_CULLED);

		// Early exit for when using TestZ
		if (TEST_Z)
		{
			cc.jmp(L_Exit);
			cc.bind(L_VisibleExit);
			cc.mov(x86::rax, VISIBLE);
		}
		cc.bind(L_Exit);
		
		ra.CleanupStack();
	}

	void GEN_RenderTriangles(
		X86Compiler &cc,
		ASM_RegAllocator &ra,
		int TEST_Z,
		int FAST_GATHER,
		X86Gp &msocPtr,
		X86Gp &inVtxPtr,
		X86Gp &inTrisPtr,
		const X86Mem &nTris,
		X86Gp &modelToClipMatrixPtr,
		X86Gp &bfWinding,
		X86Gp &clipPlaneMask,
		X86Gp &vtxLayoutPtr)
	{
#if PRECISE_COVERAGE != 0
#endif
		X86Gp clipHead = ra.newI32("clipHead");
		X86Gp clipTail = ra.newI32("clipTail");
		X86Gp triIdx = ra.newI32("triIdx");

		cc.xor_(clipHead, clipHead);
		cc.xor_(clipTail, clipTail);
		cc.xor_(triIdx, triIdx);

		Label L_TriLoopPrologue = cc.newLabel();
		Label L_TriLoopStart = cc.newLabel();
		Label L_TriLoopExit = cc.newLabel();
		Label L_FetchClip = cc.newLabel();
		Label L_FetchNoClip = cc.newLabel();
		Label L_FetchExit = cc.newLabel();

		cc.bind(L_TriLoopPrologue);
		cc.cmp(triIdx, nTris);
		cc.jl(L_TriLoopStart);
		cc.cmp(clipHead, clipTail);
		cc.jne(L_TriLoopStart);
		cc.jmp(L_TriLoopExit);
		cc.bind(L_TriLoopStart);                                                     // while(triIdx < nTris || clipHead != clipTail)
		{
			X86Ymm vtxX[3], vtxY[3], vtxZ[3];
			for (int i = 0; i < 3; ++i)
			{
				vtxX[i] = ra.newYmm("vtxX", i);
				vtxY[i] = ra.newYmm("vtxY", i);
				vtxZ[i] = ra.newYmm("vtxZ", i);
			}

			cc.cmp(clipHead, clipTail);
			cc.je(L_FetchNoClip);
			cc.bind(L_FetchClip);                                                    // if (clipHead != clipTail)
			{
				// Fetch vertices from clip buffer, and pad with vertices to fill out SIMD width
				X86Gp clippedTrisA = ra.newI32("clippedTrisA");
				X86Gp clippedTrisB = ra.newI32("clippedTrisB");
				X86Gp clippedTris = ra.newI32("clippedTris");
				cc.mov(clippedTrisA, clipHead);
				cc.sub(clippedTrisA, clipTail);
				cc.mov(clippedTrisB, clipHead);


				cc.jmp(L_FetchExit);
			}
			cc.bind(L_FetchNoClip);                                                   // else
			{
			}
			cc.bind(L_FetchExit);


		}
		cc.bind(L_TriLoopExit);

//		const unsigned int *inTrisPtr = inTris;
//		int cullResult = CullingResult::VIEW_CULLED;
//		bool fastGather = !TEXTURE_COORDINATES && vtxLayout.mStride == 16 && vtxLayout.mOffsetY == 4 && vtxLayout.mOffsetW == 12;
//
//		while (triIndex < nTris || clipHead != clipTail)
//		{
//			//////////////////////////////////////////////////////////////////////////////
//			// Assemble triangles from the index list
//			//////////////////////////////////////////////////////////////////////////////
//			__mw vtxX[3], vtxY[3], vtxW[3], vtxU[3], vtxV[3];
//			unsigned int triMask = SIMD_ALL_LANES_MASK, triClipMask = SIMD_ALL_LANES_MASK;
//
//			int numLanes = SIMD_LANES;
//			if (clipHead != clipTail)
//			{
//				int clippedTris = clipHead > clipTail ? clipHead - clipTail : MAX_CLIPPED + clipHead - clipTail;
//				clippedTris = min(clippedTris, SIMD_LANES);
//
//				// Fill out SIMD registers by fetching more triangles.
//				numLanes = max(0, min(SIMD_LANES - clippedTris, nTris - triIndex));
//				if (numLanes > 0) {
//					if (fastGather)
//						GatherVerticesFast(vtxX, vtxY, vtxW, inVtx, inTrisPtr, numLanes);
//					else
//						GatherVertices<TEXTURE_COORDINATES>(vtxX, vtxY, vtxW, vtxU, vtxV, inVtx, inTrisPtr, numLanes, vtxLayout);
//
//					TransformVerts(vtxX, vtxY, vtxW, modelToClipMatrix);
//				}
//
//				for (int clipTri = numLanes; clipTri < numLanes + clippedTris; clipTri++)
//				{
//					int triIdx = clipTail * 3;
//					for (int i = 0; i < 3; i++)
//					{
//						simd_f32(vtxX[i])[clipTri] = simd_f32(clipVtxBuffer[triIdx + i])[0];
//						simd_f32(vtxY[i])[clipTri] = simd_f32(clipVtxBuffer[triIdx + i])[1];
//						simd_f32(vtxW[i])[clipTri] = simd_f32(clipVtxBuffer[triIdx + i])[2];
//						if (TEXTURE_COORDINATES)
//						{
//							simd_f32(vtxU[i])[clipTri] = simd_f32(clipTexBuffer[triIdx + i])[0];
//							simd_f32(vtxV[i])[clipTri] = simd_f32(clipTexBuffer[triIdx + i])[1];
//						}
//					}
//					clipTail = (clipTail + 1) & (MAX_CLIPPED - 1);
//				}
//
//				triIndex += numLanes;
//				inTrisPtr += numLanes * 3;
//
//				triMask = (1U << (clippedTris + numLanes)) - 1;
//				triClipMask = (1U << numLanes) - 1; // Don't re-clip already clipped triangles
//			}
//			else
//			{
//				numLanes = min(SIMD_LANES, nTris - triIndex);
//				triMask = (1U << numLanes) - 1;
//				triClipMask = triMask;
//
//				if (fastGather)
//					GatherVerticesFast(vtxX, vtxY, vtxW, inVtx, inTrisPtr, numLanes);
//				else
//					GatherVertices<TEXTURE_COORDINATES>(vtxX, vtxY, vtxW, vtxU, vtxV, inVtx, inTrisPtr, numLanes, vtxLayout);
//
//				TransformVerts(vtxX, vtxY, vtxW, modelToClipMatrix);
//				triIndex += SIMD_LANES;
//				inTrisPtr += SIMD_LANES * 3;
//			}
//
//			//////////////////////////////////////////////////////////////////////////////
//			// Clip transformed triangles
//			//////////////////////////////////////////////////////////////////////////////
//
//			if (clipPlaneMask != ClipPlanes::CLIP_PLANE_NONE)
//				ClipTriangleAndAddToBuffer<TEXTURE_COORDINATES>(vtxX, vtxY, vtxW, vtxU, vtxV, clipVtxBuffer, clipTexBuffer, clipHead, triMask, triClipMask, clipPlaneMask);
//
//			if (triMask == 0x0)
//				continue;
//
//			//////////////////////////////////////////////////////////////////////////////
//			// Project, transform to screen space and perform backface culling. Note
//			// that we use z = 1.0 / vtx.w for depth, which means that z = 0 is far and
//			// z = 1 is near. We must also use a greater than depth test, and in effect
//			// everything is reversed compared to regular z implementations.
//			//////////////////////////////////////////////////////////////////////////////
//
//			__mw pVtxX[3], pVtxY[3], pVtxZ[3], pVtxU[3], pVtxV[3];
//#if PRECISE_COVERAGE != 0
//			__mwi ipVtxX[3], ipVtxY[3];
//			ProjectVertices(ipVtxX, ipVtxY, pVtxX, pVtxY, pVtxZ, vtxX, vtxY, vtxW);
//#else
//			ProjectVertices(pVtxX, pVtxY, pVtxZ, vtxX, vtxY, vtxW);
//#endif
//			if (TEXTURE_COORDINATES)
//				ProjectTexCoords(pVtxU, pVtxV, pVtxZ, vtxU, vtxV);
//
//			// Perform backface test.
//			__mw triArea1 = _mmw_mul_ps(_mmw_sub_ps(pVtxX[1], pVtxX[0]), _mmw_sub_ps(pVtxY[2], pVtxY[0]));
//			__mw triArea2 = _mmw_mul_ps(_mmw_sub_ps(pVtxX[0], pVtxX[2]), _mmw_sub_ps(pVtxY[0], pVtxY[1]));
//			__mw triArea = _mmw_sub_ps(triArea1, triArea2);
//			__mw ccwMask = _mmw_cmpgt_ps(triArea, _mmw_setzero_ps());
//
//#if PRECISE_COVERAGE != 0
//			triMask &= CullBackfaces(ipVtxX, ipVtxY, pVtxX, pVtxY, pVtxZ, ccwMask, bfWinding);
//#else
//			triMask &= CullBackfaces(pVtxX, pVtxY, pVtxZ, ccwMask, bfWinding);
//#endif
//
//			if (triMask == 0x0)
//				continue;
//
//			//////////////////////////////////////////////////////////////////////////////
//			// Setup and rasterize a SIMD batch of triangles
//			//////////////////////////////////////////////////////////////////////////////
//#if PRECISE_COVERAGE != 0
//			cullResult &= RasterizeTriangleBatch<TEST_Z, TEXTURE_COORDINATES>(ipVtxX, ipVtxY, pVtxX, pVtxY, pVtxZ, pVtxU, pVtxV, triMask, &mFullscreenScissor, texture);
//#else
//#define USE_ASM
//#ifdef USE_ASM
//			mASMRasterizeTriangleBatch(this, nullptr, nullptr, pVtxX, pVtxY, pVtxZ, nullptr, nullptr, triMask, &mFullscreenScissor);
//#else
//			cullResult &= RasterizeTriangleBatch<TEST_Z, TEXTURE_COORDINATES>(pVtxX, pVtxY, pVtxZ, pVtxU, pVtxV, triMask, &mFullscreenScissor, texture);
//#endif
//#endif
//
//			if (TEST_Z && cullResult == CullingResult::VISIBLE) {
//#if PRECISE_COVERAGE != 0
//				_MM_SET_ROUNDING_MODE(originalRoundingMode);
//#endif
//				return CullingResult::VISIBLE;
//			}
//		}
//
//#if PRECISE_COVERAGE != 0
//#endif
//		return (CullingResult)cullResult;
	}

	class PrintErrorHandler : public asmjit::ErrorHandler {
	public:
		// Return `true` to set last error to `err`, return `false` to do nothing.
		bool handleError(asmjit::Error err, const char* message, asmjit::CodeEmitter* origin) override {
			fprintf(stderr, "ERROR: %s\n", message);
			return false;
		}
	};

	void GenerateASM()
	{
		FILE *f = fopen("test.asm", "w+");
		FileLogger fl(f);
		PrintErrorHandler eh;

		CodeHolder code;
		code.init(mRuntime.getCodeInfo());
		code.setLogger(&fl);
		code.setErrorHandler(&eh);

		X86Compiler cc(&code);

		ASM_RegAllocator ra(cc);

#if 1

		//////////////////////////////////////////////////////////////////////////////////////////////
		// Push non-volatile registers. TODO: don't push unused
		//////////////////////////////////////////////////////////////////////////////////////////////
		cc.addFunc(FuncSignature0<int>());
		cc.push(x86::rbx);
		cc.push(x86::rbp);
		cc.push(x86::rsi);
		cc.push(x86::rdi);
		cc.push(x86::r12);
		cc.push(x86::r13);
		cc.push(x86::r14);
		cc.push(x86::r15); // 8*8 = 64, Stack is still 16 byte aligned at this point
		cc.sub(x86::rsp, 168);
		for (int i = 6; i < 16; ++i)
			cc.vmovaps(x86::dqword_ptr(x86::rsp, (15 - i) * sizeof(__m128)), x86OpData.xmm[i]);

		int StackOffset = 40 + 168 + 64; // shadow space + xmm space + gp space

		X86Gp msocPtr = ra.getI64(x86::rcx, "msocPtr");
		X86Gp ipVtxXPtr = ra.getI64(x86::rdx, "ipVtxXPtr");
		X86Gp ipVtxYPtr = ra.getI64(x86::r8, "ipVtxYPtr");
		X86Gp pVtxXPtr = ra.getI64(x86::r9, "pVtxXPtr");
		X86Gp pVtxYPtr = ra.newI64("pVtxYPtr");
		X86Gp pVtxZPtr = ra.newI64("pVtxZPtr");
		X86Gp pVtxUPtr = ra.newI64("pVtxUPtr");
		X86Gp pVtxVPtr = ra.newI64("pVtxVPtr");
		X86Gp triMask = ra.newI32("triMask");
		X86Gp scissorPtr = ra.newI64("scissorPtr");

		cc.mov(pVtxYPtr, x86::qword_ptr(x86::rsp, StackOffset));
		cc.mov(pVtxZPtr, x86::qword_ptr(x86::rsp, StackOffset + 8));
		cc.mov(pVtxUPtr, x86::qword_ptr(x86::rsp, StackOffset + 16));
		cc.mov(pVtxVPtr, x86::qword_ptr(x86::rsp, StackOffset + 24));
		cc.mov(triMask, x86::dword_ptr(x86::rsp, StackOffset + 32));
		cc.mov(scissorPtr, x86::qword_ptr(x86::rsp, StackOffset + 40));

		ASM_RasterizeTriangleBatch(cc, ra, 0, msocPtr, ipVtxXPtr, ipVtxYPtr, pVtxXPtr, pVtxYPtr, pVtxZPtr, pVtxUPtr, pVtxVPtr, triMask, scissorPtr);

		for (int i = 6; i < 16; ++i)
			cc.vmovaps(x86::dqword_ptr(x86::rsp, (15 - i) * sizeof(__m128)), x86OpData.xmm[i]);
		cc.add(x86::rsp, 168);
		cc.pop(x86::r15);
		cc.pop(x86::r14);
		cc.pop(x86::r13);
		cc.pop(x86::r12);
		cc.pop(x86::rdi);
		cc.pop(x86::rsi);
		cc.pop(x86::rbp);
		cc.pop(x86::rbx);
		cc.ret(x86::eax);
		cc.endFunc();
#else
		//////////////////////////////////////////////////////////////////////////////////////////////
		// Setup function
		//////////////////////////////////////////////////////////////////////////////////////////////
		FuncDetail func;
		func.init(FuncSignature10<int, MaskedOcclusionCullingPrivate *, __mwi*, __mwi*, __mw*, __mw*, __mw*, __mw *, __mw *, int, ScissorRect*>());
		//func.init(FuncSignature1<int, MaskedOcclusionCullingPrivate *>());

		FuncFrameInfo fnInfo;
		fnInfo.setDirtyRegs(X86Reg::kKindGp, ~0);  // TODO: pushing too much data
		fnInfo.setDirtyRegs(X86Reg::kKindVec, ~0); // TODO: pushing too much data

		//////////////////////////////////////////////////////////////////////////////////////////////
		// Setup input arguments
		//////////////////////////////////////////////////////////////////////////////////////////////
		
		X86Gp msocPtr = ra.newI64("msocPtr");      // rax
		X86Gp ipVtxXPtr = ra.newI64("ipVtxXPtr");  // rdx
		X86Gp ipVtxYPtr = ra.newI64("ipVtxYPtr");  // rbx
		X86Gp pVtxXPtr = ra.newI64("pVtxXPtr");    // rbp
		X86Gp pVtxYPtr = ra.newI64("pVtxYPtr");
		X86Gp pVtxZPtr = ra.newI64("pVtxZPtr");
		X86Gp pVtxUPtr = ra.newI64("pVtxUPtr");
		X86Gp pVtxVPtr = ra.newI64("pVtxVPtr");
		X86Gp triMask = ra.newI32("triMask");
		X86Gp scissorPtr = ra.newI64("scissorPtr");

		FuncArgsMapper fnArgs(&func);
		//fnArgs.assignAll(msocPtr, ipVtxXPtr, ipVtxYPtr, pVtxXPtr, pVtxYPtr);
		//fnArgs.assign(0, msocPtr);
		//fnArgs.assign(1, ipVtxXPtr);
		//fnArgs.assign(2, ipVtxYPtr);
		//fnArgs.assign(3, pVtxXPtr);
		//fnArgs.assign(4, pVtxYPtr);
		//fnArgs.assign(5, pVtxZPtr);
		//fnArgs.assign(6, pVtxUPtr);
		//fnArgs.assign(7, pVtxVPtr);
		//fnArgs.assign(8, triMask);
		//fnArgs.assign(9, scissorPtr);
		fnArgs.updateFrameInfo(fnInfo);

		FuncFrameLayout fnLayout;
		fnLayout.init(func, fnInfo);

		//////////////////////////////////////////////////////////////////////////////////////////////
		// Generate asm
		//////////////////////////////////////////////////////////////////////////////////////////////

		FuncUtils::emitProlog(&cc, fnLayout);
		FuncUtils::allocArgs(&cc, fnLayout, fnArgs);

		// Generate rasterizer
		//ASM_RasterizeTriangleBatch(cc, ra, 0, msocPtr, ipVtxXPtr, ipVtxYPtr, pVtxXPtr, pVtxYPtr, pVtxZPtr, pVtxUPtr, pVtxVPtr, triMask, scissorPtr);

		FuncUtils::emitEpilog(&cc, fnLayout);
#endif

		cc.finalize();

		mRuntime.add(&mASMRasterizeTriangleBatch, &code);
		fclose(f);
	}

	~MaskedOcclusionCullingPrivate() override
	{
#if QUERY_DEBUG_BUFFER != 0
		if (mQueryDebugBuffer != nullptr)
			mAlignedFreeCallback(mQueryDebugBuffer);
		mQueryDebugBuffer = nullptr;
#endif
		if (mMaskedHiZBuffer != nullptr)
			mAlignedFreeCallback(mMaskedHiZBuffer);
		mMaskedHiZBuffer = nullptr;
	}

	void SetResolution(unsigned int width, unsigned int height) override
	{
		// Resolution must be a multiple of the subtile size
		assert(width % SUB_TILE_WIDTH == 0 && height % SUB_TILE_HEIGHT == 0);
#if PRECISE_COVERAGE == 0
		// Test if combination of resolution & SLOPE_FP_BITS bits may cause 32-bit overflow. Note that the maximum resolution estimate
		// is only an estimate (not conservative). It's advicable to stay well below the limit.
		assert(width < ((1U << 31) - 1U) / ((1U << FP_BITS) * (TILE_HEIGHT + (unsigned int)(GUARD_BAND_PIXEL_SIZE + 1.0f))) - (2U * (unsigned int)(GUARD_BAND_PIXEL_SIZE + 1.0f)));
#endif

#if QUERY_DEBUG_BUFFER != 0
		// Delete debug buffer
		if (mQueryDebugBuffer != nullptr)
			mAlignedFreeCallback(mQueryDebugBuffer);
		mQueryDebugBuffer = nullptr;
#endif

		// Delete current masked hierarchical Z buffer
		if (mMaskedHiZBuffer != nullptr)
			mAlignedFreeCallback(mMaskedHiZBuffer);
		mMaskedHiZBuffer = nullptr;

		// Setup various resolution dependent constant values
		mWidth = (int)width;
		mHeight = (int)height;
		mTilesWidth = (int)(width + TILE_WIDTH - 1) >> TILE_WIDTH_SHIFT;
		mTilesHeight = (int)(height + TILE_HEIGHT - 1) >> TILE_HEIGHT_SHIFT;
		mCenterX = _mmw_set1_ps((float)mWidth  * 0.5f);
		mCenterY = _mmw_set1_ps((float)mHeight * 0.5f);
		mICenter = _mm_setr_ps((float)mWidth * 0.5f, (float)mWidth * 0.5f, (float)mHeight * 0.5f, (float)mHeight * 0.5f);
		mHalfWidth = _mmw_set1_ps((float)mWidth  * 0.5f);
#if USE_D3D != 0
		mHalfHeight = _mmw_set1_ps((float)-mHeight * 0.5f);
		mIHalfSize = _mm_setr_ps((float)mWidth * 0.5f, (float)mWidth * 0.5f, (float)-mHeight * 0.5f, (float)-mHeight * 0.5f);
#else
		mHalfHeight = _mmw_set1_ps((float)mHeight * 0.5f);
		mIHalfSize = _mm_setr_ps((float)mWidth * 0.5f, (float)mWidth * 0.5f, (float)mHeight * 0.5f, (float)mHeight * 0.5f);
#endif
		mIScreenSize = _mm_setr_epi32(mWidth - 1, mWidth - 1, mHeight - 1, mHeight - 1);

		// Setup a full screen scissor rectangle
		mFullscreenScissor.mMinX = 0;
		mFullscreenScissor.mMinY = 0;
		mFullscreenScissor.mMaxX = mTilesWidth << TILE_WIDTH_SHIFT;
		mFullscreenScissor.mMaxY = mTilesHeight << TILE_HEIGHT_SHIFT;

		// Adjust clip planes to include a small guard band to avoid clipping leaks
		float guardBandWidth = (2.0f / (float)mWidth) * GUARD_BAND_PIXEL_SIZE;
		float guardBandHeight = (2.0f / (float)mHeight) * GUARD_BAND_PIXEL_SIZE;
		mCSFrustumPlanes[1] = _mm_setr_ps(1.0f - guardBandWidth, 0.0f, 1.0f, 0.0f);
		mCSFrustumPlanes[2] = _mm_setr_ps(-1.0f + guardBandWidth, 0.0f, 1.0f, 0.0f);
		mCSFrustumPlanes[3] = _mm_setr_ps(0.0f, 1.0f - guardBandHeight, 1.0f, 0.0f);
		mCSFrustumPlanes[4] = _mm_setr_ps(0.0f, -1.0f + guardBandHeight, 1.0f, 0.0f);

		// Allocate masked hierarchical Z buffer (if zero size leave at nullptr)
		if (mTilesWidth * mTilesHeight > 0)
			mMaskedHiZBuffer = (ZTile *)mAlignedAllocCallback(64, sizeof(ZTile) * mTilesWidth * mTilesHeight);

#if QUERY_DEBUG_BUFFER != 0
		// Allocate debug buffer
		if (mTilesWidth * mTilesHeight > 0)
			mQueryDebugBuffer = (__mwi*)mAlignedAllocCallback(64, sizeof(__mwi) * mTilesWidth * mTilesHeight);
#endif
	}

	void GetResolution(unsigned int &width, unsigned int &height) override
	{
		width = mWidth;
		height = mHeight;
	}

	void ComputeBinWidthHeight(unsigned int nBinsW, unsigned int nBinsH, unsigned int & outBinWidth, unsigned int & outBinHeight) override
	{
		outBinWidth = (mWidth / nBinsW) - ((mWidth / nBinsW) % TILE_WIDTH);
		outBinHeight = (mHeight / nBinsH) - ((mHeight / nBinsH) % TILE_HEIGHT);
	}

	void SetNearClipPlane(float nearDist) override
	{
		// Setup the near frustum plane
		mNearDist = nearDist;
		mCSFrustumPlanes[0] = _mm_setr_ps(0.0f, 0.0f, 1.0f, -nearDist);
	}

	float GetNearClipPlane() override
	{
		return mNearDist;
	}

	void ClearBuffer() override
	{
		assert(mMaskedHiZBuffer != nullptr);

		// Iterate through all depth tiles and clear to default values
		for (int i = 0; i < mTilesWidth * mTilesHeight; i++)
		{
			mMaskedHiZBuffer[i].mMask = _mmw_setzero_epi32();

			// Clear z0 to beyond infinity to ensure we never merge with clear data
			mMaskedHiZBuffer[i].mZMin[0] = _mmw_set1_ps(-1.0f);
#if QUICK_MASK != 0
			// Clear z1 to nearest depth value as it is pushed back on each update
			mMaskedHiZBuffer[i].mZMin[1] = _mmw_set1_ps(FLT_MAX);
#else
			mMaskedHiZBuffer[i].mZMin[1] = _mmw_setzero_ps();
#endif
		}

#if ENABLE_STATS != 0
		memset(&mStats, 0, sizeof(OcclusionCullingStatistics));
#endif
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Polygon clipping functions
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	/*
	 * Clip a single polygon, uses SSE (simd4). Would be nice to be able to clip multiple polygons in parallel using AVX2/AVX512,
	 * but the clipping code is typically not a significant bottleneck.
	 */
	FORCE_INLINE int ClipPolygon(__m128 *outVtx, __m128 *inVtx, const __m128 &plane, int n) const
	{
		__m128 p0 = inVtx[n - 1];
		__m128 dist0 = _mmx_dp4_ps(p0, plane);

		// Loop over all polygon edges and compute intersection with clip plane (if any)
		int nout = 0;
		for (int k = 0; k < n; k++)
		{
			__m128 p1 = inVtx[k];
			__m128 dist1 = _mmx_dp4_ps(p1, plane);
			int dist0Neg = _mm_movemask_ps(dist0);
			if (!dist0Neg)	// dist0 > 0.0f
				outVtx[nout++] = p0;

			// Edge intersects the clip plane if dist0 and dist1 have opposing signs
			if (_mm_movemask_ps(_mm_xor_ps(dist0, dist1)))
			{
				// Always clip from the positive side to avoid T-junctions
				if (!dist0Neg)
				{
					__m128 t = _mm_div_ps(dist0, _mm_sub_ps(dist0, dist1));
					outVtx[nout++] = _mmx_fmadd_ps(_mm_sub_ps(p1, p0), t, p0);
				}
				else
				{
					__m128 t = _mm_div_ps(dist1, _mm_sub_ps(dist1, dist0));
					outVtx[nout++] = _mmx_fmadd_ps(_mm_sub_ps(p0, p1), t, p1);
				}
			}

			dist0 = dist1;
			p0 = p1;
		}
		return nout;
	}

	FORCE_INLINE int ClipPolygonTex(__m128 *outVtx, __m128 *inVtx, __m128 *outTex, __m128 *inTex, const __m128 &plane, int n) const
	{
		__m128 p0 = inVtx[n - 1], t0 = inTex[n - 1];
		__m128 dist0 = _mmx_dp4_ps(p0, plane);

		// Loop over all polygon edges and compute intersection with clip plane (if any)
		int nout = 0;
		for (int k = 0; k < n; k++)
		{
			__m128 p1 = inVtx[k], t1 = inTex[k];
			__m128 dist1 = _mmx_dp4_ps(p1, plane);
			int dist0Neg = _mm_movemask_ps(dist0);
			if (!dist0Neg)	// dist0 > 0.0f
			{
				outTex[nout] = t0;
				outVtx[nout++] = p0;
			}

			// Edge intersects the clip plane if dist0 and dist1 have opposing signs
			if (_mm_movemask_ps(_mm_xor_ps(dist0, dist1)))
			{
				// Always clip from the positive side to avoid T-junctions
				if (!dist0Neg)
				{
					__m128 t = _mm_div_ps(dist0, _mm_sub_ps(dist0, dist1));
					outTex[nout] = _mmx_fmadd_ps(_mm_sub_ps(t1, t0), t, t0);
					outVtx[nout++] = _mmx_fmadd_ps(_mm_sub_ps(p1, p0), t, p0);
				}
				else
				{
					__m128 t = _mm_div_ps(dist1, _mm_sub_ps(dist1, dist0));
					outTex[nout] = _mmx_fmadd_ps(_mm_sub_ps(t0, t1), t, t1);
					outVtx[nout++] = _mmx_fmadd_ps(_mm_sub_ps(p0, p1), t, p1);
				}
			}

			dist0 = dist1;
			p0 = p1;
			t0 = t1;
		}
		return nout;
	}

	/*
	 * View frustum culling: Test a SIMD-batch of triangles vs all frustum planes indicated by the mask.
	 */
	template<ClipPlanes CLIP_PLANE> void TestClipPlane(__mw *vtxX, __mw *vtxY, __mw *vtxW, unsigned int &straddleMask, unsigned int &triMask, ClipPlanes clipPlaneMask)
	{
		straddleMask = 0;
		// Skip masked clip planes
		if (!(clipPlaneMask & CLIP_PLANE))
			return;

		// Evaluate all 3 vertices against the frustum plane
		__mw planeDp[3];
		for (int i = 0; i < 3; ++i)
		{
			switch (CLIP_PLANE)
			{
			case ClipPlanes::CLIP_PLANE_LEFT:   planeDp[i] = _mmw_add_ps(vtxW[i], vtxX[i]); break;
			case ClipPlanes::CLIP_PLANE_RIGHT:  planeDp[i] = _mmw_sub_ps(vtxW[i], vtxX[i]); break;
			case ClipPlanes::CLIP_PLANE_BOTTOM: planeDp[i] = _mmw_add_ps(vtxW[i], vtxY[i]); break;
			case ClipPlanes::CLIP_PLANE_TOP:    planeDp[i] = _mmw_sub_ps(vtxW[i], vtxY[i]); break;
			case ClipPlanes::CLIP_PLANE_NEAR:   planeDp[i] = _mmw_sub_ps(vtxW[i], _mmw_set1_ps(mNearDist)); break;
			}
		}

		// Look at FP sign and determine if tri is inside, outside or straddles the frustum plane
		__mw inside = _mmw_andnot_ps(planeDp[0], _mmw_andnot_ps(planeDp[1], _mmw_not_ps(planeDp[2])));
		__mw outside = _mmw_and_ps(planeDp[0], _mmw_and_ps(planeDp[1], planeDp[2]));
		unsigned int inMask = (unsigned int)_mmw_movemask_ps(inside);
		unsigned int outMask = (unsigned int)_mmw_movemask_ps(outside);
		straddleMask = (~outMask) & (~inMask);
		triMask &= ~outMask;
	}

	/*
	 * Processes all triangles in a SIMD-batch and clips the triangles overlapping frustum planes. Clipping may add additional triangles.
	 * The first triangle is always written back into the SIMD-batch, and remaining triangles are written to the clippedVtxBuffer / clippedTexBuffer
	 * scratchpad memory.
	 */
	template<int TEXTURE_COORDINATES>
	FORCE_INLINE void ClipTriangleAndAddToBuffer(__mw *vtxX, __mw *vtxY, __mw *vtxW, __mw *vtxU, __mw *vtxV, __m128 *clippedVtxBuffer, __m128 *clippedTexBuffer, int &clipWriteIdx, unsigned int &triMask, unsigned int triClipMask, ClipPlanes clipPlaneMask)
	{
		if (!triClipMask)
			return;

		// Inside test all 3 triangle vertices against all active frustum planes
		unsigned int straddleMask[5];
		TestClipPlane<ClipPlanes::CLIP_PLANE_NEAR>(vtxX, vtxY, vtxW, straddleMask[0], triMask, clipPlaneMask);
		TestClipPlane<ClipPlanes::CLIP_PLANE_LEFT>(vtxX, vtxY, vtxW, straddleMask[1], triMask, clipPlaneMask);
		TestClipPlane<ClipPlanes::CLIP_PLANE_RIGHT>(vtxX, vtxY, vtxW, straddleMask[2], triMask, clipPlaneMask);
		TestClipPlane<ClipPlanes::CLIP_PLANE_BOTTOM>(vtxX, vtxY, vtxW, straddleMask[3], triMask, clipPlaneMask);
		TestClipPlane<ClipPlanes::CLIP_PLANE_TOP>(vtxX, vtxY, vtxW, straddleMask[4], triMask, clipPlaneMask);

		// Clip triangle against straddling planes and add to the clipped triangle buffer
		__m128 vtxBuf[2][8];
		__m128 texBuf[2][8];
		unsigned int clipMask = (straddleMask[0] | straddleMask[1] | straddleMask[2] | straddleMask[3] | straddleMask[4]) & (triClipMask & triMask);
		while (clipMask)
		{
			// Find and setup next triangle to clip
			unsigned int triIdx = find_clear_lsb(&clipMask);
			unsigned int triBit = (1U << triIdx);
			assert(triIdx < SIMD_LANES);

			int bufIdx = 0;
			int nClippedVerts = 3;
			for (int i = 0; i < 3; i++)
			{
				vtxBuf[0][i] = _mm_setr_ps(simd_f32(vtxX[i])[triIdx], simd_f32(vtxY[i])[triIdx], simd_f32(vtxW[i])[triIdx], 1.0f);
				if (TEXTURE_COORDINATES)
					texBuf[0][i] = _mm_setr_ps(simd_f32(vtxU[i])[triIdx], simd_f32(vtxV[i])[triIdx], 0.0f, 0.0f);
			}

			// Clip triangle with straddling planes.
			for (int i = 0; i < 5; ++i)
			{
				if ((straddleMask[i] & triBit) && (clipPlaneMask & (1 << i)))
				{
					if (TEXTURE_COORDINATES)
						nClippedVerts = ClipPolygonTex(vtxBuf[bufIdx ^ 1], vtxBuf[bufIdx], texBuf[bufIdx ^ 1], texBuf[bufIdx], mCSFrustumPlanes[i], nClippedVerts);
					else
						nClippedVerts = ClipPolygon(vtxBuf[bufIdx ^ 1], vtxBuf[bufIdx], mCSFrustumPlanes[i], nClippedVerts);
					bufIdx ^= 1;
				}
			}

			if (nClippedVerts >= 3)
			{
				// Write the first triangle back into the list of currently processed triangles
				for (int i = 0; i < 3; i++)
				{
					simd_f32(vtxX[i])[triIdx] = simd_f32(vtxBuf[bufIdx][i])[0];
					simd_f32(vtxY[i])[triIdx] = simd_f32(vtxBuf[bufIdx][i])[1];
					simd_f32(vtxW[i])[triIdx] = simd_f32(vtxBuf[bufIdx][i])[2];
					if (TEXTURE_COORDINATES)
					{
						simd_f32(vtxU[i])[triIdx] = simd_f32(texBuf[bufIdx][i])[0];
						simd_f32(vtxV[i])[triIdx] = simd_f32(texBuf[bufIdx][i])[1];
					}
				}
				// Write the remaining triangles into the clip buffer and process them next loop iteration
				for (int i = 2; i < nClippedVerts - 1; i++)
				{
					clippedVtxBuffer[clipWriteIdx * 3 + 0] = vtxBuf[bufIdx][0];
					clippedVtxBuffer[clipWriteIdx * 3 + 1] = vtxBuf[bufIdx][i];
					clippedVtxBuffer[clipWriteIdx * 3 + 2] = vtxBuf[bufIdx][i + 1];
					if (TEXTURE_COORDINATES)
					{
						clippedTexBuffer[clipWriteIdx * 3 + 0] = texBuf[bufIdx][0];
						clippedTexBuffer[clipWriteIdx * 3 + 1] = texBuf[bufIdx][i];
						clippedTexBuffer[clipWriteIdx * 3 + 2] = texBuf[bufIdx][i + 1];
					}
					clipWriteIdx = (clipWriteIdx + 1) & (MAX_CLIPPED - 1);
				}
			}
			else // Kill triangles that was removed by clipping
				triMask &= ~triBit;
		}
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Vertex transform & projection
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	/*
	 * Multiplies all vertices for a SIMD-batch of triangles by a 4x4 matrix. The output Z component is ignored, and the W=1 is assumed.
	 */
	FORCE_INLINE void TransformVerts(__mw *vtxX, __mw *vtxY, __mw *vtxW, const float *modelToClipMatrix)
	{
		if (modelToClipMatrix != nullptr)
		{
			for (int i = 0; i < 3; ++i)
			{
				__mw tmpX, tmpY, tmpW;
				tmpX = _mmw_fmadd_ps(vtxX[i], _mmw_set1_ps(modelToClipMatrix[0]), _mmw_fmadd_ps(vtxY[i], _mmw_set1_ps(modelToClipMatrix[4]), _mmw_fmadd_ps(vtxW[i], _mmw_set1_ps(modelToClipMatrix[8]), _mmw_set1_ps(modelToClipMatrix[12]))));
				tmpY = _mmw_fmadd_ps(vtxX[i], _mmw_set1_ps(modelToClipMatrix[1]), _mmw_fmadd_ps(vtxY[i], _mmw_set1_ps(modelToClipMatrix[5]), _mmw_fmadd_ps(vtxW[i], _mmw_set1_ps(modelToClipMatrix[9]), _mmw_set1_ps(modelToClipMatrix[13]))));
				tmpW = _mmw_fmadd_ps(vtxX[i], _mmw_set1_ps(modelToClipMatrix[3]), _mmw_fmadd_ps(vtxY[i], _mmw_set1_ps(modelToClipMatrix[7]), _mmw_fmadd_ps(vtxW[i], _mmw_set1_ps(modelToClipMatrix[11]), _mmw_set1_ps(modelToClipMatrix[15]))));
				vtxX[i] = tmpX;	vtxY[i] = tmpY;	vtxW[i] = tmpW;
			}
		}
	}

#if PRECISE_COVERAGE != 0
	/*
	 * Projects a SIMD-batch of triangles and transforms screen space/pixel coodaintes.
	 */
	FORCE_INLINE void ProjectVertices(__mwi *ipVtxX, __mwi *ipVtxY, __mw *pVtxX, __mw *pVtxY, __mw *pVtxZ, const __mw *vtxX, const __mw *vtxY, const __mw *vtxW)
	{
#if USE_D3D != 0
		static const int vertexOrder[] = { 2, 1, 0 };
#else
		static const int vertexOrder[] = { 0, 1, 2 };
#endif

		// Project vertices and transform to screen space. Snap to sub-pixel coordinates with FP_BITS precision.
		for (int i = 0; i < 3; i++)
		{
			int idx = vertexOrder[i];
			__mw rcpW = _mmw_div_ps(_mmw_set1_ps(1.0f), vtxW[i]);
			__mw screenX = _mmw_fmadd_ps(_mmw_mul_ps(vtxX[i], mHalfWidth), rcpW, mCenterX);
			__mw screenY = _mmw_fmadd_ps(_mmw_mul_ps(vtxY[i], mHalfHeight), rcpW, mCenterY);
			ipVtxX[idx] = _mmw_cvtps_epi32(_mmw_mul_ps(screenX, _mmw_set1_ps(float(1 << FP_BITS))));
			ipVtxY[idx] = _mmw_cvtps_epi32(_mmw_mul_ps(screenY, _mmw_set1_ps(float(1 << FP_BITS))));
			pVtxX[idx] = _mmw_mul_ps(_mmw_cvtepi32_ps(ipVtxX[idx]), _mmw_set1_ps(FP_INV));
			pVtxY[idx] = _mmw_mul_ps(_mmw_cvtepi32_ps(ipVtxY[idx]), _mmw_set1_ps(FP_INV));
			pVtxZ[idx] = rcpW;
		}
	}
#else
	/*
	 * Projects a SIMD-batch of triangles and transforms to screen space/pixel coodaintes.
	 */
	FORCE_INLINE void ProjectVertices(__mw *pVtxX, __mw *pVtxY, __mw *pVtxZ, const __mw *vtxX, const __mw *vtxY, const __mw *vtxW)
	{
#if USE_D3D != 0
		static const int vertexOrder[] = {2, 1, 0};
#else
		static const int vertexOrder[] = {0, 1, 2};
#endif
		// Project vertices and transform to screen space. Round to nearest integer pixel coordinate
		for (int i = 0; i < 3; i++)
		{
			int idx = vertexOrder[i];
			__mw rcpW = _mmw_div_ps(_mmw_set1_ps(1.0f), vtxW[i]);

			// The rounding modes are set to match HW rasterization with OpenGL. In practice our samples are placed
			// in the (1,0) corner of each pixel, while HW rasterizer uses (0.5, 0.5). We get (1,0) because of the
			// floor used when interpolating along triangle edges. The rounding modes match an offset of (0.5, -0.5)
			pVtxX[idx] = _mmw_ceil_ps(_mmw_fmadd_ps(_mmw_mul_ps(vtxX[i], mHalfWidth), rcpW, mCenterX));
			pVtxY[idx] = _mmw_floor_ps(_mmw_fmadd_ps(_mmw_mul_ps(vtxY[i], mHalfHeight), rcpW, mCenterY));
			pVtxZ[idx] = rcpW;
		}
	}
#endif

	FORCE_INLINE void ProjectTexCoords(__mw *pVtxU, __mw *pVtxV, __mw *pVtxZ, const __mw *vtxU, const __mw *vtxV)
	{
#if USE_D3D != 0
		static const int vertexOrder[] = { 2, 1, 0 };
#else
		static const int vertexOrder[] = { 0, 1, 2 };
#endif
		for (int i = 0; i < 3; i++)
		{
			int idx = vertexOrder[i];
			pVtxU[idx] = _mmw_mul_ps(pVtxZ[idx], vtxU[i]);
			pVtxV[idx] = _mmw_mul_ps(pVtxZ[idx], vtxV[i]);
		}
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Common SSE/AVX input assembly functions, note that there are specialized gathers for the general case in the SSE/AVX specific files
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	FORCE_INLINE void GatherVerticesFast(__mw *vtxX, __mw *vtxY, __mw *vtxW, const float *inVtx, const unsigned int *inTrisPtr, int numLanes)
	{
		// This function assumes that the vertex layout is four packed x, y, z, w-values.
		// Since the layout is known we can get some additional performance by using a
		// more optimized gather strategy.
		assert(numLanes >= 1);

		// Gather vertices
		__mw v[4], swz[4];
		for (int i = 0; i < 3; i++)
		{
			// Load 4 (x,y,z,w) vectors per SSE part of the SIMD register (so 4 vectors for SSE, 8 vectors for AVX)
			// this fetch uses templates to unroll the loop
			VtxFetch4<SIMD_LANES / 4>(v, inTrisPtr, i, inVtx, numLanes);

			// Transpose each individual SSE part of the SSE/AVX register (similar to _MM_TRANSPOSE4_PS)
			swz[0] = _mmw_shuffle_ps(v[0], v[1], 0x44);
			swz[2] = _mmw_shuffle_ps(v[0], v[1], 0xEE);
			swz[1] = _mmw_shuffle_ps(v[2], v[3], 0x44);
			swz[3] = _mmw_shuffle_ps(v[2], v[3], 0xEE);

			vtxX[i] = _mmw_shuffle_ps(swz[0], swz[1], 0x88);
			vtxY[i] = _mmw_shuffle_ps(swz[0], swz[1], 0xDD);
			vtxW[i] = _mmw_shuffle_ps(swz[2], swz[3], 0xDD);
		}
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Triangle rasterization functions
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	/*
	 * Computes the screen space bounding boxes (rectangles) for a SIMD-batch of projected triangles
	 */
	FORCE_INLINE void ComputeBoundingBox(__mwi &bbminX, __mwi &bbminY, __mwi &bbmaxX, __mwi &bbmaxY, const __mw *vX, const __mw *vY, const ScissorRect *scissor)
	{
		static const __mwi SIMD_PAD_W_MASK = _mmw_set1_epi32(~(TILE_WIDTH - 1));
		static const __mwi SIMD_PAD_H_MASK = _mmw_set1_epi32(~(TILE_HEIGHT - 1));

		// Find Min/Max vertices
		bbminX = _mmw_cvttps_epi32(_mmw_min_ps(vX[0], _mmw_min_ps(vX[1], vX[2])));
		bbminY = _mmw_cvttps_epi32(_mmw_min_ps(vY[0], _mmw_min_ps(vY[1], vY[2])));
		bbmaxX = _mmw_cvttps_epi32(_mmw_max_ps(vX[0], _mmw_max_ps(vX[1], vX[2])));
		bbmaxY = _mmw_cvttps_epi32(_mmw_max_ps(vY[0], _mmw_max_ps(vY[1], vY[2])));

		// Clamp to tile boundaries
		bbminX = _mmw_and_epi32(bbminX, SIMD_PAD_W_MASK);
		bbmaxX = _mmw_and_epi32(_mmw_add_epi32(bbmaxX, _mmw_set1_epi32(TILE_WIDTH)), SIMD_PAD_W_MASK);
		bbminY = _mmw_and_epi32(bbminY, SIMD_PAD_H_MASK);
		bbmaxY = _mmw_and_epi32(_mmw_add_epi32(bbmaxY, _mmw_set1_epi32(TILE_HEIGHT)), SIMD_PAD_H_MASK);

		// Clip to scissor
		bbminX = _mmw_max_epi32(bbminX, _mmw_set1_epi32(scissor->mMinX));
		bbmaxX = _mmw_min_epi32(bbmaxX, _mmw_set1_epi32(scissor->mMaxX));
		bbminY = _mmw_max_epi32(bbminY, _mmw_set1_epi32(scissor->mMinY));
		bbmaxY = _mmw_min_epi32(bbmaxY, _mmw_set1_epi32(scissor->mMaxY));
	}

#if PRECISE_COVERAGE != 0
	/*
	 * Sorts the vertices in a SIMD-batch of triangles so that vY[0] <= vY[1] && vY[0] < vY[2]
	 */
	FORCE_INLINE void SortVertices(__mwi *vX, __mwi *vY)
	{
		// Rotate the triangle in the winding order until v0 is the vertex with lowest Y value
		for (int i = 0; i < 2; i++)
		{
			__mwi ey1 = _mmw_sub_epi32(vY[1], vY[0]);
			__mwi ey2 = _mmw_sub_epi32(vY[2], vY[0]);
			__mwi swapMask = _mmw_or_epi32(_mmw_or_epi32(ey1, ey2), _mmw_cmpeq_epi32(simd_cast<__mwi>(ey2), SIMD_BITS_ZERO));
			__mwi sX, sY;
			sX = _mmw_blendv_epi32(vX[2], vX[0], swapMask);
			vX[0] = _mmw_blendv_epi32(vX[0], vX[1], swapMask);
			vX[1] = _mmw_blendv_epi32(vX[1], vX[2], swapMask);
			vX[2] = sX;
			sY = _mmw_blendv_epi32(vY[2], vY[0], swapMask);
			vY[0] = _mmw_blendv_epi32(vY[0], vY[1], swapMask);
			vY[1] = _mmw_blendv_epi32(vY[1], vY[2], swapMask);
			vY[2] = sY;
		}
	}

	/*
	 * Performs backface culling for a SIMD-batch of triangles, and makes sure all non-culled triangles are counter-clockwise winded.
	 */
	FORCE_INLINE int CullBackfaces(__mwi *ipVtxX, __mwi *ipVtxY, __mw *pVtxX, __mw *pVtxY, __mw *pVtxZ, const __mw &ccwMask, BackfaceWinding bfWinding)
	{
		// Reverse vertex order if non cw faces are considered front facing (rasterizer code requires CCW order)
		if (!(bfWinding & BACKFACE_CW))
		{
			__mw tmpX, tmpY, tmpZ;
			__mwi itmpX, itmpY;
			itmpX = _mmw_blendv_epi32(ipVtxX[2], ipVtxX[0], simd_cast<__mwi>(ccwMask));
			itmpY = _mmw_blendv_epi32(ipVtxY[2], ipVtxY[0], simd_cast<__mwi>(ccwMask));
			tmpX = _mmw_blendv_ps(pVtxX[2], pVtxX[0], ccwMask);
			tmpY = _mmw_blendv_ps(pVtxY[2], pVtxY[0], ccwMask);
			tmpZ = _mmw_blendv_ps(pVtxZ[2], pVtxZ[0], ccwMask);
			ipVtxX[2] = _mmw_blendv_epi32(ipVtxX[0], ipVtxX[2], simd_cast<__mwi>(ccwMask));
			ipVtxY[2] = _mmw_blendv_epi32(ipVtxY[0], ipVtxY[2], simd_cast<__mwi>(ccwMask));
			pVtxX[2] = _mmw_blendv_ps(pVtxX[0], pVtxX[2], ccwMask);
			pVtxY[2] = _mmw_blendv_ps(pVtxY[0], pVtxY[2], ccwMask);
			pVtxZ[2] = _mmw_blendv_ps(pVtxZ[0], pVtxZ[2], ccwMask);
			ipVtxX[0] = itmpX;
			ipVtxY[0] = itmpY;
			pVtxX[0] = tmpX;
			pVtxY[0] = tmpY;
			pVtxZ[0] = tmpZ;
		}

		// Return a lane mask with all front faces set
		return ((bfWinding & BACKFACE_CCW) ? 0 : _mmw_movemask_ps(ccwMask)) | ((bfWinding & BACKFACE_CW) ? 0 : ~_mmw_movemask_ps(ccwMask));
	}
#else
	/*
	 * Sorts the vertices in a SIMD-batch of triangles so that vY[0] <= vY[1] && vY[0] < vY[2]
	 */
	FORCE_INLINE void SortVertices(__mw *vX, __mw *vY)
	{
		// Rotate the triangle in the winding order until v0 is the vertex with lowest Y value
		for (int i = 0; i < 2; i++)
		{
			__mw ey1 = _mmw_sub_ps(vY[1], vY[0]);
			__mw ey2 = _mmw_sub_ps(vY[2], vY[0]);
			__mw swapMask = _mmw_or_ps(_mmw_or_ps(ey1, ey2), simd_cast<__mw>(_mmw_cmpeq_epi32(simd_cast<__mwi>(ey2), SIMD_BITS_ZERO)));
			__mw sX, sY;
			sX = _mmw_blendv_ps(vX[2], vX[0], swapMask);
			vX[0] = _mmw_blendv_ps(vX[0], vX[1], swapMask);
			vX[1] = _mmw_blendv_ps(vX[1], vX[2], swapMask);
			vX[2] = sX;
			sY = _mmw_blendv_ps(vY[2], vY[0], swapMask);
			vY[0] = _mmw_blendv_ps(vY[0], vY[1], swapMask);
			vY[1] = _mmw_blendv_ps(vY[1], vY[2], swapMask);
			vY[2] = sY;
		}
	}

	/*
	 * Performs backface culling for a SIMD-batch of triangles, and makes sure all non-culled triangles are counter-clockwise winded.
	 */
	FORCE_INLINE int CullBackfaces(__mw *pVtxX, __mw *pVtxY, __mw *pVtxZ, const __mw &ccwMask, BackfaceWinding bfWinding)
	{
		// Reverse vertex order if non cw faces are considered front facing (rasterizer code requires CCW order)
		if (!(bfWinding & BACKFACE_CW))
		{
			__mw tmpX, tmpY, tmpZ;
			tmpX = _mmw_blendv_ps(pVtxX[2], pVtxX[0], ccwMask);
			tmpY = _mmw_blendv_ps(pVtxY[2], pVtxY[0], ccwMask);
			tmpZ = _mmw_blendv_ps(pVtxZ[2], pVtxZ[0], ccwMask);
			pVtxX[2] = _mmw_blendv_ps(pVtxX[0], pVtxX[2], ccwMask);
			pVtxY[2] = _mmw_blendv_ps(pVtxY[0], pVtxY[2], ccwMask);
			pVtxZ[2] = _mmw_blendv_ps(pVtxZ[0], pVtxZ[2], ccwMask);
			pVtxX[0] = tmpX;
			pVtxY[0] = tmpY;
			pVtxZ[0] = tmpZ;
		}

		// Return a lane mask with all front faces set
		return ((bfWinding & BACKFACE_CCW) ? 0 : _mmw_movemask_ps(ccwMask)) | ((bfWinding & BACKFACE_CW) ? 0 : ~_mmw_movemask_ps(ccwMask));
	}
#endif

	/*
	 * Computes interpolation (plane) equations for a SIMD-batch of projected triangles
	 */
	FORCE_INLINE void InterpolationSetup(const __mw *pVtxX, const __mw *pVtxY, const __mw *pVtxA, __mw &aPixelDx, __mw &aPixelDy, __mw &aPixel0) const
	{
		// Setup z(x,y) = z0 + dx*x + dy*y screen space depth plane equation
		__mw x2 = _mmw_sub_ps(pVtxX[2], pVtxX[0]);
		__mw x1 = _mmw_sub_ps(pVtxX[1], pVtxX[0]);
		__mw y1 = _mmw_sub_ps(pVtxY[1], pVtxY[0]);
		__mw y2 = _mmw_sub_ps(pVtxY[2], pVtxY[0]);
		__mw a1 = _mmw_sub_ps(pVtxA[1], pVtxA[0]);
		__mw a2 = _mmw_sub_ps(pVtxA[2], pVtxA[0]);
		__mw d = _mmw_div_ps(_mmw_set1_ps(1.0f), _mmw_fmsub_ps(x1, y2, _mmw_mul_ps(y1, x2)));
		aPixelDx = _mmw_mul_ps(_mmw_fmsub_ps(a1, y2, _mmw_mul_ps(y1, a2)), d);
		aPixelDy = _mmw_mul_ps(_mmw_fmsub_ps(x1, a2, _mmw_mul_ps(a1, x2)), d);
		aPixel0 = _mmw_sub_ps(pVtxA[0], _mmw_fmadd_ps(aPixelDy, pVtxY[0], _mmw_mul_ps(aPixelDx, pVtxX[0])));
	}

	/*
	 * Updates a tile in the HiZ buffer using the quicker, but less accurate heuristic
	 */
	FORCE_INLINE void UpdateTileQuick(int tileIdx, const __mwi &coverage, const __mw &zTriv)
	{
		// Update heuristic used in the paper "Masked Software Occlusion Culling",
		// good balance between performance and accuracy
		STATS_ADD(mStats.mOccluders.mNumTilesUpdated, 1);
		assert(tileIdx >= 0 && tileIdx < mTilesWidth*mTilesHeight);

		__mwi mask = mMaskedHiZBuffer[tileIdx].mMask;
		__mw *zMin = mMaskedHiZBuffer[tileIdx].mZMin;

		// Swizzle coverage mask to 8x4 subtiles and test if any subtiles are not covered at all
		__mwi deadLane = _mmw_cmpeq_epi32(coverage, SIMD_BITS_ZERO);

		// Mask out all subtiles failing the depth test (don't update these subtiles)
		deadLane = _mmw_or_epi32(deadLane, _mmw_srai_epi32(simd_cast<__mwi>(_mmw_sub_ps(zTriv, zMin[0])), 31));
		__mwi maskedCoverage = _mmw_andnot_epi32(deadLane, coverage);

		// Use distance heuristic to discard layer 1 if incoming triangle is significantly nearer to observer
		// than the buffer contents. See Section 3.2 in "Masked Software Occlusion Culling"
		__mwi coveredLane = _mmw_cmpeq_epi32(maskedCoverage, SIMD_BITS_ONE);
		__mw diff = _mmw_fmsub_ps(zMin[1], _mmw_set1_ps(2.0f), _mmw_add_ps(zTriv, zMin[0]));
		__mwi discardLayerMask = _mmw_andnot_epi32(deadLane, _mmw_or_epi32(_mmw_srai_epi32(simd_cast<__mwi>(diff), 31), coveredLane));

		// Update the mask with incoming triangle coverage
		mask = _mmw_or_epi32(_mmw_andnot_epi32(discardLayerMask, mask), maskedCoverage);

		__mwi maskFull = _mmw_cmpeq_epi32(mask, SIMD_BITS_ONE);

		// Compute new value for zMin[1]. This has one of four outcomes: zMin[1] = min(zMin[1], zTriv),  zMin[1] = zTriv,
		// zMin[1] = FLT_MAX or unchanged, depending on if the layer is updated, discarded, fully covered, or not updated
		__mw opA = _mmw_blendv_ps(zTriv, zMin[1], simd_cast<__mw>(deadLane));
		__mw opB = _mmw_blendv_ps(zMin[1], zTriv, simd_cast<__mw>(discardLayerMask));
		__mw z1min = _mmw_min_ps(opA, opB);
		zMin[1] = _mmw_blendv_ps(z1min, _mmw_set1_ps(FLT_MAX), simd_cast<__mw>(maskFull));

		// Propagate zMin[1] back to zMin[0] if tile was fully covered, and update the mask
		zMin[0] = _mmw_blendv_ps(zMin[0], z1min, simd_cast<__mw>(maskFull));
		mMaskedHiZBuffer[tileIdx].mMask = _mmw_andnot_epi32(maskFull, mask);
	}

	/*
	 * Updates a tile in the HiZ buffer using the slower, but less accurate heuristic
	 */
	FORCE_INLINE void UpdateTileAccurate(int tileIdx, const __mwi &coverage, const __mw &zTriv)
	{
		assert(tileIdx >= 0 && tileIdx < mTilesWidth*mTilesHeight);

		__mw *zMin = mMaskedHiZBuffer[tileIdx].mZMin;
		__mwi &mask = mMaskedHiZBuffer[tileIdx].mMask;

		// Perform individual depth tests with layer 0 & 1 and mask out all failing pixels
		__mw sdist0 = _mmw_sub_ps(zMin[0], zTriv);
		__mw sdist1 = _mmw_sub_ps(zMin[1], zTriv);
		__mwi sign0 = _mmw_srai_epi32(simd_cast<__mwi>(sdist0), 31);
		__mwi sign1 = _mmw_srai_epi32(simd_cast<__mwi>(sdist1), 31);
		__mwi triMask = _mmw_and_epi32(coverage, _mmw_or_epi32(_mmw_andnot_epi32(mask, sign0), _mmw_and_epi32(mask, sign1)));

		// Early out if no pixels survived the depth test (this test is more accurate than
		// the early culling test in TraverseScanline())
		__mwi t0 = _mmw_cmpeq_epi32(triMask, SIMD_BITS_ZERO);
		__mwi t0inv = _mmw_not_epi32(t0);
		if (_mmw_testz_epi32(t0inv, t0inv))
			return;

		STATS_ADD(mStats.mOccluders.mNumTilesUpdated, 1);

		__mw zTri = _mmw_blendv_ps(zTriv, zMin[0], simd_cast<__mw>(t0));

		// Test if incoming triangle completely overwrites layer 0 or 1
		__mwi layerMask0 = _mmw_andnot_epi32(triMask, _mmw_not_epi32(mask));
		__mwi layerMask1 = _mmw_andnot_epi32(triMask, mask);
		__mwi lm0 = _mmw_cmpeq_epi32(layerMask0, SIMD_BITS_ZERO);
		__mwi lm1 = _mmw_cmpeq_epi32(layerMask1, SIMD_BITS_ZERO);
		__mw z0 = _mmw_blendv_ps(zMin[0], zTri, simd_cast<__mw>(lm0));
		__mw z1 = _mmw_blendv_ps(zMin[1], zTri, simd_cast<__mw>(lm1));

		// Compute distances used for merging heuristic
		__mw d0 = _mmw_abs_ps(sdist0);
		__mw d1 = _mmw_abs_ps(sdist1);
		__mw d2 = _mmw_abs_ps(_mmw_sub_ps(z0, z1));

		// Find minimum distance
		__mwi c01 = simd_cast<__mwi>(_mmw_sub_ps(d0, d1));
		__mwi c02 = simd_cast<__mwi>(_mmw_sub_ps(d0, d2));
		__mwi c12 = simd_cast<__mwi>(_mmw_sub_ps(d1, d2));
		// Two tests indicating which layer the incoming triangle will merge with or
		// overwrite. d0min indicates that the triangle will overwrite layer 0, and
		// d1min flags that the triangle will overwrite layer 1.
		__mwi d0min = _mmw_or_epi32(_mmw_and_epi32(c01, c02), _mmw_or_epi32(lm0, t0));
		__mwi d1min = _mmw_andnot_epi32(d0min, _mmw_or_epi32(c12, lm1));

		///////////////////////////////////////////////////////////////////////////////
		// Update depth buffer entry. NOTE: we always merge into layer 0, so if the
		// triangle should be merged with layer 1, we first swap layer 0 & 1 and then
		// merge into layer 0.
		///////////////////////////////////////////////////////////////////////////////

		// Update mask based on which layer the triangle overwrites or was merged into
		__mw inner = _mmw_blendv_ps(simd_cast<__mw>(triMask), simd_cast<__mw>(layerMask1), simd_cast<__mw>(d0min));
		mask = simd_cast<__mwi>(_mmw_blendv_ps(inner, simd_cast<__mw>(layerMask0), simd_cast<__mw>(d1min)));

		// Update the zMin[0] value. There are four outcomes: overwrite with layer 1,
		// merge with layer 1, merge with zTri or overwrite with layer 1 and then merge
		// with zTri.
		__mw e0 = _mmw_blendv_ps(z0, z1, simd_cast<__mw>(d1min));
		__mw e1 = _mmw_blendv_ps(z1, zTri, simd_cast<__mw>(_mmw_or_epi32(d1min, d0min)));
		zMin[0] = _mmw_min_ps(e0, e1);

		// Update the zMin[1] value. There are three outcomes: keep current value,
		// overwrite with zTri, or overwrite with z1
		__mw z1t = _mmw_blendv_ps(zTri, z1, simd_cast<__mw>(d0min));
		zMin[1] = _mmw_blendv_ps(z1t, z0, simd_cast<__mw>(d1min));
	}

	/*
	 * Compute texture LOD/mip level at a given pixel coordinate.
	 */
	template <int SCALE_X, int SCALE_Y>
	FORCE_INLINE __mwi ComputeTextureLOD(const __mw &x, const __mw &y, const __mw &rcpZ, TextureInterpolants &texInterpolants, const __mwi &mipLevels_1)
	{
		// Compute derivatives using arithmetic approach (allows processing individual pixels if desired)
		__mw rcpZSqr = _mmw_mul_ps(rcpZ, rcpZ);
		__mw dudx = _mmw_abs_ps(_mmw_mul_ps(rcpZSqr, _mmw_fmadd_ps(y, texInterpolants.uDerivConsts[0], texInterpolants.uDerivConsts[1])));
		__mw dvdx = _mmw_abs_ps(_mmw_mul_ps(rcpZSqr, _mmw_fmadd_ps(y, texInterpolants.vDerivConsts[0], texInterpolants.vDerivConsts[1])));
		__mw dudy = _mmw_abs_ps(_mmw_mul_ps(rcpZSqr, _mmw_fmsub_ps(x, texInterpolants.uDerivConsts[0], texInterpolants.uDerivConsts[2]))); // Actually computes negative derivative, but it's canceled by the abs
		__mw dvdy = _mmw_abs_ps(_mmw_mul_ps(rcpZSqr, _mmw_fmsub_ps(x, texInterpolants.vDerivConsts[0], texInterpolants.vDerivConsts[2]))); // Actually computes negative derivative, but it's canceled by the abs

		if (SCALE_X != 1 || SCALE_Y != 1)
		{
			dudx = _mmw_mul_ps(_mmw_set1_ps((float)SCALE_X), dudx);
			dvdx = _mmw_mul_ps(_mmw_set1_ps((float)SCALE_X), dvdx);
			dudy = _mmw_mul_ps(_mmw_set1_ps((float)SCALE_Y), dudy);
			dvdy = _mmw_mul_ps(_mmw_set1_ps((float)SCALE_Y), dvdy);
		}

		// Compute max length of derivative. This is the upper bound for the lod computation according to OpenGL 4.4 spec
		__mw maxLen = _mmw_add_ps(_mmw_max_ps(dudx, dudy), _mmw_max_ps(dvdx, dvdy));

		// Compute mip level, the log2 is computed by getting the fp32 exponent
		__mwi exponentIEEE = _mmw_sub_epi32(_mmw_set1_epi32(126), _mmw_srli_epi32(simd_cast<__mwi>(maxLen), 23));
		__mwi mipLevel = _mmw_sub_epi32(mipLevels_1, exponentIEEE);
		return _mmw_max_epi32(_mmw_setzero_epi32(), _mmw_min_epi32(mipLevels_1, mipLevel));
	}

	/*
	 * Compute the address (offset) of a specific texel at a given miplevel
	 */
	FORCE_INLINE __mwi ComputeTexelOffset(const __mw &u, const __mw &v, const __mwi &mipLevel, const __mw &texWidthf, const __mw &texHeightf, const __mwi texWidth, const __mwi &mipLevelConst)
	{
		// Apply wrapping mode (currently only repeat is supported)
		__mw wrappedU = _mmw_sub_ps(u, _mmw_floor_ps(u));
		__mw wrappedV = _mmw_sub_ps(v, _mmw_floor_ps(v));

		// Compute miplevel 0 coordinate
		__mwi ui0 = _mmw_cvttps_epi32(_mmw_mul_ps(texWidthf, wrappedU));
		__mwi vi0 = _mmw_cvttps_epi32(_mmw_mul_ps(texHeightf, wrappedV));

		// Scale coordinate by miplevel
		__mwi textureWidthN = _mmw_srlv_epi32(texWidth, mipLevel);
		__mwi uiN = _mmw_srlv_epi32(ui0, mipLevel);
		__mwi viN = _mmw_srlv_epi32(vi0, mipLevel);

		// Compute texture address for each lookup
		__mwi mipLevelOffset = _mmw_sub_epi32(mipLevelConst, _mmw_srlv_epi32(mipLevelConst, _mmw_slli_epi32(mipLevel, 1)));
		return _mmw_add_epi32(_mmw_add_epi32(_mmw_mullo_epi32(viN, textureWidthN), uiN), mipLevelOffset);
	}

	/*
	 * Performs "alpha testing". I.e. performs a pixel accurate lookup in a preprocessed occlusion 
	 * texture, where the pixel is opaque if the texture contents is 255, and transparent otherwise
	 */
	__mwi TextureAlphaTest(int tileIdx, __mwi coverageMask, __mw zDist0t, TextureInterpolants &texInterpolants, const MaskedOcclusionTextureInternal *texture)
	{
		// Do z-cull. The texture lookup is expensive, so we want to remove as much work as possible
		coverageMask = _mmw_andnot_epi32(_mmw_srai_epi32(simd_cast<__mwi>(zDist0t), 31), coverageMask);
		if (_mmw_testz_epi32(coverageMask, coverageMask))
			return coverageMask;

		// TODO: Slow & not needed
		float tilePixelX = (float)((tileIdx % mTilesWidth)*TILE_WIDTH);
		float tilePixelY = (float)((tileIdx / mTilesWidth)*TILE_HEIGHT);

		// Texture constants used for mip / adress computation
		__mwi texWidth      = _mmw_set1_epi32(texture->mWidth);
		__mw  texWidthf     = _mmw_set1_ps((float)texture->mWidth);
		__mw  texHeightf    = _mmw_set1_ps((float)texture->mWidth);
		__mwi mipLevelConst = _mmw_set1_epi32(texture->mMiplevelConst);
		__mwi mipLevels_1   = _mmw_set1_epi32(texture->mMipLevels - 1);
		__mwi byteMask      = _mmw_set1_epi32(0xFF);

		///////////////////////////////////////////////////////////////////////////////
		// Perform per-subtile conservative texture lookup (covered / transparent)
		///////////////////////////////////////////////////////////////////////////////

		// Compute pixel coordinates for 4x4 tile corners. 
		__mw subtileX0 = _mmw_add_ps(_mmw_set1_ps((float)(tilePixelX + SUB_TILE_WIDTH / 4)), SIMD_SUB_TILE_COL_OFFSET_F);
		__mw subtileX1 = _mmw_add_ps(subtileX0, _mmw_set1_ps(SUB_TILE_WIDTH / 2));
		__mw subtileY  = _mmw_add_ps(_mmw_set1_ps((float)(tilePixelY + SUB_TILE_HEIGHT/2)), SIMD_SUB_TILE_ROW_OFFSET_F);

		// Interpolate (u,v) for texture lookup = (u/z, v/z) / (1/z)
		__mw rcpZ0 = _mmw_rcp_ps(texInterpolants.zInterpolant.interpolate(subtileX0, subtileY));
		__mw u0 = _mmw_mul_ps(rcpZ0, texInterpolants.uInterpolant.interpolate(subtileX0, subtileY));
		__mw v0 = _mmw_mul_ps(rcpZ0, texInterpolants.vInterpolant.interpolate(subtileX0, subtileY));

		__mw rcpZ1 = _mmw_rcp_ps(texInterpolants.zInterpolant.interpolate(subtileX1, subtileY));
		__mw u1 = _mmw_mul_ps(rcpZ1, texInterpolants.uInterpolant.interpolate(subtileX1, subtileY));
		__mw v1 = _mmw_mul_ps(rcpZ1, texInterpolants.vInterpolant.interpolate(subtileX1, subtileY));

		// Compute texture LOD (mipmap level)
		__mwi mipLevel0 = ComputeTextureLOD<SUB_TILE_WIDTH / 2, SUB_TILE_HEIGHT>(subtileX0, subtileY, rcpZ0, texInterpolants, mipLevels_1);
		__mwi mipLevel1 = ComputeTextureLOD<SUB_TILE_WIDTH / 2, SUB_TILE_HEIGHT>(subtileX1, subtileY, rcpZ1, texInterpolants, mipLevels_1);

		// Compute address offsets for all loookups
		__mwi texelOffset0 = ComputeTexelOffset(u0, v0, mipLevel0, texWidthf, texHeightf, texWidth, mipLevelConst);
		__mwi texelOffset1 = ComputeTexelOffset(u1, v1, mipLevel1, texWidthf, texHeightf, texWidth, mipLevelConst);

		///////////////////////////////////////////////////////////////////////////////
		// Texture lookup & conservative alpha test
		///////////////////////////////////////////////////////////////////////////////

		__mwi textureVal0 = _mmw_and_epi32(byteMask, _mmw_i32gather_epi32((const int*)texture->mOcclusionData, texelOffset0, 1));
		__mwi textureVal1 = _mmw_and_epi32(byteMask, _mmw_i32gather_epi32((const int*)texture->mOcclusionData, texelOffset1, 1));

		__mwi subtileCoveredMask0 = _mmw_and_epi32(_mmw_cmpeq_epi32(textureVal0, _mmw_setzero_epi32()), _mmw_set1_epi32(0x0F0F0F0F));
		__mwi subtileCoveredMask1 = _mmw_and_epi32(_mmw_cmpeq_epi32(textureVal1, _mmw_setzero_epi32()), _mmw_set1_epi32(0xF0F0F0F0));
		__mwi subtileTransparentMask0 = _mmw_and_epi32(_mmw_cmpeq_epi32(textureVal0, byteMask), _mmw_set1_epi32(0x0F0F0F0F));
		__mwi subtileTransparentMask1 = _mmw_and_epi32(_mmw_cmpeq_epi32(textureVal1, byteMask), _mmw_set1_epi32(0xF0F0F0F0));

		__mwi subtileCoveredMask = _mmw_or_epi32(subtileCoveredMask0, subtileCoveredMask1);
		__mwi subtileTransparentMask = _mmw_or_epi32(subtileTransparentMask0, subtileTransparentMask1);

		// Remove the transparent blocks from the coverage mask
		coverageMask = _mmw_andnot_epi32(subtileTransparentMask, coverageMask);
		if (_mmw_testz_epi32(coverageMask, coverageMask))
			return coverageMask;

		// Determine which subtiles we need to do expensive pixel processing for. Pixel processing is only required 
		// if the subtile is neither completely opaque or completely transparent.
		__mwi pixelSubtiles = _mmw_andnot_epi32(_mmw_cmpeq_epi32(coverageMask, _mmw_set1_epi32(0)), _mmw_not_epi32(subtileCoveredMask));
		unsigned int pixelSubtilesMask = _mmw_movemask_ps(simd_cast<__mw>(pixelSubtiles));

		///////////////////////////////////////////////////////////////////////////////
		// Perform per-pixel tests
		///////////////////////////////////////////////////////////////////////////////

		while (pixelSubtilesMask)
		{
			unsigned int subtileIdx = find_clear_lsb(&pixelSubtilesMask);

			float subtilePixelX = simd_f32(SIMD_SUB_TILE_COL_OFFSET_F)[subtileIdx] + tilePixelX;
			float subtilePixelY = simd_f32(SIMD_SUB_TILE_ROW_OFFSET_F)[subtileIdx] + tilePixelY;

			unsigned int textureCoverageMask = 0;
			unsigned int subtileCoverageMask = (unsigned int)simd_i32(coverageMask)[subtileIdx];

			for (int px = 0; px < SUB_TILE_WIDTH; px += SIMD_PIXEL_W)
			{
#define COARSE_TEXTURELOD
#ifdef COARSE_TEXTURELOD
				__mwi mipLevel = _mmw_set1_epi32(px == 0 ? simd_i32(mipLevel0)[subtileIdx] - 2 : simd_i32(mipLevel1)[subtileIdx] - 2); // 4x4 -> 1x1 pixels is 2 miplevels
#endif

				for (int py = 0; py < SUB_TILE_HEIGHT; py += SIMD_PIXEL_H)
				{
					///////////////////////////////////////////////////////////////////////////////
					// Early exit if mask is already zero
					///////////////////////////////////////////////////////////////////////////////

					unsigned int mask = Coverage2Lanes(subtileCoverageMask >> (px + py*8)) & SIMD_ALL_LANES_MASK;
					if (!mask)
						continue;

					// Compute pixel coordinates
					__mw pixelX = _mmw_add_ps(_mmw_set1_ps(subtilePixelX + px), SIMD_PIXEL_COL_OFFSET_F);
					__mw pixelY = _mmw_add_ps(_mmw_set1_ps(subtilePixelY + py), SIMD_PIXEL_ROW_OFFSET_F);

					///////////////////////////////////////////////////////////////////////////////
					// Texture lookup: address computation
					///////////////////////////////////////////////////////////////////////////////

					// Interpolate (u,v) for texture lookup = (u/z, v/z) / (1/z)
					__mw rcpZ = _mmw_rcp_ps(texInterpolants.zInterpolant.interpolate(pixelX, pixelY));
					__mw u = _mmw_mul_ps(rcpZ, texInterpolants.uInterpolant.interpolate(pixelX, pixelY));
					__mw v = _mmw_mul_ps(rcpZ, texInterpolants.vInterpolant.interpolate(pixelX, pixelY));

#ifndef COARSE_TEXTURELOD
					// Compute texture LOD (mipmap level)
					__mwi mipLevel = ComputeTextureLOD<1, 1>(subtileX0, subtileY, rcpZ0, texInterpolants, mipLevels_1);
#endif

					// Compute texel addresses/offsets
					__mwi texelOffset = ComputeTexelOffset(u, v, mipLevel, texWidthf, texHeightf, texWidth, mipLevelConst);

					///////////////////////////////////////////////////////////////////////////////
					// Texture lookup & "alpha test"
					///////////////////////////////////////////////////////////////////////////////

					__mwi textureVal = _mmw_and_epi32(byteMask, _mmw_i32gather_epi32((const int*)texture->mOcclusionData, texelOffset, 1));
					unsigned int textureMask = _mmw_movemask_ps(simd_cast<__mw>(_mmw_cmpeq_epi32(textureVal, _mmw_setzero_epi32())));

					///////////////////////////////////////////////////////////////////////////////
					// Swizzle texture mask & accumulate result
					///////////////////////////////////////////////////////////////////////////////
					
					textureCoverageMask |= Lanes2Coverage(textureMask) << (px + py*8);
				}
			}

			// Update SIMD lane of coverage mask
			simd_i32(coverageMask)[subtileIdx] &= textureCoverageMask;
		}

		return coverageMask;
	}

	/*
	 * Traverses a scanline of 32 x SIMD_LANES pixel tiles
	 *     - Computes pixel coverage
	 *     - Performs coarse z-culling
	 *     - Updates HiZ buffer, or performs occlusion query (depending on the TEST_Z template parameter)
	 */
	template<int TEST_Z, int NRIGHT, int NLEFT, int TEXTURE_COORDINATES>
	FORCE_INLINE int TraverseScanline(int leftOffset, int rightOffset, int tileIdx, 
		int rightEvent, int leftEvent, const __mwi *events, 
		const __mw &zTriMin, const __mw &zTriMax, const __mw &iz0, float zx,
		TextureInterpolants &texInterpolants, const MaskedOcclusionTextureInternal *texture)
	{
		// Floor edge events to integer pixel coordinates (shift out fixed point bits)
		int eventOffset = leftOffset << TILE_WIDTH_SHIFT;
		__mwi right[NRIGHT], left[NLEFT];
		for (int i = 0; i < NRIGHT; ++i)
			right[i] = _mmw_max_epi32(_mmw_sub_epi32(_mmw_srai_epi32(events[rightEvent + i], FP_BITS), _mmw_set1_epi32(eventOffset)), SIMD_BITS_ZERO);
		for (int i = 0; i < NLEFT; ++i)
			left[i] = _mmw_max_epi32(_mmw_sub_epi32(_mmw_srai_epi32(events[leftEvent - i], FP_BITS), _mmw_set1_epi32(eventOffset)), SIMD_BITS_ZERO);

		__mw z0 = _mmw_add_ps(iz0, _mmw_set1_ps(zx*leftOffset));
		int tileIdxEnd = tileIdx + rightOffset;
		tileIdx += leftOffset;
		for (;;)
		{
			if (TEST_Z)
				STATS_ADD(mStats.mOccludees.mNumTilesTraversed, 1);
			else
				STATS_ADD(mStats.mOccluders.mNumTilesTraversed, 1);

			// Perform a coarse test to quickly discard occluded tiles
#if QUICK_MASK != 0
			// Only use the reference layer (layer 0) to cull as it is always conservative
			__mw zMinBuf = mMaskedHiZBuffer[tileIdx].mZMin[0];
#else
			// Compute zMin for the overlapped layers
			__mwi mask = mMaskedHiZBuffer[tileIdx].mMask;
			__mw zMin0 = _mmw_blendv_ps(mMaskedHiZBuffer[tileIdx].mZMin[0], mMaskedHiZBuffer[tileIdx].mZMin[1], simd_cast<__mw>(_mmw_cmpeq_epi32(mask, _mmw_set1_epi32(~0))));
			__mw zMin1 = _mmw_blendv_ps(mMaskedHiZBuffer[tileIdx].mZMin[1], mMaskedHiZBuffer[tileIdx].mZMin[0], simd_cast<__mw>(_mmw_cmpeq_epi32(mask, _mmw_setzero_epi32())));
			__mw zMinBuf = _mmw_min_ps(zMin0, zMin1);
#endif
			__mw dist0 = _mmw_sub_ps(zTriMax, zMinBuf);
			if (_mmw_movemask_ps(dist0) != SIMD_ALL_LANES_MASK)
			{
				// Compute coverage mask for entire 32xN using shift operations
				__mwi accumulatedMask = _mmw_sllv_ones(left[0]);
				for (int i = 1; i < NLEFT; ++i)
					accumulatedMask = _mmw_and_epi32(accumulatedMask, _mmw_sllv_ones(left[i]));
				for (int i = 0; i < NRIGHT; ++i)
					accumulatedMask = _mmw_andnot_epi32(_mmw_sllv_ones(right[i]), accumulatedMask);

				// Swizzle rasterization mask to 8x4 subtiles
				__mwi rastMask8x4 = _mmw_transpose_epi8(accumulatedMask);

				// Perform conservative texture lookup for alpha tested triangles
				if (TEXTURE_COORDINATES)
					rastMask8x4 = TextureAlphaTest(tileIdx, rastMask8x4, dist0, texInterpolants, texture);

				if (TEST_Z)
				{
					// Perform a conservative visibility test (test zMax against buffer for each covered 8x4 subtile)
					__mw zSubTileMax = _mmw_min_ps(z0, zTriMax);
					__mwi zPass = simd_cast<__mwi>(_mmw_cmpge_ps(zSubTileMax, zMinBuf));

					__mwi deadLane = _mmw_cmpeq_epi32(rastMask8x4, SIMD_BITS_ZERO);
					zPass = _mmw_andnot_epi32(deadLane, zPass);
#if QUERY_DEBUG_BUFFER != 0
					__mwi debugVal = _mmw_blendv_epi32(_mmw_set1_epi32(0), _mmw_blendv_epi32(_mmw_set1_epi32(1), _mmw_set1_epi32(2), zPass), _mmw_not_epi32(deadLane));
					mQueryDebugBuffer[tileIdx] = debugVal;
#endif

					if (!_mmw_testz_epi32(zPass, zPass))
						return CullingResult::VISIBLE;
				}
				else
				{
					// Compute interpolated min for each 8x4 subtile and update the masked hierarchical z buffer entry
					__mw zSubTileMin = _mmw_max_ps(z0, zTriMin);
#if QUICK_MASK != 0
					UpdateTileQuick(tileIdx, rastMask8x4, zSubTileMin);
#else
					UpdateTileAccurate(tileIdx, rastMask8x4, zSubTileMin);
#endif
				}
			}

			// Update buffer address, interpolate z and edge events
			tileIdx++;
			if (tileIdx >= tileIdxEnd)
				break;
			z0 = _mmw_add_ps(z0, _mmw_set1_ps(zx));
			for (int i = 0; i < NRIGHT; ++i)
				right[i] = _mmw_subs_epu16(right[i], SIMD_TILE_WIDTH);	// Trick, use sub saturated to avoid checking against < 0 for shift (values should fit in 16 bits)
			for (int i = 0; i < NLEFT; ++i)
				left[i] = _mmw_subs_epu16(left[i], SIMD_TILE_WIDTH);
		}

		return TEST_Z ? CullingResult::OCCLUDED : CullingResult::VISIBLE;
	}

	/*
	 * Traverses the triangle bounding box in tile scanline order, starting from the lowest Y coordinate. For each scanline of tiles:
	 *     - Computes the pixel coordinates of the left & right triangle edge (triEvents)
	 *     - When reaching the middle vertex, change to the final triangle edge isntead
	 *     - Invokes TraverseScanline() to traverse each individual scanline of tiles
	 */
	template<int TEST_Z, int TIGHT_TRAVERSAL, int MID_VTX_RIGHT, int TEXTURE_COORDINATES>
#if PRECISE_COVERAGE != 0
	FORCE_INLINE int RasterizeTriangle(
		unsigned int triIdx, int bbWidth, int tileRowIdx, int tileMidRowIdx, int tileEndRowIdx,
		const __mwi *eventStart, const __mw *slope, const __mwi *slopeTileDelta,
		const __mwi *edgeY, const __mwi *absEdgeX, const __mwi *slopeSign, const __mwi *eventStartRemainder, const __mwi *slopeTileRemainder,
		const __mw &zTriMin, const __mw &zTriMax, __mw &z0, float zx, float zy,
		TextureInterpolants &texInterpolants, const MaskedOcclusionTextureInternal *texture)
#else
	FORCE_INLINE int RasterizeTriangle(unsigned int triIdx, int bbWidth, int tileRowIdx, int tileMidRowIdx, int tileEndRowIdx, 
		const __mwi *eventStart, const __mwi *slope, const __mwi *slopeTileDelta,
		const __mw &zTriMin, const __mw &zTriMax, __mw &z0, float zx, float zy,
		TextureInterpolants &texInterpolants, const MaskedOcclusionTextureInternal *texture)
#endif
	{
		if (TEST_Z)
			STATS_ADD(mStats.mOccludees.mNumRasterizedTriangles, 1);
		else
			STATS_ADD(mStats.mOccluders.mNumRasterizedTriangles, 1);

		int cullResult;

#if PRECISE_COVERAGE != 0
		#define LEFT_EDGE_BIAS -1
		#define RIGHT_EDGE_BIAS 1
		#define UPDATE_TILE_EVENTS_Y(i) \
				triEventRemainder[i] = _mmw_sub_epi32(triEventRemainder[i], triSlopeTileRemainder[i]); \
				__mwi overflow##i = _mmw_srai_epi32(triEventRemainder[i], 31); \
				triEventRemainder[i] = _mmw_add_epi32(triEventRemainder[i], _mmw_and_epi32(overflow##i, triEdgeY[i])); \
				triEvent[i] = _mmw_add_epi32(triEvent[i], _mmw_add_epi32(triSlopeTileDelta[i], _mmw_and_epi32(overflow##i, triSlopeSign[i])))

		__mwi triEvent[3], triSlopeSign[3], triSlopeTileDelta[3], triEdgeY[3], triSlopeTileRemainder[3], triEventRemainder[3];
		for (int i = 0; i < 3; ++i)
		{
			triSlopeSign[i] = _mmw_set1_epi32(simd_i32(slopeSign[i])[triIdx]);
			triSlopeTileDelta[i] = _mmw_set1_epi32(simd_i32(slopeTileDelta[i])[triIdx]);
			triEdgeY[i] = _mmw_set1_epi32(simd_i32(edgeY[i])[triIdx]);
			triSlopeTileRemainder[i] = _mmw_set1_epi32(simd_i32(slopeTileRemainder[i])[triIdx]);

			__mw triSlope = _mmw_set1_ps(simd_f32(slope[i])[triIdx]);
			__mwi triAbsEdgeX = _mmw_set1_epi32(simd_i32(absEdgeX[i])[triIdx]);
			__mwi triStartRemainder = _mmw_set1_epi32(simd_i32(eventStartRemainder[i])[triIdx]);
			__mwi triEventStart = _mmw_set1_epi32(simd_i32(eventStart[i])[triIdx]);

			__mwi scanlineDelta = _mmw_cvttps_epi32(_mmw_mul_ps(triSlope, SIMD_LANE_YCOORD_F));
			__mwi scanlineSlopeRemainder = _mmw_sub_epi32(_mmw_mullo_epi32(triAbsEdgeX, SIMD_LANE_YCOORD_I), _mmw_mullo_epi32(_mmw_abs_epi32(scanlineDelta), triEdgeY[i]));

			triEventRemainder[i] = _mmw_sub_epi32(triStartRemainder, scanlineSlopeRemainder);
			__mwi overflow = _mmw_srai_epi32(triEventRemainder[i], 31);
			triEventRemainder[i] = _mmw_add_epi32(triEventRemainder[i], _mmw_and_epi32(overflow, triEdgeY[i]));
			triEvent[i] = _mmw_add_epi32(_mmw_add_epi32(triEventStart, scanlineDelta), _mmw_and_epi32(overflow, triSlopeSign[i]));
		}

#else
		#define LEFT_EDGE_BIAS 0
		#define RIGHT_EDGE_BIAS 0
		#define UPDATE_TILE_EVENTS_Y(i)		triEvent[i] = _mmw_add_epi32(triEvent[i], triSlopeTileDelta[i]);

		// Get deltas used to increment edge events each time we traverse one scanline of tiles
		__mwi triSlopeTileDelta[3];
		triSlopeTileDelta[0] = _mmw_set1_epi32(simd_i32(slopeTileDelta[0])[triIdx]);
		triSlopeTileDelta[1] = _mmw_set1_epi32(simd_i32(slopeTileDelta[1])[triIdx]);
		triSlopeTileDelta[2] = _mmw_set1_epi32(simd_i32(slopeTileDelta[2])[triIdx]);

		// Setup edge events for first batch of SIMD_LANES scanlines
		__mwi triEvent[3];
		triEvent[0] = _mmw_add_epi32(_mmw_set1_epi32(simd_i32(eventStart[0])[triIdx]), _mmw_mullo_epi32(SIMD_LANE_IDX, _mmw_set1_epi32(simd_i32(slope[0])[triIdx])));
		triEvent[1] = _mmw_add_epi32(_mmw_set1_epi32(simd_i32(eventStart[1])[triIdx]), _mmw_mullo_epi32(SIMD_LANE_IDX, _mmw_set1_epi32(simd_i32(slope[1])[triIdx])));
		triEvent[2] = _mmw_add_epi32(_mmw_set1_epi32(simd_i32(eventStart[2])[triIdx]), _mmw_mullo_epi32(SIMD_LANE_IDX, _mmw_set1_epi32(simd_i32(slope[2])[triIdx])));
#endif

		// For big triangles track start & end tile for each scanline and only traverse the valid region
		int startDelta, endDelta, topDelta, startEvent, endEvent, topEvent;
		if (TIGHT_TRAVERSAL)
		{
			startDelta = simd_i32(slopeTileDelta[2])[triIdx] + LEFT_EDGE_BIAS;
			endDelta = simd_i32(slopeTileDelta[0])[triIdx] + RIGHT_EDGE_BIAS;
			topDelta = simd_i32(slopeTileDelta[1])[triIdx] + (MID_VTX_RIGHT ? RIGHT_EDGE_BIAS : LEFT_EDGE_BIAS);

			// Compute conservative bounds for the edge events over a 32xN tile
			startEvent = simd_i32(eventStart[2])[triIdx] + min(0, startDelta);
			endEvent = simd_i32(eventStart[0])[triIdx] + max(0, endDelta) + (TILE_WIDTH << FP_BITS);
			if (MID_VTX_RIGHT)
				topEvent = simd_i32(eventStart[1])[triIdx] + max(0, topDelta) + (TILE_WIDTH << FP_BITS);
			else
				topEvent = simd_i32(eventStart[1])[triIdx] + min(0, topDelta);
		}

		if (tileRowIdx <= tileMidRowIdx)
		{
			int tileStopIdx = min(tileEndRowIdx, tileMidRowIdx);
			// Traverse the bottom half of the triangle
			while (tileRowIdx < tileStopIdx)
			{
				int start = 0, end = bbWidth;
				if (TIGHT_TRAVERSAL)
				{
					// Compute tighter start and endpoints to avoid traversing empty space
					start = max(0, min(bbWidth - 1, startEvent >> (TILE_WIDTH_SHIFT + FP_BITS)));
					end = min(bbWidth, ((int)endEvent >> (TILE_WIDTH_SHIFT + FP_BITS)));
					startEvent += startDelta;
					endEvent += endDelta;
				}

				// Traverse the scanline and update the masked hierarchical z buffer
				cullResult = TraverseScanline<TEST_Z, 1, 1, TEXTURE_COORDINATES>(start, end, tileRowIdx, 0, 2, triEvent, zTriMin, zTriMax, z0, zx, texInterpolants, texture);

				if (TEST_Z && cullResult == CullingResult::VISIBLE) // Early out if performing occlusion query
					return CullingResult::VISIBLE;

				// move to the next scanline of tiles, update edge events and interpolate z
				tileRowIdx += mTilesWidth;
				z0 = _mmw_add_ps(z0, _mmw_set1_ps(zy));
				UPDATE_TILE_EVENTS_Y(0);
				UPDATE_TILE_EVENTS_Y(2);
			}

			// Traverse the middle scanline of tiles. We must consider all three edges only in this region
			if (tileRowIdx < tileEndRowIdx)
			{
				int start = 0, end = bbWidth;
				if (TIGHT_TRAVERSAL)
				{
					// Compute tighter start and endpoints to avoid traversing lots of empty space
					start = max(0, min(bbWidth - 1, startEvent >> (TILE_WIDTH_SHIFT + FP_BITS)));
					end = min(bbWidth, ((int)endEvent >> (TILE_WIDTH_SHIFT + FP_BITS)));

					// Switch the traversal start / end to account for the upper side edge
					endEvent = MID_VTX_RIGHT ? topEvent : endEvent;
					endDelta = MID_VTX_RIGHT ? topDelta : endDelta;
					startEvent = MID_VTX_RIGHT ? startEvent : topEvent;
					startDelta = MID_VTX_RIGHT ? startDelta : topDelta;
					startEvent += startDelta;
					endEvent += endDelta;
				}

				// Traverse the scanline and update the masked hierarchical z buffer.
				if (MID_VTX_RIGHT)
					cullResult = TraverseScanline<TEST_Z, 2, 1, TEXTURE_COORDINATES>(start, end, tileRowIdx, 0, 2, triEvent, zTriMin, zTriMax, z0, zx, texInterpolants, texture);
				else
					cullResult = TraverseScanline<TEST_Z, 1, 2, TEXTURE_COORDINATES>(start, end, tileRowIdx, 0, 2, triEvent, zTriMin, zTriMax, z0, zx, texInterpolants, texture);

				if (TEST_Z && cullResult == CullingResult::VISIBLE) // Early out if performing occlusion query
					return CullingResult::VISIBLE;

				tileRowIdx += mTilesWidth;
			}

			// Traverse the top half of the triangle
			if (tileRowIdx < tileEndRowIdx)
			{
				// move to the next scanline of tiles, update edge events and interpolate z
				z0 = _mmw_add_ps(z0, _mmw_set1_ps(zy));
				int i0 = MID_VTX_RIGHT + 0;
				int i1 = MID_VTX_RIGHT + 1;
				UPDATE_TILE_EVENTS_Y(i0);
				UPDATE_TILE_EVENTS_Y(i1);
				for (;;)
				{
					int start = 0, end = bbWidth;
					if (TIGHT_TRAVERSAL)
					{
						// Compute tighter start and endpoints to avoid traversing lots of empty space
						start = max(0, min(bbWidth - 1, startEvent >> (TILE_WIDTH_SHIFT + FP_BITS)));
						end = min(bbWidth, ((int)endEvent >> (TILE_WIDTH_SHIFT + FP_BITS)));
						startEvent += startDelta;
						endEvent += endDelta;
					}

					// Traverse the scanline and update the masked hierarchical z buffer
					cullResult = TraverseScanline<TEST_Z, 1, 1, TEXTURE_COORDINATES>(start, end, tileRowIdx, MID_VTX_RIGHT + 0, MID_VTX_RIGHT + 1, triEvent, zTriMin, zTriMax, z0, zx, texInterpolants, texture);

					if (TEST_Z && cullResult == CullingResult::VISIBLE) // Early out if performing occlusion query
						return CullingResult::VISIBLE;

					// move to the next scanline of tiles, update edge events and interpolate z
					tileRowIdx += mTilesWidth;
					if (tileRowIdx >= tileEndRowIdx)
						break;
					z0 = _mmw_add_ps(z0, _mmw_set1_ps(zy));
					UPDATE_TILE_EVENTS_Y(i0);
					UPDATE_TILE_EVENTS_Y(i1);
				}
			}
		}
		else
		{
			if (TIGHT_TRAVERSAL)
			{
				// For large triangles, switch the traversal start / end to account for the upper side edge
				endEvent = MID_VTX_RIGHT ? topEvent : endEvent;
				endDelta = MID_VTX_RIGHT ? topDelta : endDelta;
				startEvent = MID_VTX_RIGHT ? startEvent : topEvent;
				startDelta = MID_VTX_RIGHT ? startDelta : topDelta;
			}

			// Traverse the top half of the triangle
			if (tileRowIdx < tileEndRowIdx)
			{
				int i0 = MID_VTX_RIGHT + 0;
				int i1 = MID_VTX_RIGHT + 1;
				for (;;)
				{
					int start = 0, end = bbWidth;
					if (TIGHT_TRAVERSAL)
					{
						// Compute tighter start and endpoints to avoid traversing lots of empty space
						start = max(0, min(bbWidth - 1, startEvent >> (TILE_WIDTH_SHIFT + FP_BITS)));
						end = min(bbWidth, ((int)endEvent >> (TILE_WIDTH_SHIFT + FP_BITS)));
						startEvent += startDelta;
						endEvent += endDelta;
					}

					// Traverse the scanline and update the masked hierarchical z buffer
					cullResult = TraverseScanline<TEST_Z, 1, 1, TEXTURE_COORDINATES>(start, end, tileRowIdx, MID_VTX_RIGHT + 0, MID_VTX_RIGHT + 1, triEvent, zTriMin, zTriMax, z0, zx, texInterpolants, texture);

					if (TEST_Z && cullResult == CullingResult::VISIBLE) // Early out if performing occlusion query
						return CullingResult::VISIBLE;

					// move to the next scanline of tiles, update edge events and interpolate z
					tileRowIdx += mTilesWidth;
					if (tileRowIdx >= tileEndRowIdx)
						break;
					z0 = _mmw_add_ps(z0, _mmw_set1_ps(zy));
					UPDATE_TILE_EVENTS_Y(i0);
					UPDATE_TILE_EVENTS_Y(i1);
				}
			}
		}

		return TEST_Z ? CullingResult::OCCLUDED : CullingResult::VISIBLE;
	}

	/*
	 * Performs triangle/rasterization setup for a SIMD-batch of projected triangles, and invokes RasterizeTriangle() to rasterize each non-culled triangle
	 *     - Computes screen space bounding box/rectangle
	 *     - Sets up z = Z(x,y) depth plane equation
	 *     - Sets up x = L(y) equations to track triangle edge x/y coordinate dependencies
	 */
	template<int TEST_Z, int TEXTURE_COORDINATES>
#if PRECISE_COVERAGE != 0
	FORCE_INLINE int RasterizeTriangleBatch(__mwi ipVtxX[3], __mwi ipVtxY[3], __mw pVtxX[3], __mw pVtxY[3], __mw pVtxZ[3], __mw pVtxU[3], __mw pVtxV[3], unsigned int triMask, const ScissorRect *scissor, const MaskedOcclusionTextureInternal *texture)
#else
	FORCE_INLINE int RasterizeTriangleBatch(__mw pVtxX[3], __mw pVtxY[3], __mw pVtxZ[3], __mw pVtxU[3], __mw pVtxV[3], unsigned int triMask, const ScissorRect *scissor, const MaskedOcclusionTextureInternal *texture)
#endif
	{
		int cullResult = CullingResult::VIEW_CULLED;

		//////////////////////////////////////////////////////////////////////////////
		// Compute bounding box and clamp to tile coordinates
		//////////////////////////////////////////////////////////////////////////////

		__mwi bbPixelMinX, bbPixelMinY, bbPixelMaxX, bbPixelMaxY;
		ComputeBoundingBox(bbPixelMinX, bbPixelMinY, bbPixelMaxX, bbPixelMaxY, pVtxX, pVtxY, scissor);

		// Clamp bounding box to tiles (it's already padded in computeBoundingBox)
		__mwi bbTileMinX = _mmw_srai_epi32(bbPixelMinX, TILE_WIDTH_SHIFT);
		__mwi bbTileMinY = _mmw_srai_epi32(bbPixelMinY, TILE_HEIGHT_SHIFT);
		__mwi bbTileMaxX = _mmw_srai_epi32(bbPixelMaxX, TILE_WIDTH_SHIFT);
		__mwi bbTileMaxY = _mmw_srai_epi32(bbPixelMaxY, TILE_HEIGHT_SHIFT);
		__mwi bbTileSizeX = _mmw_sub_epi32(bbTileMaxX, bbTileMinX);
		__mwi bbTileSizeY = _mmw_sub_epi32(bbTileMaxY, bbTileMinY);

		// Cull triangles with zero bounding box
		__mwi bboxSign = _mmw_or_epi32(_mmw_sub_epi32(bbTileSizeX, _mmw_set1_epi32(1)), _mmw_sub_epi32(bbTileSizeY, _mmw_set1_epi32(1)));
		triMask &= ~_mmw_movemask_ps(simd_cast<__mw>(bboxSign)) & SIMD_ALL_LANES_MASK;
		if (triMask == 0x0)
			return cullResult;

		if (!TEST_Z)
			cullResult = CullingResult::VISIBLE;

		//////////////////////////////////////////////////////////////////////////////
		// Set up screen space depth plane
		//////////////////////////////////////////////////////////////////////////////

		__mw zPixelDx, zPixelDy, zPixel0;
		InterpolationSetup(pVtxX, pVtxY, pVtxZ, zPixelDx, zPixelDy, zPixel0);

		// Compute z value at min corner of bounding box. Offset to make sure z is conservative for all 8x4 subtiles
		__mw bbMinXV0 = _mmw_sub_ps(_mmw_cvtepi32_ps(bbPixelMinX), pVtxX[0]);
		__mw bbMinYV0 = _mmw_sub_ps(_mmw_cvtepi32_ps(bbPixelMinY), pVtxY[0]);
		__mw zPlaneOffset = _mmw_fmadd_ps(zPixelDx, bbMinXV0, _mmw_fmadd_ps(zPixelDy, bbMinYV0, pVtxZ[0]));
		__mw zTileDx = _mmw_mul_ps(zPixelDx, _mmw_set1_ps((float)TILE_WIDTH));
		__mw zTileDy = _mmw_mul_ps(zPixelDy, _mmw_set1_ps((float)TILE_HEIGHT));
		if (TEST_Z)
		{
			zPlaneOffset = _mmw_add_ps(zPlaneOffset, _mmw_max_ps(_mmw_setzero_ps(), _mmw_mul_ps(zPixelDx, _mmw_set1_ps(SUB_TILE_WIDTH))));
			zPlaneOffset = _mmw_add_ps(zPlaneOffset, _mmw_max_ps(_mmw_setzero_ps(), _mmw_mul_ps(zPixelDy, _mmw_set1_ps(SUB_TILE_HEIGHT))));
		}
		else
		{
			zPlaneOffset = _mmw_add_ps(zPlaneOffset, _mmw_min_ps(_mmw_setzero_ps(), _mmw_mul_ps(zPixelDx, _mmw_set1_ps(SUB_TILE_WIDTH))));
			zPlaneOffset = _mmw_add_ps(zPlaneOffset, _mmw_min_ps(_mmw_setzero_ps(), _mmw_mul_ps(zPixelDy, _mmw_set1_ps(SUB_TILE_HEIGHT))));
		}

		// Compute Zmin and Zmax for the triangle (used to narrow the range for difficult tiles)
		__mw zMin = _mmw_min_ps(pVtxZ[0], _mmw_min_ps(pVtxZ[1], pVtxZ[2]));
		__mw zMax = _mmw_max_ps(pVtxZ[0], _mmw_max_ps(pVtxZ[1], pVtxZ[2]));

		//////////////////////////////////////////////////////////////////////////////
		// Set up texture (u, v) interpolation
		//////////////////////////////////////////////////////////////////////////////

		__mw uPixelDx, uPixelDy, uPixel0, vPixelDx, vPixelDy, vPixel0, uDerivConsts[3], vDerivConsts[3];
		if (TEXTURE_COORDINATES)
		{
			InterpolationSetup(pVtxX, pVtxY, pVtxU, uPixelDx, uPixelDy, uPixel0);
			InterpolationSetup(pVtxX, pVtxY, pVtxV, vPixelDx, vPixelDy, vPixel0);

			uDerivConsts[0] = _mmw_fmsub_ps(uPixelDx, zPixelDy, _mmw_mul_ps(uPixelDy, zPixelDx));
			uDerivConsts[1] = _mmw_fmsub_ps(uPixelDx, zPixel0,  _mmw_mul_ps(uPixel0,  zPixelDx));
			uDerivConsts[2] = _mmw_fmsub_ps(uPixelDy, zPixel0,  _mmw_mul_ps(uPixel0,  zPixelDy));

			vDerivConsts[0] = _mmw_fmsub_ps(vPixelDx, zPixelDy, _mmw_mul_ps(vPixelDy, zPixelDx));
			vDerivConsts[1] = _mmw_fmsub_ps(vPixelDx, zPixel0,  _mmw_mul_ps(vPixel0,  zPixelDx));
			vDerivConsts[2] = _mmw_fmsub_ps(vPixelDy, zPixel0,  _mmw_mul_ps(vPixel0,  zPixelDy));
		}

		//////////////////////////////////////////////////////////////////////////////
		// Sort vertices (v0 has lowest Y, and the rest is in winding order) and
		// compute edges. Also find the middle vertex and compute tile
		//////////////////////////////////////////////////////////////////////////////

#if PRECISE_COVERAGE != 0

		// Rotate the triangle in the winding order until v0 is the vertex with lowest Y value
		SortVertices(ipVtxX, ipVtxY);

		// Compute edges
		__mwi edgeX[3] = { _mmw_sub_epi32(ipVtxX[1], ipVtxX[0]), _mmw_sub_epi32(ipVtxX[2], ipVtxX[1]), _mmw_sub_epi32(ipVtxX[2], ipVtxX[0]) };
		__mwi edgeY[3] = { _mmw_sub_epi32(ipVtxY[1], ipVtxY[0]), _mmw_sub_epi32(ipVtxY[2], ipVtxY[1]), _mmw_sub_epi32(ipVtxY[2], ipVtxY[0]) };

		// Classify if the middle vertex is on the left or right and compute its position
		int midVtxRight = ~_mmw_movemask_ps(simd_cast<__mw>(edgeY[1]));
		__mwi midPixelX = _mmw_blendv_epi32(ipVtxX[1], ipVtxX[2], edgeY[1]);
		__mwi midPixelY = _mmw_blendv_epi32(ipVtxY[1], ipVtxY[2], edgeY[1]);
		__mwi midTileY = _mmw_srai_epi32(_mmw_max_epi32(midPixelY, SIMD_BITS_ZERO), TILE_HEIGHT_SHIFT + FP_BITS);
		__mwi bbMidTileY = _mmw_max_epi32(bbTileMinY, _mmw_min_epi32(bbTileMaxY, midTileY));

		// Compute edge events for the bottom of the bounding box, or for the middle tile in case of
		// the edge originating from the middle vertex.
		__mwi xDiffi[2], yDiffi[2];
		xDiffi[0] = _mmw_sub_epi32(ipVtxX[0], _mmw_slli_epi32(bbPixelMinX, FP_BITS));
		xDiffi[1] = _mmw_sub_epi32(midPixelX, _mmw_slli_epi32(bbPixelMinX, FP_BITS));
		yDiffi[0] = _mmw_sub_epi32(ipVtxY[0], _mmw_slli_epi32(bbPixelMinY, FP_BITS));
		yDiffi[1] = _mmw_sub_epi32(midPixelY, _mmw_slli_epi32(bbMidTileY, FP_BITS + TILE_HEIGHT_SHIFT));

		//////////////////////////////////////////////////////////////////////////////
		// Edge slope setup - Note we do not conform to DX/GL rasterization rules
		//////////////////////////////////////////////////////////////////////////////

		// Potentially flip edge to ensure that all edges have positive Y slope.
		edgeX[1] = _mmw_blendv_epi32(edgeX[1], _mmw_neg_epi32(edgeX[1]), edgeY[1]);
		edgeY[1] = _mmw_abs_epi32(edgeY[1]);

		// Compute floating point slopes
		__mw slope[3];
		slope[0] = _mmw_div_ps(_mmw_cvtepi32_ps(edgeX[0]), _mmw_cvtepi32_ps(edgeY[0]));
		slope[1] = _mmw_div_ps(_mmw_cvtepi32_ps(edgeX[1]), _mmw_cvtepi32_ps(edgeY[1]));
		slope[2] = _mmw_div_ps(_mmw_cvtepi32_ps(edgeX[2]), _mmw_cvtepi32_ps(edgeY[2]));

		// Modify slope of horizontal edges to make sure they mask out pixels above/below the edge. The slope is set to screen
		// width to mask out all pixels above or below the horizontal edge. We must also add a small bias to acount for that
		// vertices may end up off screen due to clipping. We're assuming that the round off error is no bigger than 1.0
		__mw  horizontalSlopeDelta = _mmw_set1_ps(2.0f * ((float)mWidth + 2.0f*(GUARD_BAND_PIXEL_SIZE + 1.0f)));
		__mwi horizontalSlope0 = _mmw_cmpeq_epi32(edgeY[0], _mmw_setzero_epi32());
		__mwi horizontalSlope1 = _mmw_cmpeq_epi32(edgeY[1], _mmw_setzero_epi32());
		slope[0] = _mmw_blendv_ps(slope[0], horizontalSlopeDelta, simd_cast<__mw>(horizontalSlope0));
		slope[1] = _mmw_blendv_ps(slope[1], _mmw_neg_ps(horizontalSlopeDelta), simd_cast<__mw>(horizontalSlope1));

		__mwi vy[3] = { yDiffi[0], yDiffi[1], yDiffi[0] };
		__mwi offset0 = _mmw_and_epi32(_mmw_add_epi32(yDiffi[0], _mmw_set1_epi32(FP_HALF_PIXEL - 1)), _mmw_set1_epi32((int)((~0u) << FP_BITS)));
		__mwi offset1 = _mmw_and_epi32(_mmw_add_epi32(yDiffi[1], _mmw_set1_epi32(FP_HALF_PIXEL - 1)), _mmw_set1_epi32((int)((~0u) << FP_BITS)));
		vy[0] = _mmw_blendv_epi32(yDiffi[0], offset0, horizontalSlope0);
		vy[1] = _mmw_blendv_epi32(yDiffi[1], offset1, horizontalSlope1);

		// Compute edge events for the bottom of the bounding box, or for the middle tile in case of
		// the edge originating from the middle vertex.
		__mwi slopeSign[3], absEdgeX[3];
		__mwi slopeTileDelta[3], eventStartRemainder[3], slopeTileRemainder[3], eventStart[3];
		for (int i = 0; i < 3; i++)
		{
			// Common, compute slope sign (used to propagate the remainder term when overflowing) is postive or negative x-direction
			slopeSign[i] = _mmw_blendv_epi32(_mmw_set1_epi32(1), _mmw_set1_epi32(-1), edgeX[i]);
			absEdgeX[i] = _mmw_abs_epi32(edgeX[i]);

			// Delta and error term for one vertical tile step. The exact delta is exactDelta = edgeX / edgeY, due to limited precision we
			// repersent the delta as delta = qoutient + remainder / edgeY, where quotient = int(edgeX / edgeY). In this case, since we step
			// one tile of scanlines at a time, the slope is computed for a tile-sized step.
			slopeTileDelta[i] = _mmw_cvttps_epi32(_mmw_mul_ps(slope[i], _mmw_set1_ps(FP_TILE_HEIGHT)));
			slopeTileRemainder[i] = _mmw_sub_epi32(_mmw_slli_epi32(absEdgeX[i], FP_TILE_HEIGHT_SHIFT), _mmw_mullo_epi32(_mmw_abs_epi32(slopeTileDelta[i]), edgeY[i]));

			// Jump to bottom scanline of tile row, this is the bottom of the bounding box, or the middle vertex of the triangle.
			// The jump can be in both positive and negative y-direction due to clipping / offscreen vertices.
			__mwi tileStartDir = _mmw_blendv_epi32(slopeSign[i], _mmw_neg_epi32(slopeSign[i]), vy[i]);
			__mwi tieBreaker = _mmw_blendv_epi32(_mmw_set1_epi32(0), _mmw_set1_epi32(1), tileStartDir);
			__mwi tileStartSlope = _mmw_cvttps_epi32(_mmw_mul_ps(slope[i], _mmw_cvtepi32_ps(_mmw_neg_epi32(vy[i]))));
			__mwi tileStartRemainder = _mmw_sub_epi32(_mmw_mullo_epi32(absEdgeX[i], _mmw_abs_epi32(vy[i])), _mmw_mullo_epi32(_mmw_abs_epi32(tileStartSlope), edgeY[i]));
			
			eventStartRemainder[i] = _mmw_sub_epi32(tileStartRemainder, tieBreaker);
			__mwi overflow = _mmw_srai_epi32(eventStartRemainder[i], 31);
			eventStartRemainder[i] = _mmw_add_epi32(eventStartRemainder[i], _mmw_and_epi32(overflow, edgeY[i]));
			eventStartRemainder[i] = _mmw_blendv_epi32(eventStartRemainder[i], _mmw_sub_epi32(_mmw_sub_epi32(edgeY[i], eventStartRemainder[i]), _mmw_set1_epi32(1)), vy[i]);
			
			//eventStart[i] = xDiffi[i & 1] + tileStartSlope + (overflow & tileStartDir) + _mmw_set1_epi32(FP_HALF_PIXEL - 1) + tieBreaker;
			eventStart[i] = _mmw_add_epi32(_mmw_add_epi32(xDiffi[i & 1], tileStartSlope), _mmw_and_epi32(overflow, tileStartDir));
			eventStart[i] = _mmw_add_epi32(_mmw_add_epi32(eventStart[i], _mmw_set1_epi32(FP_HALF_PIXEL - 1)), tieBreaker);
		}

#else // PRECISE_COVERAGE

		SortVertices(pVtxX, pVtxY);

		// Compute edges
		__mw edgeX[3] = { _mmw_sub_ps(pVtxX[1], pVtxX[0]), _mmw_sub_ps(pVtxX[2], pVtxX[1]), _mmw_sub_ps(pVtxX[2], pVtxX[0]) };
		__mw edgeY[3] = { _mmw_sub_ps(pVtxY[1], pVtxY[0]), _mmw_sub_ps(pVtxY[2], pVtxY[1]), _mmw_sub_ps(pVtxY[2], pVtxY[0]) };

		// Classify if the middle vertex is on the left or right and compute its position
		int midVtxRight = ~_mmw_movemask_ps(edgeY[1]);
		__mw midPixelX = _mmw_blendv_ps(pVtxX[1], pVtxX[2], edgeY[1]);
		__mw midPixelY = _mmw_blendv_ps(pVtxY[1], pVtxY[2], edgeY[1]);
		__mwi midTileY = _mmw_srai_epi32(_mmw_max_epi32(_mmw_cvttps_epi32(midPixelY), SIMD_BITS_ZERO), TILE_HEIGHT_SHIFT);
		__mwi bbMidTileY = _mmw_max_epi32(bbTileMinY, _mmw_min_epi32(bbTileMaxY, midTileY));

		//////////////////////////////////////////////////////////////////////////////
		// Edge slope setup - Note we do not conform to DX/GL rasterization rules
		//////////////////////////////////////////////////////////////////////////////

		// Compute floating point slopes
		__mw slope[3];
		slope[0] = _mmw_div_ps(edgeX[0], edgeY[0]);
		slope[1] = _mmw_div_ps(edgeX[1], edgeY[1]);
		slope[2] = _mmw_div_ps(edgeX[2], edgeY[2]);

		// Modify slope of horizontal edges to make sure they mask out pixels above/below the edge. The slope is set to screen
		// width to mask out all pixels above or below the horizontal edge. We must also add a small bias to acount for that
		// vertices may end up off screen due to clipping. We're assuming that the round off error is no bigger than 1.0
		__mw horizontalSlopeDelta = _mmw_set1_ps((float)mWidth + 2.0f*(GUARD_BAND_PIXEL_SIZE + 1.0f));
		slope[0] = _mmw_blendv_ps(slope[0], horizontalSlopeDelta, _mmw_cmpeq_ps(edgeY[0], _mmw_setzero_ps()));
		slope[1] = _mmw_blendv_ps(slope[1], _mmw_neg_ps(horizontalSlopeDelta), _mmw_cmpeq_ps(edgeY[1], _mmw_setzero_ps()));

		// Convert floaing point slopes to fixed point
		__mwi slopeFP[3];
		slopeFP[0] = _mmw_cvttps_epi32(_mmw_mul_ps(slope[0], _mmw_set1_ps(1 << FP_BITS)));
		slopeFP[1] = _mmw_cvttps_epi32(_mmw_mul_ps(slope[1], _mmw_set1_ps(1 << FP_BITS)));
		slopeFP[2] = _mmw_cvttps_epi32(_mmw_mul_ps(slope[2], _mmw_set1_ps(1 << FP_BITS)));

		// Fan out edge slopes to avoid (rare) cracks at vertices. We increase right facing slopes
		// by 1 LSB, which results in overshooting vertices slightly, increasing triangle coverage.
		// e0 is always right facing, e1 depends on if the middle vertex is on the left or right
		slopeFP[0] = _mmw_add_epi32(slopeFP[0], _mmw_set1_epi32(1));
		slopeFP[1] = _mmw_add_epi32(slopeFP[1], _mmw_srli_epi32(_mmw_not_epi32(simd_cast<__mwi>(edgeY[1])), 31));

		// Compute slope deltas for an SIMD_LANES scanline step (tile height)
		__mwi slopeTileDelta[3];
		slopeTileDelta[0] = _mmw_slli_epi32(slopeFP[0], TILE_HEIGHT_SHIFT);
		slopeTileDelta[1] = _mmw_slli_epi32(slopeFP[1], TILE_HEIGHT_SHIFT);
		slopeTileDelta[2] = _mmw_slli_epi32(slopeFP[2], TILE_HEIGHT_SHIFT);

		// Compute edge events for the bottom of the bounding box, or for the middle tile in case of
		// the edge originating from the middle vertex.
		__mwi xDiffi[2], yDiffi[2];
		xDiffi[0] = _mmw_slli_epi32(_mmw_sub_epi32(_mmw_cvttps_epi32(pVtxX[0]), bbPixelMinX), FP_BITS);
		xDiffi[1] = _mmw_slli_epi32(_mmw_sub_epi32(_mmw_cvttps_epi32(midPixelX), bbPixelMinX), FP_BITS);
		yDiffi[0] = _mmw_sub_epi32(_mmw_cvttps_epi32(pVtxY[0]), bbPixelMinY);
		yDiffi[1] = _mmw_sub_epi32(_mmw_cvttps_epi32(midPixelY), _mmw_slli_epi32(bbMidTileY, TILE_HEIGHT_SHIFT));

		__mwi eventStart[3];
		eventStart[0] = _mmw_sub_epi32(xDiffi[0], _mmw_mullo_epi32(slopeFP[0], yDiffi[0]));
		eventStart[1] = _mmw_sub_epi32(xDiffi[1], _mmw_mullo_epi32(slopeFP[1], yDiffi[1]));
		eventStart[2] = _mmw_sub_epi32(xDiffi[0], _mmw_mullo_epi32(slopeFP[2], yDiffi[0]));
#endif


		//////////////////////////////////////////////////////////////////////////////
		// Split bounding box into bottom - middle - top region.
		//////////////////////////////////////////////////////////////////////////////

		__mwi bbBottomIdx = _mmw_add_epi32(bbTileMinX, _mmw_mullo_epi32(bbTileMinY, _mmw_set1_epi32(mTilesWidth)));
		__mwi bbTopIdx = _mmw_add_epi32(bbTileMinX, _mmw_mullo_epi32(_mmw_add_epi32(bbTileMinY, bbTileSizeY), _mmw_set1_epi32(mTilesWidth)));
		__mwi bbMidIdx = _mmw_add_epi32(bbTileMinX, _mmw_mullo_epi32(midTileY, _mmw_set1_epi32(mTilesWidth)));

		//////////////////////////////////////////////////////////////////////////////
		// Loop over non-culled triangle and change SIMD axis to per-pixel
		//////////////////////////////////////////////////////////////////////////////
		while (triMask)
		{
			unsigned int triIdx = find_clear_lsb(&triMask);
			int triMidVtxRight = (midVtxRight >> triIdx) & 1;

			// Get Triangle Zmin zMax
			__mw zTriMax = _mmw_set1_ps(simd_f32(zMax)[triIdx]);
			__mw zTriMin = _mmw_set1_ps(simd_f32(zMin)[triIdx]);

			// Setup Zmin value for first set of 8x4 subtiles
			__mw z0 = _mmw_fmadd_ps(_mmw_set1_ps(simd_f32(zPixelDx)[triIdx]), SIMD_SUB_TILE_COL_OFFSET_F,
				_mmw_fmadd_ps(_mmw_set1_ps(simd_f32(zPixelDy)[triIdx]), SIMD_SUB_TILE_ROW_OFFSET_F, _mmw_set1_ps(simd_f32(zPlaneOffset)[triIdx])));
			float zx = simd_f32(zTileDx)[triIdx];
			float zy = simd_f32(zTileDy)[triIdx];

			// Get dimension of bounding box bottom, mid & top segments
			int bbWidth = simd_i32(bbTileSizeX)[triIdx];
			int bbHeight = simd_i32(bbTileSizeY)[triIdx];
			int tileRowIdx = simd_i32(bbBottomIdx)[triIdx];
			int tileMidRowIdx = simd_i32(bbMidIdx)[triIdx];
			int tileEndRowIdx = simd_i32(bbTopIdx)[triIdx];

			// Setup texture (u,v) interpolation parameters, TODO: Simdify
			TextureInterpolants texInterpolants;
			if (TEXTURE_COORDINATES)
			{
				texInterpolants.zInterpolant.mDx = _mmw_set1_ps(simd_f32(zPixelDx)[triIdx]);
				texInterpolants.zInterpolant.mDy = _mmw_set1_ps(simd_f32(zPixelDy)[triIdx]);
				texInterpolants.zInterpolant.mVal0 = _mmw_set1_ps(simd_f32(zPixel0)[triIdx]);
				texInterpolants.uInterpolant.mDx = _mmw_set1_ps(simd_f32(uPixelDx)[triIdx]);
				texInterpolants.uInterpolant.mDy = _mmw_set1_ps(simd_f32(uPixelDy)[triIdx]);
				texInterpolants.uInterpolant.mVal0 = _mmw_set1_ps(simd_f32(uPixel0)[triIdx]);
				texInterpolants.vInterpolant.mDx = _mmw_set1_ps(simd_f32(vPixelDx)[triIdx]);
				texInterpolants.vInterpolant.mDy = _mmw_set1_ps(simd_f32(vPixelDy)[triIdx]);
				texInterpolants.vInterpolant.mVal0 = _mmw_set1_ps(simd_f32(vPixel0)[triIdx]);
			
				texInterpolants.uDerivConsts[0] = _mmw_set1_ps(simd_f32(uDerivConsts[0])[triIdx]);
				texInterpolants.uDerivConsts[1] = _mmw_set1_ps(simd_f32(uDerivConsts[1])[triIdx]);
				texInterpolants.uDerivConsts[2] = _mmw_set1_ps(simd_f32(uDerivConsts[2])[triIdx]);

				texInterpolants.vDerivConsts[0] = _mmw_set1_ps(simd_f32(vDerivConsts[0])[triIdx]);
				texInterpolants.vDerivConsts[1] = _mmw_set1_ps(simd_f32(vDerivConsts[1])[triIdx]);
				texInterpolants.vDerivConsts[2] = _mmw_set1_ps(simd_f32(vDerivConsts[2])[triIdx]);
			}


			if (bbWidth > BIG_TRIANGLE && bbHeight > BIG_TRIANGLE) // For big triangles we use a more expensive but tighter traversal algorithm
			{
#if PRECISE_COVERAGE != 0
				if (triMidVtxRight)
					cullResult &= RasterizeTriangle<TEST_Z, 1, 1, TEXTURE_COORDINATES>(triIdx, bbWidth, tileRowIdx, tileMidRowIdx, tileEndRowIdx, eventStart, slope, slopeTileDelta, edgeY, absEdgeX, slopeSign, eventStartRemainder, slopeTileRemainder, zTriMin, zTriMax, z0, zx, zy, texInterpolants, texture);
				else
					cullResult &= RasterizeTriangle<TEST_Z, 1, 0, TEXTURE_COORDINATES>(triIdx, bbWidth, tileRowIdx, tileMidRowIdx, tileEndRowIdx, eventStart, slope, slopeTileDelta, edgeY, absEdgeX, slopeSign, eventStartRemainder, slopeTileRemainder, zTriMin, zTriMax, z0, zx, zy, texInterpolants, texture);
#else
				if (triMidVtxRight)
					cullResult &= RasterizeTriangle<TEST_Z, 1, 1, TEXTURE_COORDINATES>(triIdx, bbWidth, tileRowIdx, tileMidRowIdx, tileEndRowIdx, eventStart, slopeFP, slopeTileDelta, zTriMin, zTriMax, z0, zx, zy, texInterpolants, texture);
				else
					cullResult &= RasterizeTriangle<TEST_Z, 1, 0, TEXTURE_COORDINATES>(triIdx, bbWidth, tileRowIdx, tileMidRowIdx, tileEndRowIdx, eventStart, slopeFP, slopeTileDelta, zTriMin, zTriMax, z0, zx, zy, texInterpolants, texture);
#endif
			}
			else
			{
#if PRECISE_COVERAGE != 0
				if (triMidVtxRight)
					cullResult &= RasterizeTriangle<TEST_Z, 0, 1, TEXTURE_COORDINATES>(triIdx, bbWidth, tileRowIdx, tileMidRowIdx, tileEndRowIdx, eventStart, slope, slopeTileDelta, edgeY, absEdgeX, slopeSign, eventStartRemainder, slopeTileRemainder, zTriMin, zTriMax, z0, zx, zy, texInterpolants, texture);
				else
					cullResult &= RasterizeTriangle<TEST_Z, 0, 0, TEXTURE_COORDINATES>(triIdx, bbWidth, tileRowIdx, tileMidRowIdx, tileEndRowIdx, eventStart, slope, slopeTileDelta, edgeY, absEdgeX, slopeSign, eventStartRemainder, slopeTileRemainder, zTriMin, zTriMax, z0, zx, zy, texInterpolants, texture);
#else
				if (triMidVtxRight)
					cullResult &= RasterizeTriangle<TEST_Z, 0, 1, TEXTURE_COORDINATES>(triIdx, bbWidth, tileRowIdx, tileMidRowIdx, tileEndRowIdx, eventStart, slopeFP, slopeTileDelta, zTriMin, zTriMax, z0, zx, zy, texInterpolants, texture);
				else
					cullResult &= RasterizeTriangle<TEST_Z, 0, 0, TEXTURE_COORDINATES>(triIdx, bbWidth, tileRowIdx, tileMidRowIdx, tileEndRowIdx, eventStart, slopeFP, slopeTileDelta, zTriMin, zTriMax, z0, zx, zy, texInterpolants, texture);
#endif
			}

			if (TEST_Z && cullResult == CullingResult::VISIBLE)
				return CullingResult::VISIBLE;
		}

		return cullResult;
	}

	/*
	 * Rasterizes a list of triangles. Wrapper for the API function, but templated with TEST_Z to allow re-using the same rasterization code both when rasterizing occluders and preforming occlusion tests.
	 */
	template<int TEST_Z, int TEXTURE_COORDINATES>
	FORCE_INLINE CullingResult RenderTriangles(const float *inVtx, const unsigned int *inTris, int nTris, MaskedOcclusionTextureInternal *texture, const float *modelToClipMatrix, BackfaceWinding bfWinding, ClipPlanes clipPlaneMask, const VertexLayout &vtxLayout)
	{
		assert(mMaskedHiZBuffer != nullptr);

		if (TEST_Z)
			STATS_ADD(mStats.mOccludees.mNumProcessedTriangles, 1);
		else
			STATS_ADD(mStats.mOccluders.mNumProcessedTriangles, 1);

#if PRECISE_COVERAGE != 0
		int originalRoundingMode = _MM_GET_ROUNDING_MODE();
		_MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
#endif

		int clipHead = 0;
		int clipTail = 0;
		__m128 clipVtxBuffer[MAX_CLIPPED * 3], clipTexBuffer[MAX_CLIPPED * 3];

		int triIndex = 0;
		const unsigned int *inTrisPtr = inTris;
		int cullResult = CullingResult::VIEW_CULLED;
		bool fastGather = !TEXTURE_COORDINATES && vtxLayout.mStride == 16 && vtxLayout.mOffsetY == 4 && vtxLayout.mOffsetW == 12;

		while (triIndex < nTris || clipHead != clipTail)
		{
			//////////////////////////////////////////////////////////////////////////////
			// Assemble triangles from the index list
			//////////////////////////////////////////////////////////////////////////////
			__mw vtxX[3], vtxY[3], vtxW[3], vtxU[3], vtxV[3];
			unsigned int triMask = SIMD_ALL_LANES_MASK, triClipMask = SIMD_ALL_LANES_MASK;

			int numLanes = SIMD_LANES;
			if (clipHead != clipTail)
			{
				int clippedTris = clipHead > clipTail ? clipHead - clipTail : MAX_CLIPPED + clipHead - clipTail;
				clippedTris = min(clippedTris, SIMD_LANES);

				// Fill out SIMD registers by fetching more triangles.
				numLanes = max(0, min(SIMD_LANES - clippedTris, nTris - triIndex));
				if (numLanes > 0) {
					if (fastGather)
						GatherVerticesFast(vtxX, vtxY, vtxW, inVtx, inTrisPtr, numLanes);
					else
						GatherVertices<TEXTURE_COORDINATES>(vtxX, vtxY, vtxW, vtxU, vtxV, inVtx, inTrisPtr, numLanes, vtxLayout);

					TransformVerts(vtxX, vtxY, vtxW, modelToClipMatrix);
				}

				for (int clipTri = numLanes; clipTri < numLanes + clippedTris; clipTri++)
				{
					int triIdx = clipTail * 3;
					for (int i = 0; i < 3; i++)
					{
						simd_f32(vtxX[i])[clipTri] = simd_f32(clipVtxBuffer[triIdx + i])[0];
						simd_f32(vtxY[i])[clipTri] = simd_f32(clipVtxBuffer[triIdx + i])[1];
						simd_f32(vtxW[i])[clipTri] = simd_f32(clipVtxBuffer[triIdx + i])[2];
						if (TEXTURE_COORDINATES)
						{
							simd_f32(vtxU[i])[clipTri] = simd_f32(clipTexBuffer[triIdx + i])[0];
							simd_f32(vtxV[i])[clipTri] = simd_f32(clipTexBuffer[triIdx + i])[1];
						}
					}
					clipTail = (clipTail + 1) & (MAX_CLIPPED-1);
				}

				triIndex += numLanes;
				inTrisPtr += numLanes * 3;

				triMask = (1U << (clippedTris + numLanes)) - 1;
				triClipMask = (1U << numLanes) - 1; // Don't re-clip already clipped triangles
			}
			else
			{
				numLanes = min(SIMD_LANES, nTris - triIndex);
				triMask = (1U << numLanes) - 1;
				triClipMask = triMask;

				if (fastGather)
					GatherVerticesFast(vtxX, vtxY, vtxW, inVtx, inTrisPtr, numLanes);
				else
					GatherVertices<TEXTURE_COORDINATES>(vtxX, vtxY, vtxW, vtxU, vtxV, inVtx, inTrisPtr, numLanes, vtxLayout);

				TransformVerts(vtxX, vtxY, vtxW, modelToClipMatrix);
				triIndex += SIMD_LANES;
				inTrisPtr += SIMD_LANES*3;
			}

			//////////////////////////////////////////////////////////////////////////////
			// Clip transformed triangles
			//////////////////////////////////////////////////////////////////////////////

			if (clipPlaneMask != ClipPlanes::CLIP_PLANE_NONE)
				ClipTriangleAndAddToBuffer<TEXTURE_COORDINATES>(vtxX, vtxY, vtxW, vtxU, vtxV, clipVtxBuffer, clipTexBuffer, clipHead, triMask, triClipMask, clipPlaneMask);

			if (triMask == 0x0)
				continue;

			//////////////////////////////////////////////////////////////////////////////
			// Project, transform to screen space and perform backface culling. Note
			// that we use z = 1.0 / vtx.w for depth, which means that z = 0 is far and
			// z = 1 is near. We must also use a greater than depth test, and in effect
			// everything is reversed compared to regular z implementations.
			//////////////////////////////////////////////////////////////////////////////

			__mw pVtxX[3], pVtxY[3], pVtxZ[3], pVtxU[3], pVtxV[3];
#if PRECISE_COVERAGE != 0
			__mwi ipVtxX[3], ipVtxY[3];
			ProjectVertices(ipVtxX, ipVtxY, pVtxX, pVtxY, pVtxZ, vtxX, vtxY, vtxW);
#else
			ProjectVertices(pVtxX, pVtxY, pVtxZ, vtxX, vtxY, vtxW);
#endif
			if (TEXTURE_COORDINATES)
				ProjectTexCoords(pVtxU, pVtxV, pVtxZ, vtxU, vtxV);

			// Perform backface test.
			__mw triArea1 = _mmw_mul_ps(_mmw_sub_ps(pVtxX[1], pVtxX[0]), _mmw_sub_ps(pVtxY[2], pVtxY[0]));
			__mw triArea2 = _mmw_mul_ps(_mmw_sub_ps(pVtxX[0], pVtxX[2]), _mmw_sub_ps(pVtxY[0], pVtxY[1]));
			__mw triArea = _mmw_sub_ps(triArea1, triArea2);
			__mw ccwMask = _mmw_cmpgt_ps(triArea, _mmw_setzero_ps());

#if PRECISE_COVERAGE != 0
			triMask &= CullBackfaces(ipVtxX, ipVtxY, pVtxX, pVtxY, pVtxZ, ccwMask, bfWinding);
#else
			triMask &= CullBackfaces(pVtxX, pVtxY, pVtxZ, ccwMask, bfWinding);
#endif

			if (triMask == 0x0)
				continue;

			//////////////////////////////////////////////////////////////////////////////
			// Setup and rasterize a SIMD batch of triangles
			//////////////////////////////////////////////////////////////////////////////
#if PRECISE_COVERAGE != 0
			cullResult &= RasterizeTriangleBatch<TEST_Z, TEXTURE_COORDINATES>(ipVtxX, ipVtxY, pVtxX, pVtxY, pVtxZ, pVtxU, pVtxV, triMask, &mFullscreenScissor, texture);
#else
#define USE_ASM
#ifdef USE_ASM
			mASMRasterizeTriangleBatch(this, nullptr, nullptr, pVtxX, pVtxY, pVtxZ, nullptr, nullptr, triMask, &mFullscreenScissor);
#else
			cullResult &= RasterizeTriangleBatch<TEST_Z, TEXTURE_COORDINATES>(pVtxX, pVtxY, pVtxZ, pVtxU, pVtxV, triMask, &mFullscreenScissor, texture);
#endif
#endif

			if (TEST_Z && cullResult == CullingResult::VISIBLE) {
#if PRECISE_COVERAGE != 0
				_MM_SET_ROUNDING_MODE(originalRoundingMode);
#endif
				return CullingResult::VISIBLE;
			}
		}

#if PRECISE_COVERAGE != 0
		_MM_SET_ROUNDING_MODE(originalRoundingMode);
#endif
		return (CullingResult)cullResult;
	}

	CullingResult RenderTriangles(const float *inVtx, const unsigned int *inTris, int nTris, const float *modelToClipMatrix, BackfaceWinding bfWinding, ClipPlanes clipPlaneMask, const VertexLayout &vtxLayout) override
	{
		return (CullingResult)RenderTriangles<0, 0>(inVtx, inTris, nTris, nullptr, modelToClipMatrix, bfWinding, clipPlaneMask, vtxLayout);
	}

	CullingResult RenderTexturedTriangles(const float *inVtx, const unsigned int *inTris, int nTris, MaskedOcclusionTexture *texture, const float *modelToClipMatrix, BackfaceWinding bfWinding, ClipPlanes clipPlaneMask, const VertexLayout &vtxLayout) override
	{
		return (CullingResult)RenderTriangles<0, 1>(inVtx, inTris, nTris, (MaskedOcclusionTextureInternal*)texture, modelToClipMatrix, bfWinding, clipPlaneMask, vtxLayout);
	}

	CullingResult TestTriangles(const float *inVtx, const unsigned int *inTris, int nTris, const float *modelToClipMatrix, BackfaceWinding bfWinding, ClipPlanes clipPlaneMask, const VertexLayout &vtxLayout) const override
	{
#if QUERY_DEBUG_BUFFER != 0
		// Clear debug buffer (used to visualize queries). Cast required because this method is const
		memset((__mwi*)mQueryDebugBuffer, 0, sizeof(__mwi) * mTilesWidth * mTilesHeight);
#endif

		// Workaround because the RenderTriangles method is reused for both rendering occluders and performing queries, so it's not declared as const.
		// Still, it's nice for TestTriangles() to be declared as const to indicate it does not modify the HiZ buffer.
		MaskedOcclusionCullingPrivate *nonConst = (MaskedOcclusionCullingPrivate *)this;
		return (CullingResult)nonConst->RenderTriangles<1, 0>(inVtx, inTris, nTris, nullptr, modelToClipMatrix, bfWinding, clipPlaneMask, vtxLayout);
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Rectangle occlusion test
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	CullingResult TestRect(float xmin, float ymin, float xmax, float ymax, float wmin) const override
	{
		STATS_ADD(mStats.mOccludees.mNumProcessedRectangles, 1);
		assert(mMaskedHiZBuffer != nullptr);

#if QUERY_DEBUG_BUFFER != 0
		// Clear debug buffer (used to visualize queries). Cast required because this method is const
		memset((__mwi*)mQueryDebugBuffer, 0, sizeof(__mwi) * mTilesWidth * mTilesHeight);
#endif

		static const __m128i SIMD_TILE_PAD = _mm_setr_epi32(0, TILE_WIDTH, 0, TILE_HEIGHT);
		static const __m128i SIMD_TILE_PAD_MASK = _mm_setr_epi32(~(TILE_WIDTH - 1), ~(TILE_WIDTH - 1), ~(TILE_HEIGHT - 1), ~(TILE_HEIGHT - 1));
		static const __m128i SIMD_SUB_TILE_PAD = _mm_setr_epi32(0, SUB_TILE_WIDTH, 0, SUB_TILE_HEIGHT);
		static const __m128i SIMD_SUB_TILE_PAD_MASK = _mm_setr_epi32(~(SUB_TILE_WIDTH - 1), ~(SUB_TILE_WIDTH - 1), ~(SUB_TILE_HEIGHT - 1), ~(SUB_TILE_HEIGHT - 1));

		//////////////////////////////////////////////////////////////////////////////
		// Compute screen space bounding box and guard for out of bounds
		//////////////////////////////////////////////////////////////////////////////
		__m128  pixelBBox = _mmx_fmadd_ps(_mm_setr_ps(xmin, xmax, ymax, ymin), mIHalfSize, mICenter);
		__m128i pixelBBoxi = _mm_cvttps_epi32(pixelBBox);
		pixelBBoxi = _mmx_max_epi32(_mm_setzero_si128(), _mmx_min_epi32(mIScreenSize, pixelBBoxi));

		//////////////////////////////////////////////////////////////////////////////
		// Pad bounding box to (32xN) tiles. Tile BB is used for looping / traversal
		//////////////////////////////////////////////////////////////////////////////
		__m128i tileBBoxi = _mm_and_si128(_mm_add_epi32(pixelBBoxi, SIMD_TILE_PAD), SIMD_TILE_PAD_MASK);
		int txMin = simd_i32(tileBBoxi)[0] >> TILE_WIDTH_SHIFT;
		int txMax = simd_i32(tileBBoxi)[1] >> TILE_WIDTH_SHIFT;
		int tileRowIdx = (simd_i32(tileBBoxi)[2] >> TILE_HEIGHT_SHIFT)*mTilesWidth;
		int tileRowIdxEnd = (simd_i32(tileBBoxi)[3] >> TILE_HEIGHT_SHIFT)*mTilesWidth;

		if (simd_i32(tileBBoxi)[0] == simd_i32(tileBBoxi)[1] || simd_i32(tileBBoxi)[2] == simd_i32(tileBBoxi)[3])
			return CullingResult::VIEW_CULLED;

		///////////////////////////////////////////////////////////////////////////////
		// Pad bounding box to (8x4) subtiles. Skip SIMD lanes outside the subtile BB
		///////////////////////////////////////////////////////////////////////////////
		__m128i subTileBBoxi = _mm_and_si128(_mm_add_epi32(pixelBBoxi, SIMD_SUB_TILE_PAD), SIMD_SUB_TILE_PAD_MASK);
		__mwi stxmin = _mmw_set1_epi32(simd_i32(subTileBBoxi)[0] - 1); // - 1 to be able to use GT test
		__mwi stymin = _mmw_set1_epi32(simd_i32(subTileBBoxi)[2] - 1); // - 1 to be able to use GT test
		__mwi stxmax = _mmw_set1_epi32(simd_i32(subTileBBoxi)[1]);
		__mwi stymax = _mmw_set1_epi32(simd_i32(subTileBBoxi)[3]);

		// Setup pixel coordinates used to discard lanes outside subtile BB
		__mwi startPixelX = _mmw_add_epi32(SIMD_SUB_TILE_COL_OFFSET, _mmw_set1_epi32(simd_i32(tileBBoxi)[0]));
		__mwi pixelY = _mmw_add_epi32(SIMD_SUB_TILE_ROW_OFFSET, _mmw_set1_epi32(simd_i32(tileBBoxi)[2]));

		//////////////////////////////////////////////////////////////////////////////
		// Compute z from w. Note that z is reversed order, 0 = far, 1 = near, which
		// means we use a greater than test, so zMax is used to test for visibility.
		//////////////////////////////////////////////////////////////////////////////
		__mw zMax = _mmw_div_ps(_mmw_set1_ps(1.0f), _mmw_set1_ps(wmin));

		for (;;)
		{
			__mwi pixelX = startPixelX;
			for (int tx = txMin;;)
			{
				STATS_ADD(mStats.mOccludees.mNumTilesTraversed, 1);

				int tileIdx = tileRowIdx + tx;
				assert(tileIdx >= 0 && tileIdx < mTilesWidth*mTilesHeight);

				// Fetch zMin from masked hierarchical Z buffer
#if QUICK_MASK != 0
				__mw zBuf = mMaskedHiZBuffer[tileIdx].mZMin[0];
#else
				__mwi mask = mMaskedHiZBuffer[tileIdx].mMask;
				__mw zMin0 = _mmw_blendv_ps(mMaskedHiZBuffer[tileIdx].mZMin[0], mMaskedHiZBuffer[tileIdx].mZMin[1], simd_cast<__mw>(_mmw_cmpeq_epi32(mask, _mmw_set1_epi32(~0))));
				__mw zMin1 = _mmw_blendv_ps(mMaskedHiZBuffer[tileIdx].mZMin[1], mMaskedHiZBuffer[tileIdx].mZMin[0], simd_cast<__mw>(_mmw_cmpeq_epi32(mask, _mmw_setzero_epi32())));
				__mw zBuf = _mmw_min_ps(zMin0, zMin1);
#endif
				// Perform conservative greater than test against hierarchical Z buffer (zMax >= zBuf means the subtile is visible)
				__mwi zPass = simd_cast<__mwi>(_mmw_cmpge_ps(zMax, zBuf));	//zPass = zMax >= zBuf ? ~0 : 0

				// Mask out lanes corresponding to subtiles outside the bounding box
				__mwi bboxTestMin = _mmw_and_epi32(_mmw_cmpgt_epi32(pixelX, stxmin), _mmw_cmpgt_epi32(pixelY, stymin));
				__mwi bboxTestMax = _mmw_and_epi32(_mmw_cmpgt_epi32(stxmax, pixelX), _mmw_cmpgt_epi32(stymax, pixelY));
				__mwi boxMask = _mmw_and_epi32(bboxTestMin, bboxTestMax);
				zPass = _mmw_and_epi32(zPass, boxMask);

#if QUERY_DEBUG_BUFFER != 0
				__mwi debugVal = _mmw_blendv_epi32(_mmw_set1_epi32(0), _mmw_blendv_epi32(_mmw_set1_epi32(1), _mmw_set1_epi32(2), zPass), boxMask);
				mQueryDebugBuffer[tileIdx] = debugVal;
#endif
				// If not all tiles failed the conservative z test we can immediately terminate the test
				if (!_mmw_testz_epi32(zPass, zPass))
					return CullingResult::VISIBLE;

				if (++tx >= txMax)
					break;
				pixelX = _mmw_add_epi32(pixelX, _mmw_set1_epi32(TILE_WIDTH));
			}

			tileRowIdx += mTilesWidth;
			if (tileRowIdx >= tileRowIdxEnd)
				break;
			pixelY = _mmw_add_epi32(pixelY, _mmw_set1_epi32(TILE_HEIGHT));
		}

		return CullingResult::OCCLUDED;
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Sphere occlusion test
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	FORCE_INLINE void SphereBounds2D(float x, float w, float radius, float scale, float &xmin, float &xmax) const
	{
		float nx = -x / radius;
		float nw = -w / radius;
		float xw = nx * nx + nw * nw;

		float d = nw * sqrtf(xw - 1.0f);
		float tx0 = (nx - d) / xw;
		float tx1 = (nx + d) / xw;
		float wTangent0, wTangent1;
		if (nw != 0.0f)
		{
			wTangent0 = w + radius * (1.0f - tx0 * nx) / nw;
			wTangent1 = w + radius * (1.0f - tx1 * nx) / nw;
		}
		else
		{
			d = radius * sqrtf(1.0f - tx0 * tx0);
			wTangent0 = w + d;
			wTangent1 = w - d;
		}
		float xTangent0 = (x + radius * tx0) * scale;
		float xTangent1 = (x + radius * tx1) * scale;

		float proj0 = wTangent0 > 0.0f ? xTangent0 / wTangent0 : (xTangent0 < 0.0f ? -1.0f : 1.0f);
		float proj1 = wTangent1 > 0.0f ? xTangent1 / wTangent1 : (xTangent1 < 0.0f ? -1.0f : 1.0f);
		xmin = min(proj0, proj1);
		xmax = max(proj0, proj1);
	}

	CullingResult TestSphere(float viewCenterX, float viewCenterY, float viewCenterZ, float viewRadius, float xScale, float yScale) const override
	{
		STATS_ADD(mStats.mOccludees.mNumProcessedSpheres, 1);
		assert(mMaskedHiZBuffer != nullptr);

#if QUERY_DEBUG_BUFFER != 0
		// Clear debug buffer (used to visualize queries). Cast required because this method is const
		memset((__mwi*)mQueryDebugBuffer, 0, sizeof(__mwi) * mTilesWidth * mTilesHeight);
#endif

		static const __m128i SIMD_TILE_PAD = _mm_setr_epi32(0, TILE_WIDTH, 0, TILE_HEIGHT);
		static const __m128i SIMD_TILE_PAD_MASK = _mm_setr_epi32(~(TILE_WIDTH - 1), ~(TILE_WIDTH - 1), ~(TILE_HEIGHT - 1), ~(TILE_HEIGHT - 1));

		//////////////////////////////////////////////////////////////////////////////
		// Setup sphere-ray intersection and determine if camera is inside sphere
		//////////////////////////////////////////////////////////////////////////////

		float eqnCx4 = 4.0f*(viewCenterX*viewCenterX + viewCenterY*viewCenterY + viewCenterZ*viewCenterZ - viewRadius*viewRadius);
		if (eqnCx4 <= 0.0f)
			return VIEW_CULLED; // If inside sphere it's considered BF culled

		//////////////////////////////////////////////////////////////////////////////
		// Compute clip space bounding rectangle (using circle tangent lines)
		//////////////////////////////////////////////////////////////////////////////

		float xmin = -1.0f, xmax = 1.0f, ymin = -1.0f, ymax = 1.0f;
		SphereBounds2D(viewCenterX, viewCenterZ, viewRadius, xScale, xmin, xmax);
		SphereBounds2D(viewCenterY, viewCenterZ, viewRadius, yScale, ymin, ymax);

		//////////////////////////////////////////////////////////////////////////////
		// Compute screen space bounding box and guard for out of bounds
		//////////////////////////////////////////////////////////////////////////////
		__m128  pixelBBox = _mmx_fmadd_ps(_mm_setr_ps(xmin, xmax, ymax, ymin), mIHalfSize, mICenter);
		__m128i pixelBBoxi = _mm_cvttps_epi32(pixelBBox);
		pixelBBoxi = _mmx_max_epi32(_mm_setzero_si128(), _mmx_min_epi32(mIScreenSize, pixelBBoxi));

		//////////////////////////////////////////////////////////////////////////////
		// Pad bounding box to (32xN) tiles. Tile BB is used for looping / traversal
		//////////////////////////////////////////////////////////////////////////////
		__m128i tileBBoxi = _mm_and_si128(_mm_add_epi32(pixelBBoxi, SIMD_TILE_PAD), SIMD_TILE_PAD_MASK);
		int txMin = simd_i32(tileBBoxi)[0] >> TILE_WIDTH_SHIFT;
		int txMax = simd_i32(tileBBoxi)[1] >> TILE_WIDTH_SHIFT;
		int tileRowIdx = (simd_i32(tileBBoxi)[2] >> TILE_HEIGHT_SHIFT)*mTilesWidth;
		int tileRowIdxEnd = (simd_i32(tileBBoxi)[3] >> TILE_HEIGHT_SHIFT)*mTilesWidth;

		if (simd_i32(tileBBoxi)[0] == simd_i32(tileBBoxi)[1] || simd_i32(tileBBoxi)[2] == simd_i32(tileBBoxi)[3])
			return CullingResult::VIEW_CULLED;

		__mwi startPixelX = _mmw_add_epi32(SIMD_SUB_TILE_COL_OFFSET, _mmw_set1_epi32(simd_i32(tileBBoxi)[0]));
		__mwi startPixelY = _mmw_add_epi32(SIMD_SUB_TILE_ROW_OFFSET, _mmw_set1_epi32(simd_i32(tileBBoxi)[2]));

		///////////////////////////////////////////////////////////////////////////////
		// Setup clip space tile coordinates, used to calculate sphere overlap.
		///////////////////////////////////////////////////////////////////////////////

		// TODO: Move constants to setup code?
		const float viewSpaceTileX = (float)TILE_WIDTH / (simd_f32(mHalfWidth)[0] * xScale);
		const float viewSpaceTileY = (float)TILE_HEIGHT / (simd_f32(mHalfHeight)[0] * yScale);
		const float viewSpaceSubtileX = (float)SUB_TILE_WIDTH / (simd_f32(mHalfWidth)[0] * xScale);
		const float viewSpaceSubtileY = (float)SUB_TILE_HEIGHT / (simd_f32(mHalfHeight)[0] * yScale);
		float nCenterX = viewCenterX / viewCenterZ;
		float nCenterY = viewCenterY / viewCenterZ;
		__mw viewSpaceStartX = _mmw_div_ps(_mmw_sub_ps(_mmw_cvtepi32_ps(startPixelX), mCenterX), _mmw_mul_ps(mHalfWidth, _mmw_set1_ps(xScale)));
		__mw viewSpaceY      = _mmw_div_ps(_mmw_sub_ps(_mmw_cvtepi32_ps(startPixelY), mCenterY), _mmw_mul_ps(mHalfHeight, _mmw_set1_ps(yScale)));

		for (;;)
		{
			__mw viewSpaceX = viewSpaceStartX;
			for (int tx = txMin;;)
			{
				STATS_ADD(mStats.mOccludees.mNumTilesTraversed, 1);

				int tileIdx = tileRowIdx + tx;
				assert(tileIdx >= 0 && tileIdx < mTilesWidth*mTilesHeight);

				///////////////////////////////////////////////////////////////////////////////
				// Find ray direction that will generate wMin/zMax and is inside sphere
				///////////////////////////////////////////////////////////////////////////////

				// Tile corners, note max/min are swapped betweeen OGL/DX for y axis
				__mw subtileMinX = viewSpaceX;
				__mw subtileMaxX = _mmw_add_ps(subtileMinX, _mmw_set1_ps(viewSpaceSubtileX));
#if USE_D3D != 0
				__mw subtileMaxY = viewSpaceY;
				__mw subtileMinY = _mmw_add_ps(subtileMaxY, _mmw_set1_ps(viewSpaceSubtileY));
#else
				__mw subtileMinY = viewSpaceY;
				__mw subtileMaxY = _mmw_add_ps(subtileMinY, _mmw_set1_ps(viewSpaceSubtileY));
#endif

				// Dot products to classify which side sphere center is of tile boundaries
				__mw dpStartX = _mmw_fmsub_ps(subtileMinX, _mmw_set1_ps(viewCenterZ), _mmw_set1_ps(viewCenterX)); // (1, 0, startX) dot (centerX, centerY, centerZW)
				__mw dpStartY = _mmw_fmsub_ps(subtileMinY, _mmw_set1_ps(viewCenterZ), _mmw_set1_ps(viewCenterY)); // (0, 1, startY) dot (centerX, centerY, centerZW)
				__mw dpEndX   = _mmw_fmsub_ps(subtileMaxX, _mmw_set1_ps(viewCenterZ), _mmw_set1_ps(viewCenterX)); // (1, 0, endX) dot (centerX, centerY, centerZW)
				__mw dpEndY   = _mmw_fmsub_ps(subtileMaxY, _mmw_set1_ps(viewCenterZ), _mmw_set1_ps(viewCenterY)); // (0, 1, endY) dot (centerX, centerY, centerZW)

				// Setup ray through tile edge or towards sphere center
				__mw rayDirX, rayDirY;
				rayDirX = _mmw_blendv_ps(_mmw_blendv_ps(subtileMinX, _mmw_set1_ps(nCenterX), dpStartX), subtileMaxX, dpEndX);
				rayDirY = _mmw_blendv_ps(_mmw_blendv_ps(subtileMinY, _mmw_set1_ps(nCenterY), dpStartY), subtileMaxY, dpEndY);

				//////////////////////////////////////////////////////////////////////////////
				// Do ray-sphere intersection test
				//////////////////////////////////////////////////////////////////////////////

				// Setup quadratic equation: a*t^2 + b*t + c = 0
				__mw eqnA = _mmw_fmadd_ps(rayDirX, rayDirX, _mmw_fmadd_ps(rayDirY, rayDirY, _mmw_set1_ps(1.0f)));
				__mw eqnB = _mmw_mul_ps(_mmw_set1_ps(2.0f), _mmw_fmadd_ps(rayDirX, _mmw_set1_ps(viewCenterX), _mmw_fmadd_ps(rayDirY, _mmw_set1_ps(viewCenterY), _mmw_set1_ps(viewCenterZ))));

				// Find minimum valued solution:
				//     discr = b*b - 4*a*c
				//     t0 = (b - sqrt(discr)) / (2 * a) <-- Only valid solution after VF culling & not inside sphere
				//     t1 = (b + sqrt(discr)) / (2 * a)
				//     zMax = 1.0f / wMin = 1.0f / t0 = (2 * a) / (b - sqrt(discr))
				__mw discr = _mmw_fmsub_ps(eqnB, eqnB, _mmw_mul_ps(eqnA, _mmw_set1_ps(eqnCx4)));
				__mw zMax = _mmw_div_ps(_mmw_mul_ps(_mmw_set1_ps(2.0f), eqnA), _mmw_sub_ps(eqnB, _mmw_sqrt_ps(discr)));

				// Set mask in all tiles where intersection is found
				__mwi sphereMask = simd_cast<__mwi>(_mmw_cmpge_ps(discr, _mmw_set1_ps(0.0f)));
				
				///////////////////////////////////////////////////////////////////////////////
				// Test vs contents of HiZ buffer
				///////////////////////////////////////////////////////////////////////////////

				// Fetch zMin from masked hierarchical Z buffer
#if QUICK_MASK != 0
				__mw zBuf = mMaskedHiZBuffer[tileIdx].mZMin[0];
#else
				__mwi mask = mMaskedHiZBuffer[tileIdx].mMask;
				__mw zMin0 = _mmw_blendv_ps(mMaskedHiZBuffer[tileIdx].mZMin[0], mMaskedHiZBuffer[tileIdx].mZMin[1], simd_cast<__mw>(_mmw_cmpeq_epi32(mask, _mmw_set1_epi32(~0))));
				__mw zMin1 = _mmw_blendv_ps(mMaskedHiZBuffer[tileIdx].mZMin[1], mMaskedHiZBuffer[tileIdx].mZMin[0], simd_cast<__mw>(_mmw_cmpeq_epi32(mask, _mmw_setzero_epi32())));
				__mw zBuf = _mmw_min_ps(zMin0, zMin1);
#endif
				// Perform conservative greater than test against hierarchical Z buffer (zMax >= zBuf means the subtile is visible)
				__mwi zPass = simd_cast<__mwi>(_mmw_cmpge_ps(zMax, zBuf));	//zPass = zMax >= zBuf ? ~0 : 0

				// Mask out lanes corresponding to subtiles outside the bounding box
				zPass = _mmw_and_epi32(zPass, sphereMask);

#if QUERY_DEBUG_BUFFER != 0
				__mwi debugVal = _mmw_blendv_epi32(_mmw_set1_epi32(0), _mmw_blendv_epi32(_mmw_set1_epi32(1), _mmw_set1_epi32(2), zPass), sphereMask);
				mQueryDebugBuffer[tileIdx] = debugVal;
#endif
				// If not all tiles failed the conservative z test we can immediately terminate the test
				if (!_mmw_testz_epi32(zPass, zPass))
					return CullingResult::VISIBLE;

				if (++tx >= txMax)
					break;
				viewSpaceX = _mmw_add_ps(viewSpaceX, _mmw_set1_ps(viewSpaceTileX));
			}

			tileRowIdx += mTilesWidth;
			if (tileRowIdx >= tileRowIdxEnd)
				break;
			viewSpaceY = _mmw_add_ps(viewSpaceY, _mmw_set1_ps(viewSpaceTileY));
		}

		return CullingResult::OCCLUDED;
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Binning functions (for multithreading)
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	void BinTriangles(const float *inVtx, const unsigned int *inTris, int nTris, TriList *triLists, unsigned int nBinsW, unsigned int nBinsH, const float *modelToClipMatrix, BackfaceWinding bfWinding, ClipPlanes clipPlaneMask, const VertexLayout &vtxLayout) override
	{
		assert(mMaskedHiZBuffer != nullptr);

#if PRECISE_COVERAGE != 0
		int originalRoundingMode = _MM_GET_ROUNDING_MODE();
		_MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
#endif

		int clipHead = 0;
		int clipTail = 0;
		__m128 clipVtxBuffer[MAX_CLIPPED * 3], clipTexBuffer[MAX_CLIPPED * 3];

		int triIndex = 0;
		const unsigned int *inTrisPtr = inTris;
		bool fastGather = vtxLayout.mStride == 16 && vtxLayout.mOffsetY == 4 && vtxLayout.mOffsetW == 12;

		while (triIndex < nTris || clipHead != clipTail)
		{
			//////////////////////////////////////////////////////////////////////////////
			// Assemble triangles from the index list
			//////////////////////////////////////////////////////////////////////////////
			__mw vtxX[3], vtxY[3], vtxW[3], vtxU[3], vtxV[3];
			unsigned int triMask = SIMD_ALL_LANES_MASK, triClipMask = SIMD_ALL_LANES_MASK;

			int numLanes = SIMD_LANES;
			if (clipHead != clipTail)
			{
				int clippedTris = clipHead > clipTail ? clipHead - clipTail : MAX_CLIPPED + clipHead - clipTail;
				clippedTris = min(clippedTris, SIMD_LANES);

				// Fill out SIMD registers by fetching more triangles.
				numLanes = max(0, min(SIMD_LANES - clippedTris, nTris - triIndex));
				if (numLanes > 0) {
					if (fastGather)
						GatherVerticesFast(vtxX, vtxY, vtxW, inVtx, inTrisPtr, numLanes);
					else
						GatherVertices<0>(vtxX, vtxY, vtxW, vtxU, vtxV, inVtx, inTrisPtr, numLanes, vtxLayout);

					TransformVerts(vtxX, vtxY, vtxW, modelToClipMatrix);
				}

				for (int clipTri = numLanes; clipTri < numLanes + clippedTris; clipTri++)
				{
					int triIdx = clipTail * 3;
					for (int i = 0; i < 3; i++)
					{
						simd_f32(vtxX[i])[clipTri] = simd_f32(clipVtxBuffer[triIdx + i])[0];
						simd_f32(vtxY[i])[clipTri] = simd_f32(clipVtxBuffer[triIdx + i])[1];
						simd_f32(vtxW[i])[clipTri] = simd_f32(clipVtxBuffer[triIdx + i])[2];
					}
					clipTail = (clipTail + 1) & (MAX_CLIPPED - 1);
				}

				triIndex += numLanes;
				inTrisPtr += numLanes * 3;

				triMask = (1U << (clippedTris + numLanes)) - 1;
				triClipMask = (1U << numLanes) - 1; // Don't re-clip already clipped triangles
			}
			else
			{
				numLanes = min(SIMD_LANES, nTris - triIndex);
				triMask = (1U << numLanes) - 1;
				triClipMask = triMask;

				if (fastGather)
					GatherVerticesFast(vtxX, vtxY, vtxW, inVtx, inTrisPtr, numLanes);
				else
					GatherVertices<0>(vtxX, vtxY, vtxW, vtxU, vtxV, inVtx, inTrisPtr, numLanes, vtxLayout);

				TransformVerts(vtxX, vtxY, vtxW, modelToClipMatrix);

				triIndex += SIMD_LANES;
				inTrisPtr += SIMD_LANES * 3;
			}

			//////////////////////////////////////////////////////////////////////////////
			// Clip transformed triangles
			//////////////////////////////////////////////////////////////////////////////

			if (clipPlaneMask != ClipPlanes::CLIP_PLANE_NONE)
				ClipTriangleAndAddToBuffer<0>(vtxX, vtxY, vtxW, vtxU, vtxV, clipVtxBuffer, clipTexBuffer, clipHead, triMask, triClipMask, clipPlaneMask);

			if (triMask == 0x0)
				continue;

			//////////////////////////////////////////////////////////////////////////////
			// Project, transform to screen space and perform backface culling. Note
			// that we use z = 1.0 / vtx.w for depth, which means that z = 0 is far and
			// z = 1 is near. We must also use a greater than depth test, and in effect
			// everything is reversed compared to regular z implementations.
			//////////////////////////////////////////////////////////////////////////////

			__mw pVtxX[3], pVtxY[3], pVtxZ[3];
#if PRECISE_COVERAGE != 0
			__mwi ipVtxX[3], ipVtxY[3];
			ProjectVertices(ipVtxX, ipVtxY, pVtxX, pVtxY, pVtxZ, vtxX, vtxY, vtxW);
#else
			ProjectVertices(pVtxX, pVtxY, pVtxZ, vtxX, vtxY, vtxW);
#endif

			// Perform backface test.
			__mw triArea1 = _mmw_mul_ps(_mmw_sub_ps(pVtxX[1], pVtxX[0]), _mmw_sub_ps(pVtxY[2], pVtxY[0]));
			__mw triArea2 = _mmw_mul_ps(_mmw_sub_ps(pVtxX[0], pVtxX[2]), _mmw_sub_ps(pVtxY[0], pVtxY[1]));
			__mw triArea = _mmw_sub_ps(triArea1, triArea2);
			__mw ccwMask = _mmw_cmpgt_ps(triArea, _mmw_setzero_ps());

#if PRECISE_COVERAGE != 0
			triMask &= CullBackfaces(ipVtxX, ipVtxY, pVtxX, pVtxY, pVtxZ, ccwMask, bfWinding);
#else
			triMask &= CullBackfaces(pVtxX, pVtxY, pVtxZ, ccwMask, bfWinding);
#endif

			if (triMask == 0x0)
				continue;

			//////////////////////////////////////////////////////////////////////////////
			// Bin triangles
			//////////////////////////////////////////////////////////////////////////////

			unsigned int binWidth;
			unsigned int binHeight;
			ComputeBinWidthHeight(nBinsW, nBinsH, binWidth, binHeight);

			// Compute pixel bounding box
			__mwi bbPixelMinX, bbPixelMinY, bbPixelMaxX, bbPixelMaxY;
			ComputeBoundingBox(bbPixelMinX, bbPixelMinY, bbPixelMaxX, bbPixelMaxY, pVtxX, pVtxY, &mFullscreenScissor);

			while (triMask)
			{
				unsigned int triIdx = find_clear_lsb(&triMask);

				// Clamp bounding box to bins
				int startX = min(nBinsW-1, simd_i32(bbPixelMinX)[triIdx] / binWidth);
				int startY = min(nBinsH-1, simd_i32(bbPixelMinY)[triIdx] / binHeight);
				int endX = min(nBinsW, (simd_i32(bbPixelMaxX)[triIdx] + binWidth - 1) / binWidth);
				int endY = min(nBinsH, (simd_i32(bbPixelMaxY)[triIdx] + binHeight - 1) / binHeight);

				for (int y = startY; y < endY; ++y)
				{
					for (int x = startX; x < endX; ++x)
					{
						int binIdx = x + y * nBinsW;
						unsigned int writeTriIdx = triLists[binIdx].mTriIdx;
						for (int i = 0; i < 3; ++i)
						{
#if PRECISE_COVERAGE != 0
							((int*)triLists[binIdx].mPtr)[i * 3 + writeTriIdx * 9 + 0] = simd_i32(ipVtxX[i])[triIdx];
							((int*)triLists[binIdx].mPtr)[i * 3 + writeTriIdx * 9 + 1] = simd_i32(ipVtxY[i])[triIdx];
#else
							triLists[binIdx].mPtr[i * 3 + writeTriIdx * 9 + 0] = simd_f32(pVtxX[i])[triIdx];
							triLists[binIdx].mPtr[i * 3 + writeTriIdx * 9 + 1] = simd_f32(pVtxY[i])[triIdx];
#endif
							triLists[binIdx].mPtr[i * 3 + writeTriIdx * 9 + 2] = simd_f32(pVtxZ[i])[triIdx];
						}
						triLists[binIdx].mTriIdx++;
					}
				}
			}
		}
#if PRECISE_COVERAGE != 0
		_MM_SET_ROUNDING_MODE(originalRoundingMode);
#endif
	}

	void RenderTrilist(const TriList &triList, const ScissorRect *scissor) override
	{
		assert(mMaskedHiZBuffer != nullptr);

		// Setup fullscreen scissor rect as default
		scissor = scissor == nullptr ? &mFullscreenScissor : scissor;

		for (unsigned int i = 0; i < triList.mTriIdx; i += SIMD_LANES)
		{
			//////////////////////////////////////////////////////////////////////////////
			// Fetch triangle vertices
			//////////////////////////////////////////////////////////////////////////////

			unsigned int numLanes = min((unsigned int)SIMD_LANES, triList.mTriIdx - i);
			unsigned int triMask = (1U << numLanes) - 1;

			__mw pVtxX[3], pVtxY[3], pVtxZ[3], pVtxU[3], pVtxV[3];
#if PRECISE_COVERAGE != 0
			__mwi ipVtxX[3], ipVtxY[3];
			for (unsigned int l = 0; l < numLanes; ++l)
			{
				unsigned int triIdx = i + l;
				for (int v = 0; v < 3; ++v)
				{
					simd_i32(ipVtxX[v])[l] = ((int*)triList.mPtr)[v * 3 + triIdx * 9 + 0];
					simd_i32(ipVtxY[v])[l] = ((int*)triList.mPtr)[v * 3 + triIdx * 9 + 1];
					simd_f32(pVtxZ[v])[l] = triList.mPtr[v * 3 + triIdx * 9 + 2];
				}
			}

			for (int v = 0; v < 3; ++v)
			{
				pVtxX[v] = _mmw_mul_ps(_mmw_cvtepi32_ps(ipVtxX[v]), _mmw_set1_ps(FP_INV));
				pVtxY[v] = _mmw_mul_ps(_mmw_cvtepi32_ps(ipVtxY[v]), _mmw_set1_ps(FP_INV));
			}

			//////////////////////////////////////////////////////////////////////////////
			// Setup and rasterize a SIMD batch of triangles
			//////////////////////////////////////////////////////////////////////////////

			RasterizeTriangleBatch<false, 0>(ipVtxX, ipVtxY, pVtxX, pVtxY, pVtxZ, pVtxU, pVtxV, triMask, scissor);
#else
			for (unsigned int l = 0; l < numLanes; ++l)
			{
				unsigned int triIdx = i + l;
				for (int v = 0; v < 3; ++v)
				{
					simd_f32(pVtxX[v])[l] = triList.mPtr[v * 3 + triIdx * 9 + 0];
					simd_f32(pVtxY[v])[l] = triList.mPtr[v * 3 + triIdx * 9 + 1];
					simd_f32(pVtxZ[v])[l] = triList.mPtr[v * 3 + triIdx * 9 + 2];
				}
			}

			//////////////////////////////////////////////////////////////////////////////
			// Setup and rasterize a SIMD batch of triangles
			//////////////////////////////////////////////////////////////////////////////

			RasterizeTriangleBatch<false, 0>(pVtxX, pVtxY, pVtxZ, pVtxU, pVtxV, triMask, scissor, nullptr);
#endif

		}
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Debugging and statistics
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	MaskedOcclusionCulling::Implementation GetImplementation() override
	{
		return gInstructionSet;
	}

	void ComputePixelDepthBuffer(float *depthData) override
	{
		assert(mMaskedHiZBuffer != nullptr);
		for (int y = 0; y < mHeight; y++)
		{
			for (int x = 0; x < mWidth; x++)
			{
				// Compute 32xN tile index (SIMD value offset)
				int tx = x / TILE_WIDTH;
				int ty = y / TILE_HEIGHT;
				int tileIdx = ty * mTilesWidth + tx;

				// Compute 8x4 subtile index (SIMD lane offset)
				int stx = (x % TILE_WIDTH) / SUB_TILE_WIDTH;
				int sty = (y % TILE_HEIGHT) / SUB_TILE_HEIGHT;
				int subTileIdx = sty * 4 + stx;

				// Compute pixel index in subtile (bit index in 32-bit word)
				int px = (x % SUB_TILE_WIDTH);
				int py = (y % SUB_TILE_HEIGHT);
				int bitIdx = py * 8 + px;

				int pixelLayer = (simd_i32(mMaskedHiZBuffer[tileIdx].mMask)[subTileIdx] >> bitIdx) & 1;
				float pixelDepth = simd_f32(mMaskedHiZBuffer[tileIdx].mZMin[pixelLayer])[subTileIdx];

				depthData[y * mWidth + x] = pixelDepth;
			}
		}
	}

	void ComputePixelQueryBuffer(unsigned int *queryResult) override
	{
#if QUERY_DEBUG_BUFFER != 0
		assert(mQueryDebugBuffer != nullptr);
		for (int y = 0; y < mHeight; y++)
		{
			for (int x = 0; x < mWidth; x++)
			{
				// Compute 32xN tile index (SIMD value offset)
				int tx = x / TILE_WIDTH;
				int ty = y / TILE_HEIGHT;
				int tileIdx = ty * mTilesWidth + tx;

				// Compute 8x4 subtile index (SIMD lane offset)
				int stx = (x % TILE_WIDTH) / SUB_TILE_WIDTH;
				int sty = (y % TILE_HEIGHT) / SUB_TILE_HEIGHT;
				int subTileIdx = sty * 4 + stx;

				queryResult[y * mWidth + x] = simd_i32(mQueryDebugBuffer[tileIdx])[subTileIdx];
			}
		}
#else
		UNUSED_PARAMETER(queryResult);
#endif
	}

	OcclusionCullingStatistics GetStatistics() override
	{
		return mStats;
	}
};
