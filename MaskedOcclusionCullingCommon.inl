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
// Common SIMD math utility functions
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T> FORCE_INLINE T max(const T &a, const T &b) { return a > b ? a : b; }
template<typename T> FORCE_INLINE T min(const T &a, const T &b) { return a < b ? a : b; }

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Common defines and constants
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define SIMD_ALL_LANES_MASK    ((1 << SIMD_LANES) - 1)

// Tile dimensions are 32xN pixels. These values are not tweakable and the code must also be modified
// to support different tile sizes as it is tightly coupled with the SSE/AVX register size
#define TILE_WIDTH_SHIFT       5
#define TILE_WIDTH             (1 << TILE_WIDTH_SHIFT)
#define TILE_HEIGHT            (1 << TILE_HEIGHT_SHIFT)

// Sub-tiles (used for updating the masked HiZ buffer) are 8x4 tiles, so there are 4x2 sub-tiles in a tile
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
	struct ZTile
	{
		__mw        mZMin[2];
		__mwi       mMask;
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

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Constructors and state handling
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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
	 * The first triangle is always written back into the SIMD-batch, and remaining triangles are written to the clippedTrisBuffer
	 * scratchpad memory.
	 */
	template<int TEXTURE_COORDINATES>
	FORCE_INLINE void ClipTriangleAndAddToBuffer(__mw *vtxX, __mw *vtxY, __mw *vtxW, __mw *vtxU, __mw *vtxV, __m128 *clippedTrisBuffer, int &clipWriteIdx, unsigned int &triMask, unsigned int triClipMask, ClipPlanes clipPlaneMask)
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
					if (TEXTURE_COORDINATES)
					{
						clippedTrisBuffer[clipWriteIdx * 6 + 0] = vtxBuf[bufIdx][0];
						clippedTrisBuffer[clipWriteIdx * 6 + 1] = vtxBuf[bufIdx][i];
						clippedTrisBuffer[clipWriteIdx * 6 + 2] = vtxBuf[bufIdx][i + 1];
						clippedTrisBuffer[clipWriteIdx * 6 + 3] = texBuf[bufIdx][0];
						clippedTrisBuffer[clipWriteIdx * 6 + 4] = texBuf[bufIdx][i];
						clippedTrisBuffer[clipWriteIdx * 6 + 5] = texBuf[bufIdx][i + 1];
						clipWriteIdx = (clipWriteIdx + 1) & (MAX_CLIPPED - 1);
					}
					else
					{
						clippedTrisBuffer[clipWriteIdx * 3 + 0] = vtxBuf[bufIdx][0];
						clippedTrisBuffer[clipWriteIdx * 3 + 1] = vtxBuf[bufIdx][i];
						clippedTrisBuffer[clipWriteIdx * 3 + 2] = vtxBuf[bufIdx][i + 1];
						clipWriteIdx = (clipWriteIdx + 1) & (MAX_CLIPPED - 1);
					}
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
	FORCE_INLINE void InterpolationSetup(const __mw *pVtxX, const __mw *pVtxY, const __mw *pVtxA, __mw &aPixelDx, __mw &aPixelDy) const
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
		__mwi rastMask = _mmw_transpose_epi8(coverage);
		__mwi deadLane = _mmw_cmpeq_epi32(rastMask, SIMD_BITS_ZERO);

		// Mask out all subtiles failing the depth test (don't update these subtiles)
		deadLane = _mmw_or_epi32(deadLane, _mmw_srai_epi32(simd_cast<__mwi>(_mmw_sub_ps(zTriv, zMin[0])), 31));
		rastMask = _mmw_andnot_epi32(deadLane, rastMask);

		// Use distance heuristic to discard layer 1 if incoming triangle is significantly nearer to observer
		// than the buffer contents. See Section 3.2 in "Masked Software Occlusion Culling"
		__mwi coveredLane = _mmw_cmpeq_epi32(rastMask, SIMD_BITS_ONE);
		__mw diff = _mmw_fmsub_ps(zMin[1], _mmw_set1_ps(2.0f), _mmw_add_ps(zTriv, zMin[0]));
		__mwi discardLayerMask = _mmw_andnot_epi32(deadLane, _mmw_or_epi32(_mmw_srai_epi32(simd_cast<__mwi>(diff), 31), coveredLane));

		// Update the mask with incoming triangle coverage
		mask = _mmw_or_epi32(_mmw_andnot_epi32(discardLayerMask, mask), rastMask);

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

		// Swizzle coverage mask to 8x4 subtiles
		__mwi rastMask = _mmw_transpose_epi8(coverage);

		// Perform individual depth tests with layer 0 & 1 and mask out all failing pixels
		__mw sdist0 = _mmw_sub_ps(zMin[0], zTriv);
		__mw sdist1 = _mmw_sub_ps(zMin[1], zTriv);
		__mwi sign0 = _mmw_srai_epi32(simd_cast<__mwi>(sdist0), 31);
		__mwi sign1 = _mmw_srai_epi32(simd_cast<__mwi>(sdist1), 31);
		__mwi triMask = _mmw_and_epi32(rastMask, _mmw_or_epi32(_mmw_andnot_epi32(mask, sign0), _mmw_and_epi32(mask, sign1)));

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

	FORCE_INLINE __mwi TextureLookup(int tileIdx, __mwi coverageMask, __mw zDist0t,
		float zPixel0, float zPixelDx, float zPixelDy,
		float uPixel0, float uPixelDx, float uPixelDy,
		float vPixel0, float vPixelDx, float vPixelDy)
	{
		// Do z-cull. The texture lookup is expensive, so we want to remove as much work as possible
		coverageMask = _mmw_andnot_epi32(_mmw_srai_epi32(simd_cast<__mwi>(zDist0t), 31), coverageMask);
		unsigned int subtilesMask = ~_mmw_movemask_ps(simd_cast<__mw>(_mmw_cmpeq_epi32(coverageMask, _mmw_set1_epi32(0))));

		while (subtilesMask)
		{
			unsigned int subtileIdx = find_clear_lsb(&subtileIdx);

		}

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
		float zPixel0, float zPixelDx, float zPixelDy, 
		float uPixel0, float uPixelDx, float uPixelDy, 
		float vPixel0, float vPixelDx, float vPixelDy)
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

				// Perform conservative texture lookup for alpha tested triangles
				//if (TEXTURE_COORDINATES)
				//	accumulatedMask = _mmw_and_epi32(accumulatedMask, TextureLookup())

				if (TEST_Z)
				{
					// Perform a conservative visibility test (test zMax against buffer for each covered 8x4 subtile)
					__mw zSubTileMax = _mmw_min_ps(z0, zTriMax);
					__mwi zPass = simd_cast<__mwi>(_mmw_cmpge_ps(zSubTileMax, zMinBuf));

					__mwi rastMask = _mmw_transpose_epi8(accumulatedMask);
					__mwi deadLane = _mmw_cmpeq_epi32(rastMask, SIMD_BITS_ZERO);
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
					UpdateTileQuick(tileIdx, accumulatedMask, zSubTileMin);
#else
					UpdateTileAccurate(tileIdx, accumulatedMask, zSubTileMin);
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
		float zPixelDx, float zPixelDy, float uPixelDx, float uPixelDy, float vPixelDx, float vPixelDy)
#else
	FORCE_INLINE int RasterizeTriangle(unsigned int triIdx, int bbWidth, int tileRowIdx, int tileMidRowIdx, int tileEndRowIdx, 
		const __mwi *eventStart, const __mwi *slope, const __mwi *slopeTileDelta,
		const __mw &zTriMin, const __mw &zTriMax, __mw &z0, float zx, float zy,
		float zPixelDx, float zPixelDy, float uPixelDx, float uPixelDy, float vPixelDx, float vPixelDy)
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
				cullResult = TraverseScanline<TEST_Z, 1, 1, TEXTURE_COORDINATES>(start, end, tileRowIdx, 0, 2, triEvent, zTriMin, zTriMax, z0, zx);

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
					cullResult = TraverseScanline<TEST_Z, 2, 1, TEXTURE_COORDINATES>(start, end, tileRowIdx, 0, 2, triEvent, zTriMin, zTriMax, z0, zx);
				else
					cullResult = TraverseScanline<TEST_Z, 1, 2, TEXTURE_COORDINATES>(start, end, tileRowIdx, 0, 2, triEvent, zTriMin, zTriMax, z0, zx);

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
					cullResult = TraverseScanline<TEST_Z, 1, 1, TEXTURE_COORDINATES>(start, end, tileRowIdx, MID_VTX_RIGHT + 0, MID_VTX_RIGHT + 1, triEvent, zTriMin, zTriMax, z0, zx);

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
					cullResult = TraverseScanline<TEST_Z, 1, 1, TEXTURE_COORDINATES>(start, end, tileRowIdx, MID_VTX_RIGHT + 0, MID_VTX_RIGHT + 1, triEvent, zTriMin, zTriMax, z0, zx);

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
	FORCE_INLINE int RasterizeTriangleBatch(__mwi ipVtxX[3], __mwi ipVtxY[3], __mw pVtxX[3], __mw pVtxY[3], __mw pVtxZ[3], __mw pVtxU[3], __mw pVtxV[3], unsigned int triMask, const ScissorRect *scissor)
#else
	FORCE_INLINE int RasterizeTriangleBatch(__mw pVtxX[3], __mw pVtxY[3], __mw pVtxZ[3], __mw pVtxU[3], __mw pVtxV[3], unsigned int triMask, const ScissorRect *scissor)
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

		__mw zPixelDx, zPixelDy;
		InterpolationSetup(pVtxX, pVtxY, pVtxZ, zPixelDx, zPixelDy);

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

		__mw uPixelDx, uPixelDy, vPixelDx, vPixelDy;
		if (TEXTURE_COORDINATES)
		{
			InterpolationSetup(pVtxX, pVtxY, pVtxU, uPixelDx, uPixelDy);
			InterpolationSetup(pVtxX, pVtxY, pVtxV, vPixelDx, vPixelDy);
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

			// Setup texture (u,v) interpolation parameters
			float zPxDx = 0.0f, zPxDy = 0.0f, uPxDx = 0.0f, uPxDy = 0.0f, vPxDx = 0.0f, vPxDy = 0.0f;
			if (TEXTURE_COORDINATES)
			{
				zPxDx = simd_f32(zPixelDx)[triIdx];
				zPxDy = simd_f32(zPixelDy)[triIdx];
				uPxDx = simd_f32(uPixelDx)[triIdx];
				uPxDy = simd_f32(uPixelDy)[triIdx];
				vPxDx = simd_f32(vPixelDx)[triIdx];
				vPxDy = simd_f32(vPixelDy)[triIdx];
			}


			if (bbWidth > BIG_TRIANGLE && bbHeight > BIG_TRIANGLE) // For big triangles we use a more expensive but tighter traversal algorithm
			{
#if PRECISE_COVERAGE != 0
				if (triMidVtxRight)
					cullResult &= RasterizeTriangle<TEST_Z, 1, 1, TEXTURE_COORDINATES>(triIdx, bbWidth, tileRowIdx, tileMidRowIdx, tileEndRowIdx, eventStart, slope, slopeTileDelta, edgeY, absEdgeX, slopeSign, eventStartRemainder, slopeTileRemainder, zTriMin, zTriMax, z0, zx, zy, zPxDx, zPxDy, uPxDx, uPxDy, vPxDx, vPxDy);
				else
					cullResult &= RasterizeTriangle<TEST_Z, 1, 0, TEXTURE_COORDINATES>(triIdx, bbWidth, tileRowIdx, tileMidRowIdx, tileEndRowIdx, eventStart, slope, slopeTileDelta, edgeY, absEdgeX, slopeSign, eventStartRemainder, slopeTileRemainder, zTriMin, zTriMax, z0, zx, zy, zPxDx, zPxDy, uPxDx, uPxDy, vPxDx, vPxDy);
#else
				if (triMidVtxRight)
					cullResult &= RasterizeTriangle<TEST_Z, 1, 1, TEXTURE_COORDINATES>(triIdx, bbWidth, tileRowIdx, tileMidRowIdx, tileEndRowIdx, eventStart, slopeFP, slopeTileDelta, zTriMin, zTriMax, z0, zx, zy, zPxDx, zPxDy, uPxDx, uPxDy, vPxDx, vPxDy);
				else
					cullResult &= RasterizeTriangle<TEST_Z, 1, 0, TEXTURE_COORDINATES>(triIdx, bbWidth, tileRowIdx, tileMidRowIdx, tileEndRowIdx, eventStart, slopeFP, slopeTileDelta, zTriMin, zTriMax, z0, zx, zy, zPxDx, zPxDy, uPxDx, uPxDy, vPxDx, vPxDy);
#endif
			}
			else
			{
#if PRECISE_COVERAGE != 0
				if (triMidVtxRight)
					cullResult &= RasterizeTriangle<TEST_Z, 0, 1, TEXTURE_COORDINATES>(triIdx, bbWidth, tileRowIdx, tileMidRowIdx, tileEndRowIdx, eventStart, slope, slopeTileDelta, edgeY, absEdgeX, slopeSign, eventStartRemainder, slopeTileRemainder, zTriMin, zTriMax, z0, zx, zy, zPxDx, zPxDy, uPxDx, uPxDy, vPxDx, vPxDy);
				else
					cullResult &= RasterizeTriangle<TEST_Z, 0, 0, TEXTURE_COORDINATES>(triIdx, bbWidth, tileRowIdx, tileMidRowIdx, tileEndRowIdx, eventStart, slope, slopeTileDelta, edgeY, absEdgeX, slopeSign, eventStartRemainder, slopeTileRemainder, zTriMin, zTriMax, z0, zx, zy, zPxDx, zPxDy, uPxDx, uPxDy, vPxDx, vPxDy);
#else
				if (triMidVtxRight)
					cullResult &= RasterizeTriangle<TEST_Z, 0, 1, TEXTURE_COORDINATES>(triIdx, bbWidth, tileRowIdx, tileMidRowIdx, tileEndRowIdx, eventStart, slopeFP, slopeTileDelta, zTriMin, zTriMax, z0, zx, zy, zPxDx, zPxDy, uPxDx, uPxDy, vPxDx, vPxDy);
				else
					cullResult &= RasterizeTriangle<TEST_Z, 0, 0, TEXTURE_COORDINATES>(triIdx, bbWidth, tileRowIdx, tileMidRowIdx, tileEndRowIdx, eventStart, slopeFP, slopeTileDelta, zTriMin, zTriMax, z0, zx, zy, zPxDx, zPxDy, uPxDx, uPxDy, vPxDx, vPxDy);
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
	FORCE_INLINE CullingResult RenderTriangles(const float *inVtx, const unsigned int *inTris, int nTris, const float *modelToClipMatrix, BackfaceWinding bfWinding, ClipPlanes clipPlaneMask, const VertexLayout &vtxLayout)
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
		__m128 clipTriBuffer[MAX_CLIPPED * 3];

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
						GatherVertices(vtxX, vtxY, vtxW, inVtx, inTrisPtr, numLanes, vtxLayout);

					TransformVerts(vtxX, vtxY, vtxW, modelToClipMatrix);
				}

				for (int clipTri = numLanes; clipTri < numLanes + clippedTris; clipTri++)
				{
					int triIdx = clipTail * 3;
					for (int i = 0; i < 3; i++)
					{
						simd_f32(vtxX[i])[clipTri] = simd_f32(clipTriBuffer[triIdx + i])[0];
						simd_f32(vtxY[i])[clipTri] = simd_f32(clipTriBuffer[triIdx + i])[1];
						simd_f32(vtxW[i])[clipTri] = simd_f32(clipTriBuffer[triIdx + i])[2];
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
					GatherVertices(vtxX, vtxY, vtxW, inVtx, inTrisPtr, numLanes, vtxLayout);

				TransformVerts(vtxX, vtxY, vtxW, modelToClipMatrix);
				triIndex += SIMD_LANES;
				inTrisPtr += SIMD_LANES*3;
			}

			//////////////////////////////////////////////////////////////////////////////
			// Clip transformed triangles
			//////////////////////////////////////////////////////////////////////////////

			if (clipPlaneMask != ClipPlanes::CLIP_PLANE_NONE)
				ClipTriangleAndAddToBuffer<TEXTURE_COORDINATES>(vtxX, vtxY, vtxW, vtxU, vtxV, clipTriBuffer, clipHead, triMask, triClipMask, clipPlaneMask);

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
			cullResult &= RasterizeTriangleBatch<TEST_Z, TEXTURE_COORDINATES>(ipVtxX, ipVtxY, pVtxX, pVtxY, pVtxZ, pVtxU, pVtxV, triMask, &mFullscreenScissor);
#else
			cullResult &= RasterizeTriangleBatch<TEST_Z, TEXTURE_COORDINATES>(pVtxX, pVtxY, pVtxZ, pVtxU, pVtxV, triMask, &mFullscreenScissor);
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
		return (CullingResult)RenderTriangles<0, 0>(inVtx, inTris, nTris, modelToClipMatrix, bfWinding, clipPlaneMask, vtxLayout);
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
		return (CullingResult)nonConst->RenderTriangles<1, 0>(inVtx, inTris, nTris, modelToClipMatrix, bfWinding, clipPlaneMask, vtxLayout);
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
		__m128 clipTriBuffer[MAX_CLIPPED * 3];

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
						GatherVertices(vtxX, vtxY, vtxW, inVtx, inTrisPtr, numLanes, vtxLayout);

					TransformVerts(vtxX, vtxY, vtxW, modelToClipMatrix);
				}

				for (int clipTri = numLanes; clipTri < numLanes + clippedTris; clipTri++)
				{
					int triIdx = clipTail * 3;
					for (int i = 0; i < 3; i++)
					{
						simd_f32(vtxX[i])[clipTri] = simd_f32(clipTriBuffer[triIdx + i])[0];
						simd_f32(vtxY[i])[clipTri] = simd_f32(clipTriBuffer[triIdx + i])[1];
						simd_f32(vtxW[i])[clipTri] = simd_f32(clipTriBuffer[triIdx + i])[2];
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
					GatherVertices(vtxX, vtxY, vtxW, inVtx, inTrisPtr, numLanes, vtxLayout);

				TransformVerts(vtxX, vtxY, vtxW, modelToClipMatrix);

				triIndex += SIMD_LANES;
				inTrisPtr += SIMD_LANES * 3;
			}

			//////////////////////////////////////////////////////////////////////////////
			// Clip transformed triangles
			//////////////////////////////////////////////////////////////////////////////

			if (clipPlaneMask != ClipPlanes::CLIP_PLANE_NONE)
				ClipTriangleAndAddToBuffer<0>(vtxX, vtxY, vtxW, vtxU, vtxV, clipTriBuffer, clipHead, triMask, triClipMask, clipPlaneMask);

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

			RasterizeTriangleBatch<false, 0>(pVtxX, pVtxY, pVtxZ, pVtxU, pVtxV, triMask, scissor);
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
