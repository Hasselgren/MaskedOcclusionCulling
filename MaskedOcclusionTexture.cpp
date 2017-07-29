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
#include <vector>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <math.h>
#include "MaskedOcclusionCulling.h"
#include "MaskedOcclusionTextureInternal.h"
#include "CompilerSpecific.inl"

#if defined(__AVX__) || defined(__AVX2__)
	// For performance reasons, the MaskedOcclusionCullingAVX2/512.cpp files should be compiled with VEX encoding for SSE instructions (to avoid
	// AVX-SSE transition penalties, see https://software.intel.com/en-us/articles/avoiding-avx-sse-transition-penalties). However, this file
	// _must_ be compiled without VEX encoding to allow backwards compatibility. Best practice is to use lowest supported target platform
	// (/arch:SSE2) as project default, and elevate only the MaskedOcclusionCullingAVX2/512.cpp files.
	#error The MaskedOcclusionCulling.cpp should be compiled with lowest supported target platform, e.g. /arch:SSE2
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Texture class functions 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * Constructor, initializes a texture and allocates data to hold a complete mip-chain
 */
bool MaskedOcclusionTextureInternal::Initialize(unsigned int width, unsigned int height, pfnAlignedAlloc alignedAlloc, pfnAlignedFree alignedFree)
{
	mAlignedAllocCallback = alignedAlloc;
	mAlignedFreeCallback = alignedFree;
	mWidth = width;
	mHeight = height;
	mMipLevels = 1 + (int)floor(log2f(max((float)width, (float)height)));

	unsigned int mMiplevelOffset[16];

	// Compute mip level offsets & size of entire mip chain
	int totalSize = 0, mipWidth = width, mipHeight = height;
	for (int mip = 0; mip < mMipLevels; ++mip)
	{
		mMiplevelOffset[mip] = totalSize;
		totalSize += mipWidth*mipHeight;
		mipWidth = max(1, mipWidth / 2);
		mipHeight = max(1, mipHeight / 2);
	}
	mMiplevelConst = totalSize;

	for (int mip = 0; mip < mMipLevels; ++mip)
	{
		int mipOffset = mMiplevelConst - (mMiplevelConst >> (2*mip));
		assert(mipOffset >= mMiplevelOffset[mip]);
	}

	// Allocate memory for entire mip chain
	mRawData = (unsigned char *)alignedAlloc(64, sizeof(unsigned char)*totalSize);
	mOcclusionData = (unsigned char *)alignedAlloc(64, sizeof(unsigned char)*totalSize);

	return mOcclusionData != nullptr || mRawData != nullptr;
}

MaskedOcclusionTextureInternal::~MaskedOcclusionTextureInternal() 
{
	if (mRawData != nullptr)
		mAlignedFreeCallback(mRawData);
	mRawData = nullptr;
	if (mOcclusionData != nullptr)
		mAlignedFreeCallback(mOcclusionData);
	mOcclusionData = nullptr;
}

int MaskedOcclusionTextureInternal::computeMipOffset(unsigned int mipLevel)
{
	return mMiplevelConst - (mMiplevelConst >> (2 * mipLevel));
}

int convertOcclusionVal(float occlusionVal)
{
	int flooredInt = (unsigned char)min(255, max(0, (int)floor(occlusionVal)));
	int cieledInt = (unsigned char)min(255, max(0, (int)ceil(occlusionVal)));

	if (flooredInt == 0 && cieledInt != 0)
		return cieledInt;
	return flooredInt;
}

/*
 * Bloats a miplevel
 */
void MaskedOcclusionTextureInternal::FilterCorrection(unsigned int mipLevel)
{
	int mipWidth = (int)max(1, mWidth >> mipLevel);
	int mipHeight = (int)max(1, mHeight >> mipLevel);
	unsigned char *data = &mRawData[computeMipOffset(mipLevel)];
	unsigned char *occlusionData = &mOcclusionData[computeMipOffset(mipLevel)];

	for (int y = 0; y < mipHeight; ++y)
	{
		for (int x = 0; x < mipWidth; ++x)
		{
			//float occlusionVal = 0.0f;
			//for (int dy = -1; dy < 1; ++dy)
			//{
			//	for (int dx = -1; dx < 1; ++dx)
			//	{
			//		int tx = x + dx;
			//		int ty = y + dy;
			//		tx = tx < 0 ? (mipWidth + tx) : (tx >= mipWidth ? tx - mipWidth : tx);
			//		ty = ty < 0 ? (mipHeight + ty) : (ty >= mipHeight ? ty - mipHeight : ty);
			//		occlusionVal += (float)data[tx + ty*mipWidth];
			//	}
			//}
			//occlusionVal /= 9.0f;
			//occlusionData[x + y*mipWidth] = (unsigned char)min(255, max(0, (int)ceil(occlusionVal)));
			occlusionData[x + y*mipWidth] = data[x + y*mipWidth];
		}
	}
}

/*
 * Generate a single mipmap level, assumes the previous (higher resolution) map in the chain has been initialized.
 */
void MaskedOcclusionTextureInternal::GenerateMipmap(unsigned int mipLevel)
{
	int mipWidth = max(1, mWidth >> mipLevel);
	int mipHeight = max(1, mHeight >> mipLevel);
	unsigned char *mipData = &mRawData[computeMipOffset(mipLevel)];

	int prevMipWidth = max(1, mWidth >> (mipLevel - 1));
	int prevMipHeight = max(1, mHeight >> (mipLevel - 1));
	unsigned char *prevMipData = &mRawData[computeMipOffset(mipLevel - 1)];

	for (int y = 0; y < mipHeight; ++y)
	{
		for (int x = 0; x < mipWidth; ++x)
		{
			// Compute conservative box filter
			int startX = (int)floor((float)x*(float)prevMipWidth / (float)mipWidth);
			int startY = (int)floor((float)y*(float)prevMipHeight / (float)mipHeight);
			int endX = (int)ceil((float)(x + 1)*(float)prevMipWidth / (float)mipWidth);
			int endY = (int)ceil((float)(y + 1)*(float)prevMipHeight / (float)mipHeight);

			// Perform max boxfilter. Note, this could be optimized a lot, but it's supposed to happen load time.
			float occlusionVal = 0.0f;
			for (int fy = startY; fy < endY; ++fy)
				for (int fx = startX; fx < endX; ++fx)
					occlusionVal += (float)prevMipData[fx + fy*prevMipWidth];
			occlusionVal /= (float)((endY - startY)*(endX - startX));

			// Write data back
			mipData[x + y*mipWidth] = convertOcclusionVal(occlusionVal);
		}
	}
}

/*
 * Sets texture contents for a given mipmap level
 */
void MaskedOcclusionTextureInternal::SetMipLevel(unsigned int mipLevel, const unsigned char *data, float alphaThreshold)
{
	int mipWidth = max(1, mWidth >> mipLevel);
	int mipHeight = max(1, mHeight >> mipLevel);

	// Perform alpha test and copy data. Copied data = 0 for opaque texels and ~0 for transparent texels
	for (int y = 0; y < mipHeight; ++y)
		for (int x = 0; x < mipWidth; ++x)
			mRawData[x + y*mipWidth + computeMipOffset(mipLevel)] = ((float)data[x + y*mipWidth] / 255.0f) < alphaThreshold ? ~0 : 0;
}

/*
 * Automatically (conservatively) generate mipmaps from miplevel 0
 */
void MaskedOcclusionTextureInternal::GenerateMipmaps()
{
	for (int mip = 1; mip < mMipLevels; ++mip)
		GenerateMipmap(mip);
}

/*
 * Process all uploaded texture data & compute an optimized representation
 */
void MaskedOcclusionTextureInternal::Finalize()
{
	for (int mip = 0; mip < mMipLevels; ++mip)
		FilterCorrection(mip);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Texture creation functions
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

MaskedOcclusionTexture *MaskedOcclusionTexture::Create(unsigned int width, unsigned int height)
{
	return Create(width, height, aligned_alloc, aligned_free);
}

MaskedOcclusionTexture *MaskedOcclusionTexture::Create(unsigned int width, unsigned int height, pfnAlignedAlloc alignedAlloc, pfnAlignedFree alignedFree)
{
	assert(width > 0 && height > 0);

	// Allocate texture object
	MaskedOcclusionTextureInternal *texture = (MaskedOcclusionTextureInternal *)alignedAlloc(64, sizeof(MaskedOcclusionTextureInternal));
	if (texture == nullptr)
		return texture;

	new (texture) MaskedOcclusionTextureInternal();

	if (!texture->Initialize(width, height, alignedAlloc, alignedFree))
	{
		alignedFree(texture);
		texture = nullptr;
	}

	return texture;
}
