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
#pragma once 

#include "MaskedOcclusionCulling.h"

class MaskedOcclusionTextureInternal : public MaskedOcclusionTexture
{
public:

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Member variables
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	int             mWidth;                  //!< Width of image (in texels)
	int             mHeight;                 //!< Height of image (in texels)
	int             mMipLevels;              //!< Total number of mip levels
	unsigned int    mMiplevelOffset[16];     //!< Data offset to certain mip level
	unsigned char   *mRawData;               //!< Raw data pointer. Data is one byte per texel, TODO: pack to 1 bit per texel?
	unsigned char   *mOcclusionData;         //!< Raw data pointer. Data is one byte per texel, TODO: pack to 1 bit per texel?

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Functions 
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	~MaskedOcclusionTextureInternal() override;

	bool Initialize(unsigned int width, unsigned int height, pfnAlignedAlloc alignedAlloc, pfnAlignedFree alignedFree);

	void SetMipLevel(unsigned int mipLevel, const unsigned char *data, float alphaThreshold) override;

	void GenerateMipmaps() override;

	void Finalize() override;

private:
	
	void FilterCorrection(unsigned int mipLevel);
	void GenerateMipmap(unsigned int mipLevel);
};
