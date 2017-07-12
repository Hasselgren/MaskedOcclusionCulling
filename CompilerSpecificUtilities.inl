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

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Common shared include file to hide compiler/os specific functions from the rest of the code. This includes AVX/SSE intrinsics, which 
// unfortunately behaves quite differently on windows / linux compilers
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(_WIN32)	// Windows
	#include <intrin.h>
	#include <new.h>

	#define FORCE_INLINE __forceinline

	static FORCE_INLINE unsigned long find_clear_lsb(unsigned int *mask)
	{
		unsigned long idx;
		_BitScanForward(&idx, *mask);
		*mask &= *mask - 1;
		return idx;
	}

	static FORCE_INLINE void *aligned_alloc(size_t alignment, size_t size) 
	{
		return _aligned_malloc(size, alignment);
	}

	static FORCE_INLINE void aligned_free(void *ptr)
	{
		_aligned_free(ptr);
	}

#elif defined(__GNUC__)	// Linux
	#include <cpuid.h>
	#include <mm_malloc.h>
	#include <immintrin.h>
	#include <new>

	#define FORCE_INLINE inline

	static FORCE_INLINE unsigned long find_clear_lsb(unsigned int *mask)
	{
		unsigned long idx;
		idx = __builtin_ctzl(*mask);
		*mask &= *mask - 1;
		return idx;
	}

	static FORCE_INLINE void aligned_free(void *ptr)
	{
		free(ptr);
	}

	static FORCE_INLINE void __cpuidex(int* cpuinfo, int function, int subfunction)
	{
		__cpuid_count(function, subfunction, cpuinfo[0], cpuinfo[1], cpuinfo[2], cpuinfo[3]);
	}

	static FORCE_INLINE unsigned long long _xgetbv(unsigned int index)
	{
		unsigned int eax, edx;
		__asm__ __volatile__(
			"xgetbv;"
			: "=a" (eax), "=d"(edx)
			: "c" (index)
		);
		return ((unsigned long long)edx << 32) | eax;
	}

#else
	#error Unsupported compiler
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Annoying, but we need to wrap the native simd-types so we can use our operators on it without fear that we'll mix it up with compiler-generated  
// operators that do something different (like with clang/llvm). Also, GCC doesn't allow overloading operators for the __m??? types
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

union vec4f {
	__m128 native;
	float  m128_f32[4];

	vec4f() = default;
	explicit vec4f(const __m128 & ref) { native = ref; }
	operator __m128(void) const { return native; }
};

union vec4i {
	__m128i      native;
	int	         m128i_i32[4];
	unsigned int m128i_u32[4];

	vec4i() = default;
	explicit vec4i(const __m128i & ref) { native = ref; }
	operator __m128i(void) const { return native; }
};

#ifdef __AVX2__
	union vec8f {
		__m256 native;
		float  m256_f32[8];

		vec8f() = default;
		explicit vec8f(const __m256 & ref) { native = ref; }
		operator __m256(void) const { return native; }
	};

	union vec8i {
		__m256i      native;
		int          m256i_i32[8];
		unsigned int m256i_u32[8];

		vec8i() = default;
		explicit vec8i(const __m256i & ref) { native = ref; }
		operator __m256i(void) const { return native; }
	};
#endif

#ifdef __AVX512__
	union vec16f {
		__m512 native;
		float  m512_f32[16];

		vec16f() = default;
		explicit vec16f(const __m512 & ref) { native = ref; }
		operator __m512(void) const { return native; }
	};

	union vec16i {
		__m512i      native;
		int          m512i_i32[16];
		unsigned int m512i_u32[16];

		vec16i() = default;
		explicit vec16i(const __m512i & ref) { native = ref; }
		operator __m512i(void) const { return native; }
	};
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// SIMD math utility functions, operator overloading etc
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T, typename Y> FORCE_INLINE T simd_cast(Y A);

template<> FORCE_INLINE vec4f simd_cast<vec4f>(float A) { return vec4f(_mm_set1_ps(A)); }
template<> FORCE_INLINE vec4f simd_cast<vec4f>(vec4i A) { return vec4f(_mm_castsi128_ps(A.native)); }
template<> FORCE_INLINE vec4f simd_cast<vec4f>(vec4f A) { return A; }
template<> FORCE_INLINE vec4i simd_cast<vec4i>(int A)   { return vec4i(_mm_set1_epi32(A)); }
template<> FORCE_INLINE vec4i simd_cast<vec4i>(vec4f A) { return vec4i(_mm_castps_si128(A.native)); }
template<> FORCE_INLINE vec4i simd_cast<vec4i>(vec4i A) { return A; }
#ifdef __AVX2__
template<> FORCE_INLINE vec8f simd_cast<vec8f>(float A) { return vec8f(_mm256_set1_ps(A)); }
template<> FORCE_INLINE vec8f simd_cast<vec8f>(vec8i A) { return vec8f(_mm256_castsi256_ps(A.native)); }
template<> FORCE_INLINE vec8f simd_cast<vec8f>(vec8f A) { return A; }
template<> FORCE_INLINE vec8i simd_cast<vec8i>(int A)   { return vec8i(_mm256_set1_epi32(A)); }
template<> FORCE_INLINE vec8i simd_cast<vec8i>(vec8f A) { return vec8i(_mm256_castps_si256(A.native)); }
template<> FORCE_INLINE vec8i simd_cast<vec8i>(vec8i A) { return A; }
#endif
#ifdef __AVX512__
template<> FORCE_INLINE vec16f simd_cast<vec16f>(float A)  { return vec16f(_mm512_set1_ps(A)); }
template<> FORCE_INLINE vec16f simd_cast<vec16f>(vec16i A) { return vec16f(_mm512_castsi512_ps(A.native)); }
template<> FORCE_INLINE vec16f simd_cast<vec16f>(vec16f A) { return A; }
template<> FORCE_INLINE vec16i simd_cast<vec16i>(int A)    { return vec16i(_mm512_set1_epi32(A)); }
template<> FORCE_INLINE vec16i simd_cast<vec16i>(vec16f A) { return vec16i(_mm512_castps_si512(A.native)); }
template<> FORCE_INLINE vec16i simd_cast<vec16i>(vec16i A) { return A; }
#endif

// Unary operators
static FORCE_INLINE vec4f operator-(const vec4f &A) { return vec4f(_mm_xor_ps(A.native, _mm_set1_ps(-0.0f))); }
static FORCE_INLINE vec4i operator-(const vec4i &A) { return vec4i(_mm_sub_epi32(_mm_set1_epi32(0), A.native)); }
static FORCE_INLINE vec4f operator~(const vec4f &A) { return vec4f(_mm_xor_ps(A.native, _mm_castsi128_ps(_mm_set1_epi32(~0)))); }
static FORCE_INLINE vec4i operator~(const vec4i &A) { return vec4i(_mm_xor_si128(A.native, _mm_set1_epi32(~0))); }
static FORCE_INLINE vec4f abs(const vec4f &A) { return vec4f(_mm_and_ps(A.native, _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF)))); }
#ifdef __AVX2__
static FORCE_INLINE vec8f operator-(const vec8f &A) { return vec8f(_mm256_xor_ps(A.native, _mm256_set1_ps(-0.0f))); }
static FORCE_INLINE vec8i operator-(const vec8i &A) { return vec8i(_mm256_sub_epi32(_mm256_set1_epi32(0), A.native)); }
static FORCE_INLINE vec8f operator~(const vec8f &A) { return vec8f(_mm256_xor_ps(A.native, _mm256_castsi256_ps(_mm256_set1_epi32(~0)))); }
static FORCE_INLINE vec8i operator~(const vec8i &A) { return vec8i(_mm256_xor_si256(A.native, _mm256_set1_epi32(~0))); }
static FORCE_INLINE vec8f abs(const vec8f &A) { return vec8f(_mm256_and_ps(A.native, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)))); }
#endif
#ifdef __AVX512__
static FORCE_INLINE vec16f operator-(const vec16f &A) { return vec16f(_mm512_xor_ps(A.native, _mm512_set1_ps(-0.0f))); }
static FORCE_INLINE vec16i operator-(const vec16i &A) { return vec16i(_mm512_sub_epi32(_mm512_set1_epi32(0), A.native)); }
static FORCE_INLINE vec16f operator~(const vec16f &A) { return vec16f(_mm512_xor_ps(A.native, _mm512_castsi512_ps(_mm512_set1_epi32(~0)))); }
static FORCE_INLINE vec16i operator~(const vec16i &A) { return vec16i(_mm512_xor_si512(A.native, _mm512_set1_epi32(~0))); }
static FORCE_INLINE vec16f abs(const vec16f &A) { return vec16f(_mm512_and_ps(A.native, _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF)))); }
#endif

// Binary operators
#define SIMD_BINARY_OP(SIMD_TYPE, BASE_TYPE, prefix, postfix, func, op) \
	static FORCE_INLINE SIMD_TYPE operator op(const SIMD_TYPE &A, const SIMD_TYPE &B)		{ return SIMD_TYPE(_##prefix##_##func##_##postfix(A.native, B.native)); } \
	static FORCE_INLINE SIMD_TYPE operator op(const SIMD_TYPE &A, const BASE_TYPE B)		{ return SIMD_TYPE(_##prefix##_##func##_##postfix(A.native, simd_cast<SIMD_TYPE>(B).native)); } \
	static FORCE_INLINE SIMD_TYPE operator op(const BASE_TYPE &A, const SIMD_TYPE &B)		{ return SIMD_TYPE(_##prefix##_##func##_##postfix(simd_cast<SIMD_TYPE>(A).native, B.native)); } \
	static FORCE_INLINE SIMD_TYPE &operator op##=(SIMD_TYPE &A, const SIMD_TYPE &B)		{ return (A = SIMD_TYPE(_##prefix##_##func##_##postfix(A.native, B.native))); } \
	static FORCE_INLINE SIMD_TYPE &operator op##=(SIMD_TYPE &A, const BASE_TYPE B)			{ return (A = SIMD_TYPE(_##prefix##_##func##_##postfix(A.native, simd_cast<SIMD_TYPE>(B).native))); }

SIMD_BINARY_OP(vec4f, float, mm, ps, add, +)
SIMD_BINARY_OP(vec4f, float, mm, ps, sub, -)
SIMD_BINARY_OP(vec4f, float, mm, ps, mul, *)
SIMD_BINARY_OP(vec4f, float, mm, ps, div, / )
SIMD_BINARY_OP(vec4i, int, mm, epi32, add, +)
SIMD_BINARY_OP(vec4i, int, mm, epi32, sub, -)
SIMD_BINARY_OP(vec4f, float, mm, ps, and, &)
SIMD_BINARY_OP(vec4f, float, mm, ps, or , | )
SIMD_BINARY_OP(vec4f, float, mm, ps, xor, ^)
SIMD_BINARY_OP(vec4i, int, mm, si128, and, &)
SIMD_BINARY_OP(vec4i, int, mm, si128, or , | )
SIMD_BINARY_OP(vec4i, int, mm, si128, xor, ^)
#ifdef __AVX2__
SIMD_BINARY_OP(vec8f, float, mm256, ps, add, +)
SIMD_BINARY_OP(vec8f, float, mm256, ps, sub, -)
SIMD_BINARY_OP(vec8f, float, mm256, ps, mul, *)
SIMD_BINARY_OP(vec8f, float, mm256, ps, div, / )
SIMD_BINARY_OP(vec8i, int, mm256, epi32, add, +)
SIMD_BINARY_OP(vec8i, int, mm256, epi32, sub, -)
SIMD_BINARY_OP(vec8f, float, mm256, ps, and, &)
SIMD_BINARY_OP(vec8f, float, mm256, ps, or , | )
SIMD_BINARY_OP(vec8f, float, mm256, ps, xor, ^)
SIMD_BINARY_OP(vec8i, int, mm256, si256, and, &)
SIMD_BINARY_OP(vec8i, int, mm256, si256, or , | )
SIMD_BINARY_OP(vec8i, int, mm256, si256, xor, ^)
#endif
#ifdef __AVX512__
SIMD_BINARY_OP(vec16f, float, mm512, ps, add, +)
SIMD_BINARY_OP(vec16f, float, mm512, ps, sub, -)
SIMD_BINARY_OP(vec16f, float, mm512, ps, mul, *)
SIMD_BINARY_OP(vec16f, float, mm512, ps, div, / )
SIMD_BINARY_OP(vec16i, int, mm512, epi32, add, +)
SIMD_BINARY_OP(vec16i, int, mm512, epi32, sub, -)
SIMD_BINARY_OP(vec16f, float, mm512, ps, and, &)
SIMD_BINARY_OP(vec16f, float, mm512, ps, or , | )
SIMD_BINARY_OP(vec16f, float, mm512, ps, xor, ^)
SIMD_BINARY_OP(vec16i, int, mm512, si512, and, &)
SIMD_BINARY_OP(vec16i, int, mm512, si512, or , | )
SIMD_BINARY_OP(vec16i, int, mm512, si512, xor, ^)
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Also wrap all required _mm functions to avoid endless casting in the rest of the code. All this could be avoided if operator overloading was
// supported by all compilers :(
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define vec4_setzero_ps()				vec4f(_mm_setzero_ps())
#define vec4_set1_ps(A)					vec4f(_mm_set1_ps(A))
#define vec4_andnot_ps(A, B)			vec4f(_mm_andnot_ps((A).native, (B).native))
#define vec4_min_ps(A, B)				vec4f(_mm_min_ps((A).native, (B).native))
#define vec4_max_ps(A, B)				vec4f(_mm_max_ps((A).native, (B).native))
#define vec4_fmadd_ps(A, B, C)			vec4f(_mm_fmadd_ps((A).native, (B).native, (C).native))
#define vec4_fmsub_ps(A, B, C)			vec4f(_mm_fmsub_ps((A).native, (B).native, (C).native))
#define vec4_shuffle_ps(A, B, C)		vec4f(_mm_shuffle_ps((A).native, (B).native, (C)));
#define vec4_cvtepi32_ps(A)				vec4f(_mm_cvtepi32_ps((A).native));
#define vec4_cmpge_ps(A, B)				vec4f(_mm_cmpge_ps((A).native, (B).native))
#define vec4_cmpgt_ps(A, B)				vec4f(_mm_cmpgt_ps((A).native, (B).native))
#define vec4_cmpeq_ps(A, B)				vec4f(_mm_cmpeq_ps((A).native, (B).native))
#define vec4_dp_ps(A, B, C)				vec4f(_mm_dp_ps((A).native, (B).native, (C)))
#define vec4_load1_ps(A)				vec4f(_mm_load1_ps((A)))
#define vec4_loadu_ps(A)				vec4f(_mm_loadu_ps((A)))
#define vec4_storeu_ps(A, B)			_mm_storeu_ps((A), (B).native)
#define vec4_setr_ps(e0, e1, e2, e3)	vec4f(_mm_setr_ps((e0), (e1), (e2), (e3)))
#define vec4_movemask_ps(A)				_mm_movemask_ps((A).native)
#define vec4_blendv_ps(A, B, C)			vec4f(_mm_blendv_ps((A).native, (B).native, (C).native))
#define vec4_round_ps(A, B)				vec4f(_mm_round_ps((A).native, (B)))

#define vec4_setzero_epi32()			vec4i(_mm_setzero_si128())
#define vec4_set1_epi8(A)				vec4i(_mm_set1_epi8(A))
#define vec4_set1_epi32(A)				vec4i(_mm_set1_epi32(A))
#define vec4_andnot_epi32(A, B)			vec4i(_mm_andnot_si128((A).native, (B).native))
#define vec4_min_epi8(A, B)				vec4i(_mm_min_epi8((A).native, (B).native))
#define vec4_min_epi32(A, B)			vec4i(_mm_min_epi32((A).native, (B).native))
#define vec4_max_epi32(A, B)			vec4i(_mm_max_epi32((A).native, (B).native))
#define vec4_subs_epu8(A, B)			vec4i(_mm_subs_epu8((A).native, (B).native))
#define vec4_subs_epu16(A, B)			vec4i(_mm_subs_epu16((A).native, (B).native))
#define vec4_mul_epu32(A, B)			vec4i(_mm_mul_epu32((A).native, (B).native))
#define vec4_mullo_epi32(A, B)			vec4i(_mm_mullo_epi32((A).native, (B).native))
#define vec4_srai_epi32(A, B)			vec4i(_mm_srai_epi32((A).native, (B)))
#define vec4_srli_epi16(A, B)			vec4i(_mm_srli_epi16((A).native, (B)))
#define vec4_srli_epi32(A, B)			vec4i(_mm_srli_epi32((A).native, (B)))
#define vec4_srli_epi64(A, B)			vec4i(_mm_srli_epi64((A).native, (B)))
#define vec4_slli_epi32(A, B)			vec4i(_mm_slli_epi32((A).native, (B)))
#define vec4_slli_epi64(A, B)			vec4i(_mm_slli_epi64((A).native, (B)))
#define vec4_sllv_epi32(A, B)			vec4i(_mm_sllv_epi32((A).native, (B)))
#define vec4_shuffle_epi8(A, B)			vec4i(_mm_shuffle_epi8((A).native, (B).native))
#define vec4_packus_epi16(A, B)			vec4i(_mm_packus_epi16((A).native, (B).native))
#define vec4_abs_epi32(A)				vec4i(_mm_abs_epi32((A).native))
#define vec4_cmpeq_epi8(A, B)			vec4i(_mm_cmpeq_epi8((A).native, (B).native))
#define vec4_cmpeq_epi32(A, B)			vec4i(_mm_cmpeq_epi32((A).native, (B).native))
#define vec4_cmpgt_epi32(A, B)			vec4i(_mm_cmpgt_epi32((A).native, (B).native))
#define vec4_movemask_epi8(A)			_mm_movemask_epi8((A).native)
#define vec4_testz_epi32(A, B)			_mm_testz_si128((A).native, (B).native)
#define vec4_cvtps_epi32(A)				vec4i(_mm_cvtps_epi32((A).native))
#define vec4_cvttps_epi32(A)			vec4i(_mm_cvttps_epi32((A).native))
#define vec4_setr_epi32(e0, e1, e2, e3)	vec4i(_mm_setr_epi32((e0), (e1), (e2), (e3)))
#define vec4_setr_epi8(e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15) vec4i(_mm_setr_epi8( \
	(e0), (e1), (e2), (e3), (e4), (e5), (e6), (e7), \
	(e8), (e9), (e10), (e11), (e12), (e13), (e14), (e15)))

#ifdef __AVX2__
#define vec8_setzero_ps()				vec8f(_mm256_setzero_ps())
#define vec8_set1_ps(A)					vec8f(_mm256_set1_ps(A))
#define vec8_andnot_ps(A, B)			vec8f(_mm256_andnot_ps((A).native, (B).native))
#define vec8_min_ps(A, B)				vec8f(_mm256_min_ps((A).native, (B).native))
#define vec8_max_ps(A, B)				vec8f(_mm256_max_ps((A).native, (B).native))
#define vec8_fmadd_ps(A, B, C)			vec8f(_mm256_fmadd_ps((A).native, (B).native, (C).native))
#define vec8_fmsub_ps(A, B, C)			vec8f(_mm256_fmsub_ps((A).native, (B).native, (C).native))
#define vec8_shuffle_ps(A, B, C)		vec8f(_mm256_shuffle_ps((A).native, (B).native, (C)));
#define vec8_cvtepi32_ps(A)				vec8f(_mm256_cvtepi32_ps((A).native));
#define vec8_dp_ps(A, B, C)				vec8f(_mm256_dp_ps((A).native, (B).native, (C)))
#define vec8_insertf32x4_ps(A, B, C)	vec8f(_mm256_insertf128_ps((A).native, (B).native, (C)))
#define vec8_i32gather_ps(A, B, C)		vec8f(_mm256_i32gather_ps((A), (B).native, (C)))
#define vec8_movemask_ps(A)				_mm256_movemask_ps((A).native)
#define vec8_blendv_ps(A, B, C)			vec8f(_mm256_blendv_ps((A).native, (B).native, (C).native))
#define vec8_cmp_ps(A, B, C)			vec8f(_mm256_cmp_ps((A).native, (B).native, (C)))
#define vec8_round_ps(A, B)				vec8f(_mm256_round_ps((A).native, (B)))
#define vec8_setr_ps(e0, e1, e2, e3, e4, e5, e6, e7) vec8f(_mm256_setr_ps((e0), (e1), (e2), (e3), (e4), (e5), (e6), (e7)))

#define vec8_setzero_epi32()			vec8i(_mm256_setzero_si256())
#define vec8_set1_epi32(A)				vec8i(_mm256_set1_epi32(A))
#define vec8_andnot_epi32(A, B)			vec8i(_mm256_andnot_si256((A).native, (B).native))
#define vec8_min_epi32(A, B)			vec8i(_mm256_min_epi32((A).native, (B).native))
#define vec8_max_epi32(A, B)			vec8i(_mm256_max_epi32((A).native, (B).native))
#define vec8_subs_epu16(A, B)			vec8i(_mm256_subs_epu16((A).native, (B).native))
#define vec8_mullo_epi32(A, B)			vec8i(_mm256_mullo_epi32((A).native, (B).native))
#define vec8_srai_epi32(A, B)			vec8i(_mm256_srai_epi32((A).native, (B)))
#define vec8_srli_epi32(A, B)			vec8i(_mm256_srli_epi32((A).native, (B)))
#define vec8_slli_epi32(A, B)			vec8i(_mm256_slli_epi32((A).native, (B)))
#define vec8_sllv_epi32(A, B)			vec8i(_mm256_sllv_epi32((A).native, (B)))
#define vec8_shuffle_epi8(A, B)			vec8i(_mm256_shuffle_epi8((A).native, (B).native))
#define vec8_abs_epi32(A)				vec8i(_mm256_abs_epi32((A).native))
#define vec8_cmpeq_epi32(A, B)			vec8i(_mm256_cmpeq_epi32((A).native, (B).native))
#define vec8_cmpgt_epi32(A, B)			vec8i(_mm256_cmpgt_epi32((A).native, (B).native))
#define vec8_testz_epi32(A, B)			_mm256_testz_si256((A).native, (B).native)
#define vec8_cvtps_epi32(A)				vec8i(_mm256_cvtps_epi32((A).native))
#define vec8_cvttps_epi32(A)			vec8i(_mm256_cvttps_epi32((A).native))
#define vec8_i32gather_epi32(A, B, C)	vec8i(_mm256_i32gather_epi32((A), (B).native, (C)))
#define vec8_setr_epi32(e0, e1, e2, e3, e4, e5, e6, e7) vec8i(_mm256_setr_epi32((e0), (e1), (e2), (e3), (e4), (e5), (e6), (e7)))
#define vec8_setr_epi8(e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15, e16, e17, e18, e19, e20, e21, e22, e23, e24, e25, e26, e27, e28, e29, e30, e31) vec8i(_mm256_setr_epi8( \
	(e0), (e1), (e2), (e3), (e4), (e5), (e6), (e7), \
	(e8), (e9), (e10), (e11), (e12), (e13), (e14), (e15), \
	(e16), (e17), (e18), (e19), (e20), (e21), (e22), (e23), \
	(e24), (e25), (e26), (e27), (e28), (e29), (e30), (e31)))
#endif

#ifdef __AVX512__
#define vec16_setzero_ps()				vec16f(_mm512_setzero_ps())
#define vec16_set1_ps(A)				vec16f(_mm512_set1_ps(A))
#define vec16_andnot_ps(A, B)			vec16f(_mm512_andnot_ps((A).native, (B).native))
#define vec16_min_ps(A, B)				vec16f(_mm512_min_ps((A).native, (B).native))
#define vec16_max_ps(A, B)				vec16f(_mm512_max_ps((A).native, (B).native))
#define vec16_fmadd_ps(A, B, C)			vec16f(_mm512_fmadd_ps((A).native, (B).native, (C).native))
#define vec16_fmsub_ps(A, B, C)			vec16f(_mm512_fmsub_ps((A).native, (B).native, (C).native))
#define vec16_shuffle_ps(A, B, C)		vec16f(_mm512_shuffle_ps((A).native, (B).native, (C)));
#define vec16_cvtepi32_ps(A)			vec16f(_mm512_cvtepi32_ps((A).native));
#define vec16_dp_ps(A, B, C)			vec16f(_mm512_dp_ps((A).native, (B).native, (C)))
#define vec16_insertf32x4_ps(A, B, C)	vec16f(_mm512_insertf32x4((A).native, (B).native, (C)))
#define vec16_roundscale_ps(A, B)		vec16f(_mm512_roundscale_ps((A).native, B))
#define vec16_cmp_ps_mask(A, B, C)		_mm512_cmp_ps_mask((A).native, (B).native, (C))
#define vec16_mask_mov_ps(A, B, C)		vec16f(_mm512_mask_mov_ps((A).native, B, (C).native))
#define vec16_i32gather_ps(A, B, C)		vec16f(_mm512_i32gather_ps((A).native, (B), (C)))
#define vec16_set_ps(e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15) vec16f(_mm512_set_ps((e0), (e1), (e2), (e3), (e4), (e5), (e6), (e7), (e8), (e9), (e10), (e11), (e12), (e13), (e14), (e15)))
#define vec16_setr_ps(e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15) vec16f(_mm512_setr_ps((e0), (e1), (e2), (e3), (e4), (e5), (e6), (e7), (e8), (e9), (e10), (e11), (e12), (e13), (e14), (e15)))

#define vec16_setzero_epi32()			vec16i(_mm512_setzero_si512())
#define vec16_set1_epi32(A)				vec16i(_mm512_set1_epi32(A))
#define vec16_andnot_epi32(A, B)		vec16i(_mm512_andnot_si512((A).native, (B).native))
#define vec16_min_epi32(A, B)			vec16i(_mm512_min_epi32((A).native, (B).native))
#define vec16_max_epi32(A, B)			vec16i(_mm512_max_epi32((A).native, (B).native))
#define vec16_subs_epu16(A, B)			vec16i(_mm512_subs_epu16((A).native, (B).native))
#define vec16_mullo_epi32(A, B)			vec16i(_mm512_mullo_epi32((A).native, (B).native))
#define vec16_srai_epi32(A, B)			vec16i(_mm512_srai_epi32((A).native, (B)))
#define vec16_srli_epi32(A, B)			vec16i(_mm512_srli_epi32((A).native, (B)))
#define vec16_slli_epi32(A, B)			vec16i(_mm512_slli_epi32((A).native, (B)))
#define vec16_sllv_epi32(A, B)			vec16i(_mm512_sllv_epi32((A).native, (B)))
#define vec16_shuffle_epi8(A, B)		vec16i(_mm512_shuffle_epi8((A).native, (B).native))
#define vec16_abs_epi32(A)				vec16i(_mm512_abs_epi32((A).native))
#define vec16_cvtps_epi32(A)			vec16i(_mm512_cvtps_epi32((A).native))
#define vec16_cvttps_epi32(A)			vec16i(_mm512_cvttps_epi32((A).native))
#define vec16_cmp_epi32_mask(A, B, C)	_mm512_cmp_epi32_mask((A).native, (B).native, (C))
#define vec16_cmpeq_epi32_mask(A, B)	_mm512_cmpeq_epi32_mask((A).native, (B).native)
#define vec16_cmpgt_epi32_mask(A, B)	_mm512_cmpgt_epi32_mask((A).native, (B).native)
#define vec16_mask_mov_epi32(A, B, C)	vec16i(_mm512_mask_mov_epi32((A).native, B, (C).native))
#define vec16_i32gather_epi32(A, B, C)	vec16i(_mm512_i32gather_epi32((A).native, (B), (C)))
#define vec16_set_epi8(...)				vec16i(_mm512_set_epi8(__VA_ARGS__))
#define vec16_set_epi32(e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15) vec16i(_mm512_set_epi32((e0), (e1), (e2), (e3), (e4), (e5), (e6), (e7), (e8), (e9), (e10), (e11), (e12), (e13), (e14), (e15)))
#define vec16_setr_epi32(e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15) vec16i(_mm512_setr_epi32((e0), (e1), (e2), (e3), (e4), (e5), (e6), (e7), (e8), (e9), (e10), (e11), (e12), (e13), (e14), (e15)))
#endif