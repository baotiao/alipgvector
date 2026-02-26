#include "postgres.h"

#include <math.h>
#include <string.h>

#include "hnswrabitq.h"
#include "hnsw.h"
#include "storage/bufmgr.h"
#include "vector.h"

#if PG_VERSION_NUM >= 150000
#include "common/pg_prng.h"
#endif

/* Use built-ins when possible for inlining */
#if defined(HAVE__BUILTIN_POPCOUNT) && defined(HAVE_LONG_INT_64)
#define rabitq_popcount64(x) __builtin_popcountl(x)
#elif defined(HAVE__BUILTIN_POPCOUNT) && defined(HAVE_LONG_LONG_INT_64)
#define rabitq_popcount64(x) __builtin_popcountll(x)
#elif !defined(_MSC_VER)
#define rabitq_popcount64(x) pg_popcount64(x)
#else
/* Fallback for MSVC */
static inline int
rabitq_popcount64(uint64 x)
{
	x = x - ((x >> 1) & 0x5555555555555555ULL);
	x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
	x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0fULL;
	return (int) ((x * 0x0101010101010101ULL) >> 56);
}
#endif

/*
 * In-place Walsh-Hadamard Transform (unnormalized).
 *
 * Operates on data[0..dim-1] where dim must be a power of 2.
 * O(D log D) complexity.
 */
void
HnswRaBitQWalshHadamard(float *data, int dim)
{
	for (int len = 1; len < dim; len <<= 1)
	{
		for (int i = 0; i < dim; i += len << 1)
		{
			for (int j = 0; j < len; j++)
			{
				float		a = data[i + j];
				float		b = data[i + j + len];

				data[i + j] = a + b;
				data[i + j + len] = a - b;
			}
		}
	}
}

/*
 * Round dimension up to next power of 2.
 */
static int
NextPowerOf2(int dim)
{
	int			p = 1;

	while (p < dim)
		p <<= 1;
	return p;
}

/*
 * Generate three random +/-1 diagonal vectors from a seed.
 *
 * Uses PostgreSQL's PRNG seeded deterministically so the same seed
 * always produces the same diagonals.
 */
void
HnswRaBitQGenerateDiagonals(uint64 seed, int dim, float *diag1, float *diag2, float *diag3)
{
#if PG_VERSION_NUM >= 150000
	pg_prng_state prng;

	pg_prng_seed(&prng, seed);

	for (int i = 0; i < dim; i++)
	{
		diag1[i] = (pg_prng_double(&prng) < 0.5) ? -1.0f : 1.0f;
		diag2[i] = (pg_prng_double(&prng) < 0.5) ? -1.0f : 1.0f;
		diag3[i] = (pg_prng_double(&prng) < 0.5) ? -1.0f : 1.0f;
	}
#else
	unsigned int saved = random();

	srandom((unsigned int) seed);

	for (int i = 0; i < dim; i++)
	{
		diag1[i] = (((double) random()) / MAX_RANDOM_VALUE < 0.5) ? -1.0f : 1.0f;
		diag2[i] = (((double) random()) / MAX_RANDOM_VALUE < 0.5) ? -1.0f : 1.0f;
		diag3[i] = (((double) random()) / MAX_RANDOM_VALUE < 0.5) ? -1.0f : 1.0f;
	}

	srandom(saved);
#endif
}

/*
 * HD3 forward rotation: P = H * D3 * H * D2 * H * D1
 *
 * Applies the randomized orthogonal transform P to data[0..dim-1].
 * The work array must have at least padded_dim floats.
 */
void
HnswRaBitQHD3Forward(float *data, int dim, float *diag1, float *diag2, float *diag3, float *work)
{
	int			padded = NextPowerOf2(dim);
	float		scale = 1.0f / sqrtf((float) padded);

	/* Copy to work, zero-pad to power of 2 */
	memcpy(work, data, dim * sizeof(float));
	for (int i = dim; i < padded; i++)
		work[i] = 0.0f;

	/* D1 * x (diag arrays have padded elements) */
	for (int i = 0; i < padded; i++)
		work[i] *= diag1[i];

	/* H * D1 * x */
	HnswRaBitQWalshHadamard(work, padded);

	/* D2 * H * D1 * x */
	for (int i = 0; i < padded; i++)
		work[i] *= diag2[i];

	/* H * D2 * H * D1 * x */
	HnswRaBitQWalshHadamard(work, padded);

	/* D3 * H * D2 * H * D1 * x */
	for (int i = 0; i < padded; i++)
		work[i] *= diag3[i];

	/* H * D3 * H * D2 * H * D1 * x */
	HnswRaBitQWalshHadamard(work, padded);

	/* Scale by 1/sqrt(padded)^3 for three Hadamard transforms */
	/* Each WHT multiplies norms by sqrt(padded), three transforms = padded^(3/2) */
	{
		float		full_scale = scale * scale * scale;

		for (int i = 0; i < dim; i++)
			data[i] = work[i] * full_scale;
	}
}

/*
 * HD3 inverse rotation: P^T = D1 * H * D2 * H * D3 * H
 *
 * Applies the inverse of P to data[0..dim-1].
 * work must have at least padded_dim floats.
 */
void
HnswRaBitQHD3Inverse(float *data, int dim, float *diag1, float *diag2, float *diag3, float *work)
{
	int			padded = NextPowerOf2(dim);
	float		scale = 1.0f / sqrtf((float) padded);

	/* Copy to work, zero-pad */
	memcpy(work, data, dim * sizeof(float));
	for (int i = dim; i < padded; i++)
		work[i] = 0.0f;

	/* H * x */
	HnswRaBitQWalshHadamard(work, padded);

	/* D3 * H * x (diag arrays have padded elements) */
	for (int i = 0; i < padded; i++)
		work[i] *= diag3[i];

	/* H * D3 * H * x */
	HnswRaBitQWalshHadamard(work, padded);

	/* D2 * H * D3 * H * x */
	for (int i = 0; i < padded; i++)
		work[i] *= diag2[i];

	/* H * D2 * H * D3 * H * x */
	HnswRaBitQWalshHadamard(work, padded);

	/* D1 * H * D2 * H * D3 * H * x */
	{
		float		full_scale = scale * scale * scale;

		for (int i = 0; i < dim; i++)
			data[i] = work[i] * full_scale * diag1[i];
	}
}

/*
 * Quantize a single vector using RaBitQ.
 *
 * Given the per-index state (centroid, diagonals), quantize vec into:
 * - out_norm_r: ||vec - centroid||
 * - out_ip_oo_bar: <o, o_bar> correction factor
 * - out_code: D-bit binary code
 *
 * Algorithm:
 * 1. r = vec - centroid
 * 2. norm_r = ||r||
 * 3. o = r / norm_r  (unit vector)
 * 4. o' = P^T * o    (rotate with HD3)
 * 5. code[i] = (o'[i] > 0) ? 1 : 0
 * 6. o_bar = P * h where h[i] = sign(o'[i]) / sqrt(D)
 * 7. ip_oo_bar = <o, o_bar>
 */
void
HnswRaBitQQuantize(HnswRaBitQState *state, float *vec, int dim,
					float *out_norm_r, float *out_ip_oo_bar,
					unsigned char *out_code)
{
	int			padded = NextPowerOf2(dim);
	float	   *residual;
	float	   *rotated;
	float	   *o_bar;
	float	   *work;
	float		norm_r = 0.0f;
	float		inv_norm;
	float		inv_sqrt_dim;
	float		ip_oo_bar = 0.0f;
	int			code_bytes = (dim + 7) / 8;

	residual = palloc(sizeof(float) * dim);
	rotated = palloc(sizeof(float) * dim);
	o_bar = palloc(sizeof(float) * dim);
	work = palloc(sizeof(float) * padded);

	/* Step 1: residual = vec - centroid */
	for (int i = 0; i < dim; i++)
		residual[i] = vec[i] - state->centroid[i];

	/* Step 2: norm_r = ||residual|| */
	for (int i = 0; i < dim; i++)
		norm_r += residual[i] * residual[i];
	norm_r = sqrtf(norm_r);

	/* Handle zero-norm edge case */
	if (norm_r < 1e-30f)
	{
		*out_norm_r = 0.0f;
		*out_ip_oo_bar = 1.0f;
		memset(out_code, 0, code_bytes);
		pfree(residual);
		pfree(rotated);
		pfree(o_bar);
		pfree(work);
		return;
	}

	/* Step 3: o = r / norm_r */
	inv_norm = 1.0f / norm_r;
	for (int i = 0; i < dim; i++)
		residual[i] *= inv_norm;

	/* Step 4: rotated = P^T * o (using HD3 forward on o) */
	memcpy(rotated, residual, sizeof(float) * dim);
	HnswRaBitQHD3Forward(rotated, dim, state->diag1, state->diag2, state->diag3, work);

	/* Step 5: binary encode */
	memset(out_code, 0, code_bytes);
	for (int i = 0; i < dim; i++)
	{
		if (rotated[i] > 0.0f)
			out_code[i / 8] |= (1 << (i % 8));
	}

	/* Step 6: compute o_bar = P * h where h[i] = sign(rotated[i]) / sqrt(dim) */
	inv_sqrt_dim = 1.0f / sqrtf((float) dim);
	for (int i = 0; i < dim; i++)
		o_bar[i] = (rotated[i] > 0.0f) ? inv_sqrt_dim : -inv_sqrt_dim;

	/* Apply P (inverse rotation) to get o_bar in original space */
	HnswRaBitQHD3Inverse(o_bar, dim, state->diag1, state->diag2, state->diag3, work);

	/* Step 7: ip_oo_bar = <o, o_bar> */
	for (int i = 0; i < dim; i++)
		ip_oo_bar += residual[i] * o_bar[i];

	*out_norm_r = norm_r;
	*out_ip_oo_bar = ip_oo_bar;

	pfree(residual);
	pfree(rotated);
	pfree(o_bar);
	pfree(work);
}

/*
 * Compute inner product of binary code with rotated query components.
 *
 * This is the core of the distance estimation: for each bit set in code,
 * we sum the corresponding component of rotated_query.
 *
 * Uses popcount for fast processing of 64-bit chunks, with a scalar
 * fallback for the last partial chunk.
 *
 * The result approximates <o_bar, q> when properly scaled.
 */
float
HnswRaBitQBinaryInnerProduct(unsigned char *code, float *rotated_query, int dim)
{
	float		result = 0.0f;
	int			full_words = dim / 64;
	int			remaining = dim % 64;

	/* Process 64 bits at a time */
	for (int w = 0; w < full_words; w++)
	{
		uint64		word;

		memcpy(&word, code + w * 8, sizeof(uint64));

		/*
		 * For each set bit, add the corresponding rotated_query component.
		 * Since we're summing rotated_query[bit_position], we need to iterate
		 * over set bits.
		 */
		while (word)
		{
			/* Get position of lowest set bit */
			int			bit = __builtin_ctzll(word);

			result += rotated_query[w * 64 + bit];
			/* Clear lowest set bit */
			word &= word - 1;
		}
	}

	/* Handle remaining bits */
	if (remaining > 0)
	{
		int			base_idx = full_words * 64;

		for (int i = 0; i < remaining; i++)
		{
			int			byte_idx = (base_idx + i) / 8;
			int			bit_idx = (base_idx + i) % 8;

			if (code[byte_idx] & (1 << bit_idx))
				result += rotated_query[base_idx + i];
		}
	}

	return result;
}

/*
 * Prepare query state for RaBitQ distance estimation.
 *
 * Precomputes the rotated query and its norm once per search,
 * avoiding redundant work across candidate evaluations.
 */
void
HnswRaBitQPrepareQuery(HnswRaBitQState *state, float *query_vec,
						int dim, HnswRaBitQQueryState *qstate)
{
	int			padded = NextPowerOf2(dim);
	float	   *work;
	float		norm_sq = 0.0f;

	qstate->rotated_query = palloc(sizeof(float) * dim);

	/* Compute query residual: q_r = query - centroid */
	for (int i = 0; i < dim; i++)
		qstate->rotated_query[i] = query_vec[i] - state->centroid[i];

	/* Compute query_norm = ||q_r|| */
	for (int i = 0; i < dim; i++)
		norm_sq += qstate->rotated_query[i] * qstate->rotated_query[i];
	qstate->query_norm = sqrtf(norm_sq);

	/* Normalize residual */
	if (qstate->query_norm > 1e-30f)
	{
		float		inv = 1.0f / qstate->query_norm;

		for (int i = 0; i < dim; i++)
			qstate->rotated_query[i] *= inv;
	}

	/* Rotate: rotated_query = P^T * q_normalized */
	work = palloc(sizeof(float) * padded);
	HnswRaBitQHD3Forward(qstate->rotated_query, dim, state->diag1, state->diag2, state->diag3, work);
	pfree(work);

	/* Precompute sum of rotated_query components */
	qstate->sum_rotated = 0.0f;
	for (int i = 0; i < dim; i++)
		qstate->sum_rotated += qstate->rotated_query[i];
}

/*
 * Estimate squared L2 distance using RaBitQ quantized data.
 *
 * Uses the formula:
 *   ||o_r - q_r||^2 ~= norm_r^2 + norm_q^2
 *                       - 2 * norm_r * norm_q * (<o_bar, q> / ip_oo_bar)
 *
 * where <o_bar, q> is estimated from the binary code and rotated query.
 *
 * The inner product <o_bar, q> is computed as:
 *   (2 * sum_of_matching_components - sum_of_all_components) / sqrt(D)
 *
 * which leverages the fact that o_bar components are +/- 1/sqrt(D).
 */
float
HnswRaBitQEstimateL2(HnswRaBitQTuple rtup, HnswRaBitQQueryState *qstate, int dim)
{
	float		norm_r = rtup->norm_r;
	float		ip_oo_bar = rtup->ip_oo_bar;
	float		norm_q = qstate->query_norm;
	float		ip_obar_q;
	float		ip_oq;
	float		dist_sq;
	float		inv_sqrt_dim = 1.0f / sqrtf((float) dim);

	/*
	 * Compute <o_bar, q> from binary code.
	 *
	 * sum_positive = sum of rotated_query[i] where code bit i is set
	 * <o_bar, q> = (2 * sum_positive - sum_all) / sqrt(D)
	 *
	 * This follows from o_bar_i = +1/sqrt(D) if bit set, -1/sqrt(D)
	 * otherwise.
	 */
	{
		float		sum_positive;

		sum_positive = HnswRaBitQBinaryInnerProduct(rtup->code,
													qstate->rotated_query,
													dim);

		ip_obar_q = (2.0f * sum_positive - qstate->sum_rotated) * inv_sqrt_dim;
	}

	/* Correct: <o, q> ~= <o_bar, q> / ip_oo_bar */
	if (fabsf(ip_oo_bar) < 1e-10f)
		ip_oq = 0.0f;
	else
		ip_oq = ip_obar_q / ip_oo_bar;

	/* Clamp to valid range [-1, 1] */
	if (ip_oq > 1.0f)
		ip_oq = 1.0f;
	if (ip_oq < -1.0f)
		ip_oq = -1.0f;

	/* ||o_r - q_r||^2 = norm_r^2 + norm_q^2 - 2*norm_r*norm_q*<o,q> */
	dist_sq = norm_r * norm_r + norm_q * norm_q - 2.0f * norm_r * norm_q * ip_oq;

	/* Ensure non-negative */
	if (dist_sq < 0.0f)
		dist_sq = 0.0f;

	/* Return L2 distance (not squared) to match pgvector's <-> operator */
	return sqrtf(dist_sq);
}

/*
 * Load RaBitQ state from the index metapage and centroid page.
 *
 * Returns NULL if RaBitQ is not enabled for this index.
 */
HnswRaBitQState *
HnswRaBitQLoadState(Relation index)
{
	Buffer		buf;
	Page		page;
	HnswMetaPage metap;
	HnswRaBitQState *state;
	uint64		seed;
	BlockNumber centroidBlkno;
	int			dim;
	float	   *centroidData;

	buf = ReadBuffer(index, HNSW_METAPAGE_BLKNO);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	metap = HnswPageGetMeta(page);

	/* Check if RaBitQ is enabled (version 2+) */
	if (metap->version < 2 || !metap->rabitqEnabled)
	{
		UnlockReleaseBuffer(buf);
		return NULL;
	}

	seed = metap->rabitqSeed;
	centroidBlkno = metap->centroidBlkno;
	dim = metap->dimensions;

	UnlockReleaseBuffer(buf);

	/* Allocate state */
	state = palloc(sizeof(HnswRaBitQState));
	state->enabled = true;
	state->dim = dim;
	state->seed = seed;

	/* Load centroid from centroid page */
	state->centroid = palloc(sizeof(float) * dim);
	buf = ReadBuffer(index, centroidBlkno);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);

	centroidData = (float *) ((char *) PageGetContents(page) + sizeof(uint32));
	memcpy(state->centroid, centroidData, sizeof(float) * dim);
	UnlockReleaseBuffer(buf);

	/* Store index for buffer reads during search */
	state->index = index;

	/* Generate diagonals from seed */
	{
		int			padded = NextPowerOf2(dim);

		state->diag1 = palloc(sizeof(float) * padded);
		state->diag2 = palloc(sizeof(float) * padded);
		state->diag3 = palloc(sizeof(float) * padded);
		HnswRaBitQGenerateDiagonals(seed, padded, state->diag1, state->diag2, state->diag3);
	}

	return state;
}

/*
 * Free RaBitQ state.
 */
void
HnswRaBitQFreeState(HnswRaBitQState *state)
{
	if (state == NULL)
		return;

	if (state->centroid)
		pfree(state->centroid);
	if (state->diag1)
		pfree(state->diag1);
	if (state->diag2)
		pfree(state->diag2);
	if (state->diag3)
		pfree(state->diag3);
	pfree(state);
}
