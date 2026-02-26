#ifndef HNSWRABITQ_H
#define HNSWRABITQ_H

#include "postgres.h"

#include <math.h>

#include "port/pg_bitutils.h"
#include "utils/relcache.h"

/* RaBitQ tuple type, stored alongside element and neighbor tuples */
#define HNSW_RABITQ_TUPLE_TYPE 3

/* Macro to check if a tuple is a RaBitQ tuple */
#define HnswIsRaBitQTuple(tup) ((tup)->type == HNSW_RABITQ_TUPLE_TYPE)

/* Size of RaBitQ tuple for a given dimension */
#define HNSW_RABITQ_TUPLE_SIZE(dim)	MAXALIGN(offsetof(HnswRaBitQTupleData, code) + ((dim) + 7) / 8)

/*
 * On-disk RaBitQ quantized tuple.
 *
 * Stored at element offno+1, before the neighbor tuple.
 * Contains the binary-quantized vector plus correction factors
 * needed for distance estimation.
 */
typedef struct HnswRaBitQTupleData
{
	uint8		type;			/* HNSW_RABITQ_TUPLE_TYPE */
	uint8		version;		/* matches element version */
	uint16		dim;			/* number of dimensions */
	float4		norm_r;			/* ||residual|| = ||o_r - centroid|| */
	float4		ip_oo_bar;		/* <o, o_bar> correction factor */
	unsigned char code[FLEXIBLE_ARRAY_MEMBER];	/* D-bit binary code */
}			HnswRaBitQTupleData;

typedef HnswRaBitQTupleData *HnswRaBitQTuple;

/*
 * Per-index RaBitQ state.
 *
 * Loaded once per scan or insert from the metapage and centroid page.
 * Contains the centroid and pre-generated random diagonal signs
 * for the HD3 rotation.
 */
typedef struct HnswRaBitQState
{
	bool		enabled;		/* whether rabitq is active */
	int			dim;			/* vector dimensions */
	uint64		seed;			/* random seed for HD3 diagonals */
	float	   *centroid;		/* centroid vector [dim] */
	float	   *diag1;			/* random +/-1 diagonal [dim] */
	float	   *diag2;			/* random +/-1 diagonal [dim] */
	float	   *diag3;			/* random +/-1 diagonal [dim] */
	Relation	index;			/* index relation for buffer reads */
}			HnswRaBitQState;

/*
 * Per-query RaBitQ state.
 *
 * Precomputed once per query to avoid redundant work across
 * candidate distance estimations.
 */
typedef struct HnswRaBitQQueryState
{
	float	   *rotated_query;	/* P^T * q_normalized [dim] */
	float		query_norm;		/* ||q_r|| = ||query - centroid|| */
	float		sum_rotated;	/* sum of rotated_query components (for fast
								 * popcount estimation) */
}			HnswRaBitQQueryState;

/* Core RaBitQ functions */
void		HnswRaBitQWalshHadamard(float *data, int dim);
void		HnswRaBitQGenerateDiagonals(uint64 seed, int dim, float *diag1, float *diag2, float *diag3);
void		HnswRaBitQHD3Forward(float *data, int dim, float *diag1, float *diag2, float *diag3, float *work);
void		HnswRaBitQHD3Inverse(float *data, int dim, float *diag1, float *diag2, float *diag3, float *work);

/* Quantization and distance estimation */
void		HnswRaBitQQuantize(HnswRaBitQState *state, float *vec, int dim,
							   float *out_norm_r, float *out_ip_oo_bar,
							   unsigned char *out_code);
float		HnswRaBitQEstimateL2(HnswRaBitQTuple rtup,
								 HnswRaBitQQueryState *qstate, int dim);
void		HnswRaBitQPrepareQuery(HnswRaBitQState *state, float *query_vec,
								   int dim, HnswRaBitQQueryState *qstate);

/* Centroid computation */
void		HnswRaBitQComputeCentroid(float *centroid, int dim,
									  float *vectors, int nvectors);

/* State management */
HnswRaBitQState *HnswRaBitQLoadState(Relation index);
void		HnswRaBitQFreeState(HnswRaBitQState *state);

/* Popcount-based inner product of binary code with query */
float		HnswRaBitQBinaryInnerProduct(unsigned char *code,
										 float *rotated_query,
										 int dim);

#endif							/* HNSWRABITQ_H */
