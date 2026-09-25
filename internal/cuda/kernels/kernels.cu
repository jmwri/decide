// GPU kernels for the Decide encoder (forward and backward). Compiled to PTX
// offline by cmd/ptxgen (NVRTC) and embedded in the binary; the CUDA driver
// JIT-compiles the PTX for whatever GPU is installed. All math is fp32.
//
// Conventions match internal/blas and internal/nn: row-major matrices, PyTorch
// linear layout (weights are (out, in)), leading dimensions in elements.

// ---------------------------------------------------------------------------
// GEMM: C (M x N) (+)= op(A) (M x K) * op(B) (K x N)
//   TA: A is stored K x M (element (m,k) = A[k*lda + m]); else M x K.
//   TB: B is stored N x K (element (k,n) = B[n*ldb + k]); else K x N.
// 128x128 block tile, K tile 8, 256 threads, 8x8 outputs per thread, register
// prefetch of the next K tile while the current one is multiplied.

#define BM 128
#define BN 128
#define BK 8
#define PAD 4

template <bool TA, bool TB>
__device__ __forceinline__ void gemm_tile(int M, int N, int K, const float* __restrict__ A, int lda,
                                          const float* __restrict__ B, int ldb, float* __restrict__ C,
                                          int ldc, int accumulate, int bm0, int bn0) {
  __shared__ float sA[2][BK][BM + PAD];
  __shared__ float sB[2][BK][BN + PAD];
  const int tid = threadIdx.x;
  const int ty = tid >> 4, tx = tid & 15;
  float acc[8][8];
#pragma unroll
  for (int i = 0; i < 8; i++)
#pragma unroll
    for (int j = 0; j < 8; j++) acc[i][j] = 0.f;

  float ra[4], rb[4];
  const int nk = (K + BK - 1) / BK;

  // Global -> registers for K tile kt.
#define LOAD_TILE(kt)                                                                        \
  {                                                                                          \
    const int k0 = (kt) * BK;                                                                \
    _Pragma("unroll") for (int r = 0; r < 4; r++) {                                          \
      const int e = tid + 256 * r;                                                           \
      int m, k;                                                                              \
      if (TA) { k = e >> 7; m = e & 127; } else { m = e >> 3; k = e & 7; }                   \
      const int gm = bm0 + m, gk = k0 + k;                                                   \
      ra[r] = (gm < M && gk < K) ? (TA ? A[(size_t)gk * lda + gm] : A[(size_t)gm * lda + gk]) : 0.f; \
      int n, kb;                                                                             \
      if (TB) { n = e >> 3; kb = e & 7; } else { kb = e >> 7; n = e & 127; }                 \
      const int gn = bn0 + n, gkb = k0 + kb;                                                 \
      rb[r] = (gn < N && gkb < K) ? (TB ? B[(size_t)gn * ldb + gkb] : B[(size_t)gkb * ldb + gn]) : 0.f; \
    }                                                                                        \
  }
  // Registers -> shared memory buffer.
#define STORE_TILE(buf)                                                                      \
  {                                                                                          \
    _Pragma("unroll") for (int r = 0; r < 4; r++) {                                          \
      const int e = tid + 256 * r;                                                           \
      int m, k;                                                                              \
      if (TA) { k = e >> 7; m = e & 127; } else { m = e >> 3; k = e & 7; }                   \
      sA[buf][k][m] = ra[r];                                                                 \
      int n, kb;                                                                             \
      if (TB) { n = e >> 3; kb = e & 7; } else { kb = e >> 7; n = e & 127; }                 \
      sB[buf][kb][n] = rb[r];                                                                \
    }                                                                                        \
  }

  LOAD_TILE(0);
  STORE_TILE(0);
  __syncthreads();
  for (int kt = 0; kt < nk; kt++) {
    const int buf = kt & 1;
    if (kt + 1 < nk) LOAD_TILE(kt + 1);
#pragma unroll
    for (int k = 0; k < BK; k++) {
      float a[8], b[8];
#pragma unroll
      for (int i = 0; i < 8; i++) a[i] = sA[buf][k][ty + 16 * i];
#pragma unroll
      for (int j = 0; j < 8; j++) b[j] = sB[buf][k][tx + 16 * j];
#pragma unroll
      for (int i = 0; i < 8; i++)
#pragma unroll
        for (int j = 0; j < 8; j++) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
    }
    if (kt + 1 < nk) STORE_TILE(buf ^ 1);
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const int m = bm0 + ty + 16 * i;
    if (m >= M) continue;
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const int n = bn0 + tx + 16 * j;
      if (n < N) {
        float* c = C + (size_t)m * ldc + n;
        *c = accumulate ? (*c + acc[i][j]) : acc[i][j];
      }
    }
  }
}

#define GEMM_ENTRY(NAME, TA, TB)                                                                   \
  extern "C" __global__ void NAME(int M, int N, int K, const float* A, int lda, const float* B,     \
                                  int ldb, float* C, int ldc, int accumulate) {                     \
    gemm_tile<TA, TB>(M, N, K, A, lda, B, ldb, C, ldc, accumulate, blockIdx.y * BM, blockIdx.x * BN); \
  }
GEMM_ENTRY(gemm_nn, false, false)
GEMM_ENTRY(gemm_nt, false, true)
GEMM_ENTRY(gemm_tn, true, false)

// Batched variants: one item per blockIdx.z, described by 9 ints:
//   M, N, K, offA, offB, offC, lda, ldb, ldc   (offsets in elements)
// The grid covers the largest item; smaller items simply skip empty tiles.
#define GEMM_BATCH_ENTRY(NAME, TA, TB)                                                             \
  extern "C" __global__ void NAME(const int* items, const float* A, const float* B, float* C,       \
                                  int accumulate) {                                                 \
    const int* it = items + 9 * blockIdx.z;                                                         \
    const int bm0 = blockIdx.y * BM, bn0 = blockIdx.x * BN;                                         \
    if (bm0 >= it[0] || bn0 >= it[1]) return;                                                       \
    gemm_tile<TA, TB>(it[0], it[1], it[2], A + it[3], it[6], B + it[4], it[7], C + it[5], it[8],    \
                      accumulate, bm0, bn0);                                                        \
  }
GEMM_BATCH_ENTRY(gemm_nn_b, false, false)
GEMM_BATCH_ENTRY(gemm_nt_b, false, true)
GEMM_BATCH_ENTRY(gemm_tn_b, true, false)

// ---------------------------------------------------------------------------
// Reductions

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// Sum across the block; every thread receives the total. Uses 32 floats of
// shared memory and ends with a barrier so it can be called repeatedly.
__device__ float block_sum(float v, float* sh) {
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  v = warp_sum(v);
  if (lane == 0) sh[wid] = v;
  __syncthreads();
  const int nw = (blockDim.x + 31) >> 5;
  float t = (lane < nw) ? sh[lane] : 0.f;
  t = warp_sum(t);
  __syncthreads();
  return t;
}

// ---------------------------------------------------------------------------
// LayerNorm

extern "C" __global__ void ln_fwd(float* y, const float* x, const float* w, const float* b, float* mean,
                                  float* rstd, int dim, float eps) {
  __shared__ float sh[32];
  const int row = blockIdx.x;
  const float* xr = x + (size_t)row * dim;
  float* yr = y + (size_t)row * dim;
  float s = 0.f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) s += xr[i];
  const float mu = block_sum(s, sh) / (float)dim;
  float v = 0.f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float d = xr[i] - mu;
    v += d * d;
  }
  const float rs = 1.0f / sqrtf(block_sum(v, sh) / (float)dim + eps);
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    float o = (xr[i] - mu) * rs * w[i];
    if (b) o += b[i];
    yr[i] = o;
  }
  if (threadIdx.x == 0) {
    mean[row] = mu;
    rstd[row] = rs;
  }
}

// Adds the input gradient into dx and accumulates dw (and db when non-null).
// Persistent blocks loop over rows; per-thread weight-gradient partials are
// flushed with one atomic per column per block. Supports dim <= 8 * blockDim.
extern "C" __global__ void ln_bwd(float* dx, const float* dy, const float* x, const float* w,
                                  const float* mean, const float* rstd, float* dw, float* db, int rows,
                                  int dim) {
  __shared__ float sh[32];
  float pw[8], pb[8];
#pragma unroll
  for (int c = 0; c < 8; c++) {
    pw[c] = 0.f;
    pb[c] = 0.f;
  }
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    const float m = mean[row], rs = rstd[row];
    const float* xr = x + (size_t)row * dim;
    const float* gr = dy + (size_t)row * dim;
    float s1 = 0.f, s2 = 0.f;
#pragma unroll
    for (int c = 0; c < 8; c++) {
      const int i = threadIdx.x + c * blockDim.x;
      if (i < dim) {
        const float xh = (xr[i] - m) * rs, g = gr[i], gw = g * w[i];
        s1 += gw;
        s2 += gw * xh;
        pw[c] += g * xh;
        pb[c] += g;
      }
    }
    const float a = block_sum(s1, sh) / (float)dim;
    const float b2 = block_sum(s2, sh) / (float)dim;
    float* dxr = dx + (size_t)row * dim;
#pragma unroll
    for (int c = 0; c < 8; c++) {
      const int i = threadIdx.x + c * blockDim.x;
      if (i < dim) {
        const float xh = (xr[i] - m) * rs;
        dxr[i] += rs * (gr[i] * w[i] - a - xh * b2);
      }
    }
  }
#pragma unroll
  for (int c = 0; c < 8; c++) {
    const int i = threadIdx.x + c * blockDim.x;
    if (i < dim) {
      atomicAdd(dw + i, pw[c]);
      if (db) atomicAdd(db + i, pb[c]);
    }
  }
}

// ---------------------------------------------------------------------------
// Embeddings

extern "C" __global__ void embed_gather(float* x, const float* tok, const int* ids, int H) {
  const int t = blockIdx.x;
  const float* src = tok + (size_t)ids[t] * H;
  float* dst = x + (size_t)t * H;
  for (int i = threadIdx.x; i < H; i += blockDim.x) dst[i] = src[i];
}

extern "C" __global__ void embed_scatter_add(float* dtok, const float* de, const int* ids, int H) {
  const int t = blockIdx.x;
  float* dst = dtok + (size_t)ids[t] * H;
  const float* src = de + (size_t)t * H;
  for (int i = threadIdx.x; i < H; i += blockDim.x) atomicAdd(dst + i, src[i]);
}

extern "C" __global__ void gather_rows(float* out, const float* src, const int* rows, int H) {
  const int r = blockIdx.x;
  const float* s = src + (size_t)rows[r] * H;
  float* d = out + (size_t)r * H;
  for (int i = threadIdx.x; i < H; i += blockDim.x) d[i] = s[i];
}

// dst[rows[r]] = src[r]; dst must be zeroed beforehand.
extern "C" __global__ void scatter_rows(float* dst, const float* src, const int* rows, int H) {
  const int r = blockIdx.x;
  float* d = dst + (size_t)rows[r] * H;
  const float* s = src + (size_t)r * H;
  for (int i = threadIdx.x; i < H; i += blockDim.x) d[i] = s[i];
}

// ---------------------------------------------------------------------------
// RoPE on the q and k slices of qkv rows laid out [q | k | v].

extern "C" __global__ void rope(float* qkv, const float* cs, const float* sn, int T, int H, int heads,
                                int D, int inverse) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int half = D >> 1;
  if (idx >= T * heads * half) return;
  const int i = idx % half;
  const int hd = (idx / half) % heads;
  const int t = idx / (half * heads);
  const float c = cs[t * half + i];
  float s = sn[t * half + i];
  if (inverse) s = -s;
#pragma unroll
  for (int part = 0; part < 2; part++) {
    float* v = qkv + (size_t)t * 3 * H + part * H + hd * D;
    const float a = v[i], b = v[i + half];
    v[i] = a * c - b * s;
    v[i + half] = b * c + a * s;
  }
}

// ---------------------------------------------------------------------------
// Attention softmax over per-(sequence, head) score matrices.
// items: 4 ints per item = {S, offset of the S*S matrix in P, first token index, unused}.
// opt/pos are per-token arrays; plain (full-attention) sequences carry opt = -1 everywhere.

__device__ __forceinline__ bool attn_allowed(const int* opt, const int* pos, int tok, int i, int j,
                                             int sliding, int win) {
  const int oi = opt[tok + i], oj = opt[tok + j];
  bool ok = (oi == -1) ? (oj == -1) : (oj == -1 || oi == oj);
  if (ok && sliding) {
    int d = pos[tok + i] - pos[tok + j];
    if (d < 0) d = -d;
    ok = d <= win;
  }
  return ok;
}

extern "C" __global__ void softmax_masked(float* P, const int* items, const int* opt, const int* pos,
                                          int heads, int sliding, int win, float scale) {
  const int item = blockIdx.y;
  const int* it = items + 4 * (item / heads);
  const int S = it[0];
  const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (row >= S) return;
  const int lane = threadIdx.x & 31;
  const int tok = it[2];
  float* p = P + it[1] + (size_t)(item % heads) * S * S + (size_t)row * S;
  float mx = -3.0e38f;
  for (int j = lane; j < S; j += 32)
    if (attn_allowed(opt, pos, tok, row, j, sliding, win)) mx = fmaxf(mx, p[j] * scale);
  mx = warp_max(mx);
  float sum = 0.f;
  for (int j = lane; j < S; j += 32) {
    if (attn_allowed(opt, pos, tok, row, j, sliding, win)) {
      const float e = __expf(p[j] * scale - mx);
      p[j] = e;
      sum += e;
    } else {
      p[j] = 0.f;
    }
  }
  const float inv = 1.0f / warp_sum(sum);
  for (int j = lane; j < S; j += 32) p[j] *= inv;
}

// dS = P * (dP - sum(P*dP)) * scale, in place over dP.
extern "C" __global__ void softmax_bwd(float* dP, const float* P, const int* items, int heads, float scale) {
  const int item = blockIdx.y;
  const int* it = items + 4 * (item / heads);
  const int S = it[0];
  const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (row >= S) return;
  const int lane = threadIdx.x & 31;
  const size_t base = it[1] + (size_t)(item % heads) * S * S + (size_t)row * S;
  const float* p = P + base;
  float* d = dP + base;
  float dot = 0.f;
  for (int j = lane; j < S; j += 32) dot += p[j] * d[j];
  dot = warp_sum(dot);
  for (int j = lane; j < S; j += 32) d[j] = p[j] * (d[j] - dot) * scale;
}

// ---------------------------------------------------------------------------
// Elementwise

__device__ __forceinline__ float gelu_f(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f)); }
__device__ __forceinline__ float gelu_grad_f(float x) {
  const float cdf = 0.5f * (1.0f + erff(x * 0.70710678118654752f));
  const float pdf = expf(-0.5f * x * x) * 0.3989422804014327f;
  return cdf + x * pdf;
}

// wi rows are [value | gate], each I wide: act = gelu(value) * gate.
extern "C" __global__ void geglu_fwd(float* act, const float* wi, int T, int I) {
  const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (size_t)T * I) return;
  const size_t t = idx / I, i = idx % I;
  const float* row = wi + t * 2 * I;
  act[idx] = gelu_f(row[i]) * row[I + i];
}

extern "C" __global__ void geglu_bwd(float* dwi, const float* dact, const float* wi, int T, int I) {
  const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (size_t)T * I) return;
  const size_t t = idx / I, i = idx % I;
  const float* row = wi + t * 2 * I;
  const float a = row[i], g = row[I + i], da = dact[idx];
  float* d = dwi + t * 2 * I;
  d[i] = da * g * gelu_grad_f(a);
  d[I + i] = da * gelu_f(a);
}

extern "C" __global__ void add2(float* y, const float* a, const float* b, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a[i] + b[i];
}

extern "C" __global__ void add_inplace(float* y, const float* a, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] += a[i];
}

// ---------------------------------------------------------------------------
// Optimizer

extern "C" __global__ void sumsq(const float* x, long long n, float* out) {
  __shared__ float sh[32];
  float s = 0.f;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
    s += x[i] * x[i];
  s = block_sum(s, sh);
  if (threadIdx.x == 0) atomicAdd(out, s);
}

extern "C" __global__ void adamw(float* p, const float* g, float* m, float* v, long long n, float lr, float wd,
                                 float b1, float b2, float eps, float gscale, float step_size,
                                 float inv_sqrt_b2c) {
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
    const float gr = g[i] * gscale;
    const float mi = b1 * m[i] + (1.f - b1) * gr;
    const float vi = b2 * v[i] + (1.f - b2) * gr * gr;
    m[i] = mi;
    v[i] = vi;
    const float den = sqrtf(vi) * inv_sqrt_b2c + eps;
    p[i] -= lr * step_size * mi / den + wd * p[i];
  }
}
