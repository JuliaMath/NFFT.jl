/* Tests of libnfftjl against the direct sum, and timings. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "nfftjl.h"

static double urand(void) { return rand() / (RAND_MAX + 1.0); }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

/* direct NDFT (sign -1) or its adjoint (sign +1), column major, modes -N/2 ... N-N/2-1 */
static void ndft(int D, const int64_t *N, int64_t J, const double *k, const double complex *in, double complex *out, int adjoint) {
  int64_t M = 1; for (int d = 0; d < D; d++) M *= N[d];
  if (adjoint) for (int64_t m = 0; m < M; m++) out[m] = 0; else for (int64_t j = 0; j < J; j++) out[j] = 0;
  for (int64_t j = 0; j < J; j++)
    for (int64_t m = 0; m < M; m++) {
      double ph = 0; int64_t r = m;
      for (int d = 0; d < D; d++) { int64_t l = r % N[d] - N[d] / 2; r /= N[d]; ph += l * k[d + D * j]; }
      if (adjoint) out[m] += in[j] * cexp(2 * M_PI * I * ph);
      else out[j] += in[m] * cexp(-2 * M_PI * I * ph);
    }
}

static double relerr(const double complex *x, const double complex *y, int64_t n) {
  double e = 0, s = 0;
  for (int64_t i = 0; i < n; i++) { e += pow(cabs(x[i] - y[i]), 2); s += pow(cabs(y[i]), 2); }
  return sqrt(e / s);
}

static int accuracy(int D, const int64_t *N, int64_t J, double reltol, int nthreads) {
  int64_t M = 1; for (int d = 0; d < D; d++) M *= N[d];
  double *k = malloc(sizeof(double) * D * J);
  double complex *f = malloc(sizeof(double complex) * M), *g = malloc(sizeof(double complex) * M),
                 *fh = malloc(sizeof(double complex) * J), *gh = malloc(sizeof(double complex) * J);
  for (int64_t i = 0; i < D * J; i++) k[i] = urand() - 0.5;
  for (int64_t i = 0; i < M; i++) f[i] = urand() - 0.5 + I * (urand() - 0.5);
  int64_t p = nfftjl_plan_create(D, N, J, k, reltol, nthreads);
  if (p < 0) { printf("plan failed\n"); return 1; }
  nfftjl_trafo(p, f, fh); ndft(D, N, J, k, f, gh, 0);
  double e1 = relerr(fh, gh, J);
  nfftjl_adjoint(p, gh, g); ndft(D, N, J, k, gh, f, 1);
  double e2 = relerr(g, f, M);
  printf("D = %d, M = %lld, J = %lld, reltol %.0e: error trafo %.1e, adjoint %.1e\n", D, M, J, reltol, e1, e2);
  nfftjl_plan_destroy(p);
  free(k); free(f); free(g); free(fh); free(gh);
  return !(e1 < 10 * reltol && e2 < 10 * reltol);
}

static void timing(int D, const int64_t *N, int64_t J, double reltol, int nthreads) {
  int64_t M = 1; for (int d = 0; d < D; d++) M *= N[d];
  double *k = malloc(sizeof(double) * D * J);
  double complex *f = malloc(sizeof(double complex) * M), *fh = malloc(sizeof(double complex) * J);
  for (int64_t i = 0; i < D * J; i++) k[i] = urand() - 0.5;
  for (int64_t i = 0; i < M; i++) f[i] = urand() - 0.5;
  double t0 = now();
  int64_t p = nfftjl_plan_create(D, N, J, k, reltol, nthreads);
  double tp = now() - t0, tt = 1e9, ta = 1e9;
  for (int r = 0; r < 10; r++) {
    t0 = now(); nfftjl_trafo(p, f, fh); double t = now() - t0; if (t < tt) tt = t;
    t0 = now(); nfftjl_adjoint(p, fh, f); t = now() - t0; if (t < ta) ta = t;
  }
  printf("D = %d, M = %lld, J = %lld, reltol %.0e, %d threads: plan %.1f ms, trafo %.2f ms, adjoint %.2f ms\n",
         D, M, J, reltol, nthreads, 1e3 * tp, 1e3 * tt, 1e3 * ta);
  nfftjl_plan_destroy(p);
  free(k); free(f); free(fh);
}

/* single precision against the direct sum in double precision */
static int accuracy_f32(int D, const int64_t *N, int64_t J, double reltol, int nthreads) {
  int64_t M = 1; for (int d = 0; d < D; d++) M *= N[d];
  double *k = malloc(sizeof(double) * D * J);
  double complex *f = malloc(sizeof(double complex) * M), *g = malloc(sizeof(double complex) * M),
                 *fh = malloc(sizeof(double complex) * J), *gh = malloc(sizeof(double complex) * J);
  float complex *ff = malloc(sizeof(float complex) * M), *fhf = malloc(sizeof(float complex) * J);
  for (int64_t i = 0; i < D * J; i++) k[i] = urand() - 0.5;
  for (int64_t i = 0; i < M; i++) { ff[i] = urand() - 0.5 + I * (urand() - 0.5); f[i] = ff[i]; }
  int64_t p = nfftjlf_plan_create(D, N, J, k, reltol, nthreads);
  if (p < 0) { printf("plan failed\n"); return 1; }
  nfftjlf_trafo(p, ff, fhf); ndft(D, N, J, k, f, gh, 0);
  for (int64_t j = 0; j < J; j++) fh[j] = fhf[j];
  double e1 = relerr(fh, gh, J);
  for (int64_t j = 0; j < J; j++) fhf[j] = gh[j];
  nfftjlf_adjoint(p, fhf, ff); ndft(D, N, J, k, gh, f, 1);
  for (int64_t i = 0; i < M; i++) g[i] = ff[i];
  double e2 = relerr(g, f, M);
  printf("float, D = %d, M = %lld, J = %lld, reltol %.0e: error trafo %.1e, adjoint %.1e\n", D, M, J, reltol, e1, e2);
  int bad = nfftjl_trafo(p, f, fh) != -1;   /* a single-precision plan is not a double one */
  nfftjlf_plan_destroy(p);
  free(k); free(f); free(g); free(fh); free(gh); free(ff); free(fhf);
  return bad || !(e1 < 10 * reltol && e2 < 10 * reltol);
}

/* batches: data L1 x N x L3 and L1 x J x L3; every slice against the direct sum */
static int accuracy_many(int D, const int64_t *N, int64_t J, int64_t L1, int64_t L3, double reltol, int nthreads) {
  int64_t M = 1; for (int d = 0; d < D; d++) M *= N[d];
  double *k = malloc(sizeof(double) * D * J);
  double complex *f = malloc(sizeof(double complex) * L1 * M * L3), *g = malloc(sizeof(double complex) * L1 * M * L3),
                 *fh = malloc(sizeof(double complex) * L1 * J * L3), *fs = malloc(sizeof(double complex) * M),
                 *gs = malloc(sizeof(double complex) * M), *hs = malloc(sizeof(double complex) * J),
                 *rs = malloc(sizeof(double complex) * J);
  for (int64_t i = 0; i < D * J; i++) k[i] = urand() - 0.5;
  for (int64_t i = 0; i < L1 * M * L3; i++) f[i] = urand() - 0.5 + I * (urand() - 0.5);
  int64_t p = nfftjl_plan_create_many(D, N, J, k, L1, L3, reltol, nthreads);
  if (p < 0) { printf("plan failed\n"); return 1; }
  nfftjl_trafo(p, f, fh);
  nfftjl_adjoint(p, fh, g);
  double e1 = 0, e2 = 0;
  for (int64_t b = 0; b < L3; b++)
    for (int64_t a = 0; a < L1; a++) {
      for (int64_t m = 0; m < M; m++) fs[m] = f[a + L1 * (m + M * b)];
      for (int64_t j = 0; j < J; j++) hs[j] = fh[a + L1 * (j + J * b)];
      ndft(D, N, J, k, fs, rs, 0);
      double e = relerr(hs, rs, J); if (e > e1) e1 = e;
      ndft(D, N, J, k, hs, fs, 1);
      for (int64_t m = 0; m < M; m++) gs[m] = g[a + L1 * (m + M * b)];
      e = relerr(gs, fs, M); if (e > e2) e2 = e;
    }
  /* the strategy is in bits 24-27 of the handle (1 single, 2 slices, 3 lead, 4 trail) */
  printf("batch %lld x (D = %d, M = %lld) x %lld, J = %lld, strategy %lld: max error trafo %.1e, adjoint %.1e\n",
         L1, D, M, L3, J, (p >> 24) & 0xf, e1, e2);
  nfftjl_plan_destroy(p);
  free(k); free(f); free(g); free(fh); free(fs); free(gs); free(hs); free(rs);
  return !(e1 < 10 * reltol && e2 < 10 * reltol);
}

/* the real transforms against their direct sums, with a batch of L3 slices (kind 0 RNFFT, 1 NFCT, 2 NFST) */
static double cterm(int kind, int64_t n, double x) { return kind == 1 ? cos(2 * M_PI * n * x) : sin(2 * M_PI * n * x); }
static int accuracy_real(int kind, int D, const int64_t *N, int64_t J, int64_t L3, double reltol, int nthreads) {
  int64_t Nin[3], M = 1, Mr = 1;
  for (int d = 0; d < D; d++) { Nin[d] = (kind == 0 && d == 0) ? N[0] / 2 + 1 : N[d]; M *= Nin[d]; Mr *= N[d]; }
  double *k = malloc(sizeof(double) * D * J), *fhat = malloc(sizeof(double) * J * L3), *fr = malloc(sizeof(double) * M * L3),
         *gr = malloc(sizeof(double) * M * L3);
  double complex *h = malloc(sizeof(double complex) * M * L3), *gh = malloc(sizeof(double complex) * M * L3);
  for (int64_t i = 0; i < D * J; i++) k[i] = kind == 0 ? urand() - 0.5 : urand() / 2;
  for (int64_t i = 0; i < M * L3; i++) { h[i] = urand() - 0.5 + I * (urand() - 0.5); fr[i] = urand() - 0.5; }
  int64_t p = kind == 0 ? nfftjl_rnfft_plan_create_many(D, N, J, k, 1, L3, reltol, nthreads) :
              kind == 1 ? nfftjl_nfct_plan_create_many(D, N, J, k, 1, L3, reltol, nthreads) :
                          nfftjl_nfst_plan_create_many(D, N, J, k, 1, L3, reltol, nthreads);
  if (p < 0) { printf("plan failed\n"); return 1; }
  if (kind == 0) { nfftjl_rnfft_trafo(p, h, fhat); nfftjl_rnfft_adjoint(p, fhat, gh); }
  else if (kind == 1) { nfftjl_nfct_trafo(p, fr, fhat); nfftjl_nfct_adjoint(p, fhat, gr); }
  else { nfftjl_nfst_trafo(p, fr, fhat); nfftjl_nfst_adjoint(p, fhat, gr); }
  double e1 = 0, s1 = 0, e2 = 0, s2 = 0;
  for (int64_t b = 0; b < L3; b++) {
    for (int64_t j = 0; j < J; j++) {           /* trafo */
      double ref = 0;
      for (int64_t m = 0; m < M; m++) {
        int64_t r = m; double ph = 0, pr = 1;
        for (int d = 0; d < D; d++) {
          int64_t i = r % Nin[d]; r /= Nin[d];
          if (kind == 0) ph += (d == 0 ? i : i - N[d] / 2) * k[d + D * j];
          else pr *= cterm(kind, kind == 1 ? i : i + 1, k[d + D * j]);
        }
        ref += kind == 0 ? creal(h[m + M * b] * cexp(-2 * M_PI * I * ph)) : fr[m + M * b] * pr;
      }
      e1 += pow(fhat[j + J * b] - ref, 2); s1 += ref * ref;
    }
    for (int64_t m = 0; m < M; m++) {           /* adjoint */
      double complex ref = 0;
      for (int64_t j = 0; j < J; j++) {
        int64_t r = m; double ph = 0, pr = 1;
        for (int d = 0; d < D; d++) {
          int64_t i = r % Nin[d]; r /= Nin[d];
          if (kind == 0) ph += (d == 0 ? i : i - N[d] / 2) * k[d + D * j];
          else pr *= cterm(kind, kind == 1 ? i : i + 1, k[d + D * j]);
        }
        ref += kind == 0 ? fhat[j + J * b] * cexp(2 * M_PI * I * ph) : fhat[j + J * b] * pr;
      }
      double complex v = kind == 0 ? gh[m + M * b] : gr[m + M * b];
      e2 += pow(cabs(v - ref), 2); s2 += pow(cabs(ref), 2);
    }
  }
  e1 = sqrt(e1 / s1); e2 = sqrt(e2 / s2);
  printf("%s, D = %d, J = %lld, %lld slices: error trafo %.1e, adjoint %.1e\n", kind == 0 ? "RNFFT" : kind == 1 ? "NFCT" : "NFST",
         D, J, L3, e1, e2);
  int bad = nfftjl_trafo(p, h, (double complex *)fhat) != -1;   /* the NFFT functions reject it */
  nfftjl_plan_destroy(p);
  free(k); free(fhat); free(fr); free(gr); free(h); free(gh);
  return bad || !(e1 < 10 * reltol && e2 < 10 * reltol);
}

/* NNFFT against the direct sum fhat_j = sum_l f_l exp(-2 pi i k_j.y_l), batch (L1, ., L3), in double or
   single precision */
static int accuracy_nnfft(int D, int64_t J, int64_t N, int64_t L1, int64_t L3, double reltol, int f32, int nthreads) {
  int64_t L = L1 * L3;
  double *k = malloc(sizeof(double) * D * J), *y = malloc(sizeof(double) * D * N);
  double complex *f = malloc(sizeof(double complex) * N * L), *g = malloc(sizeof(double complex) * N * L),
                 *fh = malloc(sizeof(double complex) * J * L), *gh = malloc(sizeof(double complex) * J * L);
  float complex *ff = malloc(sizeof(float complex) * N * L), *fhf = malloc(sizeof(float complex) * J * L);
  for (int64_t i = 0; i < D * J; i++) k[i] = 6 * (urand() - 0.5) - 1;
  for (int64_t i = 0; i < D * N; i++) y[i] = 20 * (urand() - 0.5) + 3;
  for (int64_t i = 0; i < N * L; i++) f[i] = urand() - 0.5 + I * (urand() - 0.5);
  int64_t p = f32 ? nfftjlf_nnfft_plan_create_many(D, J, k, N, y, L1, L3, reltol, nthreads)
                  : nfftjl_nnfft_plan_create_many(D, J, k, N, y, L1, L3, reltol, nthreads);
  if (p < 0) { printf("NNFFT plan failed\n"); return 1; }
  /* direct sums, slice (a, b) */
  for (int64_t b = 0; b < L3; b++)
    for (int64_t a = 0; a < L1; a++)
      for (int64_t j = 0; j < J; j++) {
        double complex s = 0;
        for (int64_t l = 0; l < N; l++) {
          double x = 0; for (int d = 0; d < D; d++) x += k[d + D * j] * y[d + D * l];
          s += f[a + L1 * (l + N * b)] * cexp(-2 * M_PI * I * x);
        }
        gh[a + L1 * (j + J * b)] = s;
      }
  double e1, e2;
  if (f32) {
    for (int64_t i = 0; i < N * L; i++) ff[i] = f[i];
    nfftjlf_nnfft_trafo(p, ff, fhf);
    for (int64_t i = 0; i < J * L; i++) fh[i] = fhf[i];
  } else nfftjl_nnfft_trafo(p, f, fh);
  e1 = relerr(fh, gh, J * L);
  /* adjoint of gh against its direct sum */
  for (int64_t b = 0; b < L3; b++)
    for (int64_t a = 0; a < L1; a++)
      for (int64_t l = 0; l < N; l++) {
        double complex s = 0;
        for (int64_t j = 0; j < J; j++) {
          double x = 0; for (int d = 0; d < D; d++) x += k[d + D * j] * y[d + D * l];
          s += gh[a + L1 * (j + J * b)] * cexp(2 * M_PI * I * x);
        }
        f[a + L1 * (l + N * b)] = s;
      }
  if (f32) {
    for (int64_t i = 0; i < J * L; i++) fhf[i] = gh[i];
    nfftjlf_nnfft_adjoint(p, fhf, ff);
    for (int64_t i = 0; i < N * L; i++) g[i] = ff[i];
  } else nfftjl_nnfft_adjoint(p, gh, g);
  e2 = relerr(g, f, N * L);
  printf("NNFFT%s, D = %d, J = %lld, N = %lld, batch %lld x %lld, reltol %.0e: error trafo %.1e, adjoint %.1e\n",
         f32 ? " (float)" : "", D, J, N, L1, L3, reltol, e1, e2);
  int bad = nfftjl_trafo(p, f, fh) != -1;   /* the NFFT functions reject it */
  bad |= nfftjl_plan_destroy(p) != 0 || nfftjl_plan_destroy(p) != -1;
  free(k); free(y); free(f); free(g); free(fh); free(gh); free(ff); free(fhf);
  return bad || !(e1 < 10 * reltol && e2 < 10 * reltol);
}

int main(int argc, char **argv) {
  int nthreads = argc > 1 ? atoi(argv[1]) : 1, fail = 0;
  setvbuf(stdout, NULL, _IONBF, 0);
  srand(1);
  int64_t N1[] = {64}, N2[] = {16, 12}, N3[] = {8, 6, 10};
  fail |= accuracy(1, N1, 300, 1e-9, nthreads);
  fail |= accuracy(2, N2, 300, 1e-9, nthreads);
  fail |= accuracy(3, N3, 300, 1e-9, nthreads);
  fail |= accuracy(2, N2, 300, 1e-4, nthreads);
  /* the batch strategies: before (lead), after (trail), around (loop) the transformed dimensions,
     many small slices (slices) */
  int64_t B1[] = {300}, B2[] = {32, 28};
  fail |= accuracy_many(1, B1, 200, 5, 1, 1e-9, nthreads);
  fail |= accuracy_many(2, B2, 300, 1, 6, 1e-9, nthreads);
  fail |= accuracy_many(2, B2, 300, 2, 3, 1e-9, nthreads);
  fail |= accuracy_many(3, N3, 200, 1, 2, 1e-9, nthreads);
  fail |= accuracy_many(1, N1, 50, 8, 9, 1e-9, nthreads);
  for (int kind = 0; kind < 3; kind++) {
    int64_t R1[] = {40}, R2[] = {12, 10}, R3[] = {6, 5, 4};
    fail |= accuracy_real(kind, 1, R1, 150, 1, 1e-9, nthreads);
    fail |= accuracy_real(kind, 2, R2, 150, 3, 1e-9, nthreads);
    fail |= accuracy_real(kind, 3, R3, 100, 1, 1e-9, nthreads);
  }
  fail |= accuracy_nnfft(1, 200, 150, 1, 1, 1e-9, 0, nthreads);
  fail |= accuracy_nnfft(2, 200, 150, 2, 3, 1e-9, 0, nthreads);
  fail |= accuracy_nnfft(3, 100, 120, 1, 1, 1e-9, 0, nthreads);
  fail |= accuracy_nnfft(2, 200, 150, 1, 2, 1e-4, 1, nthreads);
  { /* a failing plan reports why */
    double kbad[] = {-0.25}; int64_t Nb[] = {8}; char msg[200];
    int64_t p = nfftjl_nfct_plan_create(1, Nb, 1, kbad, 1e-6, 1);
    size_t n = nfftjl_last_error(msg, sizeof msg);
    printf("failing plan: %lld, \"%s\"\n", p, msg);
    fail |= p != -1 || n == 0 || !strstr(msg, "[0, 1/2]");
  }
  fail |= accuracy_f32(1, N1, 300, 1e-5, nthreads);
  fail |= accuracy_f32(2, N2, 300, 1e-5, nthreads);
  fail |= accuracy_f32(3, N3, 300, 1e-3, nthreads);
  int64_t T1[] = {262144}, T2[] = {512, 512}, T3[] = {64, 64, 64};
  timing(1, T1, 262144, 1e-4, nthreads);   /* warm-up: the threads of FFTW start on first use */
  for (int i = 0; i < 2; i++) {
    double tol = i == 0 ? 1e-4 : 1e-8;
    timing(1, T1, 262144, tol, nthreads);
    timing(2, T2, 262144, tol, nthreads);
    timing(3, T3, 262144, tol, nthreads);
  }
  printf(fail ? "FAILED\n" : "passed\n");
  return fail;
}
