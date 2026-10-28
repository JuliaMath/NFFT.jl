/* Tests of libnfftjl against the direct sum, and timings. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
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

int main(int argc, char **argv) {
  int nthreads = argc > 1 ? atoi(argv[1]) : 1, fail = 0;
  setvbuf(stdout, NULL, _IONBF, 0);
  srand(1);
  int64_t N1[] = {64}, N2[] = {16, 12}, N3[] = {8, 6, 10};
  fail |= accuracy(1, N1, 300, 1e-9, nthreads);
  fail |= accuracy(2, N2, 300, 1e-9, nthreads);
  fail |= accuracy(3, N3, 300, 1e-9, nthreads);
  fail |= accuracy(2, N2, 300, 1e-4, nthreads);
  int64_t T1[] = {262144}, T2[] = {512, 512}, T3[] = {64, 64, 64};
  timing(1, T1, 262144, 1e-4, nthreads);   /* warm-up: the threads of FFTW start on first use */
  timing(1, T1, 262144, 1e-4, nthreads);   /* warm-up: FFTW's threads start on first use */
  for (int i = 0; i < 2; i++) {
    double tol = i == 0 ? 1e-4 : 1e-8;
    timing(1, T1, 262144, tol, nthreads);
    timing(2, T2, 262144, tol, nthreads);
    timing(3, T3, 262144, tol, nthreads);
  }
  printf(fail ? "FAILED\n" : "passed\n");
  return fail;
}
