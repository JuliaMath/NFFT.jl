/* Many transforms with the same nodes: L = 8 images of 128 x 128, stored one after the other
   (column major: 128 x 128 x 8, the batch after the transformed dimensions), with one plan,
   compared with one plan per image. */
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "nfftjl.h"

int main(void) {
  const int D = 2;
  const int64_t N[] = {128, 128}, J = 20000, L = 8, M = 128 * 128;
  double *k = malloc(sizeof(double) * D * J);
  double complex *f = malloc(sizeof(double complex) * M * L), *fhat = malloc(sizeof(double complex) * J * L),
                 *ref = malloc(sizeof(double complex) * J);
  srand(1);
  for (int64_t i = 0; i < D * J; i++) k[i] = rand() / (RAND_MAX + 1.0) - 0.5;
  for (int64_t i = 0; i < M * L; i++) f[i] = rand() / (RAND_MAX + 1.0) - 0.5;

  /* L1 = 1 (nothing before the transformed dimensions), L3 = L (the images) */
  int64_t p = nfftjl_plan_create_many(D, N, J, k, 1, L, 1e-8, 0);
  nfftjl_trafo(p, f, fhat);                   /* fhat: J x L, image l at fhat + J*l */

  int64_t q = nfftjl_plan_create(D, N, J, k, 1e-8, 0);
  double e = 0, s = 0;
  for (int64_t l = 0; l < L; l++) {
    nfftjl_trafo(q, f + M * l, ref);
    for (int64_t j = 0; j < J; j++) { e += pow(cabs(fhat[J * l + j] - ref[j]), 2); s += pow(cabs(ref[j]), 2); }
  }
  printf("difference to single plans %.1e\n", sqrt(e / s));

  nfftjl_plan_destroy(p); nfftjl_plan_destroy(q);
  free(k); free(f); free(fhat); free(ref);
  return sqrt(e / s) < 1e-12 ? 0 : 1;
}
