/* A 2D NFFT and its adjoint with libnfftjl, checked against the direct sum.
   cc -O2 getting_started.c -I../include -L../build/lib -lnfftjl -lm -Wl,-rpath,../build/lib */
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "nfftjl.h"

int main(void) {
  const int D = 2;
  const int64_t N[] = {64, 48}, J = 1000;     /* array size, number of nodes */
  const int64_t M = N[0] * N[1];

  /* column major: node j is k[2j], k[2j+1]; f[l0 + N0*l1] */
  double *k = malloc(sizeof(double) * D * J);
  double complex *f = malloc(sizeof(double complex) * M), *fhat = malloc(sizeof(double complex) * J),
                 *g = malloc(sizeof(double complex) * M);
  srand(0);
  for (int64_t i = 0; i < D * J; i++) k[i] = rand() / (RAND_MAX + 1.0) - 0.5;
  for (int64_t i = 0; i < M; i++) f[i] = rand() / (RAND_MAX + 1.0) - 0.5;

  int64_t p = nfftjl_plan_create(D, N, J, k, 1e-9, nfftjl_num_threads());
  if (p < 0) return 1;
  nfftjl_trafo(p, f, fhat);                   /* fhat_j = Σ_l f_l exp(-2πi l·k_j) */
  nfftjl_adjoint(p, fhat, g);                 /* g_l = Σ_j fhat_j exp(+2πi l·k_j) */

  /* the direct sum at the first ten nodes; l_d = -N_d/2, …, N_d - N_d/2 - 1 */
  double e = 0, s = 0;
  for (int64_t j = 0; j < 10; j++) {
    double complex ref = 0;
    for (int64_t l1 = 0; l1 < N[1]; l1++)
      for (int64_t l0 = 0; l0 < N[0]; l0++)
        ref += f[l0 + N[0] * l1] *
               cexp(-2 * M_PI * I * ((l0 - N[0] / 2) * k[D * j] + (l1 - N[1] / 2) * k[D * j + 1]));
    e += pow(cabs(fhat[j] - ref), 2); s += pow(cabs(ref), 2);
  }
  printf("relative error %.1e\n", sqrt(e / s));

  nfftjl_plan_destroy(p);
  free(k); free(f); free(fhat); free(g);
  return sqrt(e / s) < 1e-8 ? 0 : 1;
}
