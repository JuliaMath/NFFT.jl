/* NFFT.jl as a C library (built with juliac --trim). Arrays are column major (Julia order):
   nodes k[d + D*j] in [-1/2, 1/2), coefficients f of size N[0] x ... x N[D-1],
   fhat_j = sum_l f_l exp(-2 pi i l.k_j), l = -N/2 ... N-N/2-1 in every dimension. */
#ifndef NFFTJL_H
#define NFFTJL_H
#include <stdint.h>
#include <complex.h>

/* A plan for D = 1..3 dimensions, J nodes, relative error reltol, using nthreads threads
   (0: the default, nfftjl_num_threads()); -1 on error. */
int64_t nfftjl_plan_create(int D, const int64_t *N, int64_t J, const double *k, double reltol, int nthreads);
int nfftjl_trafo(int64_t plan, const double complex *f, double complex *fhat);
int nfftjl_adjoint(int64_t plan, const double complex *fhat, double complex *f);
int nfftjl_plan_destroy(int64_t plan);

/* The same in single precision (data in float complex; nodes and reltol stay double). */
int64_t nfftjlf_plan_create(int D, const int64_t *N, int64_t J, const double *k, double reltol, int nthreads);
int nfftjlf_trafo(int64_t plan, const float complex *f, float complex *fhat);
int nfftjlf_adjoint(int64_t plan, const float complex *fhat, float complex *f);
int nfftjlf_plan_destroy(int64_t plan);
/* The default number of threads of a plan: the environment variable NFFTJL_NUM_THREADS if set,
   otherwise all CPU threads (the library starts one Julia thread per CPU thread, which is also
   the upper limit for nthreads). Plans can be created and
   destroyed from any thread; one plan must not be executed on two threads at once. The Julia
   runtime starts on the first call into the library: make that call from one thread before
   others call concurrently (nfftjl_num_threads() will do). */
int nfftjl_num_threads(void);

#endif
