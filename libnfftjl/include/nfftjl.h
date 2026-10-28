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

/* A plan for many transforms with the same nodes: data of size L1 x N[0] x ... x N[D-1] x L3 and
   L1 x J x L3 (column major), the transforms along the dimensions in between. */
int64_t nfftjl_plan_create_many(int D, const int64_t *N, int64_t J, const double *k, int64_t L1, int64_t L3,
                                double reltol, int nthreads);

/* RNFFT (real values at the nodes): fhat_j = Re sum_n h_n exp(-2 pi i n.k_j) over n_0 = 0 ... N[0]/2
   and the other n_d as above; h has N[0]/2+1 x N[1] x ... complex values, fhat J real values.
   The adjoint: h_n = sum_j fhat_j exp(2 pi i n.k_j). */
int64_t nfftjl_rnfft_plan_create(int D, const int64_t *N, int64_t J, const double *k, double reltol, int nthreads);
int64_t nfftjl_rnfft_plan_create_many(int D, const int64_t *N, int64_t J, const double *k, int64_t L1, int64_t L3,
                                      double reltol, int nthreads);
int nfftjl_rnfft_trafo(int64_t plan, const double complex *h, double *fhat);
int nfftjl_rnfft_adjoint(int64_t plan, const double *fhat, double complex *h);

/* NFCT and NFST (real data, nodes in [0, 1/2]): fhat_j = sum_n f_n prod_d cos(2 pi n_d k_dj) over
   n_d = 0 ... N[d]-1 (NFCT), or with sin over n_d = 1 ... N[d] (NFST); the adjoint is the
   transpose. */
int64_t nfftjl_nfct_plan_create(int D, const int64_t *N, int64_t J, const double *k, double reltol, int nthreads);
int64_t nfftjl_nfct_plan_create_many(int D, const int64_t *N, int64_t J, const double *k, int64_t L1, int64_t L3,
                                     double reltol, int nthreads);
int nfftjl_nfct_trafo(int64_t plan, const double *f, double *fhat);
int nfftjl_nfct_adjoint(int64_t plan, const double *fhat, double *f);
int64_t nfftjl_nfst_plan_create(int D, const int64_t *N, int64_t J, const double *k, double reltol, int nthreads);
int64_t nfftjl_nfst_plan_create_many(int D, const int64_t *N, int64_t J, const double *k, int64_t L1, int64_t L3,
                                     double reltol, int nthreads);
int nfftjl_nfst_trafo(int64_t plan, const double *f, double *fhat);
int nfftjl_nfst_adjoint(int64_t plan, const double *fhat, double *f);
/* Plans of all transforms are released by nfftjl_plan_destroy. */

/* The same in single precision (data in float complex; nodes and reltol stay double). */
int64_t nfftjlf_plan_create(int D, const int64_t *N, int64_t J, const double *k, double reltol, int nthreads);
int64_t nfftjlf_plan_create_many(int D, const int64_t *N, int64_t J, const double *k, int64_t L1, int64_t L3,
                                 double reltol, int nthreads);
int nfftjlf_trafo(int64_t plan, const float complex *f, float complex *fhat);
int nfftjlf_adjoint(int64_t plan, const float complex *fhat, float complex *f);
int nfftjlf_plan_destroy(int64_t plan);
int64_t nfftjlf_rnfft_plan_create(int D, const int64_t *N, int64_t J, const double *k, double reltol, int nthreads);
int64_t nfftjlf_rnfft_plan_create_many(int D, const int64_t *N, int64_t J, const double *k, int64_t L1, int64_t L3,
                                       double reltol, int nthreads);
int nfftjlf_rnfft_trafo(int64_t plan, const float complex *h, float *fhat);
int nfftjlf_rnfft_adjoint(int64_t plan, const float *fhat, float complex *h);
int64_t nfftjlf_nfct_plan_create(int D, const int64_t *N, int64_t J, const double *k, double reltol, int nthreads);
int64_t nfftjlf_nfct_plan_create_many(int D, const int64_t *N, int64_t J, const double *k, int64_t L1, int64_t L3,
                                      double reltol, int nthreads);
int nfftjlf_nfct_trafo(int64_t plan, const float *f, float *fhat);
int nfftjlf_nfct_adjoint(int64_t plan, const float *fhat, float *f);
int64_t nfftjlf_nfst_plan_create(int D, const int64_t *N, int64_t J, const double *k, double reltol, int nthreads);
int64_t nfftjlf_nfst_plan_create_many(int D, const int64_t *N, int64_t J, const double *k, int64_t L1, int64_t L3,
                                      double reltol, int nthreads);
int nfftjlf_nfst_trafo(int64_t plan, const float *f, float *fhat);
int nfftjlf_nfst_adjoint(int64_t plan, const float *fhat, float *f);
/* The default number of threads of a plan: the environment variable NFFTJL_NUM_THREADS if set,
   otherwise all CPU threads (the library starts one Julia thread per CPU thread, which is also
   the upper limit for nthreads). Plans can be created and
   destroyed from any thread; one plan must not be executed on two threads at once. The Julia
   runtime starts on the first call into the library: make that call from one thread before
   others call concurrently (nfftjl_num_threads() will do). */
int nfftjl_num_threads(void);

#endif
