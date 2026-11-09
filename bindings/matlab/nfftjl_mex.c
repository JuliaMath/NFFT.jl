/* MEX gateway of nfftjl.Plan (MATLAB with -R2018a, Octave): nfftjl_mex(op, ...) with
     op 0: h = nfftjl_mex(0, kind, single, k, N_or_y, reltol, nthreads)
     op 1, 2: y = nfftjl_mex(op, h, kind, single, x, outsize)   (1 trafo, 2 adjoint)
     op 3: nfftjl_mex(3, h)                                     (destroy)
   kind 0 NFFT, 1 RNFFT, 2 NFCT, 3 NFST, 4 NNFFT. Data are column major as in libnfftjl. */
#include "mex.h"
#include "nfftjl.h"

typedef int64_t (*create_fn)(int, const int64_t *, int64_t, const double *, double, int);
typedef int (*apply_fn)(int64_t, const void *, void *);
#define A(f) (apply_fn)(f)

static const create_fn create[4][2] = {
  {nfftjl_plan_create, nfftjlf_plan_create}, {nfftjl_rnfft_plan_create, nfftjlf_rnfft_plan_create},
  {nfftjl_nfct_plan_create, nfftjlf_nfct_plan_create}, {nfftjl_nfst_plan_create, nfftjlf_nfst_plan_create}};
static const apply_fn apply[2][5][2] = {
  {{A(nfftjl_trafo), A(nfftjlf_trafo)}, {A(nfftjl_rnfft_trafo), A(nfftjlf_rnfft_trafo)},
   {A(nfftjl_nfct_trafo), A(nfftjlf_nfct_trafo)}, {A(nfftjl_nfst_trafo), A(nfftjlf_nfst_trafo)},
   {A(nfftjl_nnfft_trafo), A(nfftjlf_nnfft_trafo)}},
  {{A(nfftjl_adjoint), A(nfftjlf_adjoint)}, {A(nfftjl_rnfft_adjoint), A(nfftjlf_rnfft_adjoint)},
   {A(nfftjl_nfct_adjoint), A(nfftjlf_nfct_adjoint)}, {A(nfftjl_nfst_adjoint), A(nfftjlf_nfst_adjoint)},
   {A(nfftjl_nnfft_adjoint), A(nfftjlf_nnfft_adjoint)}}};
/* complex input and output per direction and kind (RNFFT: complex coefficients, real values) */
static const int cplx_in[2][5] = {{1, 1, 0, 0, 1}, {1, 0, 0, 0, 1}};
static const int cplx_out[2][5] = {{1, 0, 0, 0, 1}, {1, 1, 0, 0, 1}};

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  int op = (int)mxGetScalar(prhs[0]);
  if (op == 3) { nfftjl_plan_destroy((int64_t)mxGetScalar(prhs[1])); return; }
  int kind = (int)mxGetScalar(prhs[op == 0 ? 1 : 2]), sgl = mxGetScalar(prhs[op == 0 ? 2 : 3]) != 0;
  if (op == 0) {
    const mxArray *k = prhs[3], *n = prhs[4];
    int D = (int)mxGetM(k);
    double reltol = mxGetScalar(prhs[5]);
    int nthreads = (int)mxGetScalar(prhs[6]);
    int64_t h;
    if (kind == 4) {
      h = (sgl ? nfftjlf_nnfft_plan_create : nfftjl_nnfft_plan_create)(D, mxGetN(k), mxGetDoubles(k), mxGetN(n),
                                                                        mxGetDoubles(n), reltol, nthreads);
    } else {
      int64_t N[8];
      for (int d = 0; d < D && d < 8; d++) N[d] = (int64_t)mxGetDoubles(n)[d];
      h = create[kind][sgl](D, N, mxGetN(k), mxGetDoubles(k), reltol, nthreads);
    }
    if (h < 0) {
      char msg[1024];
      nfftjl_last_error(msg, sizeof msg);
      mexErrMsgIdAndTxt("nfftjl:plan", "%s", msg);
    }
    plhs[0] = mxCreateDoubleScalar((double)h);
    return;
  }
  const mxArray *x = prhs[4], *sz = prhs[5];
  int dir = op - 1;
  mxClassID cls = sgl ? mxSINGLE_CLASS : mxDOUBLE_CLASS;
  if (mxGetClassID(x) != cls || !mxIsComplex(x) != !cplx_in[dir][kind])
    mexErrMsgIdAndTxt("nfftjl:type", "wrong element type of the input");
  mwSize dims[8], nd = mxGetNumberOfElements(sz);
  for (mwSize d = 0; d < nd && d < 8; d++) dims[d] = (mwSize)mxGetDoubles(sz)[d];
  plhs[0] = mxCreateNumericArray(nd, dims, cls, cplx_out[dir][kind] ? mxCOMPLEX : mxREAL);
  if (apply[dir][kind][sgl]((int64_t)mxGetScalar(prhs[1]), mxGetData(x), mxGetData(plhs[0])) != 0)
    mexErrMsgIdAndTxt("nfftjl:apply", "invalid plan");
}
