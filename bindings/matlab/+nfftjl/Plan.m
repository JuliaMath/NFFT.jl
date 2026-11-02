classdef Plan < handle
  % NFFT.jl through libnfftjl. p = nfftjl.Plan(k, N, name, value, ...) plans
  %   fhat = p.trafo(f):    fhat_j = sum_l f_l exp(-2 pi i l.k_j), l_d = -N_d/2 ... N_d-N_d/2-1
  %   f = p.adjoint(fhat):  the sum with the opposite sign
  % for nodes k (D x J, in [-1/2, 1/2)) and f of size N. Options: 'kind' ('nfft', 'rnfft', 'nfct',
  % 'nfst', 'nnfft'), 'reltol' (1e-8), 'single' (false), 'nthreads' (0: the library default).
  % RNFFT: f of size floor(N(1)/2)+1 x N(2) x ..., real fhat. NFCT, NFST: real data, k in [0, 1/2],
  % adjoint = transpose. NNFFT: nfftjl.Plan(k, y, 'kind', 'nnfft') with sources y (D x N).
  properties (SetAccess = private)
    h = -1
    kind
    single
    insize
    J
  end
  methods
    function p = Plan(k, N, varargin)
      o = struct('kind', 'nfft', 'reltol', 1e-8, 'single', false, 'nthreads', 0);
      for i = 1:2:numel(varargin), o.(varargin{i}) = varargin{i + 1}; end
      p.kind = find(strcmp(o.kind, {'nfft', 'rnfft', 'nfct', 'nfst', 'nnfft'})) - 1;
      p.single = logical(o.single);
      p.J = size(k, 2);
      if p.kind == 4, p.insize = [size(N, 2) 1]; else, p.insize = [N(:)' 1]; end
      if p.kind == 1, p.insize(1) = floor(N(1) / 2) + 1; end
      p.h = nfftjl_mex(0, p.kind, p.single, double(k), double(N), o.reltol, o.nthreads);
    end
    function fhat = trafo(p, f)
      fhat = p.apply(1, f, p.insize, [p.J 1], ~ismember(p.kind, [2 3]));
    end
    function f = adjoint(p, fhat)
      f = p.apply(2, fhat, [p.J 1], p.insize, ~ismember(p.kind, [1 2 3]));
    end
    function delete(p)
      if p.h >= 0, nfftjl_mex(3, p.h); end
    end
  end
  methods (Access = private)
    function y = apply(p, op, x, insize, outsize, cplx)
      if numel(x) ~= prod(insize), error('nfftjl:size', 'expected %d values', prod(insize)); end
      cls = 'double';
      if p.single, cls = 'single'; end
      x = cast(x, cls);
      if cplx, x = complex(x); end
      y = nfftjl_mex(op, p.h, p.kind, p.single, x, outsize);
    end
  end
end
