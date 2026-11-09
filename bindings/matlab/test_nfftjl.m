% nfftjl.Plan against the direct sums: NFFT, RNFFT, NFCT, NFST and NNFFT, transform and adjoint,
% 1D-3D, double and single precision. After build_nfftjl:  octave --eval "run('test_nfftjl.m')"
addpath(fileparts(mfilename('fullpath')));
J = 60; nfail = 0;
relerr = @(a, b) norm(double(a(:)) - b(:)) / norm(b(:));
for D = 1:3
  N = [12 9 6]; N = N(1:D);
  r = cell(1, D); c = cell(1, D);
  for sgl = [false true]
    tol = 1e-8; bound = 1e-7;
    if sgl, tol = 1e-4; bound = 1e-3; end
    k = rand(D, J) - 0.5; y = 4 * (rand(D, 30) - 0.5);
    for kind = {'nfft', 'rnfft', 'nfct', 'nfst', 'nnfft'}
      kn = kind{1}; kk = k; arg = N;
      for d = 1:D
        r{d} = -floor(N(d) / 2):N(d) - floor(N(d) / 2) - 1;
        if strcmp(kn, 'rnfft') && d == 1, r{d} = 0:floor(N(1) / 2); end
        if strcmp(kn, 'nfct'), r{d} = 0:N(d) - 1; end
        if strcmp(kn, 'nfst'), r{d} = 1:N(d); end
      end
      if strcmp(kn, 'nnfft')
        L = y'; arg = y;
      else
        [c{:}] = ndgrid(r{:});
        L = cell2mat(cellfun(@(x) x(:), c, 'UniformOutput', false));
      end
      if any(strcmp(kn, {'nfct', 'nfst'}))
        kk = abs(k);  % nodes in [0, 1/2]
        E = ones(J, size(L, 1));
        for d = 1:D
          if strcmp(kn, 'nfct'), E = E .* cos(2 * pi * kk(d, :)' * L(:, d)');
          else, E = E .* sin(2 * pi * kk(d, :)' * L(:, d)'); end
        end
      else
        E = exp(-2i * pi * kk' * L');
      end
      p = nfftjl.Plan(kk, arg, 'kind', kn, 'reltol', tol, 'single', sgl);
      x = randn([p.insize 1]); v = randn(J, 1);
      if ~any(strcmp(kn, {'nfct', 'nfst'})), x = x + 1i * randn(size(x)); end
      if any(strcmp(kn, {'nfft', 'nnfft'})), v = v + 1i * randn(J, 1); end
      fh = E * x(:);
      if strcmp(kn, 'rnfft'), fh = real(fh); end
      e = [relerr(p.trafo(x), fh), relerr(p.adjoint(v), E' * v)];
      status = 'ok';
      if any(e >= bound), status = 'FAILED'; nfail = nfail + 1; end
      fprintf('%dD %-6s %-6s trafo %.1e adjoint %.1e %s\n', D, kn, class(p.trafo(x)), e, status);
      clear p
    end
  end
end
if nfail > 0, error('%d cases failed', nfail); end
disp('all passed');
