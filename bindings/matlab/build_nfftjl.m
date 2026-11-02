function build_nfftjl(libdir)
% Builds nfftjl_mex in this folder against libnfftjl in libdir (default ../../libnfftjl/build/lib).
here = fileparts(mfilename('fullpath'));
if nargin < 1, libdir = fullfile(here, '..', '..', 'libnfftjl', 'build', 'lib'); end
args = {['-I' fullfile(here, '..', '..', 'libnfftjl', 'include')], ['-L' libdir], '-lnfftjl', ...
        fullfile(here, 'nfftjl_mex.c')};
if exist('OCTAVE_VERSION', 'builtin')
  mkoctfile('--mex', '-R2018a', '-o', fullfile(here, 'nfftjl_mex'), ['-Wl,-rpath,' libdir], args{:});
else
  mex('-R2018a', '-outdir', here, ['LDFLAGS=$LDFLAGS -Wl,-rpath,' libdir], args{:});
end
end
