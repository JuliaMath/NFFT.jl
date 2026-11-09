# cuFINUFFT on the GPU, same problems as the suite `cuda` (J = |N| uniform nodes, tolerance sweep) or,
# with `batch`, as the suite `batch` (L transforms with the same nodes, n_trans = L): type 2 with
# isign -1 is the NFFT, type 1 with isign +1 its adjoint, x = 2πk, modes in increasing order. Error
# of the first transform against the direct sum on 64 sampled outputs, time the minimum over repeated
# executions (plan and setpts excluded, data on the device). Output: CSV on stdout.
#   python benchmark/cufinufft_bench.py [big|batch] > cufinufft.csv
import sys, time
import numpy as np, cupy as cp, cufinufft

mode = sys.argv[1] if len(sys.argv) > 1 else ""
SIZES = {1: (2**22,), 2: (2048, 2048), 3: (160, 160, 160)} if mode == "big" else {1: (2**18,), 2: (512, 512), 3: (64, 64, 64)}
TOLS = {"Float64": [10.0**-e for e in range(2, 15)], "Float32": [10.0**-e for e in range(2, 7)]}
# as BATCH_CASES in src/suites.jl: name, N, J, L
BATCH = [("oct", (2048,), 2048, [1, 8, 64, 512]), ("mri", (256, 256), 65536, [1, 4, 16, 32]),
         ("volume", (64, 64, 64), 2**18, [1, 4, 8])]
PRECISIONS = (("Float64", (np.float64, np.complex128)), ("Float32", (np.float32, np.complex64)))

def tmin(f, budget=0.1):
    f(); cp.cuda.Device().synchronize()
    best, t0, n = np.inf, time.perf_counter(), 0
    while n < 3 or time.perf_counter() - t0 < budget:
        s = time.perf_counter(); f(); cp.cuda.Device().synchronize(); best = min(best, time.perf_counter() - s); n += 1
    return best

def problems(rng, N, J):
    D = len(N)
    k = rng.random((D, J)) - 0.5
    freqs = [np.arange(-(n // 2), n - n // 2) for n in N]
    js = rng.choice(J, 64, replace=False)
    ls = [tuple(rng.integers(0, n) for n in N) for _ in range(64)]
    grid = np.stack([g.ravel() for g in np.meshgrid(*freqs, indexing="ij")])            # D x M
    lv = np.array([[freqs[d][l[d]] for d in range(D)] for l in ls])                        # 64 x D
    return k, js, ls, grid, lv

def run(name, N, J, L, T, rt, ct, tol, k, f, y, js, ls, grid, lv):
    D = len(N)
    # reference from the nodes as the library receives them (rounded to T)
    x = (2 * np.pi * k).astype(rt)
    ref_t = np.array([np.exp(-1j * (x[:, j].astype(np.float64) @ grid)) @ f[0].ravel() for j in js])
    ref_a = np.array([np.exp(1j * (l @ x.astype(np.float64))) @ y[0] for l in lv])
    xs = [cp.asarray(x[d]) for d in range(D)]
    fd = cp.asarray(f.astype(ct) if L > 1 else f[0].astype(ct)); yd = cp.asarray(y.astype(ct) if L > 1 else y[0].astype(ct))
    tag = f"cuFINUFFT,{name},{D},{'x'.join(map(str, N))},{J},{L},{T},{tol:g}"
    try:
        t0 = time.perf_counter()
        p2 = cufinufft.Plan(2, N, n_trans=L, eps=tol, isign=-1, dtype=ct); p2.setpts(*xs)
        p1 = cufinufft.Plan(1, N, n_trans=L, eps=tol, isign=1, dtype=ct); p1.setpts(*xs)
        cp.cuda.Device().synchronize(); tp = time.perf_counter() - t0
        out = p2.execute(fd).get(); g = p1.execute(yd).get()
        out = out.reshape(L, J)[0]; g = g.reshape((L,) + N)[0]
        et = np.linalg.norm(out[js] - ref_t) / np.linalg.norm(ref_t)
        ea = np.linalg.norm(np.array([g[l] for l in ls]) - ref_a) / np.linalg.norm(ref_a)
        tt = tmin(lambda: p2.execute(fd)); ta = tmin(lambda: p1.execute(yd))
        print(f"{tag},{et:.3e},{ea:.3e},{tp:.5f},{tt:.6f},{ta:.6f}", flush=True)
        del p1, p2
    except Exception as e:
        print(f"{tag},,,,,,# {str(e)[:80]}", flush=True)

print("lib,case,D,N,J,L,T,tol,err_trafo,err_adj,t_plan,t_trafo,t_adj")
rng = np.random.default_rng(1234)
if mode == "batch":
    for name, N, J, Ls in BATCH:
        k, js, ls, grid, lv = problems(rng, N, J)
        Lmax = max(Ls)
        f = rng.standard_normal((Lmax,) + N) + 1j * rng.standard_normal((Lmax,) + N)
        y = rng.standard_normal((Lmax, J)) + 1j * rng.standard_normal((Lmax, J))
        for L in Ls:
            for T, (rt, ct) in PRECISIONS:
                for tol in (1e-4, 1e-8) if T == "Float64" else (1e-4,):
                    run(name, N, J, L, T, rt, ct, tol, k, f[:L], y[:L], js, ls, grid, lv)
else:
    for D in (1, 2, 3):
        N = SIZES[D]; J = int(np.prod(N))
        k, js, ls, grid, lv = problems(rng, N, J)
        f = (rng.standard_normal(N) + 1j * rng.standard_normal(N))[None]
        y = (rng.standard_normal(J) + 1j * rng.standard_normal(J))[None]
        for T, (rt, ct) in PRECISIONS:
            for tol in TOLS[T]:
                run("cuda", N, J, 1, T, rt, ct, tol, k, f, y, js, ls, grid, lv)
