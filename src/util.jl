# Split a range into at most k contiguous chunks of nearly equal length.
chunks(r, k) = [r[i:min(i + cld(length(r), k) - 1, end)] for i in 1:cld(length(r), k):length(r)]

# Smallest even n′ ≥ n whose prime factors are 2, 3, 5 and 7 (FFT-friendly grid sizes).
function smooth_size(n)
  m = n + isodd(n)
  while !issmooth(m)
    m += 2
  end
  m
end
function issmooth(n)
  for p in (2, 3, 5, 7)
    while n % p == 0
      n ÷= p
    end
  end
  n == 1
end

function threaded_fill!(a, v, nthreads)
  Threads.@threads for s in chunks(eachindex(a), 4nthreads)
    @inbounds for i in s
      a[i] = v
    end
  end
  a
end
