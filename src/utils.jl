const _use_threads = Ref(false)

macro cthreads(loop::Expr) 
  return esc(quote
      if NFFT._use_threads[]
          @batch per=thread $loop
      else
          @inbounds $loop
      end
  end)
end