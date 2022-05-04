# This file contains different window functions.
# The function getWindow returns a pair of window functions based on a string

function getWindow(window::Symbol)
    if window == :gauss
        return window_gauss, window_gauss_hat
    elseif window == :spline
        return window_spline, window_spline_hat
    elseif window == :kaiser_bessel_rev
        return window_kaiser_bessel_rev, window_kaiser_bessel_rev_hat
    elseif window == :kaiser_bessel
        return window_kaiser_bessel, window_kaiser_bessel_hat
    elseif window == :cosh_type
        return window_cosh_type, window_cosh_type_hat
    else 
        error("Window $(window) not yet implemented!")
        return window_cosh_type, window_cosh_type_hat
    end
end

function window_kaiser_bessel(x,n,m,σ)
  m_by_n = m/n
  b = pi*(2-1/σ)
  if abs(x) < m_by_n
      arg = sqrt(m^2-n^2*x^2)
      arg_times_pi = arg*pi
      y = sinh(b*arg)/arg_times_pi
  elseif abs(x) > m_by_n
      y = zero(x)
  else
      y = b/pi
  end
  return y
end

function window_kaiser_bessel_hat(k,n,m,σ)
    b = pi*(2-1/σ)
    return besseli(0,m*sqrt(b^2-(2*pi*k/n)^2))
end

function window_kaiser_bessel_rev(x,n,m,σ)
    b = pi*(2-1/σ)
    if abs(x) < m/n
        arg = m*b*sqrt(1-(n*x/m)^2)
        y = 0.5/m*besseli(0,arg)
    else
        y = zero(x)
    end
    return y
end

function window_kaiser_bessel_rev_hat(k,n,m,σ)
  b = pi*(2-1/σ)

  arg = sqrt(complex((2*pi*m*k/n)^2-(m*b)^2)) # Fix this to work on the real line.
  return real(sinc(arg/pi))
end


function window_gauss(x,n,m,σ)
    b = m / pi
    if abs(x) < m/n
        y = 1 / sqrt(pi*b) * exp(-(n*x)^2 / b)
    else
        y =  zero(x)
    end
    return y
end

function window_gauss_hat(k,n,m,σ)
    b = m / pi
    return exp(-(pi*k/(n))^2 * b)
end

function cbspline(m,x)
    if m == 1
        if x>=0 && x<1
            y = one(x)
        else
            y = zero(x)
        end
    else
        y = x/(m-1)*cbspline(m-1,x) + (m-x)/(m-1)*cbspline(m-1,x-1)
    end
    return y
end

function window_spline(x,n,m,σ)
    if abs(x) < m/n
        y = cbspline(2*m, n*x+m)
    else
        y = zero(x)
    end
    return y
end

function window_spline_hat(k,n,m,σ)
    return (sinc(k/n))^(2*m)
end

# modified cosh_type window proposed in https://www-user.tu-chemnitz.de/~potts/paper/nffterror.pdf
# equation 5.22 and following

function window_cosh_type(x,n,m,σ)
  m_by_n = m/n

  β = pi*m*(2-1/σ)
  if abs(x) < m_by_n
      arg = (n*x) / m
      α = sqrt(1-arg^2)
      y = 1/(cosh(β)-1) * (cosh(β*α)-1)/(α)
  else
      y = zero(x)
  end
  return y
end

function window_cosh_type_hat(k,n,m,σ)
  β = pi*m*(2-1/σ)
  γ = β/(2*π)
  ζ = π/(cosh(β)-1) * m 

  arg = m * k / n

  if abs(arg) < γ
    y = ζ * ( besseli(0, sqrt(β^2-(2*π*arg)^2)) - besselj(0,2*π*arg) )
  elseif abs(arg) > γ
    y =  ζ * ( besselj(0, sqrt((2*π*arg)^2-β^2)) - besselj(0,2*π*arg) )
  else
    y = ζ * (1 - besselj(0,β))
  end

  return y
end


