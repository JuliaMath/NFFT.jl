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

function window_kaiser_bessel(k,Ñ,m,σ)
  m_by_Ñ = m/Ñ
  b = pi*(2-1/σ)
  if abs(k) < m_by_Ñ
      arg = sqrt(m^2-Ñ^2*k^2)
      arg_times_pi = arg*pi
      y = sinh(b*arg)/arg_times_pi
  elseif abs(k) > m_by_Ñ
      y = zero(k)
  else
      y = b/pi
  end
  return y
end

function window_kaiser_bessel_hat(n,Ñ,m,σ)
    b = pi*(2-1/σ)
    return besseli(0,m*sqrt(b^2-(2*pi*n/Ñ)^2))
end

function window_kaiser_bessel_rev(k,Ñ,m,σ)
    b = pi*(2-1/σ)
    if abs(k) < m/Ñ
        arg = m*b*sqrt(1-(Ñ*k/m)^2)
        y = 0.5/m*besseli(0,arg)
    else
        y = zero(k)
    end
    return y
end

function window_kaiser_bessel_rev_hat(n,Ñ,m,σ)
  b = pi*(2-1/σ)

  arg = sqrt(complex((2*pi*m*n/Ñ)^2-(m*b)^2)) # Fix this to work on the real line.
  return real(sinc(arg/pi))
end


function window_gauss(k,Ñ,m,σ)
    b = m / pi
    if abs(k) < m/Ñ
        y = 1 / sqrt(pi*b) * exp(-(Ñ*k)^2 / b)
    else
        y =  zero(k)
    end
    return y
end

function window_gauss_hat(n,Ñ,m,σ)
    b = m / pi
    return exp(-(pi*n/Ñ)^2 * b)
end

function cbspline(m,k)
    if m == 1
        if k>=0 && k<1
            y = one(k)
        else
            y = zero(k)
        end
    else
        y = k/(m-1)*cbspline(m-1,k) + (m-k)/(m-1)*cbspline(m-1,k-1)
    end
    return y
end

function window_spline(k,Ñ,m,σ)
    if abs(k) < m/Ñ
        y = cbspline(2*m, Ñ*k+m)
    else
        y = zero(k)
    end
    return y
end

function window_spline_hat(n,Ñ,m,σ)
    return (sinc(n/Ñ))^(2*m)
end

# modified cosh_type window proposed in https://www-user.tu-chemnitz.de/~potts/paper/Ñffterror.pdf
# equation 5.22 and following

function window_cosh_type(k,Ñ,m,σ)
  m_by_Ñ = m/Ñ

  β = pi*m*(2-1/σ)
  if abs(k) < m_by_Ñ
      arg = (Ñ*k) / m
      α = sqrt(1-arg^2)
      y = 1/(cosh(β)-1) * (cosh(β*α)-1)/(α)
  else
      y = zero(k)
  end
  return y
end

function window_cosh_type_hat(n,Ñ,m,σ)
  β = pi*m*(2-1/σ)
  γ = β/(2*π)
  ζ = π/(cosh(β)-1) * m 

  arg = m * n / Ñ

  if abs(arg) < γ
    y = ζ * ( besseli(0, sqrt(β^2-(2*π*arg)^2)) - besselj(0,2*π*arg) )
  elseif abs(arg) > γ
    y =  ζ * ( besselj(0, sqrt((2*π*arg)^2-β^2)) - besselj(0,2*π*arg) )
  else
    y = ζ * (1 - besselj(0,β))
  end

  return y
end


