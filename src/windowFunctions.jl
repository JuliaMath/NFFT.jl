if VERSION >= v"0.6.0-"
    using SpecialFunctions
end

# This file contains different window functions.
# The function getWindow returns a pair of window functions based on a string

function getWindow(window::Symbol)
    if window == :gauss
        return window_gauss, window_gauss_hat
    elseif window == :spline
        return window_spline, window_spline_hat
    elseif window == :kaiser_bessel_rev
        return window_kaiser_bessel_rev, window_kaiser_bessel_rev_hat
    else # default to kaiser_bessel
        return window_kaiser_bessel, window_kaiser_bessel_hat
    end
end

function window_kaiser_bessel(x,n,m,sigma)
    b = pi*(2-1/sigma)
    arg = m^2-n^2*x^2
    if abs(x) < m/n
        y = sinh(b*sqrt(arg))/sqrt(arg)/pi
    elseif abs(x) > m/n
        y = zero(x)
    else
        y = b/pi
    end
    return y
end

function window_kaiser_bessel_hat(k,n,m,sigma)
    b = pi*(2-1/sigma)
    return besseli(0,m*sqrt(b^2-(2*pi*k/n)^2))
end

function window_kaiser_bessel_rev(x,n,m,sigma)
    b = pi*(2-1/sigma)
    if abs(x) < m/n
        arg = m*b*sqrt(1-(n*x/m)^2)
        y = 0.5/m*besseli(0,arg)
    else
        y = zero(x)
    end
    return y
end

function window_kaiser_bessel_rev_hat(k,n,m,sigma)
    b = pi*(2-1/sigma)

    arg = sqrt(complex((2*pi*m*k/n)^2-(m*b)^2))
    return sinc(arg/pi)
end


function window_gauss(x,n,m,sigma)
    b = m / pi
    if abs(x) < m/n
        y = 1 / sqrt(pi*b) * exp(-(n*x)^2 / b)
    else
        y =  zero(x)
    end
    return y
end

function window_gauss_hat(k,n,m,sigma)
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

function window_spline(x,n,m,sigma)
    if abs(x) < m/n
        y = cbspline(2*m, n*x+m)
    else
        y = zero(x)
    end
    return y
end

function window_spline_hat(k,n,m,sigma)
    return (sinc(k/n))^(2*m)
end
