

# represents s*log(ε) + c
# or possibly s*log(M) + c
struct LogNumber
    s::Complex128
    c::Complex128
end

(l::LogNumber)(ε) = l.s*log(ε) + l.c

for f in (:+, :-)
    @eval begin
        $f(a::LogNumber, b::LogNumber) = LogNumber($f(a.s, b.s), $f(a.c, b.c))
        $f(l::LogNumber, b::Number) = LogNumber(l.s, $f(l.c, b))
        $f(a::Number, l::LogNumber) = LogNumber($f(l.s), $f(a, l.c))
    end
end

*(l::LogNumber, b::Number) = LogNumber(l.s*b, l.c*b)
*(a::Number, l::LogNumber) = LogNumber(a*l.s, a*l.c)
/(l::LogNumber, b::Number) = LogNumber(l.s/b, l.c/b)

# This is a relative version of dual number, in the sense that its value*(1+epsilon)
struct RiemannDual{T} <: Number
    value::T
    epsilon::T
end

RiemannDual(x, y) = RiemannDual(promote(x, y)...)

RiemannDual(x::Dual) = RiemannDual(value(x), epsilon(x))
Dual(x::RiemannDual) = Dual(value(x), epsilon(x))
dual(x::RiemannDual) = Dual(x)

# the relative perturbation
value(r::RiemannDual) = r.value
epsilon(r::RiemannDual) = r.epsilon
undirected(r::RiemannDual) = value(r)

for f in (:-,)
    @eval $f(x::RiemannDual) = RiemannDual($f(value(x)),$f(epsilon(x)))
end

for f in (:sqrt,)
    @eval $f(x::RiemannDual) = RiemannDual($f(dual(x)))
end

for f in (:+, :-, :*)
    @eval begin
        $f(x::RiemannDual, y::RiemannDual) = RiemannDual($f(dual(x),dual(y)))
        $f(x::RiemannDual, p::Number) = RiemannDual($f(dual(x),p))
        $f(p::Number, x::RiemannDual) = RiemannDual($f(p,dual(x)))
    end
end

function inv(z::RiemannDual)
    value(z) == 0 && return RiemannDual(inv(value(z)),inv(epsilon(z)))
    RiemannDual(inv(dual(z)))
end

/(z::RiemannDual, x::RiemannDual) = z*inv(x)
/(z::RiemannDual, x::Number) = z*inv(x)
/(x::Number, z::RiemannDual) = x*inv(z)

function log(z::RiemannDual)
    @assert value(z) == 0 || isinf(value(z))
    LogNumber(1,log(abs(epsilon(z))) + im*angle(epsilon(z)))
end

function atanh(z::RiemannDual)
    if value(z) ≈ 1

    elseif value(z) ≈ -1

    else
        error("Not implemented")
    end
end

log1p(z::RiemannDual) = log(z+1)

SingularIntegralEquations.HypergeometricFunctions.speciallog(x::RiemannDual) =
    (s = sqrt(x); 3(atanh(s)-s)/s^3)


# # (s*log(M) + c)*(p*M
# function /(l::LogNumber, b::RiemannDual)
#     @assert isinf(value(b))
#     LogNumber(l.s/b, l.c/b)
