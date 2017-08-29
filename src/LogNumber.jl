

# represents s*log(ε) + c as ε -> 0
struct LogNumber <: Number
    s::Complex128
    c::Complex128
end


@inline logpart(z::Number) = zero(z)
@inline finitepart(z::Number) = z

@inline logpart(l::LogNumber) = l.s
@inline finitepart(l::LogNumber) = l.c


Base.promote_rule(::Type{LogNumber}, ::Type{<:Number}) = LogNumber
Base.convert(::Type{LogNumber}, z::LogNumber) = z
Base.convert(::Type{LogNumber}, z::Number) = LogNumber(0, z)

==(a::LogNumber, b::LogNumber) = logpart(a) == logpart(b) && finitepart(a) == finitepart(b)

(l::LogNumber)(ε) = logpart(l)*log(ε) + finitepart(l)

for f in (:+, :-)
    @eval begin
        $f(a::LogNumber, b::LogNumber) = LogNumber($f(a.s, b.s), $f(a.c, b.c))
        $f(l::LogNumber, b::Number) = LogNumber(l.s, $f(l.c, b))
        $f(a::Number, l::LogNumber) = LogNumber($f(l.s), $f(a, l.c))
    end
end

-(l::LogNumber) = LogNumber(-l.s, -l.c)

*(l::LogNumber, b::Number) = LogNumber(l.s*b, l.c*b)
*(a::Number, l::LogNumber) = LogNumber(a*l.s, a*l.c)
/(l::LogNumber, b::Number) = LogNumber(l.s/b, l.c/b)

function exp(l::LogNumber)::Complex128
    if real(l.s) > 0
        0.0+0.0im
    elseif real(l.s) < 0
        Inf+Inf*im
    elseif real(l.s) == 0 && imag(l.s) == 0
        log(l.c)
    else
        NaN + NaN*im
    end
end

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
undirected(r::RiemannDual) = undirected(value(r))

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


# loses sign information
for f in (:real, :imag, :abs)
    @eval $f(z::RiemannDual) = $f(value(z))
end

function log(z::RiemannDual)
    @assert value(z) == 0 || isinf(value(z))
    LogNumber(1,log(abs(epsilon(z))) + im*angle(epsilon(z)))
end

function atanh(z::RiemannDual)
    if value(z) ≈ 1
        LogNumber(-0.5,log(2)/2  - log(abs(epsilon(z)))/2 - im/2*angle(-epsilon(z)))
    elseif value(z) ≈ -1
        LogNumber(0.5,-log(2)/2  + log(abs(epsilon(z)))/2 + im/2*angle(epsilon(z)))
    else
        error("Not implemented")
    end
end





log1p(z::RiemannDual) = log(z+1)

SingularIntegralEquations.HypergeometricFunctions.speciallog(x::RiemannDual) =
    (s = sqrt(x); 3(atanh(s)-value(s))/value(s)^3)


# # (s*log(M) + c)*(p*M
# function /(l::LogNumber, b::RiemannDual)
#     @assert isinf(value(b))
#     LogNumber(l.s/b, l.c/b)
