

# represents s*ε^α + c as ε -> 0
struct PowerNumber <: Number
    s::ComplexF64
    c::ComplexF64
    α::Float64
end

@inline exppart(l::Number) = 0

@inline powerpart(l::PowerNumber) = l.s
@inline finitepart(l::PowerNumber) = l.c
@inline exppart(l::PowerNumber) = l.α

Base.promote_rule(::Type{PowerNumber}, ::Type{<:Number}) = PowerNumber
Base.convert(::Type{PowerNumber}, z::PowerNumber) = z
Base.convert(::Type{PowerNumber}, z::Number) = PowerNumber(0, z, 0)

==(a::PowerNumber, b::PowerNumber) = powerpart(a) == powerpart(b) && exppart(a) == exppart(b) && finitepart(a) == finitepart(b)
Base.isapprox(a::PowerNumber, b::PowerNumber; opts...) = ≈(powerpart(a), powerpart(b); opts...) && ≈(exppart(a), exppart(b); opts...) && ≈(finitepart(a), finitepart(b); opts...)

(l::PowerNumber)(ε) = powerpart(l)*(ε^(l.α)) + finitepart(l)

for f in (:+, :-)
    @eval begin
        function $f(a::PowerNumber, b::PowerNumber) 
            if a.α == b.α PowerNumber($f(a.s, b.s), $f(a.c, b.c), a.α) 
            elseif a.α < b.α && b.α ≥ 0
                PowerNumber(a.s, $f(a.c, b.c), a.α) 
            elseif b.α < a.α && a.α ≥ 0
                PowerNumber($f(b.s), $f(a.c, b.c), b.α) 
            else
                error("Not implemented")
            end
        end
        $f(l::PowerNumber, b::Number) = PowerNumber(l.s, $f(l.c, b), l.α)
        $f(a::Number, l::PowerNumber) = PowerNumber($f(l.s), $f(a, l.c), l.α)
    end
end

-(l::PowerNumber) = PowerNumber(-l.s, -l.c, l.α)

for Typ in (:Bool, :Number)
    @eval begin
        *(l::PowerNumber, b::$Typ) = PowerNumber(l.s*b, l.c*b, l.α)
        *(a::$Typ, l::PowerNumber) = PowerNumber(a*l.s, a*l.c, l.α)
    end
end

*(a::RiemannDual, b::PowerNumber) = realpart(a)*b
*(a::PowerNumber, b::RiemannDual) = a*realpart(b)

/(l::PowerNumber, b::Number) = PowerNumber(l.s/b, l.c/b, l.α)
/(l::PowerNumber, b::PowerNumber) = l*inv(b)
/(l::Number, b::PowerNumber) = l*inv(b)

exp(l::PowerNumber)::ComplexF64 = exp(l.c)

for op in (:real, :imag, :conj)
    @eval $op(l::PowerNumber) = PowerNumber($op(powerpart(l)), $op(finitepart(l)), exppart(l))
end

function sqrt(x::RiemannDual{T}) where T 
    if realpart(x) == 0 
        PowerNumber(sqrt(epsilon(x)),zero(T),one(T)/2) 
    elseif isinf(realpart(x))
        PowerNumber(sqrt(epsilon(x)),zero(T),-one(T)/2) 
    else
        RiemannDual(sqrt(dual(x)))
    end
end
^(x::RiemannDual{T}, α::Number) where T = realpart(x) == 0 ? PowerNumber(epsilon(x)^α,zero(T),α) : RiemannDual(dual(x)^α)

function inv(x::PowerNumber) 
    a,b, α = powerpart(x),finitepart(x),exppart(x)
    if b == 0 || α < 0 
        PowerNumber(inv(a), zero(b), -α) 
    else # α > 0
        PowerNumber(-b/a^2, inv(b), α)
    end
end

function *(x::PowerNumber, y::PowerNumber)
    a,b, α = powerpart(x),finitepart(x),exppart(x)
    c,d, γ = powerpart(y),finitepart(y),exppart(y)
    α ≤ γ || return y*x
    if α == 0.0
        (a+b) * y
    elseif α ≥ α+γ > 0
        PowerNumber(a*c,b*d, α+γ)
    elseif α+γ == 0
        PowerNumber(a*d,a*c+b*d, α)
    else
        error("Not implemented for $α, $γ")
    end
end

Base.show(io::IO, x::PowerNumber) = print(io, "($(powerpart(x)))ε^$(exppart(x)) + ($(finitepart(x)))")