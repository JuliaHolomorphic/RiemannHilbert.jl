

# represents s*ε^α + c as ε -> 0
struct PowerNumber <: Number
    s::ComplexF64
    c::ComplexF64
    α::Float64
end

@inline powerpart(z::Number) = zero(z)
@inline finitepart(z::Number) = z
@inline exppart(l::Number) = l

@inline powerpart(l::PowerNumber) = l.s
@inline finitepart(l::PowerNumber) = l.c
@inline exppart(l::PowerNumber) = l.α

Base.promote_rule(::Type{PowerNumber}, ::Type{<:Number}) = PowerNumber
Base.convert(::Type{PowerNumber}, z::PowerNumber) = z
Base.convert(::Type{PowerNumber}, z::Number) = PowerNumber(0, z, 1)

==(a::PowerNumber, b::PowerNumber) = powerpart(a) == powerpart(b) && exppart(a) == exppart(b) && finitepart(a) == finitepart(b)
Base.isapprox(a::PowerNumber, b::PowerNumber; opts...) = ≈(powerpart(a), powerpart(b); opts...) && ≈(exppart(a), exppart(b); opts...) && ≈(finitepart(a), finitepart(b); opts...)

(l::PowerNumber)(ε) = powerpart(l)*(ε^(l.α)) + finitepart(l)

for f in (:+, :-)
    @eval begin
        $f(a::PowerNumber, b::PowerNumber) = if a.α == b.α PowerNumber($f(a.s, b.s), $f(a.c, b.c), a.α) else error("Exponents must be equal.") end
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

/(l::PowerNumber, b::Number) = PowerNumber(l.s/b, l.c/b, l.α)

exp(l::PowerNumber)::ComplexF64 = exp(l.c)

Base.show(io::IO, x::PowerNumber) = print(io, "($(powerpart(x)))ε^$(exppart(x)) + ($(finitepart(x)))")