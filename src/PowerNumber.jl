# represents s*ε^α + c as ε -> 0
struct PowerNumber <: Number
    s::ComplexF64
    α::Float64
    c::ComplexF64
end

@inline powerpart(z::Number) = zero(z)
@inline exppart(l::Number) = l
@inline finitepart(z::Number) = z

@inline powerpart(l::PowerNumber) = l.s
@inline exppart(l::PowerNumber) = l.α
@inline finitepart(l::PowerNumber) = l.c

Base.promote_rule(::Type{PowerNumber}, ::Type{<:Number}) = PowerNumber
Base.convert(::Type{PowerNumber}, z::PowerNumber) = z
Base.convert(::Type{PowerNumber}, z::Number) = PowerNumber(0, 1, z)

==(a::PowerNumber, b::PowerNumber) = powerpart(a) == powerpart(b) && exppart(a) == exppart(b) && finitepart(a) == finitepart(b)
Base.isapprox(a::PowerNumber, b::PowerNumber; opts...) = ≈(powerpart(a), powerpart(b); opts...) && ≈(exppart(a), exppart(b); opts...) && ≈(finitepart(a), finitepart(b); opts...)