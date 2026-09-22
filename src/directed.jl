"""
`Directed` represents a number that is a limit from either left (s=true) or right (s=false)
For functions with branch cuts, it is assumed that the value is on the branch cut,
Therefore not requiring tolerances.  This will naturally give the analytic continuation.
"""
struct Directed{s,T} <: Number
    x::T
    Directed{s,T}(x::T) where {s,T} = new{s,T}(x)
    Directed{s,T}(x::Number) where {s,T} = new{s,T}(T(x))
end


Directed{s}(x) where {s} = Directed{s,eltype(x)}(x)

convert(::Type{Directed{s,T}}, x::Directed{s}) where {s,T} = Directed{s,T}(T(x.x))
convert(::Type{Directed{s,T}}, x::T) where {s,T} = Directed{s,T}(x)
convert(::Type{Directed{s,T}}, x::Real) where {s,T} = Directed{s,T}(T(x))
convert(::Type{Directed{s,T}}, x::Complex) where {s,T} = Directed{s,T}(T(x))
convert(::Type{T}, x::Directed{s,T}) where {T<:Number,s} = x.x

const ⁺ = Directed{true}(true)
const ⁻ = Directed{false}(true)

orientationsign(::Type{Directed{true}}) = 1
orientationsign(::Type{Directed{false}}) = -1
orientation(::Type{Directed{s}}) where {s} = s
orientation(::Directed{s}) where {s} = s

# removes direction from a number
undirected(x::Number) = x
undirected(x::Directed) = undirected(x.x)  # x might also have directeion
reverseorientation(x::Number) = x
reverseorientation(x::Directed{s}) where {s} = Directed{!s}(reverseorientation(x.x))


for OP in (:*, :+, :-, :/)
    @eval begin
        $OP(a::Directed{s}) where {s} = Directed{s}($OP(a.x))
        $OP(a::Directed{s}, b::Directed{s}) where {s} = Directed{s}($OP(a.x,b.x))
        $OP(a::Directed{s}, b::Number) where {s} = Directed{s}($OP(a.x,b))
        $OP(a::Directed{s}, b::PowerNumber) where {s} = Directed{s}($OP(a.x,b))
        $OP(a::Number, b::Directed{s}) where {s} = Directed{s}($OP(a,b.x))
        $OP(a::PowerNumber, b::Directed{s}) where {s} = Directed{s}($OP(a,b.x))
    end
end

muladd(a::Number, b::Directed, c::Number) = a*b + c

*(a::Directed, b::LogNumber) = undirected(a) * b # temp work around
*(a::LogNumber, b::Directed) = a * undirected(b) # temp work around


real(::Type{Directed{s,T}}) where {s,T} = real(T)


# abs, real and imag delete orientation.
for OP in (:(Base.isfinite), :(Base.isinf), :(Base.abs), :(Base.real), :(Base.imag), :(Base.angle))
    @eval $OP(a::Directed) = $OP(a.x)
end

# conj(a::Directed{s}) where s = Directed{!s}(conj(a.x))


# branchcuts of log, sqrt, etc. are oriented from (0,-∞)
# log(-x) = log|x| ± im*π
function Base.log(x::Directed{s}) where s
    r = log(abs(x.x))
    r - (2s-1) * convert(typeof(r), π) * im
end
Base.log1p(x::Directed) = log(1+x)
Base.sqrt(x::Directed{true}) = real(x.x) ≥ 0 ? sqrt(complex(x.x)) : -im*sqrt(-x.x)
Base.sqrt(x::Directed{false}) = real(x.x) ≥ 0 ? sqrt(complex(x.x)) : im*sqrt(-x.x)
^(x::Directed{true}, a::Integer) = x.x^a
^(x::Directed{false}, a::Integer) = x.x^a
^(x::Directed{true}, a::Number) = exp(-a*π*im)*(-x.x)^a
^(x::Directed{false}, a::Number) = exp(a*π*im)*(-x.x)^a


# Support for _2F1
import HypergeometricFunctions: log1pover, logandpoly, mxa_₂F₁, _₂F₁general, abeqcd, log1p, _₂F₁maclaurin, _₂F₁Inf, _₂F₁one, _₂F₁taylor, speciallog

speciallog(x::Directed) = (s = sqrt(-x); 3(s-atan(s))/s^3)
log1pover(s::Directed) = log1p(s)/undirected(s)
logandpoly(x::Directed) = undirected(x) == 0 ? one(x) : 6*(-2undirected(x)+(undirected(x)-2)*log1p(-x))/undirected(x)^3

function directed_mxa_₂F₁(a,b,c,z)
    if isequal(c,2)
        if abeqcd(a,b,1) # 6. 15.4.1
            return log1p(-z)
        end
    elseif isequal(c,4)
        if abeqcd(a,b,2)
            return 6*(-2 + (1-2/undirected(z))*log1p(-z))
        end
    end
    undirected(-z)^a*_₂F₁(a,b,c,z)
end

function directed_₂F₁general(a::Number,b::Number,c::Number,z)
    T = promote_type(typeof(a),typeof(b),typeof(c),typeof(undirected(z)))

    real(b) < real(a) && (return _₂F₁general(b,a,c,z))
    real(c) < real(a)+real(b) && (return exp((c-a-b)*log1p(-z))*_₂F₁general(c-a,c-b,c,z))

    if abs(z) ≤ ρ || -a ∈ ℕ₀ || -b ∈ ℕ₀
        _₂F₁maclaurin(a,b,c,undirected(z))
    elseif abs(z/(z-1)) ≤ ρ
        exp(-a*log1p(-z))_₂F₁maclaurin(a,c-b,c,undirected(z/(z-1)))
    elseif abs(inv(z)) ≤ ρ
        _₂F₁Inf(a,b,c,z)
    elseif abs(1-inv(z)) ≤ ρ
        exp(-a*log1p(-z))*_₂F₁Inf(a,c-b,c,reverseorientation(z/(z-1)))
    elseif abs(1-z) ≤ ρ
        _₂F₁one(a,b,c,z)
    elseif abs(inv(1-z)) ≤ ρ
        exp(-a*log1p(-z))*_₂F₁one(a,c-b,c,reverseorientation(z/(z-1)))
    else
        _₂F₁taylor(a,b,c,undirected(z))
    end
end

mxa_₂F₁(a, b, c, z::Directed) = directed_mxa_₂F₁(a,b,c,z)
_₂F₁general(a::Number, b::Number, c::Number, z::Directed) = directed_₂F₁general(a,b,c,z)



# work around for SIEs: the recurrence itself is an ordinary scalar recurrence, so the
# orientation is dropped from the seed data as well as from the point
RecurrenceRelationshipArrays.RecurrenceArray(z::Directed, (A,B,C), data::AbstractVector) =
    RecurrenceArray(undirected(z), (A,B,C), undirected.(data))


RecurrenceRelationshipArrays.RecurrenceArray(z::AbstractVector{<:Directed}, (A,B,C), data::AbstractMatrix) =
    RecurrenceArray(undirected.(z), (A,B,C), undirected.(data))