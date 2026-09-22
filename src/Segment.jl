# Ported from ApproxFunBase.jl (`src/Domains/Segment.jl`, plus the `indomain`
# method for segments from `src/Domain.jl`).  The ApproxFun-specific material —
# `AnyDomain`/`AnySegment`/`isambiguous`, the `Space` overloads and
# `strictconvert` — has been dropped, as has `SVector` support, since
# RiemannHilbert only ever needs segments in ℝ and ℂ.

export Segment


abstract type AbstractSegment{T} <: Domain{T} end

const IntervalOrSegment{T} = Union{AbstractInterval{T}, AbstractSegment{T}}


## Standard interval
# T Must be a Vector space
"""
	Segment(a,b)

represents a line segment from `a` to `b`.  In the case where `a` and `b`
are real and `a < b`, then this is is equivalent to an `Interval(a,b)`.
"""
struct Segment{T} <: AbstractSegment{T}
	a::T
	b::T
	Segment{T}(a,b) where {T} = new{T}(a,b)
end

Segment(a::Complex{IT1}, b::Complex{IT2}) where {IT1<:Integer,IT2<:Integer} =
	Segment(ComplexF64(a), ComplexF64(b)) #convenience method
Segment(a::Integer, b::Integer) = Segment(Float64(a),Float64(b)) #convenience method
Segment(a::Complex{IT}, b) where {IT<:Integer} = Segment(ComplexF64(a),b) #convenience method
Segment(a, b::Complex{IT}) where {IT<:Integer} = Segment(a,ComplexF64(b)) #convenience method
Segment(a, b) = Segment{promote_type(typeof(a),typeof(b))}(a,b)


convert(::Type{Domain{T}}, d::Segment) where {T<:Number} = Segment{T}(leftendpoint(d),rightendpoint(d))
convert(::Type{Segment{T}}, d::Segment) where {T<:Number} = Segment{T}(leftendpoint(d),rightendpoint(d))
convert(::Type{Segment}, d::AbstractInterval) = Segment(leftendpoint(d), rightendpoint(d))
convert(::Type{Segment{T}}, d::AbstractInterval) where T = convert(Segment{T}, convert(Segment, d))

Segment(d::AbstractInterval) = convert(Segment, d)

convert(::Type{Interval}, d::Segment{<:Real}) = d.a < d.b ? d.a .. d.b : d.b .. d.a
Interval(d::Segment) = convert(Interval, d)


## Information
@inline leftendpoint(d::Segment) = d.a
@inline rightendpoint(d::Segment) = d.b
@inline endpoints(d::Segment) = d.a, d.b

@inline minimum(d::Segment) = min(leftendpoint(d),rightendpoint(d))
@inline maximum(d::Segment) = max(leftendpoint(d),rightendpoint(d))

# `real` so that this also works for segments in the complex plane
isempty(d::Segment) = isapprox(leftendpoint(d), rightendpoint(d); atol=200eps(real(eltype(d))))

issubset(a::Segment,b::Segment) = leftendpoint(a) ∈ b && rightendpoint(a) ∈ b


arclength(d::AbstractInterval) = width(d)
arclength(d::Segment) = norm(complexlength(d))
complexlength(d::IntervalOrSegment) = rightendpoint(d)-leftendpoint(d)
mean(d::IntervalOrSegment) = (rightendpoint(d)+leftendpoint(d))/2
angle(d::IntervalOrSegment) = angle(complexlength(d))
sign(d::IntervalOrSegment) = sign(complexlength(d))

## Map interval
# The first definition  is the more general

mobius(d::ChebyshevInterval{T},x) where {T<:Real} = x
fromcanonical(d::ChebyshevInterval{T},x) where {T<:Real} = x
fromcanonicalD(d::ChebyshevInterval{T},x) where {T<:Real} = one(x)
tocanonical(d::ChebyshevInterval{T},x) where {T<:Real} = x
tocanonicalD(d::ChebyshevInterval{T},x) where {T<:Real} = one(x)

tocanonical(d::IntervalOrSegment{T},x) where {T} = 2norm(x-leftendpoint(d))/arclength(d)-1
tocanonical(d::IntervalOrSegment{T},x::Number) where {T<:Complex} = 2norm(x-leftendpoint(d))/arclength(d)-1
mobius(d::IntervalOrSegment,x) = (2x - leftendpoint(d) - rightendpoint(d))/complexlength(d)
tocanonical(d::IntervalOrSegment{T},x) where {T<:Real} = mobius(d,x)
tocanonicalD(d::IntervalOrSegment{T},x) where {T<:Real} = 2/complexlength(d)
fromcanonical(d::IntervalOrSegment{T},x) where {T<:Number} = mean(d) + complexlength(d)x/2
fromcanonicalD(d::IntervalOrSegment,x) = complexlength(d) / 2


# a point is in a segment if it maps into the canonical interval to within a
# tolerance that accounts for how much the map stretches
function DomainSets.indomain(x, d::AbstractSegment)
    T = float(real(prectype(d)))
    y = tocanonical(d,x)
    ry = real(y)
    iy = imag(y)
    # scale based on stretch of map on projection to interval
    sc = norm(fromcanonicalD(d, ry < -1 ? -one(ry) : (ry > 1 ? one(ry) : ry)))
    dy = fromcanonical(d,y)
    ((isinf(norm(dy)) && isinf(norm(x))) || norm(dy-x) ≤ 1000eps(T)*max(norm(x),1)) &&
        -one(T)-100eps(T)/sc ≤ ry ≤ one(T)+100eps(T)/sc &&
        -100eps(T)/sc ≤ iy ≤ 100eps(T)/sc
end


==(d::Segment, m::Segment) = leftendpoint(d) == leftendpoint(m) && rightendpoint(d) == rightendpoint(m)
function isapprox(d::Segment, m::Segment)
    tol=10E-12
    norm(leftendpoint(d)-leftendpoint(m))<tol && norm(rightendpoint(d)-rightendpoint(m))<tol
end

for op in (:(==), :isapprox)
    @eval begin
        $op(d::Segment, m::AbstractInterval) = $op(d, Segment(m))
        $op(m::AbstractInterval, d::Segment) = $op(Segment(m), d)
    end
end



## algebra

for op in (:*, :+, :-)
    @eval begin
        $op(c::Number,d::Segment) = broadcast($op,c,d)
        $op(d::Segment,c::Number) = broadcast($op,d,c)
        broadcasted(::typeof($op), c::Number, d::Segment) = Segment($op(c,leftendpoint(d)),$op(c,rightendpoint(d)))
        broadcasted(::typeof($op), d::Segment, c::Number) = Segment($op(leftendpoint(d),c),$op(rightendpoint(d),c))
    end
end

broadcasted(::typeof(^), c::Number, d::Segment) = Segment(c^leftendpoint(d),c^rightendpoint(d))
function broadcasted(::typeof(^), d::Segment, c::Number)
    a,b = endpoints(d)
    if a < 0 < b
        Segment(0, b^c)
    elseif b < 0 < a
        Segment(a^c, 0)
    else
        Segment(a^c,b^c)
    end
end

broadcasted(::typeof(Base.literal_pow), ::typeof(^), d::Segment, ::Val{K}) where K =
    broadcasted(^, d, K)

/(d::Segment,c::Number) = broadcast(/,d,c)
broadcasted(::typeof(/), d::Segment,c::Number) = Segment(leftendpoint(d)/c,rightendpoint(d)/c)

sqrt(d::Segment) = broadcast(sqrt, d)
broadcasted(::typeof(sqrt), d::Segment)=Segment(sqrt(leftendpoint(d)),sqrt(rightendpoint(d)))

+(d1::Segment,d2::Segment)=Segment(d1.a+d2.a,d1.b+d2.b)
broadcasted(::typeof(+),d1::Segment,d2::Segment) = Segment(d1.a+d2.a,d1.b+d2.b)


DomainSets.map_domain(map::DomainSets.AbstractAffineMap, domain::AbstractSegment) =
    Segment(map(leftendpoint(domain)),map(rightendpoint(domain)))

## intersect/union

reverseorientation(d::IntervalOrSegment) = Segment(rightendpoint(d),leftendpoint(d))

intersect(a::Segment{<:Real}, b::Segment{<:Real}) = intersect(Interval(a), Interval(b))
intersect(a::AbstractInterval, b::Segment{<:Real}) = intersect(a, Interval(b))
intersect(a::Segment{<:Real}, b::AbstractInterval) = intersect(Interval(a), b)
setdiff(a::Segment{<:Real}, b::Segment{<:Real})  = setdiff(Interval(a), Interval(b))
setdiff(a::AbstractInterval, b::Segment{<:Real})  = setdiff(a, Interval(b))
setdiff(a::Segment{<:Real}, b::AbstractInterval)  = setdiff(Interval(a), b)


## sort
isless(d1::Segment{T1},d2::Segment{T2}) where {T1<:Real,T2<:Real} =
    d1 ≤ leftendpoint(d2) && d1 ≤ rightendpoint(d2)
isless(d1::Segment{T},x::Real) where {T<:Real}=leftendpoint(d1) ≤ x && rightendpoint(d1) ≤ x
isless(x::Real,d1::Segment{T}) where {T<:Real}=x≤leftendpoint(d1) && x≤rightendpoint(d1)


choice(d::Segment{T}) where T = fromcanonical(d, choice(ChebyshevInterval{real(T)}()))

first(x::Inclusion{<:Any, <:Segment}) = leftendpoint(x.domain)
last(x::Inclusion{<:Any, <:Segment}) = rightendpoint(x.domain)

QuasiArrays.cardinality(::Segment) = ℵ₁
legendre(d::Segment{T}) where T = Legendre{float(real(T))}()[affine(d,ChebyshevInterval{real(T)}()), :]
ContinuumArrays.basis_axes(ax::Inclusion{<:Any,<:Segment}, v) = convert(AbstractQuasiMatrix{ContinuumArrays._any_eltype(v)}, legendre(ax))