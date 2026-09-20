module RiemannHilbert
using SingularIntegrals, HypergeometricFunctions, PowerNumbers, RecurrenceRelationshipArrays,
        IntervalSets, DomainSets, LinearAlgebra, Statistics

import Base: values, convert, getindex, setindex!, *, +, -, ==, <, <=, >, |, !, !=, eltype,
                >=, /, ^, \, ∪, size, reindex, tail, broadcast, broadcast!,
                isinf, in, real, imag, muladd, conj, isempty, issubset, isapprox, isless,
                intersect, setdiff, minimum, maximum, angle, sign, sqrt
import Base.Broadcast: broadcasted
import IntervalSets: leftendpoint, rightendpoint, endpoints, width, Interval
import DomainSets: Domain, ChebyshevInterval, prectype
import LinearAlgebra: norm
import Statistics: mean
export ⁺, ⁻, Directed, undirected, Segment,
        mobius, tocanonical, tocanonicalD, fromcanonical, fromcanonicalD,
        arclength, complexlength, reverseorientation

include("Segment.jl")
include("directed.jl")

# import ApproxFunBase: mobius, pieces, npieces, piece, BlockInterlacer, interlacer, pieces_npoints,
#                     ArraySpace, tocanonical, components_npoints, ScalarFun, VectorFun, MatrixFun,
#                     dimension, evaluate, prectype, cfstype, Space, SumSpace, spacescompatible,
#                     pieces
# import ApproxFunOrthogonalPolynomials: PolynomialSpace, recA, recB, recC, IntervalOrSegment


# # we need to import all special functions to use Calculus.symbolic_derivatives_1arg
# # we can't do importall Base as we replace some Base definitions
# import Base: sinpi, cospi, exp,
#                 asinh, acosh,atanh,
#                 sin, cos, sinh, cosh,
#                 exp2, exp10, log2, log10,
#                 tan, tanh, csc, asin, acsc, sec, acos, asec,
#                 cot, atan, acot, sinh, csch, asinh, acsch,
#                 sech, acosh, asech, tanh, coth, atanh, acoth,
#                 expm1, log1p, sinc, cosc,
#                 abs, sign, log, expm1, tan, abs2, sqrt, angle, max, min, cbrt, log,
#                 atan, acos, asin, inv, real, imag, abs, conj

# import LinearAlgebra: conj, transpose

# import SpecialFunctions: airy, besselh, erfcx, dawson, erf, erfi,
#                 airyai, airybi, airyaiprime, airybiprime,
#                 hankelh1, hankelh2, besselj, bessely, besseli, besselk,
#                 besselkx, hankelh1x, hankelh2x, lfact,
#                 erfinv, erfcinv, erfc, beta, lbeta,
#                 eta, zeta, gamma,  lgamma, polygamma, invdigamma, digamma, trigamma

# import DualNumbers: Dual, realpart, epsilon, dual
# import FillArrays: AbstractFill

# export cauchymatrix, rhmatrix, rhsolve, ℂ, istieltjes, KdV

intervalsign(d::AbstractInterval) = 1
intervalsign(d::AbstractSegment) = sign(d)
orientedleftendpoint(d::IntervalOrSegment) = leftendpoint(d) + intervalsign(d)ϵ
orientedrightendpoint(d::IntervalOrSegment) = rightendpoint(d) - intervalsign(d)ϵ


# # use 2nd kind to include endpoints
# collocationpoints(d::IntervalOrSegmentDomain, m::Int) = points(d, m; kind=2)
# collocationpoints(d::UnionDomain, ms::AbstractVector{Int}) = vcat(collocationpoints.(pieces(d), ms)...)
# collocationpoints(d::UnionDomain, m::Int) = collocationpoints(d, pieces_npoints(d,m))

# collocationpoints(sp::Space, m) = collocationpoints(domain(sp), m)


# collocationvalues(f::ScalarFun, n) = f.(collocationpoints(space(f), n))
# collocationvalues(f::Fun{<:Chebyshev}, n) = ichebyshevtransform!(coefficients(pad(f,n)); kind=2)
# function collocationvalues(f::VectorFun, n)
#     m = n÷size(f,1)
#     mapreduce(f̃ -> collocationvalues(f̃,m), vcat, f)
# end
# function collocationvalues(f::MatrixFun, n)
#     M = size(f,2)
#     ret = Array{cfstype(f)}(undef, n, M)
#     for J=1:M
#         ret[:,J] = collocationvalues(f[:,J], n)
#     end
#     ret
# end

# collocationvalues(f::Fun{<:PiecewiseSpace}, n) = vcat(collocationvalues.(components(f), pieces_npoints(domain(f),n))...)

# fprightstieltjesmoment!(V, sp) = stieltjesmoment!(V, sp, Directed{false}(orientedrightendpoint(domain(sp))), finitepart)
# fpleftstieltjesmoment!(V, sp) = stieltjesmoment!(V, sp, Directed{false}(orientedleftendpoint(domain(sp))), finitepart)
# fprightstieltjesmoment!(V, sp, d) = stieltjesmoment!(V, sp, orientedrightendpoint(d), finitepart)
# fpleftstieltjesmoment!(V, sp, d) = stieltjesmoment!(V, sp, orientedleftendpoint(d), finitepart)

# function fpstieltjesmatrix!(C, sp, d)
#     m, n = size(C)
#     pts = collocationpoints(d, m)
#     if d == domain(sp)
#         fprightstieltjesmoment!(view(C,1,:), sp)
#         for k=2:m-1
#             stieltjesmoment!(view(C,k,:), sp, Directed{false}(pts[k]))
#         end
#         fpleftstieltjesmoment!(view(C,m,:), sp)
#     elseif leftendpoint(d) ∈ domain(sp) && rightendpoint(d) ∈ domain(sp)
#         fprightstieltjesmoment!(view(C,1,:), sp, d)
#         for k=2:m-1
#             stieltjesmoment!(view(C,k,:), sp, pts[k])
#         end
#         fpleftstieltjesmoment!(view(C,m,:), sp, d)
#     elseif leftendpoint(d) ∈ domain(sp)
#         for k=1:m-1
#             stieltjesmoment!(view(C,k,:), sp, pts[k])
#         end
#         fpleftstieltjesmoment!(view(C,m,:), sp, d)
#     elseif rightendpoint(d) ∈ domain(sp)
#         fprightstieltjesmoment!(view(C,1,:), sp, d)
#         for k=2:m
#             stieltjesmoment!(view(C,k,:), sp, pts[k])
#         end
#     else
#         for k=1:m
#             stieltjesmoment!(view(C,k,:), sp, pts[k])
#         end
#     end
#     C
# end

# fpstieltjesmatrix!(C, sp) = fpstieltjesmatrix!(C, sp, domain(sp))

# fpstieltjesmatrix(sp::Space, d::Domain, n::Int, m::Int) =
#     fpstieltjesmatrix!(Array{ComplexF64}(undef, n, m), sp, d)

# fpstieltjesmatrix(sp::Space, n::Int, m::Int) =
#     fpstieltjesmatrix!(Array{ComplexF64}(undef, n, m), sp, domain(sp))


# # we group points together by piece
# function fpstieltjesmatrix!(C, sp::PiecewiseSpace, ns::AbstractVector{Int}, ms::AbstractVector{Int})
#     N, M = length(ns), length(ms)
#     @assert N == M == npieces(sp)
#     n, m = sum(ns), sum(ms)
#     @assert size(C) == (n,m)

#     for J = 1:M
#         jr = component_indices(sp, J, 1:ms[J])
#         k_start = 1
#         for K = 1:N
#             k_end = k_start + ns[K] - 1
#             kr = k_start:k_end
#             fpstieltjesmatrix!(view(C, kr, jr), component(sp, J),  domain(component(sp, K)))
#             k_start = k_end+1
#         end
#     end

#     C
# end


# fpstieltjesmatrix(sp::PiecewiseSpace, ns::AbstractVector{Int}, ms::AbstractVector{Int}) =
#     fpstieltjesmatrix!(Array{ComplexF64}(undef, sum(ns), sum(ms)), sp, ns, ms)

# fpstieltjesmatrix!(C, sp::PiecewiseSpace) = fpstieltjesmatrix!(C, sp, pieces_npoints(sp, size(C,1)), pieces_npoints(sp, size(C,2)))
# fpstieltjesmatrix(sp::PiecewiseSpace, n::Int, m::Int) = fpstieltjesmatrix(sp, pieces_npoints(sp, n), pieces_npoints(sp, m))


# # we group indices together by piece
# function fpstieltjesmatrix(sp::ArraySpace, ns::AbstractArray{Int}, ms::AbstractArray{Int})
#     @assert size(ns) == size(ms) == size(sp)
#     N = length(ns)

#     n, m = sum(ns), sum(ms)
#     C = zeros(ComplexF64, n, m)

#     for J = 1:N
#         jr = component_indices(sp, J, 1:ms[J]) ∩ (1:m)
#         k_start = sum(view(ns,1:J-1))+1
#         kr = k_start:k_start+ns[J]-1
#         fpstieltjesmatrix!(view(C, kr, jr), sp[J])
#     end

#     C
# end

# fpstieltjesmatrix(sp::ArraySpace, n::Int, m::Int) =
#     fpstieltjesmatrix(sp, reshape(pieces_npoints(sp, n), size(sp)), reshape(pieces_npoints(sp, m), size(sp)))


# cauchymatrix(x...) = stieltjesmatrix(x...)/(-2π*im)
# function fpcauchymatrix(x...)
#     C = fpstieltjesmatrix(x...)
#     C ./= (-2π*im)
#     C
# end

# ## riemannhilbert
# function multiplicationmatrix(G, n)
#     N, M = size(G)
#     @assert N == M
#     sp = space(G)
#     ret = spzeros(cfstype(G), n, n)
#     m = n ÷ N
#     pts = collocationpoints(sp, m)
#     for K=1:N,J=1:M
#         kr = (K-1)*m .+ (1:m)
#         jr = (J-1)*m .+ (1:m)
#         V = view(ret, kr, jr)
#         view(V, diagind(V)) .= collocationvalues(G[K,J],m)
#     end
#     ret
# end

# function rhmatrix(g::ScalarFun, n)
#     sp = rhspace(g)
#     C₋ = fpcauchymatrix(sp, n, n)
#     g_v = collocationvalues(g-1, n)
#     E = evaluationmatrix(sp, n)
#     C₋ .= g_v .* C₋
#     E .- C₋
# end

# function rhmatrix(g::MatrixFun, n)
#     sp = vector_rhspace(g)
#     C₋ = fpcauchymatrix(sp, n, n)
#     G = multiplicationmatrix(g-I, n)
#     E = evaluationmatrix(sp, n)
#     E .- G*C₋
# end

# function rh_sie_solve(G::MatrixFun, n)
#     sp = vector_rhspace(G)
#     cfs = rhmatrix(G, n) \ (collocationvalues(G-I, n))
#     U = hcat([Fun(sp, cfs[:,J]) for J=1:size(G,2)]...)
# end

# struct RHProblem{GTyp,CM,RHMTyp}
#     G::GTyp
#     C₋::CM
#     RP::RHMTyp
# end

# # function RHProblem(G)
# #     RHProblem(G,

# scalar_rhspace(d::AbstractInterval) = Legendre(d)
# scalar_rhspace(d::UnionDomain) = PiecewiseSpace(Legendre.(components(d)))
# array_rhspace(sz, d::Domain) = ArraySpace(scalar_rhspace(d), sz)
# vector_rhspace(sz1, d::Domain) = ArraySpace(scalar_rhspace(d), sz1)
# vector_rhspace(f::Fun) = vector_rhspace(size(f,1), domain(f))

# rhspace(g::Fun{<:ArraySpace}) = array_rhspace(size(g), domain(g))
# rhspace(g::Fun) = scalar_rhspace(domain(g))

# rhsolve(g::ScalarFun, n) = 1+cauchy(Fun(rhspace(g), rhmatrix(g, n) \ (collocationvalues(g-1, n))))
# function rhsolve(G::MatrixFun, n)
#     U = rh_sie_solve(G, n)
#     I+cauchy(U)
# end



# ## AffineSpace

# struct AffineSpace{DD,RR} <: Space{DD,RR}
#     domain::DD
# end

# AffineSpace(d::Domain) = AffineSpace{typeof(d),prectype(d)}(d)
# spacescompatible(::AffineSpace, ::AffineSpace) = true


# dimension(::AffineSpace) = 2

# function evaluate(v::AbstractVector{T}, s::AffineSpace, x::V) where {T,V}
#     @assert length(v) ≤ 2
#     (isempty(v) || x ∉ domain(s)) && return zero(promote_type(T,V))
#     length(v) == 1 && return v[1] + zero(x)
#     v[1] + v[2]*x
# end

# Fun(::typeof(identity), S::AffineSpace) = Fun(S, [0.0,1.0])
# Fun(::typeof(identity), S::ComplexPlane) = Fun(Space(S), [0.0,1.0])

# Space(d::ComplexPlane) = AffineSpace(d)

# *(φ::Fun, z::Fun{<:AffineSpace}) = z*φ

# function *(z::Fun{<:AffineSpace}, φ::Fun{<:JacobiQ})
#     a = coefficient(z,1)
#     b = coefficient(z,2)
#     u = istieltjes(φ)
#     x = Fun(domain(u))
#     b*sum(u)+stieltjes(a*u + b*x*u)
# end

# *(z::Fun{<:AffineSpace}, φ::Fun{<:ConstantSpace}) = Fun(space(z),Number(φ)*coefficients(z))
# *(z::Fun{<:AffineSpace}, Φ::Fun{<:SumSpace}) = mapreduce(f -> z*f, +, components(Φ))
# *(z::Fun{<:AffineSpace}, Φ::Fun{<:ArraySpace}) = Fun(z.*Array(Φ))

# include("KdV.jl")

# function unorientedangles(ds, z₀)
#     ret = Vector{Float64}()
#     for d in ds
#         if z₀ ≈ leftendpoint(d)
#             push!(ret, angle(rightendpoint(d)-z₀))
#         elseif z₀ ≈ rightendpoint(d)
#             push!(ret, angle(leftendpoint(d)-z₀))
#         else
#             throw(ArgumentError("Must contain"))
#         end
#     end
#     ret
# end

# function productcondition(G, z₀)
#     Gs = filter(g -> z₀ ∈ domain(g),  pieces(G))
#     p =  sortperm( unorientedangles(domain.(Gs), z₀))

#     g₀ = Matrix{ComplexF64}(I, size(G))
#     for g in Gs[p]
#         if leftendpoint(domain(g)) ≈ z₀ 
#             g₀ = g₀ * first(g)
#         else
#             g₀ = g₀ * inv(last(g))
#         end
#     end
#     g₀
# end

# function productcondition(G)
#     error("Implement")
# end

end #module
