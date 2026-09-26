module RiemannHilbert
using SingularIntegrals, HypergeometricFunctions, PowerNumbers, Infinities, RecurrenceRelationshipArrays,
        IntervalSets, DomainSets, LinearAlgebra, Statistics,
        ContinuumArrays, QuasiArrays, ClassicalOrthogonalPolynomials, BlockArrays, LazyBandedMatrices

import Base: values, convert, getindex, setindex!, *, +, -, ==, <, <=, >, |, !, !=, eltype,
                >=, /, ^, \, ∪, size, reindex, tail, broadcast, broadcast!,
                isinf, in, real, imag, muladd, conj, isempty, issubset, isapprox, isless,
                intersect, setdiff, minimum, maximum, angle, sign, sqrt,
                first, last
import Base.Broadcast: broadcasted
import IntervalSets: leftendpoint, rightendpoint, endpoints, width, Interval
import DomainSets: Domain, ChebyshevInterval, prectype, choice
import LinearAlgebra: norm
import Statistics: mean
import ClassicalOrthogonalPolynomials: legendre, AbstractJacobiWeight
using PowerNumbers: realpart
export ⁺, ⁻, Directed, undirected, Segment,
        mobius, tocanonical, tocanonicalD, fromcanonical, fromcanonicalD,
        arclength, complexlength, reverseorientation, collocationpoints,
        rhmatrix, rhsolve
        
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

# export cauchymatrix, ℂ, istieltjes, KdV

intervalsign(d::AbstractInterval) = 1
intervalsign(d::AbstractSegment) = sign(d)
orientedleftendpoint(d::IntervalOrSegment) = leftendpoint(d) + intervalsign(d)ϵ
orientedrightendpoint(d::IntervalOrSegment) = rightendpoint(d) - intervalsign(d)ϵ
orientedleftendpoint(d::Inclusion) = orientedleftendpoint(d.domain)
orientedrightendpoint(d::Inclusion) = orientedrightendpoint(d.domain)

collocationpoints(d::Inclusion, m::Int) = collocationpoints(d.domain, m)
# use 2nd kind to include endpoints
collocationpoints(::ChebyshevInterval{T}, m::Int) where T = reverse(ChebyshevGrid{2,float(real(T))}(m))
function collocationpoints(d::IntervalOrSegment{T}, m::Int) where T
    i = ChebyshevInterval{real(T)}()
    affine(i, d)[collocationpoints(i, m)]
end

collocationpoints(d::UnionDomain, M::Int) = BlockVcat(collocationpoints.(components(d), M)...)

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

function fpstieltjesmatrix((m,n), d::IntervalOrSegment{T}) where T
    sp = legendre(d)
    x = collocationpoints(d, m)
    [permutedims(realpart.(stieltjes(sp, Directed{false}(orientedleftendpoint(d)))[1:n]));
     stieltjes(sp, Directed{false}.(x[2:end-1]))[:,1:n];
     permutedims(realpart.(stieltjes(sp, Directed{false}(orientedrightendpoint(d)))[1:n]))]
end

function fpstieltjesmatrix((m,n), d::IntervalOrSegment, r::IntervalOrSegment)
    sp = legendre(d)
    d == r && return fpstieltjesmatrix((m,n), d)
    x = collocationpoints(r, m)
    if leftendpoint(r) ∈ d && rightendpoint(r) ∈ d
        [permutedims(realpart.(stieltjes(sp, orientedleftendpoint(r))[1:n]));
         stieltjes(sp, x[2:end-1])[:,1:n];
         permutedims(realpart.(stieltjes(sp, orientedrightendpoint(r))[1:n]))]
    elseif leftendpoint(r) ∈ d
        [permutedims(realpart.(stieltjes(sp, orientedleftendpoint(r))[1:n]));
         stieltjes(sp, x[2:end])[:,1:n]]
    elseif rightendpoint(r) ∈ d
        [stieltjes(sp, x[1:end-1])[:,1:n];
         permutedims(realpart.(stieltjes(sp, orientedrightendpoint(r))[1:n]))]
    else
        stieltjes(sp, x)[:,1:n]
    end
end



function fpstieltjesmatrix((m,n), d::UnionDomain)
    mortar([fpstieltjesmatrix((m,n), b, a) for a in components(d), b in components(d)])
end

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

function fpcauchymatrix(x...)
    C = fpstieltjesmatrix(x...)
    C ./= (-2π*im)
    C
end

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

evaluationmatrix_domain(d, P, n) = P[collocationpoints(d, n),1:n]
evaluationmatrix_domain(::UnionDomain, P, n) = mortar(Diagonal([evaluationmatrix.(P.args, n)...]))
evaluationmatrix(P, n) = evaluationmatrix_domain(domain(P), P, n)

collocationvalues_domain(d, g, n) = g[collocationpoints(d, n)]
collocationvalues_domain(::UnionDomain, g, n) = vcat(collocationvalues.(components(g), n)...)
collocationvalues(g, n) = collocationvalues_domain(domain(g), g, n)

function rhmatrix(g, n)
    sp = basis(g)
    d = domain(sp)
    C₋ = fpcauchymatrix((n, n), d)
    𝐱 = collocationpoints(d, n)
    𝐠 = collocationvalues(g, n) .- 1
    E = evaluationmatrix(sp, n)
    C₋ .= 𝐠 .* C₋
    E .- C₋
end

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

function rhsolve(g, n)
    sp = basis(g)
    𝐱 = collocationpoints(domain(sp), n)
    g_v = g[𝐱] .- 1
    u = sp[:,1:n] * (rhmatrix(g,n) \ g_v)
    z -> 1 + cauchy(u,z)
end


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
