module RiemannHilbert
using Base, ApproxFun, SingularIntegralEquations, DualNumbers


import SingularIntegralEquations: stieltjesforward, stieltjesbackward, undirected, Directed, stieltjesmoment!
import ApproxFun: mobius, pieces, npieces, piece, BlockInterlacer, Repeated, UnitCount, interlacer, IntervalDomain, pieces_npoints,
                    ArraySpace, tocanonical, components_npoints, ScalarFun, VectorFun, MatrixFun
import ApproxFun: PolynomialSpace, recA, recB, recC

import Base: values, convert, getindex, setindex!, *, +, -, ==, <, <=, >, |, !, !=, eltype, start, next, done,
                >=, /, ^, \, ∪, transpose, size, to_indexes, reindex, tail, broadcast, broadcast!

# we need to import all special functions to use Calculus.symbolic_derivatives_1arg
# we can't do importall Base as we replace some Base definitions
import Base: sinpi, cospi, airy, besselh, exp,
                asinh, acosh,atanh, erfcx, dawson, erf, erfi,
                sin, cos, sinh, cosh, airyai, airybi, airyaiprime, airybiprime,
                hankelh1, hankelh2, besselj, bessely, besseli, besselk,
                besselkx, hankelh1x, hankelh2x, exp2, exp10, log2, log10,
                tan, tanh, csc, asin, acsc, sec, acos, asec,
                cot, atan, acot, sinh, csch, asinh, acsch,
                sech, acosh, asech, tanh, coth, atanh, acoth,
                expm1, log1p, lfact, sinc, cosc, erfinv, erfcinv, beta, lbeta,
                eta, zeta, gamma,  lgamma, polygamma, invdigamma, digamma, trigamma,
                abs, sign, log, expm1, tan, abs2, sqrt, angle, max, min, cbrt, log,
                atan, acos, asin, erfc, inv, real, imag, abs, conj

import DualNumbers: Dual, value, epsilon, dual

export cauchymatrix, rhmatrix, rhsolve

include("LogNumber.jl")



function component_indices(it::BlockInterlacer, N::Int, kr::UnitRange)
    ret = Array{Int}(0)
    ind = 1
    k_end = last(kr)
    for (M,j) in it
        N == M && j > k_end && return ret
        N == M && j ∈ kr && push!(ret, ind)
        ind += 1
    end
    ret
end


function component_indices(it::BlockInterlacer{NTuple{N,Repeated{Bool}}}, k::Int, kr::UnitRange) where N
    b = length(it.blocks)
    k + (first(kr)-1)*b:b:k + (last(kr)-1)*b
end



function component_indices(it::BlockInterlacer{NTuple{N,Repeated{Bool}}}, k::Int, kr::UnitCount) where N
    b = length(it.blocks)
    k + (first(kr)-1)*b:b:∞
end

component_indices(sp::Space, k...) = component_indices(interlacer(sp), k...)

# # function fpstieltjes(f::Fun,z::Dual)
# #     x = mobius(domain(f),z)
# #     if !isinf(mobius(domain(f),Inf))
# #         error("Not implemented")
# #     end
# #     cfs = coefficients(f,Chebyshev)
# #     if realpart(x) ≈ 1
# #         c = -(log(dualpart(x))-log(2)) * sum(cfs)
# #         r = 0.0
# #         for k=2:2:length(cfs)-1
# #             r += 1/(k-1)
# #             c += -r*4*cfs[k+1]
# #         end
# #         r = 1.0
# #         for k=1:2:length(cfs)-1
# #             r += 1/(k-2)
# #             c += -(r+1/(2k))*4*cfs[k+1]
# #         end
# #         c
# #     elseif realpart(x) ≈ -1
# #         v = -(log(-dualpart(x))-log(2))
# #         if !isempty(cfs)
# #             c = -v*cfs[1]
# #         end
# #         r = 0.0
# #         for k=2:2:length(cfs)-1
# #             r += 1/(k-1)
# #             c += r*4*cfs[k+1]
# #             c += -v*cfs[k+1]
# #         end
# #         r = 1.0
# #         for k=1:2:length(cfs)-1
# #             r += 1/(k-2)
# #             c += -(r+1/(2k))*4*cfs[k+1]
# #             c += v*cfs[k+1]
# #         end
# #         c
# #     else
# #         error("Not implemented")
# #     end
# # end
# #
# # fpcauchy(x...) = fpstieltjes(x...)/(-2π*im)
#
#
#
# function stieltjesmatrix(space,pts::Vector,s::Bool)
#     n=length(pts)
#     C=Array(Complex128,n,n)
#     for k=1:n
#          C[k,:] = stieltjesforward(s,space,n,pts[k])
#     end
#     C
# end
#
# function stieltjesmatrix(space,pts::Vector)
#     n=length(pts)
#     C=zeros(Complex128,n,n)
#     for k=1:n
#         cfs = stieltjesbackward(space,pts[k])
#         C[k,1:min(length(cfs),n)] = cfs
#     end
#
#     C
# end


# stieltjesmatrix(space,n::Integer,s::Bool)=stieltjesmatrix(space,points(space,n),s)
# stieltjesmatrix(space,space2,n::Integer)=stieltjesmatrix(space,points(space2,n))



orientedfirst(d::Segment) = RiemannDual(first(d), sign(d))
orientedlast(d::Segment) = RiemannDual(last(d), -sign(d))


# use 2nd kind to include endpoints
collocationpoints(d::IntervalDomain, m::Int) = points(d, m; kind=2)
collocationpoints(d::UnionDomain, ms::AbstractVector{Int}) = vcat(collocationpoints.(pieces(d), ms)...)
collocationpoints(d::UnionDomain, m::Int) = collocationpoints(d, pieces_npoints(d,m))

collocationpoints(sp::Space, m) = collocationpoints(domain(sp), m)


collocationvalues(f::ScalarFun, n) = f.(collocationpoints(space(f), n))
function collocationvalues(f::VectorFun, n)
    pts = collocationpoints(space(f), n÷size(f,1))
    mapreduce(f̃ -> f̃.(pts), vcat, f)
end
function collocationvalues(f::MatrixFun, n)
    M = size(f,2)
    ret = Array{eltype(f)}(n, M)
    for J=1:M
        ret[:,J] = collocationvalues(f[:,J], n)
    end
    ret
end

collocationvalues(f::Fun{<:PiecewiseSpace}, n) = vcat(collocationvalues.(components(f), pieces_npoints(domain(f),n))...)

function evaluationmatrix!(E, sp::PolynomialSpace, x)
    x .= tocanonical.(sp, x)

    E[:,1] = 1
    E[:,2] .= (recA(Float64,sp,0) .* x .+ recB(Float64,sp,0)) .* view(E,:,1)
    for j = 3:size(E,2)
        E[:,j] .= (recA(Float64,sp,j-2) .* x .+ recB(Float64,sp,j-2)) .* view(E,:,j-1) .- recC(Float64,sp,j-2).*view(E,:,j-2)
    end
    E
end


evaluationmatrix!(E, sp::PolynomialSpace) =
    evaluationmatrix!(E, sp, collocationpoints(sp, size(E,1)))

evaluationmatrix(sp::PolynomialSpace, x, n) =
    evaluationmatrix!(Array{Float64}(length(x), n), sp,x)


function evaluationmatrix!(C, sp::PiecewiseSpace, ns::AbstractVector{Int}, ms::AbstractVector{Int})
    N, M = length(ns), length(ms)
    @assert N == M == npieces(sp)
    n, m = sum(ns), sum(ms)
    @assert size(C) == (n,m)

    C .= 0

    for J = 1:M
        jr = component_indices(sp, J, 1:ms[J])
        k_start = sum(view(ns,1:J-1))+1
        kr = k_start:k_start+ns[J]-1
        evaluationmatrix!(view(C, kr, jr), component(sp, J))
    end

    C
end


function evaluationmatrix!(C, sp::ArraySpace, ns::AbstractVector{Int}, ms::AbstractVector{Int})
    @assert size(ns) == size(ms) == size(sp)
    N = length(ns)

    n, m = sum(ns), sum(ms)
    @assert size(C) == (n,m)

    C .= 0

    for J = 1:N
        jr = component_indices(sp, J, 1:ms[J]) ∩ (1:m)
        k_start = sum(view(ns,1:J-1))+1
        kr = k_start:k_start+ns[J]-1
        evaluationmatrix!(view(C, kr, jr), sp[J])
    end

    C
end

evaluationmatrix!(C, sp::PiecewiseSpace) =
    evaluationmatrix!(C, sp, pieces_npoints(sp, size(C,1)), pieces_npoints(sp, size(C,2)))

evaluationmatrix!(C, sp::ArraySpace) =
    evaluationmatrix!(C, sp, components_npoints(sp, size(C,1)), components_npoints(sp, size(C,2)))


evaluationmatrix(sp::Space, n::Int) = evaluationmatrix!(Array{Float64}(n,n), sp)


function fpstieltjesmatrix!(C, sp, d)
    m, n = size(C)
    pts = collocationpoints(d, m)
    if d == domain(sp)
        stieltjesmoment!(view(C,1,:), sp, Directed{false}(orientedlast(d)), finitepart)
        for k=2:m-1
            stieltjesmoment!(view(C,k,:), sp, Directed{false}(pts[k]))
        end
        stieltjesmoment!(view(C,m,:), sp, Directed{false}(orientedfirst(d)), finitepart)
    elseif first(d) == domain(sp) && last(d) ∈ domain(sp)
        stieltjesmoment!(view(C,1,:), sp, orientedlast(d), finitepart)
        for k=2:m-1
            stieltjesmoment!(view(C,k,:), sp, pts[k])
        end
        stieltjesmoment!(view(C,m,:), sp, orientedfirst(d), finitepart)
    elseif first(d) ∈ domain(sp)
        for k=1:m-1
            stieltjesmoment!(view(C,k,:), sp, pts[k])
        end
        stieltjesmoment!(view(C,m,:), sp, orientedfirst(d), finitepart)
    elseif last(d) ∈ domain(sp)
        stieltjesmoment!(view(C,1,:), sp, orientedlast(d), finitepart)
        for k=2:m
            stieltjesmoment!(view(C,k,:), sp, pts[k])
        end
    else
        for k=1:m
            stieltjesmoment!(view(C,k,:), sp, pts[k])
        end
    end
    C
end

fpstieltjesmatrix!(C, sp) = fpstieltjesmatrix!(C, sp, domain(sp))

fpstieltjesmatrix(sp::Space, d::Domain, n::Int, m::Int) = fpstieltjesmatrix!(Array{Complex128}(n, m), sp, d)

fpstieltjesmatrix(sp::Space, n::Int, m::Int) = fpstieltjesmatrix!(Array{Complex128}(n, m), sp, domain(sp))


# we group points together by piece
function fpstieltjesmatrix!(C, sp::PiecewiseSpace, ns::AbstractVector{Int}, ms::AbstractVector{Int})
    N, M = length(ns), length(ms)
    @assert N == M == npieces(sp)
    n, m = sum(ns), sum(ms)
    @assert size(C) == (n,m)

    for J = 1:M
        jr = component_indices(sp, J, 1:ms[J])
        k_start = 1
        for K = 1:N
            k_end = k_start + ns[K] - 1
            kr = k_start:k_end
            fpstieltjesmatrix!(view(C, kr, jr), component(sp, J),  domain(component(sp, K)))
            k_start = k_end+1
        end
    end

    C
end


fpstieltjesmatrix(sp::PiecewiseSpace, ns::AbstractVector{Int}, ms::AbstractVector{Int}) =
    fpstieltjesmatrix!(Array{Complex128}(sum(ns), sum(ms)), sp, ns, ms)

fpstieltjesmatrix!(C, sp::PiecewiseSpace) = fpstieltjesmatrix!(C, sp, pieces_npoints(sp, size(C,1)), pieces_npoints(sp, size(C,2)))
fpstieltjesmatrix(sp::PiecewiseSpace, n::Int, m::Int) = fpstieltjesmatrix(sp, pieces_npoints(sp, n), pieces_npoints(sp, m))


# we group indices together by piece
function fpstieltjesmatrix(sp::ArraySpace, ns::AbstractArray{Int}, ms::AbstractArray{Int})
    @assert size(ns) == size(ms) == size(sp)
    N = length(ns)

    n, m = sum(ns), sum(ms)
    C = zeros(Complex128, n, m)

    for J = 1:N
        jr = component_indices(sp, J, 1:ms[J]) ∩ (1:m)
        k_start = sum(view(ns,1:J-1))+1
        kr = k_start:k_start+ns[J]-1
        fpstieltjesmatrix!(view(C, kr, jr), sp[J])
    end

    C
end



fpstieltjesmatrix(sp::ArraySpace, n::Int, m::Int) =
    fpstieltjesmatrix(sp, reshape(pieces_npoints(sp, n), size(sp)), reshape(pieces_npoints(sp, m), size(sp)))




cauchymatrix(x...) = stieltjesmatrix(x...)/(-2π*im)
function fpcauchymatrix(x...)
    C = fpstieltjesmatrix(x...)
    C ./= (-2π*im)
    C
end



## riemannhilbert


function multiplicationmatrix(G, n)
    N, M = size(G)
    @assert N == M
    sp = space(G)
    ret = spzeros(eltype(G), n, n)
    m = n ÷ N
    pts = collocationpoints(sp, m)
    for K=1:N,J=1:M
        kr = (K-1)*m + (1:m)
        jr = (J-1)*m + (1:m)
        V = view(ret, kr, jr)
        view(V, diagind(V)) .= G[K,J].(pts)
    end
    ret
end

function rhmatrix(g::ScalarFun, n)
    sp = space(g)
    C₋ = fpcauchymatrix(sp, n, n)
    g_v = collocationvalues(g-1, n)
    E = evaluationmatrix(sp, n)
    C₋ .= g_v .* C₋
    E .- C₋
end

function rhmatrix(g::MatrixFun, n)
    sp = space(g)[:,1]
    C₋ = fpcauchymatrix(sp, n, n)
    G = multiplicationmatrix(g-I, n)
    E = evaluationmatrix(sp, n)
    E .- G*C₋
end


rhsolve(g::ScalarFun, n) = 1+cauchy(Fun(space(g), rhmatrix(g, n) \ (collocationvalues(g-1, n))))
function rhsolve(G::MatrixFun, n)
    cfs = rhmatrix(G, n) \ (collocationvalues(G-I, n))
    U = hcat([Fun(space(G)[:,J], cfs[:,J]) for J=1:size(G,2)]...)
    I+cauchy(U)
end

end #module
