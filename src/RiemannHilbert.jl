module RiemannHilbert
using Base, ApproxFun, SingularIntegralEquations, DualNumbers


import SingularIntegralEquations: stieltjesforward, stieltjesbackward, undirected, Directed, stieltjesmoment!
import ApproxFun: mobius

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

export cauchymatrix

include("LogNumber.jl")


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



function fpstietjesmatrix!(C, sp, d)
    m, n = size(C)
    pts = points(d, m; kind=2)
    if d == domain(sp)
        stieltjesmoment!(view(C,1,:), sp, Directed{false}(orientedlast(d)), finitepart)
        for k=2:m-1
            stieltjesmoment!(view(C,k,:), sp, Directed{false}(pts[k]))
        end
        stieltjesmoment!(view(C,m,:), sp, Directed{false}(orientedfirst(d)), finitepart)
    elseif first(d) ∈ domain(sp) && last(d) ∈ domain(sp)
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



function fpstietjesmatrix(sp::Space, ns::AbstractVector{Int}, ms::AbstractVector{Int})
    N, M = length(ns), length(ms)
    @assert N == M == ncomponents(sp)
    n, m = sum(ns), sum(ms)
    C = Array{Complex128}(n, m)

    k_start = j_start = 1
    for k = 1:N
        k_end = k_start + ns[k] - 1
        for j=1:N
            j_end = j_start + ms[j] - 1
            fpstietjesmatrix!(view(C, k_start:k_end, j_start:j_end), component(sp, j), domain(component(sp, k)))
            j_start = j_end+1
        end
        k_start = k_end+1
        j_start = 1
    end
    C
end

function fpstietjesmatrix(sp::Space, n::Int, m::Int)
    N = ncomponents(sp)
    nn = n ÷ N
    ns = Array{Int}(N)
    ns[1] = nn + rem(n,N)
    ns[2:end] = nn

    mm = n ÷ N
    ms = Array{Int}(N)
    ms[1] = mm + rem(m,N)
    ms[2:end] = mm

    fpstietjesmatrix(sp, ns, ms)
end





cauchymatrix(x...) = stieltjesmatrix(x...)/(-2π*im)

end #module
