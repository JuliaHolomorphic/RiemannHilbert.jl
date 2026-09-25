using RiemannHilbert, PowerNumbers, SingularIntegrals, ClassicalOrthogonalPolynomials, ContinuumArrays, Statistics, BlockArrays, Test, DomainSets, LinearAlgebra
import PowerNumbers: logpart, realpart
import IntervalSets: leftendpoint, rightendpoint, endpoints, Interval
import RiemannHilbert: intervalsign, orientedleftendpoint, orientedrightendpoint, fpstieltjesmatrix

@testset "Segment" begin
    @testset "construction and conversion" begin
        @test Segment(1,2) === Segment{Float64}(1.0,2.0)
        @test Segment(1,2.0) === Segment(1.0,2.0)
        @test Segment(im,2im) === Segment{ComplexF64}(im,2im)
        @test Segment(1,2im) === Segment{ComplexF64}(1,2im)
        @test Segment(0..1) === Segment(0.0,1.0)
        @test Segment(ChebyshevInterval()) === Segment(-1.0,1.0)
        @test convert(Interval, Segment(2,1)) === 1.0..2.0
        @test Interval(Segment(1,2)) === 1.0..2.0
        @test convert(Segment{ComplexF64}, Segment(1,2)) === Segment(1.0+0im,2.0+0im)
        @test convert(Segment{Float64}, 0..1) === Segment(0.0,1.0)
    end

    @testset "information" begin
        d = Segment(1,2)
        @test leftendpoint(d) == 1 && rightendpoint(d) == 2
        @test endpoints(d) === (1.0,2.0)
        @test minimum(Segment(2,1)) == 1 && maximum(Segment(2,1)) == 2
        @test !isempty(d)
        @test isempty(Segment(1,1))
        @test isempty(Segment(0.0im,0.0im)) # eps of a complex segment is the eps of its precision
        @test Segment(1,2) ⊆ Segment(0,3)
        @test !(Segment(1,4) ⊆ Segment(0,3))
        @test arclength(d) == 1
        @test arclength(0..1) == 1
        @test arclength(Segment(0,1+im)) ≈ sqrt(2)
        @test complexlength(Segment(0,1+im)) == 1+im
        @test mean(d) == 1.5
        @test angle(Segment(0,1im)) ≈ π/2
        @test sign(Segment(0,1im)) == im
    end

    @testset "in" begin
        @test 0.5 ∈ Segment(0,1)
        @test 0 ∈ Segment(0,1)
        @test !(1.5 ∈ Segment(0,1))
        @test !(0.5im ∈ Segment(0,1))
        @test 0.5im ∈ Segment(0,1im)
        @test !(0.5 ∈ Segment(0,1im))
    end

    @testset "equality" begin
        @test Segment(1,2) == Segment(1,2)
        @test Segment(1,2) != Segment(2,1)
        @test Segment(1,2) == 1..2
        @test 1..2 == Segment(1,2)
        @test Segment(1,2) ≈ 1..2
        @test Segment(1,2+1E-14) ≈ Segment(1,2)
    end

    @testset "algebra" begin
        d = Segment(1,2)
        @test 2d === Segment(2.0,4.0)
        @test d*2 === Segment(2.0,4.0)
        @test d+1 === Segment(2.0,3.0)
        @test 1-d === Segment(0.0,-1.0)
        @test d/2 === Segment(0.5,1.0)
        @test sqrt(Segment(1,4)) === Segment(1.0,2.0)
        @test Segment(1,2).^2 === Segment(1.0,4.0)
        @test Segment(-1,2).^2 === Segment(0.0,4.0) # a segment straddling 0 squares to [0,max]
        @test Segment(2,-1).^2 === Segment(4.0,0.0)
        @test 2 .^ Segment(1,2) === Segment(2.0,4.0)
        @test Segment(1,2) + Segment(2,3) === Segment(3.0,5.0)
    end

    @testset "orientation" begin
        @test reverseorientation(Segment(1,2)) === Segment(2.0,1.0)
        @test reverseorientation(0..1) === Segment(1.0,0.0)
        @test intervalsign(Segment(1,2)) == 1
        @test intervalsign(Segment(0,1im)) == im
        @test orientedleftendpoint(Segment(1,2)) == 1+ϵ
        @test orientedrightendpoint(Segment(1,2)) == 2-ϵ
        @test orientedleftendpoint(Segment(0,1im)) == im*ϵ
    end

    @testset "canonical maps" begin
        d = Segment(1,2)
        @test mobius(d, 1.5) == 0
        @test tocanonical(d, 1) == -1
        @test tocanonical(d, 2) == 1
        @test tocanonicalD(d, 1.5) == 2
        @test fromcanonical(d, 0) == 1.5
        @test fromcanonicalD(d, 0) == 0.5
        @test fromcanonical(d, tocanonical(d, 1.25)) ≈ 1.25
        c = Segment(0,1im)
        @test tocanonical(c, 0.5im) ≈ 0
        @test fromcanonical(c, 1) ≈ 1im
        @test tocanonical(ChebyshevInterval(), 0.3) == 0.3
        @test fromcanonical(ChebyshevInterval(), 0.3) == 0.3
        @test tocanonicalD(ChebyshevInterval(), 0.3) == 1
        @test fromcanonicalD(ChebyshevInterval(), 0.3) == 1
    end

    @testset "set operations and sorting" begin
        @test Segment(0,2) ∩ Segment(1,3) == 1..2
        @test Segment(0,2) ∩ (1..3) == 1..2
        @test (0..2) ∩ Segment(1,3) == 1..2
        @test setdiff(Segment(0,3), Segment(1,2)) == setdiff(0..3, 1..2)
        @test sort([Segment(2,3), Segment(0,1)]) == [Segment(0,1), Segment(2,3)]
        @test Segment(0,1) < 2
        @test 0 < Segment(1,2)
    end
end


@testset "weights" begin
    o = LegendreWeight()
    @test stieltjes(o, 2+ϵ) isa LogNumber
    @test stieltjes(o, 2+ϵ) ≈ stieltjes(o, 2)
    @test stieltjes(o, 1+ϵ) == LogNumber(-1, log(2))

    h = 1E-10
    @test stieltjes(o, 1+ϵ)(h) == -log(h) + log(2)

    w  = ChebyshevTWeight()
    @test stieltjes(w, 2+ϵ) isa PowerNumber
    @test stieltjes(w, 2+ϵ) ≈ stieltjes(w, 2)
    @test stieltjes(w, 1+ϵ) == PowerNumber(π/sqrt(2), -1/2)
    @test stieltjes(w, 1+ϵ)+2 == PowerNumber(π/sqrt(2), 2, -1/2, 0)
end

@testset "Legendre Cauchy" begin
    P = Legendre()
    f = expand(P, exp)
    h = 1E-10
    for z in (1+ϵ, -1-ϵ, -1 + (1+im)*ϵ, 1+2ϵ, -1-2ϵ, -1 + 2*(1-im)*ϵ)
        @test stieltjes(f, z)(h) ≈ stieltjes(f, z(h))
        @test cauchy(f, z)(h) ≈ cauchy(f, z(h))
    end
    for z in (2+ϵ, 2+im+ϵ)
        @test stieltjes(f, 2+ϵ)(h) ≈ stieltjes(f, 2)
        @test cauchy(f, 2+ϵ)(h) ≈ cauchy(f, 2)
    end
end

@testset "Chebyshev" begin
    h = 1E-10
    W = Weighted(ChebyshevT())
    g = expand(W, x -> exp(x) / sqrt(1-x^2))
    @test stieltjes(g, 1+ϵ)(h) ≈ stieltjes(g, 1+h) rtol=1E-5
end

@testset "Directed and PowerNumber" begin
    @test undirected(Directed{false}(ϵ)) == 0

    @test real(LogNumber(2im,im+1)) == LogNumber(0,1)
    @test imag(LogNumber(2im,im+1)) == LogNumber(2,1)
    @test conj(LogNumber(2im,im+1)) == LogNumber(-2im,1-im)

    @test log((-ϵ) * ⁻) == LogNumber(1,π*im)
    @test log((-ϵ) * ⁺) == LogNumber(1,-π*im)

    @test log(Directed{false}((-1-eps()*im)ϵ)) ≈ LogNumber(1,π*im)
    @test log(Directed{false}((-1+eps()*im)ϵ)) ≈ LogNumber(1,π*im)

    @test log(Directed{true}((-1-eps()*im)ϵ)) ≈ LogNumber(1,-π*im)
    @test log(Directed{true}((-1+eps()*im)ϵ)) ≈ LogNumber(1,-π*im)


    z = Directed{false}(1-2ϵ)

    for k = 1:2, s = (false,true)
        z = Directed{s}(-1+2ϵ)
        l = stieltjes(Legendre()[:,k], z)
        h = 0.00000001
        @test l(h) ≈ stieltjes(Legendre()[:,k], -1 + (z.x.B)h + (s ? 1 : -1)*eps()*im) atol=1E-5
    end


    @test RiemannHilbert.orientedleftendpoint(ChebyshevInterval()) ≡ -1.0+ϵ
    @test RiemannHilbert.orientedrightendpoint(ChebyshevInterval()) ≡ 1.0-ϵ
end

@testset "finitepart stieltjes" begin
    h = 1E-10
    w = LegendreWeight()
    @test stieltjes(w, (-1+ϵ) * ⁻)(h) ≈ stieltjes(w, -1 + h - im*h^2) 
    @test stieltjes(w, (-1+ϵ) * ⁺)(h) ≈ stieltjes(w, -1 + h + im*h^2)
    @test stieltjes(w, (1-ϵ) * ⁻)(h) ≈ stieltjes(w, 1 - h - im*h^2) 
    @test stieltjes(w, (1-ϵ) * ⁺)(h) ≈ stieltjes(w, 1 - h + im*h^2)

    f  = expand(Legendre(), exp)
    @test stieltjes(f, (-1+ϵ) * ⁻)(h) ≈ stieltjes(f, -1 + h - im*h^2)
    
    f1 = expand(legendre(-1..0), exp)
    f2 = expand(legendre(0..1), exp)

    # Each half-interval transform has a log singularity at the shared endpoint 0. Their
    # singular parts cancel, and the finite parts add up to the transform over (-1,1)
    # approached from below.
    for z in (-im*ϵ, exp(-0.1im)*ϵ)
        s = stieltjes(f1, z) + stieltjes(f2, z)
        @test logpart(s) ≈ 0 atol=1E-12
        @test stieltjes(f, 0.0⁻) ≈ realpart(s)
    end

    # approaching 0 along the real axis instead, so f1 needs the branch cut orientation
    s = stieltjes(f1, Directed{false}(-ϵ)) + stieltjes(f2, -ϵ)
    @test logpart(s) ≈ 0 atol=1E-12
    @test stieltjes(f, 0.0⁻) ≈ realpart(s)

    # the same, on the piecewise basis rather than piece by piece
    fp = expand(PiecewiseInterlace(legendre(-1..0), legendre(0..1)), exp)
    # atol because the log parts cancel to zero: comparing two roundoff-level values of
    # opposite sign with a relative tolerance can never succeed
    @test stieltjes(fp, -im*ϵ) ≈ stieltjes(f1, -im*ϵ) + stieltjes(f2, -im*ϵ) atol=1E-12
    @test stieltjes(f, 0.0⁻) ≈ realpart(stieltjes(fp, -im*ϵ))
end

@testset "Segment stieltjes" begin
    Γ = Segment(0, im)
    f = expand(exp(im*z) for z in Γ)
    @test f[0.1im] ≈ exp(-0.1)

    @test stieltjes(f, 0.1) ≈ sum(exp(im*t)/(0.1-t) for t in Γ) ≈ sum(exp(-t)/(0.1-im*t)*im for t in 0..1)
    @test hilbert(f, 0.1im) ≈ -im*(cauchy(f, 0.1im-eps()) + cauchy(f,0.1im+eps()))
end

@testset "Interval FPStieltjes" begin
    Γ = ChebyshevInterval()
    f = expand(exp(-40(x-0.1)^2) for x in Γ)

    h = 1E-10
    
    @test map(l -> l(h), stieltjes(Legendre(), (-1+ϵ) * ⁻)[1:100]) ≈ stieltjes(Legendre(), -1+h-im*h^2)[1:100] atol=1E-4

    n = 100
    r = Segment(im,2im)
    @test fpstieltjesmatrix((n,n), Γ, r) * coefficients(f)[1:n] ≈ stieltjes(f, collocationpoints(r, n))

    r = Segment(-1,-1+im)
    C = fpstieltjesmatrix((n,n), Γ, r)
    @test (C * coefficients(f)[1:n]) ≈ stieltjes(f, [-1+eps()im; collocationpoints(r, n)[2:end]])
    @test norm(C) ≤ 100

    r = Segment(1,1+im)
    C = fpstieltjesmatrix((n,n), Γ, r)
    @test (C * coefficients(f)[1:n]) ≈ stieltjes(f, [1+eps()im; collocationpoints(r, n)[2:end]])
    @test norm(C) ≤ 100

    r = Γ
    C = fpstieltjesmatrix((n,n), Γ, r)
    @test (C * coefficients(f)[1:n]) ≈ stieltjes(f, collocationpoints(r, n) .- eps()*im)
    @test norm(C) ≤ 200
    

    d = 0..1
    f = expand(exp(-200(x-0.6)^2) for x in d)
    C = fpstieltjesmatrix((n,n), d)
    @test norm(C) ≤ 200
    c = C*coefficients(f)[1:n]
    @test c ≈ stieltjes(f, collocationpoints(d, n) .- eps()*im)

    r = -2..(-1)
    C = fpstieltjesmatrix((n,n), d, r)
    @test stieltjes(f, collocationpoints(r, n)) ≈ C * coefficients(f)[1:n]
    
    r = -1..0
    C = fpstieltjesmatrix((n,n), d, r)
    @test stieltjes(f, collocationpoints(r, n)[1:end-1]) ≈ C[1:end-1,:] * coefficients(f)[1:n]
    @test stieltjes(f, -eps()) ≈ only(C[end:end,:] * coefficients(f)[1:n])

    # reversed orientation
    d = Segment(1,0)
    f = expand(exp(-200(x-0.6)^2) for x in d)
    C = fpstieltjesmatrix((n,n), d)
    @test norm(C) ≤ 200
    c = C*coefficients(f)[1:n]
    @test c ≈ stieltjes(f, collocationpoints(d, n) .+ eps()*im)
end

@testset "Two interval" begin
    @testset "-1..0 and 0..1" begin
        f = expand(exp(-40(x-0.1)^2) for x in UnionDomain(-1..0, 0..1))
        d = domain(f)

        n = 100
        @time C = fpstieltjesmatrix((n,n), d)
        @test C[Block(1,1)] == fpstieltjesmatrix((n,n), -1..0)
        @test C[Block(1,2)] == fpstieltjesmatrix((n,n), 0..1, -1..0)
        @test C[Block(2,1)] == fpstieltjesmatrix((n,n), -1..0, 0..1)
        @test C[Block(2,2)] == fpstieltjesmatrix((n,n), 0..1)

        v = components(f)
        @test C * [coefficients(v[1])[1:n]; coefficients(v[2])[1:n]] ≈ [stieltjes(f, x) for x in collocationpoints(domain(f), Block(n)) .- eps()im]
        @test norm(C) ≤ 300

        g = expand(exp(-40(x-0.1)^2) for x in -1..1)
        @test stieltjes(f, -im) ≈ stieltjes(g, -im)
        @test stieltjes(f, -eps()im) ≈ stieltjes(g, -eps()im)
        h = 0.00000001
        # log part == 0 means ≈ fails
        @test stieltjes(f, (-ϵ) * ⁻)(h) ≈ (stieltjes(v[1], (-ϵ) * ⁻) + stieltjes(v[2], (0.0-ϵ)))(h)
        @test stieltjes(v[1], (-ϵ) * ⁻)(h) ≈ stieltjes(v[1], -h-h^2*im) rtol=1E-6
        @test stieltjes(v[2], -ϵ)(h) ≈ stieltjes(v[2], -h) rtol=1E-6
        @test stieltjes(v[1], (-ϵ) * ⁻) + stieltjes(v[2], (0.0-ϵ)) ≈ stieltjes(g, 0.0 * ⁻)  ≈ stieltjes(f, -eps()im) ≈ stieltjes(g, -eps()im)
    end

    @testset "Segment(0,-1) and 0..1" begin
        f = expand(sign(x)*exp(-40(x-0.1)^2) for x in Segment(0,-1) ∪ (0..1))
        d = domain(f)

        n = 100
        @time C = fpstieltjesmatrix((n,n), d)

        @test C[Block(1,1)] == fpstieltjesmatrix((n,n), Segment(0,-1))
        @test C[Block(1,2)] == fpstieltjesmatrix((n,n), 0..1, Segment(0,-1))
        @test C[Block(2,1)] == fpstieltjesmatrix((n,n), Segment(0,-1), 0..1)
        @test C[Block(2,2)] == fpstieltjesmatrix((n,n), 0..1)
    end
end

@testset "Vector-valued stieltjes" begin
    𝐟 = expand([exp(-40(x-0.1)^2); cos(x-0.1)*exp(-40(x-0.1)^2)] for x in ChebyshevInterval())
    @test cauchy(𝐟, 0.2 * ⁺) ≈  cauchy(𝐟, 0.2 + eps()im) ≈ [cauchy(first.(𝐟), 0.2+eps()im); cauchy(last.(𝐟), 0.2+eps()im)]
    @test cauchy(𝐟, 0.2 * ⁺) - cauchy(𝐟, 0.2 * ⁻) ≈ 𝐟[0.2]
    @test cauchy(𝐟, 0.2 * ⁺) + cauchy(𝐟, 0.2 * ⁻) ≈ im*hilbert(𝐟, 0.2)

    @test realpart.(stieltjes(𝐟, (-1+ϵ) * ⁻)) ≈ π*hilbert(𝐟,-1+eps())


    𝐠 = expand([exp(-40(x-0.1)^2); cos(x-0.1)*exp(-40(x-0.1)^2)] for x in UnionDomain(-1..0, 0..1))
    @test sum(𝐟) ≈ sum(𝐠) ≈ [sum(first.(𝐠)), sum(last.(𝐠))]
    @test stieltjes(𝐟, im) ≈ stieltjes(𝐠, im)
    @test hilbert(𝐠, 0.2) ≈ hilbert(𝐟, 0.2)
    @test realpart.(stieltjes(𝐠, (0.2-ϵ) * ⁻)) ≈ stieltjes(𝐟, 0.2 * ⁻)
    stieltjes(𝐠, 0.2-eps()im)
end

@testset "Vector-valued fpstieltjes" begin
    𝐟 = expand([exp(-40(x-0.1)^2); cos(x-0.1)*exp(-40(x-0.1)^2)] for x in ChebyshevInterval())
    sp = basis(𝐟)
    n = 200
    # fpstieltjesmatrix((n,n), sp)
#     C = fpstieltjesmatrix(sp, ns, ns)
#     C1 = fpstieltjesmatrix(sp[1], ns[1], ns[1])
#     C2 = fpstieltjesmatrix(sp[2], ns[2], ns[2])

#     @test C*coefficients(f) ≈ [C1*coefficients(f[1]) ; C2*coefficients(f[2])]

#     sp = ArraySpace(Legendre(), 2, 2)
#     f = Fun(x->[exp(-40(x-0.1)^2) sin(x-0.1)exp(-40(x-0.1)^2); cos(x-0.1)*exp(-40(x-0.1)^2) airyai(x-0.1)*exp(-40(x-0.1)^2)], sp)
#     ns = ncoefficients.(Array(f))
#     C = fpstieltjesmatrix(sp, ns, ns)
#     C11 = fpstieltjesmatrix(sp[1,1], ns[1,1], ns[1,1])
#     C21 = fpstieltjesmatrix(sp[2,1], ns[2,1], ns[2,1])
#     C12 = fpstieltjesmatrix(sp[1,2], ns[1,2], ns[1,2])
#     C22 = fpstieltjesmatrix(sp[2,2], ns[2,2], ns[2,2])

#     @test C[1:ns[1], 1:4:end] ≈ C11
#     @test norm(C[1:ns[1], 2:4:end]) ≤ 100eps()
#     @test norm(C[1:ns[1], 3:4:end]) ≤ 100eps()
#     @test norm(C[1:ns[1], 4:4:end]) ≤ 100eps()

#     @test C[ns[1]+1:ns[1]+ns[2], 2:4:end] ≈ C21

#     @test C*coefficients(f) ≈ [C11*coefficients(f[1,1]) ; C21*coefficients(f[2,1]) ; C12*coefficients(f[1,2]) ; C22*coefficients(f[2,2])]


#     sp = ArraySpace(Legendre(-1 .. 0) ∪ Legendre(0 .. 1), 2)
#     f = Fun(x->[exp(-40(x-0.1)^2); cos(x-0.1)*exp(-40(x-0.1)^2)], sp)
#     ns = ncoefficients.(Array(f))
#     C = fpstieltjesmatrix(sp, ns, ns)

#     C1 = fpstieltjesmatrix(sp[1], ns[1], ns[1])
#     C2 = fpstieltjesmatrix(sp[2], ns[2], ns[2])


#     @test C*coefficients(f) ≈ [C1*coefficients(f[1]) ; C2*coefficients(f[2])]

#     sp = ArraySpace(Legendre(-1 .. 0) ∪ Legendre(0 .. 1), 2, 2)
#     f = Fun(x->[exp(-40(x-0.1)^2) sin(x-0.1)exp(-40(x-0.1)^2); cos(x-0.1)*exp(-40(x-0.1)^2) airyai(x-0.1)*exp(-40(x-0.1)^2)], sp)
#     ns = ncoefficients.(Array(f))
#     C = fpstieltjesmatrix(sp, ns, ns)

#     C11 = fpstieltjesmatrix(sp[1,1], ns[1,1], ns[1,1])
#     C21 = fpstieltjesmatrix(sp[2,1], ns[2,1], ns[2,1])
#     C12 = fpstieltjesmatrix(sp[1,2], ns[1,2], ns[1,2])
#     C22 = fpstieltjesmatrix(sp[2,2], ns[2,2], ns[2,2])

#     @test C*coefficients(f) ≈ [C11*coefficients(f[1,1]) ; C21*coefficients(f[2,1]) ; C12*coefficients(f[1,2]) ; C22*coefficients(f[2,2])]
end

@testset "rhsolve" begin    
    @testset "-1 .. 1" begin
        sp = Legendre()
        g = expand(sp, x-> 1 -0.3exp(-40x^2))
        n = 200
        𝐱 = collocationpoints(domain(sp), n)
        g_v = g[𝐱] .- 1
        
        u = sp[:,1:n] * (rhmatrix(g,n) \ g_v)
        @test 1 + cauchy(u, 0.1+0.0im) ≈ (1 + cauchy(u, 0.1-0.0im))*g[0.1]
        φ = z -> 1 + cauchy(u,z)
        @test φ(0.1⁺)  ≈ g[0.1]φ(0.1⁻)

        φ = rhsolve(g, n)
        @test φ(0.1⁺)  ≈ g[0.1]φ(0.1⁻)
    end

#     @testset "-1 .. 0 and 0 .. 1" begin
#         sp = Legendre(-1 .. 0) ∪ Legendre(0 .. 1)
#         u_1 = u
#         u_ex = Fun(x->u_1(x), sp)
#         g_1 = 1-0.3Fun(x->exp(-40x^2), Legendre())

#         g = 1-0.3Fun(x->exp(-40x^2), sp)

#         n = 2ncoefficients(g)
#         g = pad(g,n)
#         u_ex = pad(u_ex,n)
#         @test (1+cauchy(u_ex,0.1⁺)) ≈ g(0.1)*(1+cauchy(u_ex,0.1⁻))

#         C₋ = fpcauchymatrix(sp, n, n)
#         pts = RiemannHilbert.collocationpoints(sp, n)

#         @test transpose(C₋[1,:])*coefficients(u_ex) ≈ cauchy(u_ex,pts[1]-eps()im)
#         @test transpose(C₋[5,:])*coefficients(u_ex) ≈ cauchy(u_ex,pts[5]-eps()im)
#         @test transpose(C₋[end,:])*coefficients(u_ex) ≈ cauchy(u_ex, 0.0-eps()im)
#         @test transpose(C₋[end-1,:])*coefficients(u_ex) ≈ cauchy(u_ex,pts[end-1]-eps()im)
#         @test transpose(C₋[n÷2,:])*coefficients(u_ex) ≈ cauchy(u_ex,-1.0-eps())
#         @test transpose(C₋[(n÷2)+1,:])*coefficients(u_ex) ≈ cauchy(u_ex,1.0+eps())

#         @test C₋*coefficients(u_ex) ≈ cauchy.(u_ex, pts.-eps()im)
#         g_v = RiemannHilbert.collocationvalues(g-1, n)
#         @test g_v ≈ g.(pts).-1
#         G = Diagonal(g_v)

#         @test G*g_v ≈ (g.(pts) .- 1).^2

#         g1 = component(g,2)
#         pts1 = RiemannHilbert.collocationpoints(component(sp,2), n÷2)
#         E1=RiemannHilbert.evaluationmatrix(component(sp,1), length(pts1))

#         @test E1*g1.coefficients ≈ g1.(pts1)

#         E = RiemannHilbert.evaluationmatrix(sp, n)
#         @test E*coefficients(g) ≈ g.(pts)

#         @test (g.(pts).-1).*(C₋*coefficients(g)) ≈ (g.(pts).-1).*cauchy.(g, pts.-0.000000001im)
#         @test G*(C₋*coefficients(g)) ≈ (g.(pts).-1).*cauchy.(g, pts.-0.000000001im)

#         L = E - G*C₋
#         @test L*coefficients(g) ≈ g.(pts) - (g.(pts).-1).*cauchy.(g, pts.-0.000000001im)

#         @test L == rhmatrix(g,n)

#         u = Fun(sp, rhmatrix(g,n) \ g_v)
#         φ = z -> 1 + cauchy(u,z)
#         @test φ(0.1⁺)  ≈ g(0.1)φ(0.1⁻)

#         @test 1+cauchy(u)(0.1+0.2im) ≈ φ(0.1+0.2im)
#         @test (1+cauchy(u) )(0.1+0.2im) ≈ φ(0.1+0.2im)
#         @test (1+cauchy(u) )(0.1⁻) ≈ φ(0.1⁻)

#         @time φ = rhsolve(g, 2ncoefficients(g))
#         @test φ(0.1⁺)  ≈ g(0.1)φ(0.1⁻)
#     end

#     @testset "0..-1 and 0..1" begin
#         sp = Legendre(Segment(0 , -1)) ∪ Legendre(0 .. 1)
#         g = Fun(x->x ≥ 0 ? 1-0.3exp(-40x^2) : inv(1-0.3exp(-40x^2)), sp)
#         u_1 = u
#         u_ex = Fun(x->sign(x)u_1(x), sp)

#         n = 2ncoefficients(g)
#         g = pad(g,n)
#         u_ex = pad(u_ex,n)
#         @test (1+cauchy(u_ex,0.1⁺)) ≈ g(0.1)*(1+cauchy(u_ex,0.1⁻))

#         C₋ = fpcauchymatrix(sp, n, n)
#         pts = RiemannHilbert.collocationpoints(sp, n)

#         @test transpose(C₋[1,:])*coefficients(u_ex) ≈ cauchy(u_ex,pts[1]+eps()im)
#         @test transpose(C₋[5,:])*coefficients(u_ex) ≈ cauchy(u_ex,pts[5]+eps()im)
#         @test transpose(C₋[n÷2,:])*coefficients(u_ex) ≈ cauchy(u_ex,0.0+eps()im)
#         @test transpose(C₋[(n÷2)+1,:])*coefficients(u_ex) ≈ cauchy(u_ex,1.0+eps())
#         @test transpose(C₋[end,:])*coefficients(u_ex) ≈ cauchy(u_ex, pts[end]-eps()im)
#         @test transpose(C₋[end,:])*coefficients(u_ex) ≈ cauchy(u_ex, 0.0-eps()im)
#         @test transpose(C₋[end-1,:])*coefficients(u_ex) ≈ cauchy(u_ex,pts[end-1]-eps()im)

#         @test C₋*coefficients(u_ex) ≈ [cauchy.(u_ex, pts[1:n÷2].+eps()im); cauchy.(u_ex, pts[(n÷2)+1:end].-eps()im)]
#         g_v = RiemannHilbert.collocationvalues(g-1, n)

#         g1 = component(g,1)
#         pts1 = RiemannHilbert.collocationpoints(component(sp,1), n÷2)
#         E1=RiemannHilbert.evaluationmatrix(component(sp,1), length(pts1))

#         @test E1*g1.coefficients ≈ g1.(pts1)

#         g2 = component(g,2)
#         pts2 = RiemannHilbert.collocationpoints(component(sp,2), n÷2)

#         @test g_v ≈ [g1.(pts1).-1; g2.(pts2).-1]
#         G = Diagonal(g_v)

#         @test g1.(pts1) ≈ g.(pts1)
#         @test_broken g2.(pts2) ≈ g.(pts2) # evaluation at 0
#         @test_broken G*g_v ≈ (g.(pts) .- 1).^2

#         E = RiemannHilbert.evaluationmatrix(sp, n)
#         @test E[1:end-1,:]*coefficients(g) ≈ g.(pts[1:end-1])
#         @test transpose(E[end,:])*coefficients(g) ≈ component(g,2)(0.0)

#         @test (g2.(pts2).-1).*(C₋[(n÷2)+1:end,:]*coefficients(u_ex)) ≈
#             (g2.(pts2).-1).*cauchy.(u_ex, pts2.-eps()im)
#         @test (g.(pts1).-1).*(C₋[1:n÷2,:]*coefficients(u_ex)) ≈
#             (g.(pts1).-1).*cauchy.(u_ex, pts1.+0.000000001im)
#         @test (g.(pts2).-1).*(C₋[(n÷2)+1:end,:]*coefficients(u_ex)) ≈
#             (g.(pts2).-1).*cauchy.(u_ex, pts2.-0.000000001im)

#         u_ex1,u_ex2 = components(u_ex)
#         @test (C₋*coefficients(u_ex))[ncoefficients(u_ex1)+1:end] ≈ cauchy.(u_ex, pts2.-eps()im)
#         @test G*C₋*coefficients(u_ex) ≈ [(g1.(pts1).-1).*cauchy.(u_ex, pts1.+eps()im);
#                                          (g2.(pts2).-1).*cauchy.(u_ex, pts2.-eps()im)]

#         L = E - G*C₋
#         @test L*coefficients(u_ex) ≈ [u_ex1.(pts1) - (g1.(pts1).-1).*cauchy.(u_ex, pts1.+eps()im);
#                                         u_ex2.(pts2) - (g2.(pts2).-1).*cauchy.(u_ex, pts2.-eps()im)]

#         @test L == rhmatrix(g,n)

#         u = Fun(sp, rhmatrix(g,n) \ g_v)
#         φ1 = z -> 1 + cauchy(u,z)
#         @test φ1(0.1⁺)  ≈ g(0.1)φ1(0.1⁻)
#         @test φ1(-0.1⁺)  ≈ g(-0.1)φ1(-0.1⁻)

#         @test 1+cauchy(u)(0.1+0.2im) ≈ φ1(0.1+0.2im)
#         @test (1+cauchy(u) )(0.1+0.2im) ≈ φ1(0.1+0.2im)
#         @test (1+cauchy(u) )(0.1⁻) ≈ φ1(0.1⁻)
#         @test (1+cauchy(u) )(0.1⁺) ≈ φ1(0.1⁺)
#         @test (1+cauchy(u) )(-0.1⁻) ≈ φ1(-0.1⁻)
#         @test (1+cauchy(u) )(-0.1⁺) ≈ φ1(-0.1⁺)

#         @time φ = rhsolve(g, 2ncoefficients(g))
#         @test φ(0.1⁺)  ≈ g(0.1)φ(0.1⁻)
#     end

#     @testset "Chebyshev g" begin
#         G = Fun(x -> 1 + exp(-30x^2), -1..1)
#         Φ = rhsolve(G, 200)
#         @test Φ(0.1⁺) ≈ G(0.1)Φ(0.1⁻)
#     end
end

# @testset "Matrix rhsolve" begin
#     sp = ArraySpace(Legendre(), 2)
#     f = Fun(Fun(x->[cos(x);sin(x)], Chebyshev()), ArraySpace(Legendre(), 2))
#     G = Fun(Fun(x->[1 exp(-40x^2); 0.1exp(-40x^2) 1], Chebyshev()), ArraySpace(Legendre(), 2, 2))

#     n = 2ncoefficients(G)
#     E = RiemannHilbert.evaluationmatrix(sp, n)
#     pts = RiemannHilbert.collocationpoints(sp, n÷2)

#     @test E*coefficients(pad(f,n)) ≈ [f[1].(pts); f[2].(pts)]


#     M = RiemannHilbert.multiplicationmatrix(G-I, n)
#     @test M*(E*coefficients(pad(f,n))) ≈ mapreduce(f -> f.(pts), vcat, (G-I)*f)

#     L = E - M*fpcauchymatrix(sp, n,n)
#     u1 = L \ mapreduce(f -> f.(pts), vcat, (G-I)[:,1])
#     u2 = L \ mapreduce(f -> f.(pts), vcat, (G-I)[:,2])
#     z=2+I
#         Φ = z -> I + [cauchy(Fun(sp, u1),z) cauchy(Fun(sp, u2),z)]

#     @test Φ(0.1+eps()im) ≈ G(0.1)*Φ(0.1-eps()im)

#     @test RiemannHilbert.rhmatrix(G, n) == L

#     @test RiemannHilbert.collocationvalues(G-I, n) ≈
#         [RiemannHilbert.collocationvalues((G-I)[:,1], n) RiemannHilbert.collocationvalues((G-I)[:,2], n)]

#     Φ = rhsolve(G,n)
#     @test G(0.1)*Φ(0.1⁻) ≈ Φ(0.1⁺)
# end

# @testset "Matrix rhsolve two interval" begin
#     sp = ArraySpace(Legendre(-1 .. 0) ∪ Legendre(0 .. 1), 2)
#     G = Fun(x->[1 exp(-40x^2); 0.1exp(-40x^2) 1], ArraySpace(sp[1], 2, 2))

#     n=2ncoefficients(G)
#     Φ = rhsolve(G,n)

#     @test G(0.1)*Φ(0.1⁻) ≈ Φ(0.1⁺)
#     U = RiemannHilbert.rh_sie_solve(G,n)

#     U1 = U[:,1]
#     g = G
#     sp = space(g)[:,1]
#     C₋ = fpcauchymatrix(sp, n, n)
#     G = RiemannHilbert.multiplicationmatrix(g-I, n)
#     E = RiemannHilbert.evaluationmatrix(sp, n)

#     pts = RiemannHilbert.collocationpoints(sp, n÷2)
#     @test last(E*coefficients(U1)) ≈ U1(0.00000000001)[2]
#     @test last(C₋*coefficients(U1)) ≈ cauchy(U1[2], 0.00000000001-eps()*im)
#     @test cauchy(U1[1], 0.00000000001-eps()*im) ≈ (C₋*coefficients(U1))[n÷2]
#     @test last(G*C₋*coefficients(U1)) ≈ ((g(0.00000000001)-I)*cauchy(U1, 0.00000000001-eps()*im))[2]


#     @test RiemannHilbert.collocationvalues((g-I)[:,1], n)[end] ≈ 0.1

#     sp = ArraySpace(Legendre(Segment(0 , -1)) ∪ Legendre(0 .. 1), 2)
#     G = g = Fun(x-> x ≥ 0 ? [1 exp(-40x^2); 0.1exp(-40x^2) 1] : inv([1 exp(-40x^2); 0.1exp(-40x^2) 1]), ArraySpace(sp[1], 2, 2))
#     n=2ncoefficients(G)
#     Ũ1 = Fun(x-> x ≥ 0 ? U1(x) : -U1(x), sp, n÷2)
#     @test cauchy(Ũ1, 0.1⁻) ≈ cauchy(Ũ1, 0.1-eps()im)
#     @test cauchy(Ũ1, 0.1+eps()im) ≈ cauchy(U1, 0.1+eps()im)
#     @test cauchy(Ũ1, -0.1+eps()im) ≈ cauchy(U1, -0.1+eps()im)

#     L = rhmatrix(G, n)
#     rhs = RiemannHilbert.collocationvalues((g-I)[:,1], n)
#     @test rhs[end] ≈ 0.1

#     Φ = rhsolve(G,n)

#     g = G
#     sp = space(g)[:,1]
#     C₋ = fpcauchymatrix(sp, n, n)
#     G = RiemannHilbert.multiplicationmatrix(g-I, n)
#     E = RiemannHilbert.evaluationmatrix(sp, n)

#     pts = RiemannHilbert.collocationpoints(sp, n÷2)

#     @test last(E*coefficients(Ũ1)) ≈ Ũ1(0.00000000001)[2]
#     @test last(C₋*coefficients(Ũ1)) ≈ cauchy(Ũ1[2], 0.00000000001-eps()*im)
#     @test cauchy(Ũ1[1], 0.00000000001-eps()*im) ≈ (C₋*coefficients(Ũ1))[n÷2]
#     @test G[[n÷2,n],[n÷2,n]] ≈ (g(0.00000000001)-I)

#     @test last(G*C₋*coefficients(Ũ1)) ≈ transpose((g(0.00000000001)-I)[2,:])*cauchy(Ũ1, 0.00000000001-eps()*im)

#     @test transpose(L[end,:])*coefficients(Ũ1) ≈ Ũ1(0.00000000001)[2] - transpose((g(0.00000000001)-I)[2,:])*cauchy(Ũ1, 0.00000000001-eps()*im)

#     @test U1(0.00000000001)[2] - transpose((g(0.00000000001)-I)[2,:])*cauchy(U1, 0.00000000001-eps()*im) ≈ 0.1
#     @test g(0.1)*Φ(0.1⁻) ≈ Φ(0.1⁺)
# end

# @testset "4 rays" begin
#     Γ = Segment(0, 2.5exp(im*π/6)) ∪ Segment(0, 2.5exp(5im*π/6)) ∪
#             Segment(0, 2.5exp(-5im*π/6)) ∪ Segment(0, 2.5exp(-im*π/6))
#     sp = ArraySpace(PiecewiseSpace(Legendre.(components(Γ))), 2,2)

#     s₁ = im
#     s₃ = -im

#     G = Fun( z -> if angle(z) ≈ π/6
#                         [1 0; s₁*exp(8im/3*z^3) 1]
#                     elseif angle(z) ≈ 5π/6
#                         [1 0; s₃*exp(8im/3*z^3) 1]
#                     elseif angle(z) ≈ -π/6
#                         [1 -s₃*exp(-8im/3*z^3); 0 1]
#                     elseif angle(z) ≈ -5π/6
#                         [1 -s₁*exp(-8im/3*z^3); 0 1]
#                     end
#                         , sp)

#     Φ = transpose(rhsolve(transpose(G), 2*4*100))

#     s=exp(im*π/6)
#     @test Φ((s)⁻)*G(s) ≈ Φ((s)⁺)
#     @test map(g->first(components(g)[1]), G)*map(g->first(components(g)[2]), G)*
#         map(g->first(components(g)[3]), G)*map(g->first(components(g)[4]), G) ≈ Matrix(I,2,2)

#     for _=1:5
#         @test RiemannHilbert.evaluationmatrix(sp[:,1], 32) ≈ RiemannHilbert.evaluationmatrix(sp[:,1], 32)
#         @test RiemannHilbert.rhmatrix(G, 32) ≈ RiemannHilbert.rhmatrix(G, 32)
#     end

#     @test RiemannHilbert.rhmatrix(transpose(G), 900) ≈ RiemannHilbert.rhmatrix(transpose(G), 900)
#     U = RiemannHilbert.rh_sie_solve(transpose(G), 2*4*100)
#     @test -0.36706155154807807 ≈ sum(U[1,2])/(-π*im)

#     G̃ = Fun( z -> if angle(z) ≈ π/6
#                         [1 0; s₁*exp(8im/3*z^3) 1]
#                     elseif angle(z) ≈ 5π/6
#                         [1 0; s₃*exp(8im/3*z^3) 1]
#                     elseif angle(z) ≈ -π/6
#                         [1 -s₃*exp(-8im/3*z^3); 0 1]
#                     elseif angle(z) ≈ -5π/6
#                         [1 -s₁*exp(-8im/3*z^3); 0 1]
#                     end
#                         , Γ)

#     Ũ = RiemannHilbert.rh_sie_solve(transpose(G̃), 2*4*100)

#     @test rhmatrix(transpose(G), 2*4*100) ≈ rhmatrix(transpose(G̃), 2*4*100)
#     @test -0.36706155154807807 ≈ sum(Ũ[1,2])/(-π*im)

#     V = istieltjes(Φ)
#     x = Fun(domain(V))

#     @test 10.0I + sum.(Array(V)) + stieltjes(x*V, 10.0) ≈ 10.0Φ(10.0)


#     # check rhmatrix relationship for below
#     U = V*(-2π*im)
#     U1 = U[1,:]
#     n= ncoefficients(U1)
#     L = rhmatrix(transpose(G),n)
#     vals = collocationvalues(transpose(G)-I, n)
#     @test L*coefficients(U1) ≈ vals[:,1]
# end


# @testset "6 rays" begin
#     @testset "HM on 6 rays" begin
#         s₁,s₂,s₃ = -im,0,im
#         @assert s₁ - s₂ + s₃ + s₁*s₂*s₃ ≈ 0

#         # construct true solution using 4 rays
#         Γ = Segment(0, 2.5exp(im*π/6)) ∪ Segment(0, 2.5exp(5im*π/6)) ∪
#         Segment(0, 2.5exp(-5im*π/6)) ∪ Segment(0, 2.5exp(-im*π/6))
#         sp = ArraySpace(PiecewiseSpace(Legendre.(components(Γ))), 2,2)

#         G = Fun( z -> if angle(z) ≈ π/6
#                             [1 0; s₁*exp(8im/3*z^3) 1]
#                         elseif angle(z) ≈ 5π/6
#                             [1 0; s₃*exp(8im/3*z^3) 1]
#                         elseif angle(z) ≈ -π/6
#                             [1 -s₃*exp(-8im/3*z^3); 0 1]
#                         elseif angle(z) ≈ -5π/6
#                             [1 -s₁*exp(-8im/3*z^3); 0 1]
#                         end
#                             , sp)

#         Φ = transpose(rhsolve(transpose(G), 2*6*100))


#         # debug
#         V4 = istieltjes(Φ)
#         x = 0.0

#         @test stieltjes(V4,1+im)+I == Φ(1+im)


#         Γ = Segment(0, 2.5exp(im*π/6))   ∪
#         Segment(0, 2.5exp(im*π/2))       ∪
#         Segment(0, 2.5exp(5im*π/6))      ∪
#         Segment(0, 2.5exp(-5im*π/6))     ∪
#         Segment(0, 2.5exp(-im*π/2))      ∪
#         Segment(0, 2.5exp(-im*π/6));

#         G = Fun( z -> if angle(z) ≈ π/6
#                         [1                             0;
#                         s₁*exp(8im/3*z^3+2im*x*z)     1]
#                     elseif angle(z) ≈ π/2
#                         [1                 s₂*exp(-8im/3*z^3-2im*x*z);
#                         0                 1]
#                     elseif angle(z) ≈ 5π/6
#                         [1                             0;
#                         s₃*exp(8im/3*z^3+2im*x*z)     1]
#                     elseif angle(z) ≈ -π/6
#                         [1                -s₃*exp(-8im/3*z^3-2im*x*z);
#                         0                 1]
#                     elseif angle(z) ≈ -π/2
#                         [1                             0;
#                         -s₂*exp(8im/3*z^3+2im*x*z)    1]
#                     elseif angle(z) ≈ -5π/6
#                         [1                -s₁*exp(-8im/3*z^3-2im*x*z);
#                         0                 1]
#                     end
#                         , Γ);
#         sp = ArraySpace(PiecewiseSpace(Legendre.(components(Γ))), 2,2)
#         V = Fun(V4, sp)

#         @test stieltjes(V,1+im)+I ≈ Φ(1+im)
        
#         U = V*(-2π*im)

#         U1 = U[1,:]
#         @test cauchy(U1,1+im)+[1,0] ≈ Φ(1+im)[1,:]
#         @test abs(sum(first.(components(U1[1])))) ≤ 100eps()
#         @test abs(sum(first.(components(U1[2])))) ≤ 100eps()


#         U11 = U1[1]
#         n = ncoefficients(U11)
#         C₋ = fpcauchymatrix(space(U11), n, n)
#         pts = collocationpoints(space(U11), n)
#         c = C₋*coefficients(U11)
        
#         @test c[1] ≈ finitepart(cauchy(U11,orientedrightendpoint(component(Γ,1))))
#         @test c[2] ≈ cauchy(U11,pts[2])
#         @test c[150] ≈ finitepart(cauchy(U11,orientedleftendpoint(component(Γ,1))⁻))
#         @test c[151] ≈ finitepart(cauchy(U11,orientedrightendpoint(component(Γ,2))))

#         n = ncoefficients(U1)
#         L = rhmatrix(transpose(G),n)
#         vals = collocationvalues(transpose(G)-I, n)
#         pts = collocationpoints(space(U1),n)

#         c = L*coefficients(U1)
#         @test c ≈ vals[:,1]

#         @test coefficients(U1) ≈ L \ vals[:,1]
#     end
# end

# include("test_nls.jl")