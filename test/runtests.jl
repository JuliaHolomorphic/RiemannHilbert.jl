using ApproxFun, SingularIntegralEquations, DualNumbers, RiemannHilbert, LinearAlgebra, FastTransforms, SpecialFunctions, Test
    import ApproxFun: ArraySpace, pieces
    import RiemannHilbert: RiemannDual, LogNumber, fpstieltjesmatrix!, fpstieltjesmatrix, orientedrightendpoint, finitepart, fpcauchymatrix
    import SingularIntegralEquations: stieltjesmoment, stieltjesmoment!, undirected, Directed, ⁺, ⁻, istieltjes
    import SingularIntegralEquations.HypergeometricFunctions: speciallog


@testset "RiemannDual" begin
    for h in (0.1,0.01), a in (2exp(0.1im),1.1)
        @test log(RiemannDual(0,a))(h) ≈ log(h*a)
        @test log(RiemannDual(Inf,a))(h) ≈ log(a/h)
    end

    for h in (0.1,0.01), a in (2exp(0.1im),1.1)
        @test log1p(RiemannDual(-1,a))(h) ≈ log(h*a)
        @test log1p(RiemannDual(Inf,a))(h) ≈ log(a/h)
    end


    h = 0.0000001
    for z in (RiemannDual(-1,-1), RiemannDual(1,1), RiemannDual(-1,2exp(0.1im)), RiemannDual(1,2exp(0.1im))),
            k = 0:1
        l = stieltjesjacobimoment(0,0,k,z)
        @test l(h) ≈ stieltjesjacobimoment(0,0,k,realpart(z)+epsilon(z)h) atol=1E-5
    end

    h=0.0001
    for z in (RiemannDual(1,3exp(0.2im)), RiemannDual(1,0.5exp(-1.3im)),
                RiemannDual(-1,3exp(0.2im)), RiemannDual(-1,0.5exp(-1.3im)),
                RiemannDual(-1,1), RiemannDual(1,-1))
        @test atanh(z)(h) ≈  atanh(realpart(z)+epsilon(z)h) atol = 1E-4
    end

    z = RiemannDual(1,-0.25)
    h = 0.0000001
    @test speciallog(z)(h) ≈ speciallog(realpart(z)+epsilon(z)h) atol=1E-4

    h = 0.00001
    for z in (RiemannDual(-1,-1), RiemannDual(-1,exp(0.1im)), RiemannDual(-1,exp(-0.1im)))
        @test stieltjesjacobimoment(0.5,0,0,z)(h) ≈ stieltjesjacobimoment(0.5,0,0,realpart(z)+epsilon(z)h) atol=1E-4
    end
end



@testset "Legendre Cauchy" begin
    f = Fun(exp,Legendre())

    h = 0.00001
    for z in  (RiemannDual(-1,-1), RiemannDual(-1,1+im), RiemannDual(-1,1-im))
        @test cauchy(f, z)(h) ≈ cauchy(f, realpart(z) + epsilon(z)h) atol=1E-4
    end
end


@testset "Directed and RiemannDual" begin
    @test undirected(Directed{false}(RiemannDual(0,-1))) == 0

    @test log(Directed{false}(RiemannDual(0,-1))) == LogNumber(1,π*im)
    @test log(Directed{true}(RiemannDual(0,-1))) == LogNumber(1,-π*im)

    @test log(Directed{false}(RiemannDual(0,-1-eps()*im))) == LogNumber(1,π*im)
    @test log(Directed{false}(RiemannDual(0,-1+eps()*im))) == LogNumber(1,π*im)

    @test log(Directed{true}(RiemannDual(0,-1-eps()*im))) == LogNumber(1,-π*im)
    @test log(Directed{true}(RiemannDual(0,-1+eps()*im))) == LogNumber(1,-π*im)


    z = Directed{false}(RiemannDual(1,-2))

    for k=0:1, s=(false,true)
        z = Directed{s}(RiemannDual(-1,2))
        l = stieltjesmoment(Legendre(),k,z)
        h = 0.00000001
        @test l(h) ≈ stieltjesmoment(Legendre(),k,-1 + epsilon(z.x)h + (s ? 1 : -1)*eps()*im) atol=1E-5
    end


    @test RiemannHilbert.orientedleftendpoint(ChebyshevInterval()) == RiemannDual(-1.0,1)
    @test RiemannHilbert.orientedrightendpoint(ChebyshevInterval()) == RiemannDual(1.0,-1)
end

@testset "Interval FPStieltjes" begin
    Γ = ChebyshevInterval()
    f = Fun(x->exp(-40(x-0.1)^2), Legendre())
    C = Array{ComplexF64}(undef, ncoefficients(f), ncoefficients(f))
    d = Segment(im,2im)

    fpstieltjesmatrix!(C, space(f), d)
    c = Fun(d, ApproxFun.chebyshevtransform(C*coefficients(f); kind=2))
    @test c(1.5im) ≈ stieltjes(f,1.5im)

    d = Segment(-1,-1+im)
    fpstieltjesmatrix!(C, space(f), d)
    @test norm(C) ≤ 100
    c = Fun(d, ApproxFun.chebyshevtransform(C*coefficients(f); kind=2))
    @test c(-1+0.5im) ≈ stieltjes(f,-1+0.5im)

    d = Segment(1,1+im)
    fpstieltjesmatrix!(C, space(f), d)
    @test norm(C) ≤ 200
    c = Fun(d, ApproxFun.chebyshevtransform(C*coefficients(f); kind=2))
    @test c(1+0.5im) ≈ stieltjes(f,1+0.5im)

    d = ChebyshevInterval()
    fpstieltjesmatrix!(C, space(f), d)
    @test norm(C) ≤ 200
    c = Fun(d, ApproxFun.chebyshevtransform(C*coefficients(f); kind=2))
    @test c(0.5) ≈ stieltjes(f,0.5⁻)


    d = Segment(0,1)
    f = Fun(x->exp(-200(x-0.6)^2), Legendre(d))
    C = fpstieltjesmatrix(space(f), ncoefficients(f), ncoefficients(f))
    @test norm(C) ≤ 200
    c = Fun(d, chebyshevtransform(C*coefficients(f); kind=2))
    @test c(0.5) ≈ stieltjes(f,0.5⁻)
end

@testset "finitepart stieltjes" begin
    f = Fun(exp,Legendre())
    f1 = Fun(exp,Legendre(-1..0))
    f2 = Fun(exp,Legendre(0..1))
    fp = f1+f2

    @test stieltjes(f,0.0⁻) ≈ finitepart(stieltjes(f1,RiemannDual(0.0,-im)) + stieltjes(f2,RiemannDual(0.0,-im)))
    @test stieltjes(f,0.0⁻) ≈ finitepart(stieltjes(f1,RiemannDual(0.0,exp(-0.1im))) + stieltjes(f2,RiemannDual(0.0,exp(-0.1im))))
    @test stieltjes(f,0.0⁻) ≈ finitepart(stieltjes(f1,Directed{false}(RiemannDual(0.0,-1.0))) + stieltjes(f2,RiemannDual(0.0,-1.0)))

    @test stieltjes(fp,RiemannDual(0.0,-im)) ≈ stieltjes(f1,RiemannDual(0.0,-im)) + stieltjes(f2,RiemannDual(0.0,-im))
    @test stieltjes(f,0.0⁻) ≈ finitepart(stieltjes(fp,RiemannDual(0.0,-im)))
end

@testset "Two interval" begin
    sp = Legendre(-1 .. 0) ⊕ Legendre(0 .. 1)
    f = Fun(x->exp(-40(x-0.1)^2), sp)
    v = components(f)
    ns = ncoefficients.(v)
    C = fpstieltjesmatrix(space(f), ns, ns)
    @test norm(C) ≤ 200

    c_vals = C*coefficients(f)
    pts = RiemannHilbert.collocationpoints(space(f), ns)

    @test c_vals[1] ≈ finitepart(stieltjes(f,Directed{false}(RiemannDual(0.0,-im))))
    @test c_vals[1] ≈ finitepart(stieltjes(f,Directed{false}(RiemannDual(0.0,exp(-0.1im)))))
    @test c_vals[1] ≈ finitepart(stieltjes(f,RiemannDual(0.0,-im)))
    @test c_vals[2:ns[1]-1] ≈  stieltjes.(f,pts[2:ns[1]-1]⁻)
    @test c_vals[ns[1]] ≈ finitepart(stieltjes(f,RiemannDual(-1.0,-1.0)))

    @test c_vals[ns[1]+1] ≈ finitepart(stieltjes(f,RiemannDual(1.0,1.0)))
    @test c_vals[ns[1]+2:end-1] ≈  stieltjes.(f,pts[ns[1]+2:end-1]⁻)
    @test c_vals[end] ≈ finitepart(stieltjes(f,RiemannDual(0.0,-im)))

    h =0.00001
    @test stieltjes(v[1], Directed{false}(RiemannDual(0.0,-1.0))) ≈ stieltjes(v[1], RiemannDual(0.0,-1.0-eps()*im))

    @test finitepart(stieltjes(v[1], Directed{false}(RiemannDual(0.0,-1.0)))+ stieltjes(v[2], RiemannDual(0.0,-1.0))) ≈
        stieltjes(v[1], -0.00000000001im)+stieltjes(v[2], -0.00000000001im)


    @test finitepart(stieltjes(v[1], Directed{false}(RiemannDual(0.0,-1.0)))+ stieltjes(v[2], RiemannDual(0.0,-1.0))) ≈
        stieltjes(f, -0.00000000001im)

    @test c_vals[1] ≈ stieltjes(f, -0.0000000001im)
    @test c_vals[end] ≈ stieltjes(f, -0.0000000001im)

    @test finitepart(stieltjes(f, RiemannDual(0.0,-im))) ≈ finitepart(stieltjes(v[1], RiemannDual(0.0,-im))+stieltjes(v[2], RiemannDual(0.0,-im)))


    C11 = fpstieltjesmatrix(space(v[1]), ncoefficients(v[1]), ncoefficients(v[1]))
    C12 = fpstieltjesmatrix(space(v[2]), domain(v[1]), ncoefficients(v[1]), ncoefficients(v[2]))

    @test finitepart(stieltjes(v[1], Directed{false}(RiemannDual(0.0,-1.0)))) ≈ (C11*coefficients(v[1]))[1]
    @test finitepart(stieltjes(v[2], RiemannDual(0.0,-1.0))) ≈ dotu(stieltjesmoment!(Array{ComplexF64}(undef,ncoefficients(v[2])), space(v[2]), orientedrightendpoint(domain(v[1])), finitepart),
                coefficients(v[2]))

    @test C12[1,:] ≈ stieltjesmoment!(Array{ComplexF64}(undef,ncoefficients(v[2])), space(v[2]), orientedrightendpoint(domain(v[1])), finitepart)

    @test finitepart(stieltjes(v[2], RiemannDual(0.0,-1.0))) ≈ (C12*coefficients(v[2]))[1]
    @test C[1,:] ≈ ApproxFun.interlace(C11[1,:], C12[1,:])
end

@testset "Piecewise" begin
    sp = Legendre(0 .. -1) ⊕ Legendre(0 .. 1)
    f = Fun(x->sign(x)*exp(-40(x-0.1)^2), sp)
    v = components(f)
    ns = ncoefficients.(v)
    C = fpstieltjesmatrix(space(f), ns, ns)
    @test norm(C) ≤ 100

    @test c_vals[1] ≈ finitepart(stieltjes(f,RiemannDual(-1.0,-1.0)))
    @test c_vals[2:ns[1]-1] ≈  stieltjes.(f,pts[2:ns[1]-1]⁻)
    @test c_vals[ns[1]] ≈ finitepart(stieltjes(f,RiemannDual(0.0,+im)))
    @test c_vals[ns[1]+1] ≈ finitepart(stieltjes(f,RiemannDual(1.0,1.0)))
    @test c_vals[ns[1]+2:end-1] ≈  stieltjes.(f,pts[ns[1]+2:end-1]⁻)
    @test c_vals[end] ≈ finitepart(stieltjes(f,RiemannDual(0.0,-im)))


    h =0.00001
    @test stieltjes(v[1], Directed{false}(RiemannDual(0.0,-1.0))) ≈ stieltjes(v[1], RiemannDual(0.0,-1.0+eps()*im))
    @test stieltjes(v[1], Directed{true}(RiemannDual(0.0,-1.0))) ≈ stieltjes(v[1], RiemannDual(0.0,-1.0-eps()*im))

    @test stieltjes(v[2], RiemannDual(0.0,-1.0))(h) ≈ stieltjes(v[2], -h) atol=1E-3

    @test stieltjes(v[1], Directed{true}(RiemannDual(0.0,-1.0)))(h) ≈ stieltjes(v[1], -h-eps()*im) atol=1E-3


    @test finitepart(stieltjes(v[1], Directed{true}(RiemannDual(0.0,-1.0)))+ stieltjes(v[2], RiemannDual(0.0,-1.0))) ≈
        stieltjes(v[1], -0.00000000001im)+stieltjes(v[2], -0.00000000001im)


    @test finitepart(stieltjes(v[1], Directed{true}(RiemannDual(0.0,-1.0)))+ stieltjes(v[2], RiemannDual(0.0,-1.0))) ≈
        stieltjes(f, -0.00000000001im)

    @test c_vals[ns[1]] ≈ stieltjes(f, +0.0000000001im)
    @test c_vals[end] ≈ stieltjes(f, -0.0000000001im)


    sp = Legendre(-1 .. 0) ⊕ Legendre(0 .. 1)
    f = Fun(x->exp(-40(x-0.1)^2), sp)
    v = components(f)
    ns = ncoefficients.(v)
    C = fpstieltjesmatrix(space(f), ns, ns)
    @test norm(C) ≤ 100

    c_vals = C*coefficients(f)
    pts = RiemannHilbert.collocationpoints(space(f), ns)

    @test c_vals[1] ≈ finitepart(stieltjes(f,RiemannDual(0.0,-im)))
    @test c_vals[2:ns[1]-1] ≈  stieltjes.(f,pts[2:ns[1]-1]⁻)
    @test c_vals[ns[1]] ≈ finitepart(stieltjes(f,RiemannDual(-1.0,-1.0)))
    @test c_vals[ns[1]+1] ≈ finitepart(stieltjes(f,RiemannDual(1.0,1.0)))
    @test c_vals[ns[1]+2:end-1] ≈  stieltjes.(f,pts[ns[1]+2:end-1]⁻)
    @test c_vals[end] ≈ finitepart(stieltjes(f,RiemannDual(0.0,-im)))
end

@testset "ArraySpace" begin
    sp = ArraySpace(Legendre() ,2)
    f = Fun(x->[exp(-40(x-0.1)^2); cos(x-0.1)*exp(-40(x-0.1)^2)], sp)
    ns = ncoefficients.(Array(f))
    C = fpstieltjesmatrix(sp, ns, ns)
    C1 = fpstieltjesmatrix(sp[1], ns[1], ns[1])
    C2 = fpstieltjesmatrix(sp[2], ns[2], ns[2])

    @test C*coefficients(f) ≈ [C1*coefficients(f[1]) ; C2*coefficients(f[2])]

    sp = ArraySpace(Legendre(), 2, 2)
    f = Fun(x->[exp(-40(x-0.1)^2) sin(x-0.1)exp(-40(x-0.1)^2); cos(x-0.1)*exp(-40(x-0.1)^2) airyai(x-0.1)*exp(-40(x-0.1)^2)], sp)
    ns = ncoefficients.(Array(f))
    C = fpstieltjesmatrix(sp, ns, ns)
    C11 = fpstieltjesmatrix(sp[1,1], ns[1,1], ns[1,1])
    C21 = fpstieltjesmatrix(sp[2,1], ns[2,1], ns[2,1])
    C12 = fpstieltjesmatrix(sp[1,2], ns[1,2], ns[1,2])
    C22 = fpstieltjesmatrix(sp[2,2], ns[2,2], ns[2,2])

    @test C[1:ns[1], 1:4:end] ≈ C11
    @test norm(C[1:ns[1], 2:4:end]) ≤ 100eps()
    @test norm(C[1:ns[1], 3:4:end]) ≤ 100eps()
    @test norm(C[1:ns[1], 4:4:end]) ≤ 100eps()

    @test C[ns[1]+1:ns[1]+ns[2], 2:4:end] ≈ C21

    @test C*coefficients(f) ≈ [C11*coefficients(f[1,1]) ; C21*coefficients(f[2,1]) ; C12*coefficients(f[1,2]) ; C22*coefficients(f[2,2])]


    sp = ArraySpace(Legendre(-1 .. 0) ∪ Legendre(0 .. 1), 2)
    f = Fun(x->[exp(-40(x-0.1)^2); cos(x-0.1)*exp(-40(x-0.1)^2)], sp)
    ns = ncoefficients.(Array(f))
    C = fpstieltjesmatrix(sp, ns, ns)

    C1 = fpstieltjesmatrix(sp[1], ns[1], ns[1])
    C2 = fpstieltjesmatrix(sp[2], ns[2], ns[2])


    @test C*coefficients(f) ≈ [C1*coefficients(f[1]) ; C2*coefficients(f[2])]

    sp = ArraySpace(Legendre(-1 .. 0) ∪ Legendre(0 .. 1), 2, 2)
    f = Fun(x->[exp(-40(x-0.1)^2) sin(x-0.1)exp(-40(x-0.1)^2); cos(x-0.1)*exp(-40(x-0.1)^2) airyai(x-0.1)*exp(-40(x-0.1)^2)], sp)
    ns = ncoefficients.(Array(f))
    C = fpstieltjesmatrix(sp, ns, ns)

    C11 = fpstieltjesmatrix(sp[1,1], ns[1,1], ns[1,1])
    C21 = fpstieltjesmatrix(sp[2,1], ns[2,1], ns[2,1])
    C12 = fpstieltjesmatrix(sp[1,2], ns[1,2], ns[1,2])
    C22 = fpstieltjesmatrix(sp[2,2], ns[2,2], ns[2,2])

    @test C*coefficients(f) ≈ [C11*coefficients(f[1,1]) ; C21*coefficients(f[2,1]) ; C12*coefficients(f[1,2]) ; C22*coefficients(f[2,2])]
end

@testset "rhsolve" begin
    sp = Legendre()
    g = 1-0.3Fun(x->exp(-40x^2), sp)

    n = 2ncoefficients(g)
    g = pad(g,n)
    C₋ = fpcauchymatrix(sp, n, n)
    pts = RiemannHilbert.collocationpoints(sp, n)
    @test C₋[2:end-1,:]*coefficients(g) ≈ cauchy.(g, pts[2:end-1]⁻)
    g_v = RiemannHilbert.collocationvalues(g-1, n)
    @test g_v ≈ g.(pts)-1
    G = diagm(g_v)

    @test G*g_v ≈ (g.(pts) .- 1).^2

    E = RiemannHilbert.evaluationmatrix(sp, pts, length(pts))
    @test E*coefficients(g) ≈ g.(pts)

    @test (g.(pts)-1).*(C₋*coefficients(g)) ≈ (g.(pts)-1).*cauchy.(g, pts-0.000000001im)
    @test G*(C₋*coefficients(g)) ≈ (g.(pts)-1).*cauchy.(g, pts-0.000000001im)



    L = E - G*C₋
    @test L*coefficients(g) ≈ g.(pts) - (g.(pts)-1).*cauchy.(g, pts-0.000000001im)

    @test L == rhmatrix(g,n)

    u = Fun(sp, rhmatrix(g,n) \ g_v)
    φ = z -> 1 + cauchy(u,z)
    @test φ(0.1⁺)  ≈ g(0.1)φ(0.1⁻)


    @test 1+cauchy(u)(0.1+0.2im) ≈ φ(0.1+0.2im)
    @test (1+cauchy(u) )(0.1+0.2im) ≈ φ(0.1+0.2im)
    @test (1+cauchy(u) )(0.1⁻) ≈ φ(0.1⁻)
end

sp = Legendre(-1 .. 0) ∪ Legendre(0 .. 1)
u_1 = u
u_ex = Fun(x->u_1(x), sp)
g_1 = 1-0.3Fun(x->exp(-40x^2), Legendre())

g = 1-0.3Fun(x->exp(-40x^2), sp)

n = 2ncoefficients(g)
g = pad(g,n)
u_ex = pad(u_ex,n)
@test (1+cauchy(u_ex,0.1⁺)) ≈ g(0.1)*(1+cauchy(u_ex,0.1⁻))

C₋ = fpcauchymatrix(sp, n, n)
pts = RiemannHilbert.collocationpoints(sp, n)

@test C₋[1,:].'*coefficients(u_ex) ≈ cauchy(u_ex,pts[1]-eps()im)
@test C₋[5,:].'*coefficients(u_ex) ≈ cauchy(u_ex,pts[5]-eps()im)
@test C₋[end,:].'*coefficients(u_ex) ≈ cauchy(u_ex, 0.0-eps()im)
@test C₋[end-1,:].'*coefficients(u_ex) ≈ cauchy(u_ex,pts[end-1]-eps()im)
@test C₋[n÷2,:].'*coefficients(u_ex) ≈ cauchy(u_ex,-1.0-eps())
@test C₋[(n÷2)+1,:].'*coefficients(u_ex) ≈ cauchy(u_ex,1.0+eps())


@test C₋*coefficients(u_ex) ≈ cauchy.(u_ex, pts-eps()im)
g_v = RiemannHilbert.collocationvalues(g-1, n)
@test g_v ≈ g.(pts)-1
G = diagm(g_v)

@test G*g_v ≈ (g.(pts) .- 1).^2


g1 = component(g,2)
pts1 = RiemannHilbert.collocationpoints(component(sp,2), n÷2)
E1=RiemannHilbert.evaluationmatrix(component(sp,1), length(pts1))

@test E1*g1.coefficients ≈ g1.(pts1)


E = RiemannHilbert.evaluationmatrix(sp, n)
@test E*coefficients(g) ≈ g.(pts)

@test (g.(pts)-1).*(C₋*coefficients(g)) ≈ (g.(pts)-1).*cauchy.(g, pts-0.000000001im)
@test G*(C₋*coefficients(g)) ≈ (g.(pts)-1).*cauchy.(g, pts-0.000000001im)



L = E - G*C₋
@test L*coefficients(g) ≈ g.(pts) - (g.(pts)-1).*cauchy.(g, pts-0.000000001im)

@test L == rhmatrix(g,n)

u = Fun(sp, rhmatrix(g,n) \ g_v)
φ = z -> 1 + cauchy(u,z)
@test φ(0.1⁺)  ≈ g(0.1)φ(0.1⁻)


@test 1+cauchy(u)(0.1+0.2im) ≈ φ(0.1+0.2im)
@test (1+cauchy(u) )(0.1+0.2im) ≈ φ(0.1+0.2im)
@test (1+cauchy(u) )(0.1⁻) ≈ φ(0.1⁻)



@time φ = rhsolve(g, 2ncoefficients(g))
@test φ(0.1⁺)  ≈ g(0.1)φ(0.1⁻)


sp = Legendre(0 .. -1) ∪ Legendre(0 .. 1)
g = Fun(x->x ≥ 0 ? 1-0.3exp(-40x^2) : inv(1-0.3exp(-40x^2)), sp)

u_ex = Fun(x->sign(x)u_1(x), sp)


n = 2ncoefficients(g)
g = pad(g,n)
u_ex = pad(u_ex,n)
@test (1+cauchy(u_ex,0.1⁺)) ≈ g(0.1)*(1+cauchy(u_ex,0.1⁻))

C₋ = fpcauchymatrix(sp, n, n)
pts = RiemannHilbert.collocationpoints(sp, n)

@test C₋[1,:].'*coefficients(u_ex) ≈ cauchy(u_ex,pts[1]+eps()im)
@test C₋[5,:].'*coefficients(u_ex) ≈ cauchy(u_ex,pts[5]+eps()im)
@test C₋[n÷2,:].'*coefficients(u_ex) ≈ cauchy(u_ex,0.0+eps()im)
@test C₋[(n÷2)+1,:].'*coefficients(u_ex) ≈ cauchy(u_ex,1.0+eps())
@test C₋[end,:].'*coefficients(u_ex) ≈ cauchy(u_ex, pts[end]-eps()im)
@test C₋[end,:].'*coefficients(u_ex) ≈ cauchy(u_ex, 0.0-eps()im)
@test C₋[end-1,:].'*coefficients(u_ex) ≈ cauchy(u_ex,pts[end-1]-eps()im)



@test C₋*coefficients(u_ex) ≈ [cauchy.(u_ex, pts[1:n÷2]+eps()im); cauchy.(u_ex, pts[(n÷2)+1:end]-eps()im)]
g_v = RiemannHilbert.collocationvalues(g-1, n)

@test g_v ≈ [g1.(pts1)-1; g2.(pts2)-1]
G = diagm(g_v)

@test G*g_v ≈ (g.(pts) .- 1).^2


g1 = component(g,1)
pts1 = RiemannHilbert.collocationpoints(component(sp,1), n÷2)
E1=RiemannHilbert.evaluationmatrix(component(sp,1), length(pts1))

@test E1*g1.coefficients ≈ g1.(pts1)

g2 = component(g,2)
pts2 = RiemannHilbert.collocationpoints(component(sp,2), n÷2)


E = RiemannHilbert.evaluationmatrix(sp, n)
@test E[1:end-1,:]*coefficients(g) ≈ g.(pts[1:end-1])
@test E[end,:].'*coefficients(g) ≈ component(g,2)(0.0)

@test (g2.(pts2)-1).*(C₋[(n÷2)+1:end,:]*coefficients(u_ex)) ≈ (g2.(pts2)-1).*cauchy.(u_ex, pts2-eps()im)
@test (g.(pts1)-1).*(C₋[1:n÷2,:]*coefficients(u_ex)) ≈ (g.(pts1)-1).*cauchy.(u_ex, pts1+0.000000001im)
@test (g.(pts2)-1).*(C₋[(n÷2)+1:end,:]*coefficients(u_ex)) ≈ (g.(pts2)-1).*cauchy.(u_ex, pts2-0.000000001im)
@test G*C₋*coefficients(u_ex) ≈ [u_ex1.(pts1) - (g1.(pts1)-1).*cauchy.(u_ex, pts1+eps()im);
                                u_ex2.(pts2) - (g2.(pts2)-1).*cauchy.(u_ex, pts2-eps()im)]

u_ex1,u_ex2 = components(u_ex)
L = E - G*C₋
@test L*coefficients(u_ex) ≈ [u_ex1.(pts1) - (g1.(pts1)-1).*cauchy.(u_ex, pts1+eps()im);
                                u_ex2.(pts2) - (g2.(pts2)-1).*cauchy.(u_ex, pts2-eps()im)]

@test L == rhmatrix(g,n)

u = Fun(sp, rhmatrix(g,n) \ g_v)
φ1 = z -> 1 + cauchy(u,z)
@test φ1(0.1⁺)  ≈ g(0.1)φ1(0.1⁻)
@test φ1(-0.1⁺)  ≈ g(-0.1)φ1(-0.1⁻)


@test 1+cauchy(u)(0.1+0.2im) ≈ φ1(0.1+0.2im)
@test (1+cauchy(u) )(0.1+0.2im) ≈ φ1(0.1+0.2im)
@test (1+cauchy(u) )(0.1⁻) ≈ φ1(0.1⁻)
@test (1+cauchy(u) )(0.1⁺) ≈ φ1(0.1⁺)
@test (1+cauchy(u) )(-0.1⁻) ≈ φ1(-0.1⁻)
@test (1+cauchy(u) )(-0.1⁺) ≈ φ1(-0.1⁺)


@time φ = rhsolve(g, 2ncoefficients(g))
@test φ(0.1⁺)  ≈ g(0.1)φ(0.1⁻)




sp = ArraySpace(Legendre(), 2)
f = Fun(Fun(x->[cos(x);sin(x)], Chebyshev()), ArraySpace(Legendre(), 2))
G = Fun(Fun(x->[1 exp(-40x^2); 0.1exp(-40x^2) 1], Chebyshev()), ArraySpace(Legendre(), 2, 2))

n = 2ncoefficients(G)
E = RiemannHilbert.evaluationmatrix(sp, n)
pts = RiemannHilbert.collocationpoints(sp, n÷2)

@test E*coefficients(pad(f,n)) ≈ [f[1].(pts); f[2].(pts)]



M = RiemannHilbert.multiplicationmatrix(G-I, n)
@test M*E*coefficients(pad(f,n)) ≈ mapreduce(f -> f.(pts), vcat, (G-I)*f)


L = E - M*fpcauchymatrix(sp, n,n)
u1 = L \ mapreduce(f -> f.(pts), vcat, (G-I)[:,1])
u2 = L \ mapreduce(f -> f.(pts), vcat, (G-I)[:,2])
z=2+I
    Φ = z -> I + [cauchy(Fun(sp, u1),z) cauchy(Fun(sp, u2),z)]

@test Φ(0.1+eps()im) ≈ G(0.1)*Φ(0.1-eps()im)

@test RiemannHilbert.rhmatrix(G, n) == L

@test RiemannHilbert.collocationvalues(G-I, n) ≈
    [RiemannHilbert.collocationvalues((G-I)[:,1], n) RiemannHilbert.collocationvalues((G-I)[:,2], n)]

Φ = rhsolve(G,n)
@test G(0.1)*Φ(0.1⁻) ≈ Φ(0.1⁺)

sp = ArraySpace(Legendre(-1 .. 0) ∪ Legendre(0 .. 1), 2)
G = Fun(x->[1 exp(-40x^2); 0.1exp(-40x^2) 1], ArraySpace(sp[1], 2, 2))

n=2ncoefficients(G)
Φ = rhsolve(G,n)

@test G(0.1)*Φ(0.1⁻) ≈ Φ(0.1⁺)
U = RiemannHilbert.rh_sie_solve(G,n)

U1 = U[:,1]
g = G
sp = space(g)[:,1]
C₋ = fpcauchymatrix(sp, n, n)
G = RiemannHilbert.multiplicationmatrix(g-I, n)
E = RiemannHilbert.evaluationmatrix(sp, n)

pts = RiemannHilbert.collocationpoints(sp, n÷2)
@test last(E*coefficients(U1)) ≈ U1(0.00000000001)[2]
@test last(C₋*coefficients(U1)) ≈ cauchy(U1[2], 0.00000000001-eps()*im)
@test cauchy(U1[1], 0.00000000001-eps()*im) ≈ (C₋*coefficients(U1))[n÷2]
@test last(G*C₋*coefficients(U1)) ≈ ((g(0.00000000001)-I)*cauchy(U1, 0.00000000001-eps()*im))[2]


@test RiemannHilbert.collocationvalues((g-I)[:,1], n)[end] ≈ 0.1


sp = ArraySpace(Legendre(0 .. -1) ∪ Legendre(0 .. 1), 2)
G = g = Fun(x-> x ≥ 0 ? [1 exp(-40x^2); 0.1exp(-40x^2) 1] : inv([1 exp(-40x^2); 0.1exp(-40x^2) 1]), ArraySpace(sp[1], 2, 2))
n=2ncoefficients(G)
Ũ1 = Fun(x-> x ≥ 0 ? U1(x) : -U1(x), sp, n÷2)
@test cauchy(Ũ1, 0.1⁻) ≈ cauchy(Ũ1, 0.1-eps()im)
@test cauchy(Ũ1, 0.1+eps()im) ≈ cauchy(U1, 0.1+eps()im)
@test cauchy(Ũ1, -0.1+eps()im) ≈ cauchy(U1, -0.1+eps()im)

L = rhmatrix(G, n)
rhs = RiemannHilbert.collocationvalues((g-I)[:,1], n)
@test rhs[end] ≈ 0.1
L*coefficients(Ũ1) - rhs |>norm

Φ = rhsolve(G,n)

g = G
sp = space(g)[:,1]
C₋ = fpcauchymatrix(sp, n, n)
G = RiemannHilbert.multiplicationmatrix(g-I, n)
E = RiemannHilbert.evaluationmatrix(sp, n)

pts = RiemannHilbert.collocationpoints(sp, n÷2)

@test last(E*coefficients(Ũ1)) ≈ Ũ1(0.00000000001)[2]
@test last(C₋*coefficients(Ũ1)) ≈ cauchy(Ũ1[2], 0.00000000001-eps()*im)
@test cauchy(Ũ1[1], 0.00000000001-eps()*im) ≈ (C₋*coefficients(Ũ1))[n÷2]
@test G[[n÷2,n],[n÷2,n]] ≈ (g(0.00000000001)-I)

@test last(G*C₋*coefficients(Ũ1)) ≈ (g(0.00000000001)-I)[2,:].'*cauchy(Ũ1, 0.00000000001-eps()*im)

L[end,:].'*coefficients(Ũ1) ≈ Ũ1(0.00000000001)[2] - (g(0.00000000001)-I)[2,:].'*cauchy(Ũ1, 0.00000000001-eps()*im)


U1(0.00000000001)[2] - (g(0.00000000001)-I)[2,:].'*cauchy(U1, 0.00000000001-eps()*im)

rhs[end]


@test g(0.1)*Φ(0.1⁻) ≈ Φ(0.1⁺)


G(0.1)*Φ(0.1⁻) , Φ(0.1⁺)



Γ = Segment(0, 2.5exp(im*π/6)) ∪ Segment(0, 2.5exp(5im*π/6)) ∪
        Segment(0, 2.5exp(-5im*π/6)) ∪ Segment(0, 2.5exp(-im*π/6))
sp = ArraySpace(PiecewiseSpace(Legendre.(components(Γ))), 2,2)

s₁ = im
s₃ = -im

G = Fun( z -> if angle(z) ≈ π/6
                    [1 0; s₁*exp(8im/3*z^3) 1]
                elseif angle(z) ≈ 5π/6
                    [1 0; s₃*exp(8im/3*z^3) 1]
                elseif angle(z) ≈ -π/6
                    [1 -s₃*exp(-8im/3*z^3); 0 1]
                elseif angle(z) ≈ -5π/6
                    [1 -s₁*exp(-8im/3*z^3); 0 1]
                end
                    , sp)

Φ = rhsolve(G.', 2*4*100).'


s=exp(im*π/6)
    @test Φ((s)⁻)*G(s) ≈ Φ(s*⁺)


@test map(g->first(components(g)[1]), G)*map(g->first(components(g)[2]), G)*
    map(g->first(components(g)[3]), G)*map(g->first(components(g)[4]), G) ≈ eye(2)

for _=1:5
    @test RiemannHilbert.evaluationmatrix(sp[:,1], 32) ≈ RiemannHilbert.evaluationmatrix(sp[:,1], 32)
    @test RiemannHilbert.rhmatrix(G, 32) ≈ RiemannHilbert.rhmatrix(G, 32)
end


@test RiemannHilbert.rhmatrix(G.', 900) ≈ RiemannHilbert.rhmatrix(G.', 900)



U = RiemannHilbert.rh_sie_solve(G.', 2*4*100)
@test -0.36706155154807807 ≈ sum(U[1,2])/(-π*im)


G̃ = Fun( z -> if angle(z) ≈ π/6
                    [1 0; s₁*exp(8im/3*z^3) 1]
                elseif angle(z) ≈ 5π/6
                    [1 0; s₃*exp(8im/3*z^3) 1]
                elseif angle(z) ≈ -π/6
                    [1 -s₃*exp(-8im/3*z^3); 0 1]
                elseif angle(z) ≈ -5π/6
                    [1 -s₁*exp(-8im/3*z^3); 0 1]
                end
                    , Γ)

Ũ = RiemannHilbert.rh_sie_solve(G̃.', 2*4*100)


@which RiemannHilbert.rh_sie_solve(G̃.', 2*4*100)

@test rhmatrix(G.', 2*4*100) ≈ rhmatrix(G̃.', 2*4*100)

@test -0.36706155154807807 ≈ sum(Ũ[1,2])/(-π*im)

Φ[1,1]

SingularIntegralEquations.istieltjes(Φ[1,1])
stieltjes(SingularIntegralEquations.istieltjes(Φ[1,1]),2+5)


V = SingularIntegralEquations.istieltjes(Φ)
x = Fun(domain(V))

10.0I + sum.(Array(V)) + stieltjes(x*V, 10.0)





@test (z*Φ)(10.0) ≈ 10.0Φ(10.0)






spacescompatible(AffineSpace(ComplexPlane()), AffineSpace(ComplexPlane()))s
@which z*component(Array(Φ)[1,1],5)





v = map(f -> z*f, components(Φ[1,2]))


(v[1] + v[3])(10.0)
ApproxFun.spacescompatible(v[1], v[2])

Fun(v[1], space(v[1]))
sp = union(space(v[1]), space(v[2]))
Fun(v[1], sp)
ApproxFun.isconvertible(space(v[1]), sp.spaces[2])
maxspace(space(v[1]).spaces[2], sp.spaces[2])

sp.spaces[1]
ApproxFun.canonicalspace(sp)

coefficients(v[1], sp)
space(v[1] + v[3])

space(v[1])
@which ApproxFun.spacescompatible(component(space(v[1]),2), component(space(v[2]),2))

v[3](10.0)



(z*φ)(10.0) - 10.0φ(10.0)
using SO
@which stieltjes(istieltjes(Φ))

c = zero(prectype(space(z)))
r = Array{Fun}(0)
Φ = Φ[1,2]
for φ in components(Φ)
    a,b = components(z*φ)
    c += Number(a)
    push!(r, b)
end

⊕(r...)


istieltjes(U)

 component(Φ,1) + component(Φ,2)

(z*Φ)


(z*Φ)(20.0)
20.0Φ(20.0)


(z*component(Φ[1,2],4))(10.0)+(z*component(Φ[1,2],3))(10.0)


a1,b1 = components(z*component(Φ[1,2],4))
a2,b2 = components(z*component(Φ[1,2],3))
space(b2)
space(b1)


space(component(Φ[1,2],3))
space(component(Φ[1,2],3))



(b2⊕b1)
(b2+b1)(10.0)
b2(10.0) + b1(10.0)
10.0component(Φ[1,2],4)(10.0)

ApproxFun.prectype(space(z))

v = map(f -> z*f, components(φ))

component(v[1],2)+component(v[2],2)

φ = Φ[1,1]

components(φ)
φ = component(φ,1)
a = coefficient(z,1)
b = coefficient(z,2)
u = istieltjes(φ)
x = Fun(domain(u))
@which domain(stieltjes(a*u + b*x*u).space )

stieltjes(a*u + b*x*u).space

z*components(φ)[1] |>domain
domain.(map(f -> z*f, components(φ)))
z*φ



p = components(u)
u = p[1]
x = Fun(domain(u))
stieltjes(x*u)
u.coefficients[1]

(sum(u)+stieltjes(x*u))(10.0)

10.0stieltjes(u, 10.0)

Fun(AffineSpace(ComplexPlane()), Float64[1,2])(0.1)


0.1 ∈ ComplexPlane()1!~1

Fun(::typeof(identity), ::ComplexPlane)

Fun(identity,ComplexPlane())

f = Φ[1,2]


10Φ(10.0)

stieltjes(10V[1,2],10.0)


stieltjes((10-x)V[1,2],10.0) + stieltjes(x*V[1,2],10.0)



stieltjes((10-x)V[1,2],10.0)



10.0linesum.(Array(V))

10.0*Φ(10.0)


Φ(7)


Φ[1,2](7)

space(Φ)
space(component(Ũ[1,1],1))
coefficients(component(Ũ[1,1],1))
@which stieltjes(Fun(space(component(Ũ[1,1],1)),collect(coefficients(component(Ũ[1,1],1)))))

RiemannHilbert.ComplexPlane()

z/(z-x) == (z-x)/(z-x) + x/(z-x)


@test cond(rhmatrix(G.', 2*4*100)) ≤ 1000
cond(rhmatrix(G.', 2*4*200))

Φ((s)⁻)*G(s) ≈ Φ(s*⁺)


2*100000.0Φ(100000.0)[1,2]


n = 2*4*20
U = pad((G-I)[:,1],n)
E = RiemannHilbert.evaluationmatrix(space(U), n)

ret = Array{ComplexF64}(0)
    for K=1:2, J=1:4
        Up = component(U[K],J)
        p = collocationpoints(space(Up), (n ÷ 2) ÷ 4)
        append!(ret,Up.(p))
    end
    norm(E*coefficients(U) - ret)




L = RiemannHilbert.rhmatrix(G.', 2*4*90)
@test cond(L) ≤ 200

sp = space(U)
C₋ = fpcauchymatrix(sp, n, n)
ret = Array{ComplexF64}(0)
    for K=1:2, J=1:4
        Up = component(U[K],J)
        p = collocationpoints(space(Up), (n ÷ 2) ÷ 4)
        append!(ret,cauchy.(p))
    end
    norm(C₋*coefficients(U) - ret)




pts = collocationpoints(sp, n÷2)


scatter(abs.(E*coefficients(U)  - [U[1].(pts);U[2].(pts)]))


E*coefficients(U)  - [U[1].(pts);U[2].(pts)]

@which rhsolve(G.', 2*4*20)

Φ = rhsolve(G.', 2*4*250).'

s=exp(im*π/6)
    Φ((s)⁻)*G(s) - Φ(s*⁺) |>norm

2*4*300

g = G.'
sp = space(g)[:,1]
n=900
C₋ = fpcauchymatrix(sp, n, n)
G = RiemannHilbert.multiplicationmatrix(g-I, n)
E = RiemannHilbert.evaluationmatrix(sp, n)
E .- G*C₋
pts = points(sp, (n÷2))

scatter(real.(pts), imag.(pts))

E*coefficients(U) - [U[1].(pts);U[2].(pts)]|>norm

@which RiemannHilbert.evaluationmatrix(sp, n)

L*coefficients(U)


L = RiemannHilbert.rhmatrix(G.', 900)
    cond(L)
4*4*2
n = 4*4*2






@test Φ((s)⁻) ≈ Φ(s+eps())
@test Φ(s*⁺) ≈ Φ(s-eps())

Φ(2.499999exp(im*π/6)⁺)- Φ(2.499999exp(im*π/6)⁻)
using Plots
scatter(abs.(Φ.coefficients);yscale=:log10)

*G(s) - Φ(s*⁺)

coefficients(Φ)[4500:end]|>norm


coefficients(G.')
using SO, Plots
plot(abs.(coefficients(Φ)))

@test Φ((s)⁻)*G(s) ≈ Φ(s*⁺)

G(s)[2,1] ≈ s₁*exp(8im/3*s^3)

G(s)*Φ((s)⁻) - Φ(s*⁺)


Φ(1.0+0.1im)
Φ(s*⁺)



G.'(exp(π/6*im))

ns = ncoefficients.(Array(G)))
@time C = fpstieltjesmatrix(space(G),ns,ns)


G[1,1].(pts)

import RiemannHilbert: pieces_npoints, collocationpoints, pieces



v_sp = ArraySpace(space(G)[:,1])

@time C⁻ = fpstieltjesmatrix(v_sp, ns[:,1], ns[:,1]); C⁻ ./= (-2π*im);
@time vals = collocationvalues.(G.', ns)
@time G_m = hvcat((2,2), diagm.(vals)...)
@time L = I - G_m*C⁻

hvcat((2,2), ones(2,2), ones(2,2), ones(2,2), ones(2,2))
rhs = collocationvalues((G.'-I)[:,1], ns[:,1])

L \rhs


collocationvalues(G, ns)
sp = space(G)
s = component(sp[1],1)

k,j = 1,1

f = G[1]
n = pieces_npoints(sp[k,j],ns[k,j])

pts = collocationpoints(s, ns11[1])
    G11 = f.(pts)




G(exp(im*π/6))
scatter(real(pts), imag(pts))

pts1 = RiemannHilbert.collocationpoints(,


G[1,1].(pts)

using Plots

scatter(real(pts), imag(pts))

Array(G)



###

Γ = Segment(0, 2.5exp(im*π/6))   ∪
    Segment(0, 2.5exp(5im*π/6))  ∪
    Segment(0, 2.5exp(-5im*π/6)) ∪
    Segment(0, 2.5exp(-im*π/6))

s₁ = im
s₃ = -im

G = Fun( z -> if angle(z) ≈ π/6
                    [1                 0;
                     s₁*exp(8im/3*z^3) 1]
                elseif angle(z) ≈ 5π/6
                    [1                 0;
                     s₃*exp(8im/3*z^3) 1]
                elseif angle(z) ≈ -π/6
                    [1                -s₃*exp(-8im/3*z^3);
                     0                 1]
                elseif angle(z) ≈ -5π/6
                    [1                -s₁*exp(-8im/3*z^3);
                     0                 1]
                end
                    , Γ);


Φ = rhsolve(G.', 800).'


U = istieltjes(Φ)


RiemannDual(0,exp(im*π/6))⁺

Φ₁⁺ = I + finitepart.(stieltjes(U,RiemannDual(0,exp(im*π/6))⁺))
Φ₁⁻ = I + finitepart.(stieltjes(U,RiemannDual(0,exp(im*π/6))⁻))


Φ₁⁺



Φ(10.0) ≈ I+stieltjes(U,10.0)
a,b,c,d = pieces(U)

angle(domain(a))

π/6.0
finitepart.(stieltjes(a,RiemannDual(0,exp(im*π/6))⁺) +
    stieltjes(b,RiemannDual(0,exp(im*π/6))) +
    stieltjes(c,RiemannDual(0,exp(im*π/6))) +
    stieltjes(d,RiemannDual(0,exp(im*π/6))))
stieltjes(U,)
Φ₁⁺
Φ(0.00001ζ+100im*eps())-I


Φ₁⁻*(pieces(G)[1])(0.)




Φ(ζ+100im*eps()) ≈ Φ(ζ-100im*eps())G(ζ)


Φ(ζ*⁺)






Φ = rhsolve(G.', 800).'

ζ = exp(im*π/6)
G(ζ)




Φ(ζ*⁺) - Φ((ζ)⁻)*G(ζ)|>norm
