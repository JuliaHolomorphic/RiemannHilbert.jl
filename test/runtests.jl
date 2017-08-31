using ApproxFun, SingularIntegralEquations, DualNumbers, RiemannHilbert, Base.Test
    import RiemannHilbert: RiemannDual, LogNumber, fpstietjesmatrix!, fpstietjesmatrix, orientedlast, finitepart
    import SingularIntegralEquations: stieltjesmoment, stieltjesmoment!, undirected, Directed, ⁻
    import SingularIntegralEquations.HypergeometricFunctions: speciallog

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
    @test l(h) ≈ stieltjesjacobimoment(0,0,k,value(z)+epsilon(z)h) atol=1E-5
end

h=0.0001
for z in (RiemannDual(1,3exp(0.2im)), RiemannDual(1,0.5exp(-1.3im)),
            RiemannDual(-1,3exp(0.2im)), RiemannDual(-1,0.5exp(-1.3im)),
            RiemannDual(-1,1), RiemannDual(1,-1))
    @test atanh(z)(h) ≈  atanh(value(z)+epsilon(z)h) atol = 1E-4
end

z = RiemannDual(1,-0.25)
h = 0.0000001
@test speciallog(z)(h) ≈ speciallog(value(z)+epsilon(z)h) atol=1E-4

h = 0.00001
for z in (RiemannDual(-1,-1), RiemannDual(-1,exp(0.1im)), RiemannDual(-1,exp(-0.1im)))
    @test stieltjesjacobimoment(0.5,0,0,z)(h) ≈ stieltjesjacobimoment(0.5,0,0,value(z)+epsilon(z)h) atol=1E-4
end





f = Fun(exp,Legendre())

h = 0.00001
for z in  (RiemannDual(-1,-1), RiemannDual(-1,1+im), RiemannDual(-1,1-im))
    @test cauchy(f, z)(h) ≈ cauchy(f, value(z) + epsilon(z)h) atol=1E-4
end



# Directed and RiemannDual

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


@test RiemannHilbert.orientedfirst(Segment()) == RiemannDual(-1.0,1)
@test RiemannHilbert.orientedlast(Segment()) == RiemannDual(1.0,-1)


Γ = Segment()
f = Fun(x->exp(-40(x-0.1)^2), Legendre())
C = Array{Complex128}(ncoefficients(f), ncoefficients(f))
d = Segment(im,2im)

fpstietjesmatrix!(C, space(f), d)
c = Fun(d, ApproxFun.chebyshevtransform(C*coefficients(f); kind=2))
@test c(1.5im) ≈ stieltjes(f,1.5im)

d = Segment(-1,-1+im)
fpstietjesmatrix!(C, space(f), d)
@test norm(C) ≤ 100
c = Fun(d, ApproxFun.chebyshevtransform(C*coefficients(f); kind=2))
@test c(-1+0.5im) ≈ stieltjes(f,-1+0.5im)

d = Segment(1,1+im)
fpstietjesmatrix!(C, space(f), d)
@test norm(C) ≤ 100
c = Fun(d, ApproxFun.chebyshevtransform(C*coefficients(f); kind=2))
@test c(1+0.5im) ≈ stieltjes(f,1+0.5im)

d = Segment()
fpstietjesmatrix!(C, space(f), d)
@test norm(C) ≤ 100
c = Fun(d, ApproxFun.chebyshevtransform(C*coefficients(f); kind=2))
@test c(0.5) ≈ stieltjes(f,0.5*⁻)


d = Segment(0,1)
f = Fun(x->exp(-200(x-0.6)^2), Legendre(d))
C = fpstietjesmatrix(space(f), ncoefficients(f), ncoefficients(f))
@test norm(C) ≤ 100
c = Fun(d, chebyshevtransform(C*coefficients(f); kind=2))
@test c(0.5) ≈ stieltjes(f,0.5*⁻)






f = Fun(exp,Legendre())
f1 = Fun(exp,Legendre(-1..0))
f2 = Fun(exp,Legendre(0..1))


@test stieltjes(f,0.0⁻) ≈ finitepart(stieltjes(f1,RiemannDual(0.0,-im)) + stieltjes(f2,RiemannDual(0.0,-im)))
@test stieltjes(f,0.0⁻) ≈ finitepart(stieltjes(f1,RiemannDual(0.0,exp(-0.1im))) + stieltjes(f2,RiemannDual(0.0,exp(-0.1im))))
@test stieltjes(f,0.0⁻) ≈ finitepart(stieltjes(f1,Directed{false}(RiemannDual(0.0,-1.0))) + stieltjes(f2,RiemannDual(0.0,-1.0)))

sp = Legendre(-1 .. 0) ⊕ Legendre(0 .. 1)
f = Fun(x->exp(-40(x-0.1)^2), sp)
v = components(f)
ns = ncoefficients.(v)
C = fpstietjesmatrix(space(f), ns, ns)
@test norm(C) ≤ 100

cfs = vcat(coefficients.(v)...)
c_vals = C*cfs
h =0.00001
@test stieltjes(v[1], Directed{false}(RiemannDual(0.0,-1.0))) ≈ stieltjes(v[1], RiemannDual(0.0,-1.0-eps()*im))

@test finitepart(stieltjes(v[1], Directed{false}(RiemannDual(0.0,-1.0)))+ stieltjes(v[2], RiemannDual(0.0,-1.0))) ≈
    stieltjes(v[1], -0.00000000001im)+stieltjes(v[2], -0.00000000001im)


@test finitepart(stieltjes(v[1], Directed{false}(RiemannDual(0.0,-1.0)))+ stieltjes(v[2], RiemannDual(0.0,-1.0))) ≈
    stieltjes(f, -0.00000000001im)

@test c_vals[1] ≈ stieltjes(f, -0.0000000001im)
@test c_vals[end] ≈ stieltjes(f, -0.0000000001im)

@test finitepart(stieltjes(f, RiemannDual(0.0,-im))) ≈ finitepart(stieltjes(v[1], RiemannDual(0.0,-im))+stieltjes(v[2], RiemannDual(0.0,-im)))


C11 = fpstietjesmatrix(space(v[1]), ncoefficients(v[1]), ncoefficients(v[1]))
C12 = fpstietjesmatrix(space(v[2]), domain(v[1]), ncoefficients(v[1]), ncoefficients(v[2]))

@test finitepart(stieltjes(v[1], Directed{false}(RiemannDual(0.0,-1.0)))) ≈ (C11*coefficients(v[1]))[1]
@test finitepart(stieltjes(v[2], RiemannDual(0.0,-1.0))) ≈ dotu(stieltjesmoment!(Array{Complex128}(44), space(v[2]), orientedlast(domain(v[1])), finitepart),
            coefficients(v[2]))

@test C12[1,:] ≈ stieltjesmoment!(Array{Complex128}(44), space(v[2]), orientedlast(domain(v[1])), finitepart)

@test finitepart(stieltjes(v[2], RiemannDual(0.0,-1.0))) ≈ (C12*coefficients(v[2]))[1]
@test C[1,:] ≈ [C11[1,:]; C12[1,:]]



sp = Legendre(0 .. -1) ⊕ Legendre(0 .. 1)
f = Fun(x->sign(x)*exp(-40(x-0.1)^2), sp)
v = components(f)
ns = ncoefficients.(v)
C = fpstietjesmatrix(space(f), ns, ns)
@test norm(C) ≤ 100

cfs = vcat(coefficients.(v)...)
c_vals = C*cfs
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
