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




sp = Legendre(0 .. -1) ⊕ Legendre(0..1)
f = Fun(x->exp(-40(x-0.1)^2), sp)
ns = ncoefficients.(components(f))
C = fpstietjesmatrix(space(f), ns, ns)
@test norm(C) ≤ 100

cfs = vcat(coefficients.(components(f))...)
c_vals = C*cfs

@test c_vals[1] ≈ stieltjes(f, -1.0000000001)

@test c_vals[43] ≈ stieltjes(f, points(Domain(0.. -1), ns[1]; kind=2)[43]*⁻)

c_vals[43]
c_vals[44]
c_vals[end]

c1 = Fun(Chebyshev(domain(component(sp,1))),chebyshevtransform(c_vals[1:ns[1]]; kind=2))
c2 = Fun(Chebyshev(domain(component(sp,2))),c_cfs[ns[1]+1:end])




@test RiemannHilbert.fpstietjesmatrix(Legendre(0..1), [44], [44]) == C[1:44,1:44]





stieltjes(f, -0.5*⁻)

stieltjes(component(f, -0.5*⁻)


c2(-0.5)


@time fpcauchy(f, dual(z))

@time cauchy(g, z)
@time fpcauchy(g, dual(z))



Γ = Segment(0,1) ∪ Segment(0,im) ∪ Segment(0,-1 -im)


sp = Legendre(component(Γ, 1))
d = component(Γ, 2)



n=200
    C = Array{Complex128}(n*ncomponents(Γ),n*ncomponents(Γ))
    k, j = 1, 2
    @time fpstietjesmatrix!(view(C, n*(j-1) .+ (1:n), n*(k-1) .+ (1:n)),
                            Legendre(component(Γ,k)), component(Γ,j))




n = 1000
    C = Array{Complex128}(n, n)
    @time fpstietjesmatrix!(C, Legendre(component(Γ,1)), component(Γ,2))
Profile.print()

C


view(C, 10(j-1) .+ (1:10), 10(k-1) .+ (1:10))




m = n =10

import RiemannHilbert: finitepart


C

pts[2]


first(d)

C


m, n = size(C)
pts = points(d, m; kind=2)





view(C,2,:)




import SingularIntegralEquations.stieltjesmoment!

stieltjesmoment!(sp,

pts[2]

for k=2:m-1
    C[


using Plots
plot(Γ;xlims=(-2,2),ylims=(-2,2))








f=Fun(exp,Legendre())
    @which stieltjes(space(f),f.coefficients,1000.0)


f = f.coefficients
S = Legendre()
@which SingularIntegralEquations.stieltjesintervalrecurrence(Legendre(),coefficients(f,S,Legendre()),1000.0)


@time SingularIntegralEquations.stieltjesbackward(S,1000.0)[1:10000]

stieltjesmatrix!(Legendre(0..1), Segment(0,im), C)

S = Legendre(0..1)


SingularIntegralEquations.dotu([1.],[1.,2.])




points(Segment(0,im), m)
















plot(Γ)


using Plots

g(-1)/(2π)

#
# S=Laurent(Circle())
#
# Hilbert(S)
#
# Cp=Cauchy(true,S)
#
# Cp.ops[2]
#
# Cm=Cauchy(Legendre(),-)
# L=I-(G-I)*Cm
#
# L=I-(G-I)*Cm+Derivative()
#
# collocation(L,100)\(G-I)
#
#
# function collocation(L::Operator,pts)
#     n=length(pts)
#     ret=Array(Float64,n,n)
#     for j=1:n
#         Lf=L*Fun([zeros(j-1);1.],domainspace(L))
#         ret[:,j]=Lf(pts)
#     end
#     ret
# end
#
# collocation(I+Derivative(Chebyshev()),points(Interval(),10))

# d = ComplexPlane() \ [-1,1]
# [Evaluation(∞);Restriction(d,+)-Restriction(d,-)*(G-I)]\[I]
