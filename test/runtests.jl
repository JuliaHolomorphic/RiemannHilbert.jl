using ApproxFun, SingularIntegralEquations, DualNumbers, RiemannHilbert, Base.Test
    import RiemannHilbert: fpcauchy, RiemannDual, LogNumber
    import SingularIntegralEquations: stieltjesmoment, undirected, Directed
    import SingularIntegralEquations.HypergeometricFunctions: speciallog


f=Fun(exp)
ε=0.00000000001;
@test cauchy(f,1.0+ε)-f(1.)/(2π*im)*log(ε) ≈ fpcauchy(f,dual(1.0,1.0)) atol=1E-6
@test cauchy(f,1.0+(1.+1.0im)*ε)-f(1.)/(2π*im)*log(ε) ≈ fpcauchy(f,dual(1.0,(1.+1.0im))) atol=1E-6
@test cauchy(f,-1.0-ε)+f(-1.)/(2π*im)*log(ε) ≈ fpcauchy(f,dual(-1.0,-1.0)) atol=1E-6
@test cauchy(f,-1.0+(1.+1.0im)*ε)+f(-1.)/(2π*im)*log(ε) ≈ fpcauchy(f,dual(-1.0,(1.+1.0im))) atol=1E-6






@which fpcauchy(f,dual(1.0,1.0))



for h in (0.1,0.01), a in (2exp(0.1im),1.1)
    @test log(RiemannDual(0,a))(h) ≈ log(h*a)
end


for h in (0.1,0.01), a in (2exp(0.1im),1.1)
    @test log1p(RiemannDual(-1,a))(h) ≈ log(h*a)
end


z = RiemannDual(-1,-1)
l = stieltjesjacobimoment(0,0,0,z)
h = 0.0000001
@test l(h) ≈ stieltjesjacobimoment(0,0,0,value(z)+epsilon(z)h) atol=1E-5

l = stieltjesjacobimoment(0,0,1,z)
@test l(h) ≈ stieltjesjacobimoment(0,0,1,value(z)+epsilon(z)h) atol=1E-5



h = 0.00001
@test l(h) ≈ stieltjesjacobimoment(0,0,0,value(z)+epsilon(z)h) atol=1E-5



for z in (RiemannDual(1,3exp(0.2im)), RiemannDual(1,0.5exp(-1.3im)),
            RiemannDual(-1,3exp(0.2im)), RiemannDual(-1,0.5exp(-1.3im)),
            RiemannDual(-1,1), RiemannDual(1,-1))
    @test atanh(z)(h) ≈  atanh(value(z)+epsilon(z)h) atol = 1E-5
end

z = RiemannDual(1,-0.25)
@test speciallog(z)(h) ≈ speciallog(value(z)+epsilon(z)h) atol=1E-4


for z in (RiemannDual(-1,-1), RiemannDual(-1,exp(0.1im)), RiemannDual(-1,exp(-0.1im)))
    @test stieltjesjacobimoment(0.5,0,0,z)(h) ≈ stieltjesjacobimoment(0.5,0,0,value(z)+epsilon(z)h) atol=1E-4
end



f = Fun(exp,Legendre())


for z in  (RiemannDual(-1,-1), RiemannDual(-1,1+im), RiemannDual(-1,1-im))
    @test cauchy(f, z)(h) ≈ cauchy(f, value(z) + epsilon(z)h) atol=1E-5
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




@time g = Fun(f, Chebyshev())
@time cauchy(f, z)

@time fpcauchy(f, dual(z))

@time cauchy(g, z)
@time fpcauchy(g, dual(z))



Γ = Segment(0,1) ∪ Segment(0,im) ∪ Segment(0,-1 -im)


sp = Legendre(component(Γ, 1))
d = component(Γ, 2)


orientedfirst(d::Segment) = RiemannDual(first(d), angle(d))
orientedlast(d::Segment) = RiemannDual(last(d), -angle(d))



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



n=200
    C = Array{Complex128}(n*ncomponents(Γ),n*ncomponents(Γ))
    k, j = 1, 2
    @time fpstietjesmatrix!(view(C, n*(j-1) .+ (1:n), n*(k-1) .+ (1:n)),
                            Legendre(component(Γ,k)), component(Γ,j))




C = Array{Complex128}(10, 10)
    fpstietjesmatrix!(C, Legendre(), Segment())
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
