using ApproxFun, SingularIntegralEquations, DualNumbers, RiemannHilbert, Base.Test
    import RiemannHilbert: fpcauchy, RiemannDual
    import SingularIntegralEquations: stieltjesmoment, undirected, Directed



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


stieltjesmoment(Legendre(),0,Directed{true}(0.1))


z = RiemannDual(-1,-1)
l = stieltjesjacobimoment(0,0,0,z)
h = 0.00001
@test l(h) ≈ stieltjesjacobimoment(0,0,0,value(z)+epsilon(z)h) atol=1E-5


z = RiemannDual(-1,-1)
l = stieltjesjacobimoment(0.5,0,0,z)


h=0.000000000001;atanh(1+h*exp(0.1im))

-0.5log(h)+log(2)/2 - im/2*angle(-exp(0.1im))







log((1-0.5)+log(0.5+1))

z = -1.00001
x = 2./(1-z)
sqrt(-x)

stieltjesjacobimoment(0.5,0,0,-1.0001-0im)

SingularIntegralEquations.HypergeometricFunctions.speciallog(0.9999-0im)

n=0
_₂F₁(n+1,n+0.5+1,2n+0.5+2,x)


SingularIntegralEquations.HypergeometricFunctions.speciallog(-2.0-0im)

SingularIntegralEquations.HypergeometricFunctions.speciallog(2.0+0im)

stieltjesjacobimoment(0.5,0,0,z)


stieltjesmoment(Legendre(),0,1+h*exp(0.1im))
-stieltjesmoment(Legendre(),0,-1-h*exp(0.1im))

stieltjesmoment(Legendre(),1,1+h*exp(0.1im))
stieltjesmoment(Legendre(),1,-1-h*exp(0.1im))




n = 0
x = 2/(1-z)
l = SingularIntegralEquations._₂F₁(n+1,n+1,2n+2,x)

s = -x
log1p(s)/undirected(s)

log(z)*


1 - 2/(1-z) == (1-z - 2)/(1-z) == - (1+z)/(1-z)

(s = -z; log1p(s)/undirected(s))



M=10000
    log(M*(1+im))-(log(M)+log(abs(1+im)) + im*angle(1+im))

#


1/dual(0,1)

x

-z

z = dual(1,1)
x = 2./(1-z);

n=0;
    α = β = 0
_₂F₁(n+1,n+α+1,2n+α+β+2,x)

normalization(n,α,β)*(-x)^(n+1)*


stieltjesmoment(sp,1,z)





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
