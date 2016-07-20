using ApproxFun,SingularIntegralEquations,DualNumbers,RiemannHilbert,Base.Test
    import RiemannHilbert: fpcauchy



f=Fun(exp)
ε=0.00000000001;
@test_approx_eq_eps cauchy(f,1.0+ε)-f(1.)/(2π*im)*log(ε) fpcauchy(f,dual(1.0,1.0)) 1E-6
@test_approx_eq_eps cauchy(f,1.0+(1.+1.im)*ε)-f(1.)/(2π*im)*log(ε) fpcauchy(f,dual(1.0,(1.+1.im))) 1E-6
@test_approx_eq_eps cauchy(f,-1.0-ε)+f(-1.)/(2π*im)*log(ε) fpcauchy(f,dual(-1.0,-1.0)) 1E-6
@test_approx_eq_eps cauchy(f,-1.0+(1.+1.im)*ε)+f(-1.)/(2π*im)*log(ε) fpcauchy(f,dual(-1.0,(1.+1.im))) 1E-6



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
