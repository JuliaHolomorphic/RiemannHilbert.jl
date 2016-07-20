using ApproxFun,SingularIntegralEquations,DualNumbers,RiemannHilbert,Base.Test
    import RiemannHilbert: fpcauchy



f=Fun(exp)
ε=0.00000000001;
@test_approx_eq_eps cauchy(f,1.0+ε)-f(1.)/(2π*im)*log(ε) fpcauchy(f,dual(1.0,1.0)) 1E-6
@test_approx_eq_eps cauchy(f,1.0+(1.+1.im)*ε)-f(1.)/(2π*im)*log(ε) fpcauchy(f,dual(1.0,(1.+1.im))) 1E-6
@test_approx_eq_eps cauchy(f,-1.0-ε)+f(-1.)/(2π*im)*log(ε) fpcauchy(f,dual(-1.0,-1.0)) 1E-6
@test_approx_eq_eps cauchy(f,-1.0+(1.+1.im)*ε)+f(-1.)/(2π*im)*log(ε) fpcauchy(f,dual(-1.0,(1.+1.im))) 1E-6
