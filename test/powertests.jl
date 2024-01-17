using ApproxFun, SingularIntegralEquations, DualNumbers, PowerNumbers, RiemannHilbert, LinearAlgebra, FastTransforms, SpecialFunctions, Test
import ApproxFunBase: ArraySpace, pieces, dotu, interlace
import RiemannHilbert: productcondition, fpstieltjesmatrix!, fpstieltjesmatrix, orientedleftendpoint, orientedrightendpoint, fpcauchymatrix, collocationvalues, collocationpoints
import SingularIntegralEquations: stieltjesmoment, stieltjesmoment!, undirected, Directed, ⁺, ⁻, istieltjes
import SingularIntegralEquations.HypergeometricFunctions: speciallog
import PowerNumbers: PowerNumber, LogNumber, realpart, epsilon, alpha

@testset "stieltjesjacobimoment" begin

    h = 0.0000001
    for z in (PowerNumber(-1,-1,1), PowerNumber(1,1,1), PowerNumber(-1,2exp(0.1im),1), PowerNumber(1,2exp(0.1im),1)),
            k = 0:1
        l = stieltjesjacobimoment(0,0,k,z)
        @test l(h) ≈ stieltjesjacobimoment(0,0,k,realpart(z)+epsilon(z)h^alpha(z)) atol=1E-5
    end

    h = 0.00001
    for z in (PowerNumber(-1,-1,1), PowerNumber(-1,exp(0.1im),1), PowerNumber(-1,exp(-0.1im),1))
        @test stieltjesjacobimoment(0.5,0,0,z)(h) ≈ stieltjesjacobimoment(0.5,0,0,realpart(z)+epsilon(z)h^alpha(z)) atol=1E-4
    end

    z = PowerNumber(1,1,1)
    ZD = Dual(1,1)
    k = 1
    x = 2/(1-z)
    XD = 2/(1-ZD)
    #HypergeometricFunctions.mxa_₂F₁(n+1,n+α+1,2n+α+β+2,x)
    α,β,n = 0,0,k
    a,b,c,x = n+1,n+α+1,2n+α+β+2,2/(1-z)
    abeqcd(a,b,cd) = isequal(a,b) && isequal(b,cd)
    abeqcd(a,b,c,d) = isequal(a,c) && isequal(b,d)
    if isequal(c,2)
        if abeqcd(a,b,1) # 6. 15.4.1
            print("A")
        end
    elseif isequal(c,4)
        if abeqcd(a,b,2)
            print("B")
        end
    else print("C")
    end
    6*(-2 + (1-2/undirected(x))*log1p(-x))
end 

@testset "Interval FPStieltjes" begin
    f = Fun(x->exp(-40(x-0.1)^2), Legendre())
    C = Array{ComplexF64}(undef, ncoefficients(f), ncoefficients(f))
    d = Segment(-1,-1+im)
    fpstieltjesmatrix!(C, space(f), d)
end