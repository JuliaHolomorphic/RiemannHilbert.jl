using ApproxFun, SingularIntegralEquations, DualNumbers, PowerNumbers, RiemannHilbert, LinearAlgebra, FastTransforms, SpecialFunctions, Test
import ApproxFunBase: ArraySpace, pieces, dotu, interlace
import RiemannHilbert: productcondition, fpstieltjesmatrix!, fpstieltjesmatrix, orientedleftendpoint, orientedrightendpoint, fpcauchymatrix, collocationvalues, collocationpoints
import SingularIntegralEquations: stieltjesmoment, stieltjesmoment!, undirected, Directed, ⁺, ⁻, istieltjes
import SingularIntegralEquations.HypergeometricFunctions: speciallog
import PowerNumbers: PowerNumber, LogNumber, realpart, epsilon, alpha


@testset "PowerNumber -> PowerNumber" begin

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

end 

@testset "Legendre Cauchy" begin
    f = Fun(exp,Legendre())

    h = 0.00001
    for z in  (PowerNumber(-1,-1,1), PowerNumber(-1,1+im,1), PowerNumber(-1,1-im,1))
        @test cauchy(f, z)(h) ≈ cauchy(f, realpart(z) + epsilon(z)h) atol=1E-4
    end
end

@testset "Directed and PowerNumber" begin
    z = Directed{false}(PowerNumber(1,-2, 1))

    for k=0:1, s=(false,true)
        z = Directed{s}(PowerNumber(-1,2, 1))
        l = stieltjesmoment(Legendre(),k,z)
        h = 0.00000001
        @test l(h) ≈ stieltjesmoment(Legendre(),k,-1 + epsilon(z.x)h + (s ? 1 : -1)*eps()*im) atol=1E-5
    end


    @test RiemannHilbert.orientedleftendpoint(ChebyshevInterval()) == PowerNumber(-1.0,1,1)
    @test RiemannHilbert.orientedrightendpoint(ChebyshevInterval()) == PowerNumber(1.0,-1,1)
end

@testset "PowerNumbers arithmetic" begin
    h = 0.000000001
    @test inv(PowerNumber(2.0,1.0,-1/2))(h) ≈ inv(h^(-1/2) + 2) rtol = 0.001
    @test inv(PowerNumber(0.0,1.0,1/2))(h) ≈ inv(h^(1/2)) rtol = 0.001
    @test inv(PowerNumber(2.0,1.0,1/2))(h) ≈ inv(h^(1/2)+2) rtol = 0.001


    @test sqrt(PowerNumber(0.0,2.0,1)) ≈ PowerNumber(0.0,sqrt(2),1/2) 
    @test sqrt(PowerNumber(0.0,2.0,1))(h) ≈ sqrt(2*h)
    @test PowerNumber(0.0,2.0,1)^(1/3) ≈ PowerNumber(0.0,2^(1/3),1/3) 
    @test (PowerNumber(0.0,2.0,1)^(1/3))(h) ≈ (2*h)^(1/3)

    @test real(PowerNumber(2im,im+1,0.5)) == PowerNumber(0,1,0.5)
    @test imag(PowerNumber(2im,im+1,0.5)) == PowerNumber(2,1,0.5)
    @test conj(PowerNumber(2im,im+1,0.5)) == PowerNumber(-2im,1-im,0.5)
    
    @test (3*PowerNumber(0.01+3im,0.2im+1,0.4) - 4*PowerNumber(0.6-1.5im,1.5-0.3im,0.4))/3 == PowerNumber(-0.79+5.0im,-1.0 + 0.6im,0.4)

    @test SingularIntegralEquations.sqrtx2(PowerNumber(1.0,2.0,1)) ≈ PowerNumber(0.0,2.0,0.5)

    x = Fun()
    f = 1/sqrt(1-x^2)
    @test stieltjes(f,PowerNumber(1.0,2.0,1))(h) ≈ stieltjes(f,1+2h) rtol = 0.001
    f = exp(x)/sqrt(1-x^2)
    @test stieltjes(f,PowerNumber(1.0,2.0,1))(h) ≈ stieltjes(f,1+2h) rtol = 0.001

    f = sqrt(1-x^2)
    @test stieltjes(f,PowerNumber(1.0,2.0,1))(h) ≈ stieltjes(f,1+2h) rtol = 0.001
    f = exp(x)*sqrt(1-x^2)
    @test stieltjes(f,PowerNumber(1.0,2.0,1))(h) ≈ stieltjes(f,1+2h) rtol = 0.001

    f = Fun(JacobiWeight(0.0,0.5,Jacobi(0.0,0.5)), [1.0])
    stieltjes(f, PowerNumber(1.0,2.0,1))
    n = 0
    β,α = 0.0,0.5; a,b,c = n+1,n+α+1,2n+α+β+2
    z = PowerNumber(1.0,2.0,1); z= 2/(1-z);
end

@testset "Interval FPStieltjes" begin
    Γ = ChebyshevInterval()
    f = Fun(x->exp(-40(x-0.1)^2), Legendre())
    C = Array{ComplexF64}(undef, ncoefficients(f), ncoefficients(f))
    d = Segment(im,2im)

    fpstieltjesmatrix!(C, space(f), d)
    c = Fun(d, chebyshevtransform(C*coefficients(f); kind=2))
    @test c(1.5im) ≈ stieltjes(f,1.5im)

    d = Segment(-1,-1+im)
    fpstieltjesmatrix!(C, space(f), d)
    @test norm(C) ≤ 100
    c = Fun(d, chebyshevtransform(C*coefficients(f); kind=2))
    @test c(-1+0.5im) ≈ stieltjes(f,-1+0.5im)

    d = Segment(1,1+im)
    fpstieltjesmatrix!(C, space(f), d)
    @test norm(C) ≤ 200
    c = Fun(d, chebyshevtransform(C*coefficients(f); kind=2))
    @test c(1+0.5im) ≈ stieltjes(f,1+0.5im)

    d = ChebyshevInterval()
    fpstieltjesmatrix!(C, space(f), d)
    @test norm(C) ≤ 200
    c = Fun(d, chebyshevtransform(C*coefficients(f); kind=2))
    @test c(0.5) ≈ stieltjes(f,0.5⁻)


    d = Segment(0,1)
    f = Fun(x->exp(-200(x-0.6)^2), Legendre(d))
    C = fpstieltjesmatrix(space(f), ncoefficients(f), ncoefficients(f))
    @test norm(C) ≤ 200
    c = Fun(d, chebyshevtransform(C*coefficients(f); kind=2))
    @test c(0.5) ≈ stieltjes(f,0.5⁻)
end

@testset "Product condition" begin
    (s1,s2,s3)=(-2im,0,2im); x=-20; n=450
    z_0 = sqrt(-x)/2
    ΓD  = Segment(-z_0, z_0)  
    Γ1  = Segment(z_0, z_0 + 2.5exp(im*π/4))
    ΓUi = Segment(z_0, z_0 + 2.5exp(im*3π/4))
    ΓU  = Segment(-z_0, -z_0 + 2.5exp(im*π/4))
    Γ3  = Segment(-z_0, -z_0 + 2.5exp(im*3π/4))
    Γ4  = Segment(-z_0, -z_0 + 2.5exp(-im*3π/4))
    ΓL  = Segment(-z_0, -z_0 + 2.5exp(-im*π/4))
    ΓLi = Segment(z_0, z_0 + 2.5exp(-im*3π/4)) 
    Γ6  = Segment(z_0, z_0 + 2.5exp(-im*π/4)) 
    Γ = ΓD ∪ Γ1 ∪ ΓUi ∪ ΓU ∪ Γ3 ∪ Γ4 ∪ ΓL ∪ ΓLi ∪ Γ6
    
    Θ(z) = 8/3*z^3+2*x*z
    
    D(z)     = [1-s1*s3 0; 0 1/(1-s1*s3)]
    S1(z)    = [1 0; s1*exp(im*Θ(z)) 1]
    Ui(z)    = [1 s3*exp(-im*Θ(z))/(1-s1*s3); 0 1]
    U(z)     = [1 -s3*exp(-im*Θ(z))/(1-s1*s3); 0 1]
    S3(z)    = [1 0; s3*exp(im*Θ(z)) 1]
    S4(z)    = [1 -s1*exp(-im*Θ(z)); 0 1]
    L(z)     = [1 0; s1*exp(im*Θ(z))/(1-s1*s3) 1]
    Li(z)    = [1 0; -s1*exp(im*Θ(z))/(1-s1*s3) 1]
    S6(z)    = [1 -s3*exp(-im*Θ(z)); 0 1]
    
    G = Fun( z -> if z in component(Γ, 1)    D(z)
              elseif z in component(Γ, 2)    S1(z)
              elseif z in component(Γ, 3)    Ui(z)
              elseif z in component(Γ, 4)    U(z)
              elseif z in component(Γ, 5)    S3(z)
              elseif z in component(Γ, 6)    S4(z)
              elseif z in component(Γ, 7)    L(z)
              elseif z in component(Γ, 8)    Li(z)
              elseif z in component(Γ, 9)    S6(z)
              end, Γ);

    @test productcondition(G)
end

@testset "realpart stieltjes" begin
    f = Fun(exp,Legendre())
    f1 = Fun(exp,Legendre(-1..0))
    f2 = Fun(exp,Legendre(0..1))
    fp = f1+f2

    @test stieltjes(f,0.0⁻) ≈ realpart(stieltjes(f1,PowerNumber(0.0,-im,1)) + stieltjes(f2,PowerNumber(0.0,-im,1)))
    @test stieltjes(f,0.0⁻) ≈ realpart(stieltjes(f1,PowerNumber(0.0,exp(-0.1im),1)) + stieltjes(f2,PowerNumber(0.0,exp(-0.1im),1)))
    @test stieltjes(f,0.0⁻) ≈ realpart(stieltjes(f1,Directed{false}(PowerNumber(0.0,-1.0,1))) + stieltjes(f2,PowerNumber(0.0,-1.0,1)))

    @test stieltjes(fp,PowerNumber(0.0,-im,1)) ≈ stieltjes(f1,PowerNumber(0.0,-im,1)) + stieltjes(f2,PowerNumber(0.0,-im,1))
    @test stieltjes(f,0.0⁻) ≈ realpart(stieltjes(fp,PowerNumber(0.0,-im,1)))
end

@testset "Two interval" begin
    @testset "-1..0 and 0..1" begin
        sp = Legendre(-1 .. 0) ⊕ Legendre(0 .. 1)
        f = Fun(x->exp(-40(x-0.1)^2), sp)
        v = components(f)
        ns = ncoefficients.(v)
        C11 = fpstieltjesmatrix(space(v[1]), ncoefficients(v[1]), ncoefficients(v[1]))
        c_vals11 = C11*v[1].coefficients
        h = 0.00000001; @test stieltjes(v[1],h*im) ≈ stieltjes(v[1],PowerNumber(0.0,im,1))(h) atol=1E-6
        @test realpart(stieltjes(v[1],Directed{false}(PowerNumber(0.0,-1,1)))) ≈ c_vals11[1]

        C22 = fpstieltjesmatrix(space(v[2]), ncoefficients(v[2]), ncoefficients(v[2]))
        c_vals22 = C22*v[2].coefficients
        h = 0.00000001; @test stieltjes(v[2],h*im) ≈ stieltjes(v[2],PowerNumber(0.0,im,1))(h) atol=1E-6
        @test realpart(stieltjes(v[2],Directed{false}(PowerNumber(0.0,1,1)))) ≈ c_vals22[end]

        C12 = fpstieltjesmatrix(space(v[2]), domain(v[1]), ncoefficients(v[1]), ncoefficients(v[2]))        
        c_vals12 = C12*v[2].coefficients
        h = 0.00000001; @test stieltjes(v[2],-h) ≈ stieltjes(v[2],PowerNumber(0.0,-1,1))(h) atol=1E-6
        @test realpart(stieltjes(v[2],PowerNumber(0.0,-1,1))) ≈ c_vals12[1]

        C = fpstieltjesmatrix(space(f), ns, ns)
        @test norm(C) ≤ 200

        @test C[1:ns[1],1:2:end] == C11

        c_vals = C*coefficients(f)
        pts = RiemannHilbert.collocationpoints(space(f), ns)

        f_ex = Fun(x->exp(-40(x-0.1)^2), Legendre())
        @test stieltjes(f_ex, Directed{false}(0.0)) ≈ realpart(stieltjes(f,Directed{false}(PowerNumber(0.0,-im,1))))
        @test realpart(stieltjes(v[1],PowerNumber(0.0,-im,1))+stieltjes(v[2],PowerNumber(0.0,-im,1))) ≈ stieltjes(f_ex,Directed{false}(0.0))

        @test c_vals[1] ≈ realpart(stieltjes(f,Directed{false}(PowerNumber(0.0,-im,1))))
        @test c_vals[1] ≈ realpart(stieltjes(f,Directed{false}(PowerNumber(0.0,exp(-0.1im),1))))
        @test c_vals[1] ≈ realpart(stieltjes(f,PowerNumber(0.0,-im,1)))
        @test c_vals[2:ns[1]-1] ≈  stieltjes.(f,pts[2:ns[1]-1]⁻)
        @test c_vals[ns[1]] ≈ realpart(stieltjes(f,PowerNumber(-1.0,-1.0,1)))

        @test c_vals[ns[1]+1] ≈ realpart(stieltjes(f,PowerNumber(1.0,1.0,1)))
        @test c_vals[ns[1]+2:end-1] ≈  stieltjes.(f,pts[ns[1]+2:end-1]⁻)
        @test c_vals[end] ≈ realpart(stieltjes(f,PowerNumber(0.0,-im,1)))

        h =0.00001
        @test stieltjes(v[1], Directed{false}(PowerNumber(0.0,-1.0,1))) ≈ stieltjes(v[1], PowerNumber(0.0,-1.0-eps()*im,1))

        @test realpart(stieltjes(v[1], Directed{false}(PowerNumber(0.0,-1.0,1)))+ stieltjes(v[2], PowerNumber(0.0,-1.0,1))) ≈
            stieltjes(v[1], -0.00000000001im)+stieltjes(v[2], -0.00000000001im)


        @test realpart(stieltjes(v[1], Directed{false}(PowerNumber(0.0,-1.0,1)))+ stieltjes(v[2], PowerNumber(0.0,-1.0,1))) ≈
            stieltjes(f, -0.00000000001im)

        @test c_vals[1] ≈ stieltjes(f, -0.0000000001im)
        @test c_vals[end] ≈ stieltjes(f, -0.0000000001im)

        @test realpart(stieltjes(f, PowerNumber(0.0,-im,1))) ≈ 
            realpart(stieltjes(v[1], PowerNumber(0.0,-im,1))+stieltjes(v[2], PowerNumber(0.0,-im,1)))


        C11 = fpstieltjesmatrix(space(v[1]), ncoefficients(v[1]), ncoefficients(v[1]))
        C12 = fpstieltjesmatrix(space(v[2]), domain(v[1]), ncoefficients(v[1]), ncoefficients(v[2]))

        @test realpart(stieltjes(v[1], Directed{false}(PowerNumber(0.0,-1.0,1)))) ≈ (C11*coefficients(v[1]))[1]
        @test realpart(stieltjes(v[2], PowerNumber(0.0,-1.0,1))) ≈ dotu(stieltjesmoment!(Array{ComplexF64}(undef,ncoefficients(v[2])), space(v[2]), orientedrightendpoint(domain(v[1])), realpart),
                    coefficients(v[2]))

        @test C12[1,:] ≈ stieltjesmoment!(Array{ComplexF64}(undef,ncoefficients(v[2])), space(v[2]), orientedrightendpoint(domain(v[1])), realpart)

        @test realpart(stieltjes(v[2], PowerNumber(0.0,-1.0,1))) ≈ (C12*coefficients(v[2]))[1]
        @test C[1,:] ≈ interlace(C11[1,:], C12[1,:])
    end

    @testset "0..1 and 0..1" begin
        sp = Legendre(Segment(0 , -1)) ⊕ Legendre(0 .. 1)

        f = Fun(x->sign(x)*exp(-40(x-0.1)^2), sp)
        v = components(f)
        ns = ncoefficients.(v)
        C = fpstieltjesmatrix(space(f), ns, ns)
        @test norm(C) ≤ 200
        c_vals = C*coefficients(f)
        pts = RiemannHilbert.collocationpoints(space(f), ns)

        @test stieltjes(f,-1-eps()) ≈ stieltjes(f,PowerNumber(-1.0,-1.0,1)).c
        @test c_vals[1] ≈ realpart(stieltjes(f,PowerNumber(-1.0,-1.0,1)))
        @test c_vals[2:ns[1]-1] ≈ stieltjes.(f,pts[2:ns[1]-1]⁻)
        @test c_vals[ns[1]] ≈ realpart(stieltjes(f,PowerNumber(0.0,+im,1)))
        @test c_vals[ns[1]+1] ≈ realpart(stieltjes(f,PowerNumber(1.0,1.0,1)))
        @test c_vals[ns[1]+2:end-1] ≈  stieltjes.(f,pts[ns[1]+2:end-1]⁻)
        @test c_vals[end] ≈ realpart(stieltjes(f,PowerNumber(0.0,-im,1)))


        h =0.00001
        @test stieltjes(v[1], Directed{false}(PowerNumber(0.0,-1.0,1))) ≈ stieltjes(v[1], PowerNumber(0.0,-1.0+eps()*im,1))
        @test stieltjes(v[1], Directed{true}(PowerNumber(0.0,-1.0,1))) ≈ stieltjes(v[1], PowerNumber(0.0,-1.0-eps()*im,1))

        @test stieltjes(v[2], PowerNumber(0.0,-1.0,1))(h) ≈ stieltjes(v[2], -h) atol=1E-3

        @test stieltjes(v[1], Directed{true}(PowerNumber(0.0,-1.0,1)))(h) ≈ stieltjes(v[1], -h-eps()*im) atol=1E-3


        @test realpart(stieltjes(v[1], Directed{true}(PowerNumber(0.0,-1.0,1)))+ stieltjes(v[2], PowerNumber(0.0,-1.0,1))) ≈
            stieltjes(v[1], -0.00000000001im)+stieltjes(v[2], -0.00000000001im)


        @test realpart(stieltjes(v[1], Directed{true}(PowerNumber(0.0,-1.0,1)))+ stieltjes(v[2], PowerNumber(0.0,-1.0,1))) ≈
            stieltjes(f, -0.00000000001im)

        @test c_vals[ns[1]] ≈ stieltjes(f, +0.0000000001im)
        @test c_vals[end] ≈ stieltjes(f, -0.0000000001im)
    end

    @testset "-1..0 and 0..1" begin
        sp = Legendre(-1 .. 0) ⊕ Legendre(0 .. 1)
        f = Fun(x->exp(-40(x-0.1)^2), sp)
        v = components(f)
        ns = ncoefficients.(v)
        C = fpstieltjesmatrix(space(f), ns, ns)
        @test norm(C) ≤ 200

        c_vals = C*coefficients(f)
        pts = RiemannHilbert.collocationpoints(space(f), ns)

        @test c_vals[1] ≈ realpart(stieltjes(f,PowerNumber(0.0,-im,1)))
        @test c_vals[2:ns[1]-1] ≈  stieltjes.(f,pts[2:ns[1]-1]⁻)
        @test c_vals[ns[1]] ≈ realpart(stieltjes(f,PowerNumber(-1.0,-1.0,1)))
        @test c_vals[ns[1]+1] ≈ realpart(stieltjes(f,PowerNumber(1.0,1.0,1)))
        @test c_vals[ns[1]+2:end-1] ≈  stieltjes.(f,pts[ns[1]+2:end-1]⁻)
        @test c_vals[end] ≈ realpart(stieltjes(f,PowerNumber(0.0,-im,1)))
    end
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
    g_v = RiemannHilbert.collocationvalues(g-1, n)
    u = Fun(sp, rhmatrix(g,n) \ g_v)

    @testset "-1 .. 1" begin
        n = 2ncoefficients(g)
        g = pad(g,n)
        C₋ = fpcauchymatrix(sp, n, n)
        pts = RiemannHilbert.collocationpoints(sp, n)
        @test C₋[2:end-1,:]*coefficients(g) ≈ cauchy.(g, pts[2:end-1]⁻)
        g_v = RiemannHilbert.collocationvalues(g-1, n)
        @test g_v ≈ g.(pts).-1
        G = Diagonal(g_v)

        @test G*g_v ≈ (g.(pts) .- 1).^2

        E = RiemannHilbert.evaluationmatrix(sp, pts, length(pts))
        @test E*coefficients(g) ≈ g.(pts)

        @test (g.(pts).-1).*(C₋*coefficients(g)) ≈ (g.(pts).-1).*cauchy.(g, pts.-0.000000001im)
        @test G*(C₋*coefficients(g)) ≈ (g.(pts).-1).*cauchy.(g, pts.-0.000000001im)

        L = E - G*C₋
        @test L*coefficients(g) ≈ g.(pts) - (g.(pts).-1).*cauchy.(g, pts.-0.000000001im)

        @test L == rhmatrix(g,n)

        
        φ = z -> 1 + cauchy(u,z)
        @test φ(0.1⁺)  ≈ g(0.1)φ(0.1⁻)


        @test 1+cauchy(u)(0.1+0.2im) ≈ φ(0.1+0.2im)
        @test (1+cauchy(u) )(0.1+0.2im) ≈ φ(0.1+0.2im)
        @test (1+cauchy(u) )(0.1⁻) ≈ φ(0.1⁻)
    end

    @testset "-1 .. 0 and 0 .. 1" begin
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

        @test transpose(C₋[1,:])*coefficients(u_ex) ≈ cauchy(u_ex,pts[1]-eps()im)
        @test transpose(C₋[5,:])*coefficients(u_ex) ≈ cauchy(u_ex,pts[5]-eps()im)
        @test transpose(C₋[end,:])*coefficients(u_ex) ≈ cauchy(u_ex, 0.0-eps()im)
        @test transpose(C₋[end-1,:])*coefficients(u_ex) ≈ cauchy(u_ex,pts[end-1]-eps()im)
        @test transpose(C₋[n÷2,:])*coefficients(u_ex) ≈ cauchy(u_ex,-1.0-eps())
        @test transpose(C₋[(n÷2)+1,:])*coefficients(u_ex) ≈ cauchy(u_ex,1.0+eps())

        @test C₋*coefficients(u_ex) ≈ cauchy.(u_ex, pts.-eps()im)
        g_v = RiemannHilbert.collocationvalues(g-1, n)
        @test g_v ≈ g.(pts).-1
        G = Diagonal(g_v)

        @test G*g_v ≈ (g.(pts) .- 1).^2

        g1 = component(g,2)
        pts1 = RiemannHilbert.collocationpoints(component(sp,2), n÷2)
        E1=RiemannHilbert.evaluationmatrix(component(sp,1), length(pts1))

        @test E1*g1.coefficients ≈ g1.(pts1)

        E = RiemannHilbert.evaluationmatrix(sp, n)
        @test E*coefficients(g) ≈ g.(pts)

        @test (g.(pts).-1).*(C₋*coefficients(g)) ≈ (g.(pts).-1).*cauchy.(g, pts.-0.000000001im)
        @test G*(C₋*coefficients(g)) ≈ (g.(pts).-1).*cauchy.(g, pts.-0.000000001im)

        L = E - G*C₋
        @test L*coefficients(g) ≈ g.(pts) - (g.(pts).-1).*cauchy.(g, pts.-0.000000001im)

        @test L == rhmatrix(g,n)

        u = Fun(sp, rhmatrix(g,n) \ g_v)
        φ = z -> 1 + cauchy(u,z)
        @test φ(0.1⁺)  ≈ g(0.1)φ(0.1⁻)

        @test 1+cauchy(u)(0.1+0.2im) ≈ φ(0.1+0.2im)
        @test (1+cauchy(u) )(0.1+0.2im) ≈ φ(0.1+0.2im)
        @test (1+cauchy(u) )(0.1⁻) ≈ φ(0.1⁻)

        @time φ = rhsolve(g, 2ncoefficients(g))
        @test φ(0.1⁺)  ≈ g(0.1)φ(0.1⁻)
    end

    @testset "0..-1 and 0..1" begin
        sp = Legendre(Segment(0 , -1)) ∪ Legendre(0 .. 1)
        g = Fun(x->x ≥ 0 ? 1-0.3exp(-40x^2) : inv(1-0.3exp(-40x^2)), sp)
        u_1 = u
        u_ex = Fun(x->sign(x)u_1(x), sp)

        n = 2ncoefficients(g)
        g = pad(g,n)
        u_ex = pad(u_ex,n)
        @test (1+cauchy(u_ex,0.1⁺)) ≈ g(0.1)*(1+cauchy(u_ex,0.1⁻))

        C₋ = fpcauchymatrix(sp, n, n)
        pts = RiemannHilbert.collocationpoints(sp, n)

        @test transpose(C₋[1,:])*coefficients(u_ex) ≈ cauchy(u_ex,pts[1]+eps()im)
        @test transpose(C₋[5,:])*coefficients(u_ex) ≈ cauchy(u_ex,pts[5]+eps()im)
        @test transpose(C₋[n÷2,:])*coefficients(u_ex) ≈ cauchy(u_ex,0.0+eps()im)
        @test transpose(C₋[(n÷2)+1,:])*coefficients(u_ex) ≈ cauchy(u_ex,1.0+eps())
        @test transpose(C₋[end,:])*coefficients(u_ex) ≈ cauchy(u_ex, pts[end]-eps()im)
        @test transpose(C₋[end,:])*coefficients(u_ex) ≈ cauchy(u_ex, 0.0-eps()im)
        @test transpose(C₋[end-1,:])*coefficients(u_ex) ≈ cauchy(u_ex,pts[end-1]-eps()im)

        @test C₋*coefficients(u_ex) ≈ [cauchy.(u_ex, pts[1:n÷2].+eps()im); cauchy.(u_ex, pts[(n÷2)+1:end].-eps()im)]
        g_v = RiemannHilbert.collocationvalues(g-1, n)

        g1 = component(g,1)
        pts1 = RiemannHilbert.collocationpoints(component(sp,1), n÷2)
        E1=RiemannHilbert.evaluationmatrix(component(sp,1), length(pts1))

        @test E1*g1.coefficients ≈ g1.(pts1)

        g2 = component(g,2)
        pts2 = RiemannHilbert.collocationpoints(component(sp,2), n÷2)

        @test g_v ≈ [g1.(pts1).-1; g2.(pts2).-1]
        G = Diagonal(g_v)

        @test g1.(pts1) ≈ g.(pts1)
        @test_broken g2.(pts2) ≈ g.(pts2) # evaluation at 0
        @test_broken G*g_v ≈ (g.(pts) .- 1).^2

        E = RiemannHilbert.evaluationmatrix(sp, n)
        @test E[1:end-1,:]*coefficients(g) ≈ g.(pts[1:end-1])
        @test transpose(E[end,:])*coefficients(g) ≈ component(g,2)(0.0)

        @test (g2.(pts2).-1).*(C₋[(n÷2)+1:end,:]*coefficients(u_ex)) ≈
            (g2.(pts2).-1).*cauchy.(u_ex, pts2.-eps()im)
        @test (g.(pts1).-1).*(C₋[1:n÷2,:]*coefficients(u_ex)) ≈
            (g.(pts1).-1).*cauchy.(u_ex, pts1.+0.000000001im)
        @test (g.(pts2).-1).*(C₋[(n÷2)+1:end,:]*coefficients(u_ex)) ≈
            (g.(pts2).-1).*cauchy.(u_ex, pts2.-0.000000001im)

        u_ex1,u_ex2 = components(u_ex)
        @test (C₋*coefficients(u_ex))[ncoefficients(u_ex1)+1:end] ≈ cauchy.(u_ex, pts2.-eps()im)
        @test G*C₋*coefficients(u_ex) ≈ [(g1.(pts1).-1).*cauchy.(u_ex, pts1.+eps()im);
                                         (g2.(pts2).-1).*cauchy.(u_ex, pts2.-eps()im)]

        L = E - G*C₋
        @test L*coefficients(u_ex) ≈ [u_ex1.(pts1) - (g1.(pts1).-1).*cauchy.(u_ex, pts1.+eps()im);
                                        u_ex2.(pts2) - (g2.(pts2).-1).*cauchy.(u_ex, pts2.-eps()im)]

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
    end

    @testset "Chebyshev g" begin
        G = Fun(x -> 1 + exp(-30x^2), -1..1)
        Φ = rhsolve(G, 200)
        @test Φ(0.1⁺) ≈ G(0.1)Φ(0.1⁻)
    end
end

@testset "Matrix rhsolve" begin
    sp = ArraySpace(Legendre(), 2)
    f = Fun(Fun(x->[cos(x);sin(x)], Chebyshev()), ArraySpace(Legendre(), 2))
    G = Fun(Fun(x->[1 exp(-40x^2); 0.1exp(-40x^2) 1], Chebyshev()), ArraySpace(Legendre(), 2, 2))

    n = 2ncoefficients(G)
    E = RiemannHilbert.evaluationmatrix(sp, n)
    pts = RiemannHilbert.collocationpoints(sp, n÷2)

    @test E*coefficients(pad(f,n)) ≈ [f[1].(pts); f[2].(pts)]


    M = RiemannHilbert.multiplicationmatrix(G-I, n)
    @test M*(E*coefficients(pad(f,n))) ≈ mapreduce(f -> f.(pts), vcat, (G-I)*f)

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
end

@testset "Matrix rhsolve two interval" begin
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

    sp = ArraySpace(Legendre(Segment(0 , -1)) ∪ Legendre(0 .. 1), 2)
    G = g = Fun(x-> x ≥ 0 ? [1 exp(-40x^2); 0.1exp(-40x^2) 1] : inv([1 exp(-40x^2); 0.1exp(-40x^2) 1]), ArraySpace(sp[1], 2, 2))
    n=2ncoefficients(G)
    Ũ1 = Fun(x-> x ≥ 0 ? U1(x) : -U1(x), sp, n÷2)
    @test cauchy(Ũ1, 0.1⁻) ≈ cauchy(Ũ1, 0.1-eps()im)
    @test cauchy(Ũ1, 0.1+eps()im) ≈ cauchy(U1, 0.1+eps()im)
    @test cauchy(Ũ1, -0.1+eps()im) ≈ cauchy(U1, -0.1+eps()im)

    L = rhmatrix(G, n)
    rhs = RiemannHilbert.collocationvalues((g-I)[:,1], n)
    @test rhs[end] ≈ 0.1

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

    @test last(G*C₋*coefficients(Ũ1)) ≈ transpose((g(0.00000000001)-I)[2,:])*cauchy(Ũ1, 0.00000000001-eps()*im)

    @test transpose(L[end,:])*coefficients(Ũ1) ≈ Ũ1(0.00000000001)[2] - transpose((g(0.00000000001)-I)[2,:])*cauchy(Ũ1, 0.00000000001-eps()*im)

    @test U1(0.00000000001)[2] - transpose((g(0.00000000001)-I)[2,:])*cauchy(U1, 0.00000000001-eps()*im) ≈ 0.1
    @test g(0.1)*Φ(0.1⁻) ≈ Φ(0.1⁺)
end

@testset "4 rays" begin
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

    Φ = transpose(rhsolve(transpose(G), 2*4*100))

    s=exp(im*π/6)
    @test Φ((s)⁻)*G(s) ≈ Φ((s)⁺)
    @test map(g->first(components(g)[1]), G)*map(g->first(components(g)[2]), G)*
        map(g->first(components(g)[3]), G)*map(g->first(components(g)[4]), G) ≈ Matrix(I,2,2)

    for _=1:5
        @test RiemannHilbert.evaluationmatrix(sp[:,1], 32) ≈ RiemannHilbert.evaluationmatrix(sp[:,1], 32)
        @test RiemannHilbert.rhmatrix(G, 32) ≈ RiemannHilbert.rhmatrix(G, 32)
    end

    @test RiemannHilbert.rhmatrix(transpose(G), 900) ≈ RiemannHilbert.rhmatrix(transpose(G), 900)
    U = RiemannHilbert.rh_sie_solve(transpose(G), 2*4*100)
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

    Ũ = RiemannHilbert.rh_sie_solve(transpose(G̃), 2*4*100)

    @test rhmatrix(transpose(G), 2*4*100) ≈ rhmatrix(transpose(G̃), 2*4*100)
    @test -0.36706155154807807 ≈ sum(Ũ[1,2])/(-π*im)

    V = istieltjes(Φ)
    x = Fun(domain(V))

    @test 10.0I + sum.(Array(V)) + stieltjes(x*V, 10.0) ≈ 10.0Φ(10.0)


    # check rhmatrix relationship for below
    U = V*(-2π*im)
    U1 = U[1,:]
    n= ncoefficients(U1)
    L = rhmatrix(transpose(G),n)
    vals = collocationvalues(transpose(G)-I, n)
    @test L*coefficients(U1) ≈ vals[:,1]
end


@testset "6 rays" begin
    @testset "HM on 6 rays" begin
        s₁,s₂,s₃ = -im,0,im
        @assert s₁ - s₂ + s₃ + s₁*s₂*s₃ ≈ 0

        # construct true solution using 4 rays
        Γ = Segment(0, 2.5exp(im*π/6)) ∪ Segment(0, 2.5exp(5im*π/6)) ∪
        Segment(0, 2.5exp(-5im*π/6)) ∪ Segment(0, 2.5exp(-im*π/6))
        sp = ArraySpace(PiecewiseSpace(Legendre.(components(Γ))), 2,2)

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

        Φ = transpose(rhsolve(transpose(G), 2*6*100))


        # debug
        V4 = istieltjes(Φ)
        x = 0.0

        @test stieltjes(V4,1+im)+I == Φ(1+im)


        Γ = Segment(0, 2.5exp(im*π/6))   ∪
        Segment(0, 2.5exp(im*π/2))       ∪
        Segment(0, 2.5exp(5im*π/6))      ∪
        Segment(0, 2.5exp(-5im*π/6))     ∪
        Segment(0, 2.5exp(-im*π/2))      ∪
        Segment(0, 2.5exp(-im*π/6));

        G = Fun( z -> if angle(z) ≈ π/6
                        [1                             0;
                        s₁*exp(8im/3*z^3+2im*x*z)     1]
                    elseif angle(z) ≈ π/2
                        [1                 s₂*exp(-8im/3*z^3-2im*x*z);
                        0                 1]
                    elseif angle(z) ≈ 5π/6
                        [1                             0;
                        s₃*exp(8im/3*z^3+2im*x*z)     1]
                    elseif angle(z) ≈ -π/6
                        [1                -s₃*exp(-8im/3*z^3-2im*x*z);
                        0                 1]
                    elseif angle(z) ≈ -π/2
                        [1                             0;
                        -s₂*exp(8im/3*z^3+2im*x*z)    1]
                    elseif angle(z) ≈ -5π/6
                        [1                -s₁*exp(-8im/3*z^3-2im*x*z);
                        0                 1]
                    end
                        , Γ);
        sp = ArraySpace(PiecewiseSpace(Legendre.(components(Γ))), 2,2)
        V = Fun(V4, sp)

        @test stieltjes(V,1+im)+I ≈ Φ(1+im)
        
        U = V*(-2π*im)

        U1 = U[1,:]
        @test cauchy(U1,1+im)+[1,0] ≈ Φ(1+im)[1,:]
        @test abs(sum(first.(components(U1[1])))) ≤ 100eps()
        @test abs(sum(first.(components(U1[2])))) ≤ 100eps()


        U11 = U1[1]
        n = ncoefficients(U11)
        C₋ = fpcauchymatrix(space(U11), n, n)
        pts = collocationpoints(space(U11), n)
        c = C₋*coefficients(U11)
        
        @test c[1] ≈ realpart(cauchy(U11,orientedrightendpoint(component(Γ,1))))
        @test c[2] ≈ cauchy(U11,pts[2])
        @test c[150] ≈ realpart(cauchy(U11,orientedleftendpoint(component(Γ,1))⁻))
        @test c[151] ≈ realpart(cauchy(U11,orientedrightendpoint(component(Γ,2))))

        n = ncoefficients(U1)
        L = rhmatrix(transpose(G),n)
        vals = collocationvalues(transpose(G)-I, n)
        pts = collocationpoints(space(U1),n)

        c = L*coefficients(U1)
        @test c ≈ vals[:,1]

        @test coefficients(U1) ≈ L \ vals[:,1]
    end
end

include("test\\test_nls.jl")