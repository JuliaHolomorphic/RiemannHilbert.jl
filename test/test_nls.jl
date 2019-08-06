using RiemannHilbert, ApproxFun, ApproxFunBase, LinearAlgebra, SingularIntegralEquations, Test
import ApproxFunBase: pieces

@testset "NLS" begin
    x,t = 0.0,0.0
    r = k -> 0.5exp(-k^2)
    r̄ = k -> conj(r(conj(k)))
    θ = k -> 4t*k^2+2x*k
    Gf = k -> [1-abs2(r(k))                      -conj(r(k))*exp(-im*θ(k));
                    r(k)*exp(im*θ(k))      1  ]

    g = Fun(Gf, -5.5..5.5)                
    @test first(g) ≈ last(g) ≈ [1 0; 0 1]
    φ = transpose(rhsolve(transpose(g), 500))
    istieltjes(φ)(0.1)
    @test φ(1+im) ≈ [0.9847854664616504-0.014938097892888065im -0.0778564768587141-0.05302708037085426im;
                0.07824233623082648+0.05395699475793745im 1.0120649351483406+0.006873030868410332im] # mathematica

    x,t = 1.0,1.0

    g = Fun(Gf, -5.5..5.5)                
    @test first(g) ≈ last(g) ≈ [1 0; 0 1]
    φ = transpose(rhsolve(transpose(g), 1000))
    @test φ(1+im) ≈ [0.9952708020858667-0.00828828357987084im -0.036773740429971044-0.01579754685573759im;
                0.009328821936648003+0.04514458247615043im 1.0050689684367242+ 0.006553783105674419im]

    @test norm(φ.coefficients[end-50:end]) ≤ 1E-14
    @test φ(0.1⁺) ≈ φ(0.1⁻)g(0.1)

    z₀ = -x/(4t)
    Γ = Segment(z₀,z₀+3exp(3im*π/4)) ∪ Segment(z₀,-5.5) ∪ Segment(z₀,z₀+3exp(-3im*π/4)) ∪ Segment(z₀,z₀+3exp(-im*π/4)) ∪ Segment(z₀,z₀+3exp(im*π/4)) 

    U = z -> [1 -r̄(z)/(1-r(z)r̄(z))*exp(-im*θ(z)); 0 1]
    D = z -> [(1-r(z)r̄(z)) 0; 0 1/(1-r(z)r̄(z))]
    L = z -> [1 0; r(z)/(1-r(z)r̄(z))*exp(im*θ(z)) 1]
    M = z -> [1 -r̄(z)*exp(-im*θ(z)); 0 1]
    P = z -> [1 0; r(z)*exp(im*θ(z)) 1]
    G = Fun(function(z)        
            z ∈ component(Γ,1) && return inv(U(z))
            z ∈ component(Γ,2) && return inv(D(z))
            z ∈ component(Γ,3) && return inv(L(z))
            z ∈ component(Γ,4) && return M(z)
            z ∈ component(Γ,5) && return P(z)
            error("Not in contour") 
        end, Γ)

    @test prod(first.(pieces(G))) ≈ [1 0; 0 1]
    @test norm(last.(pieces(G)) .- Ref(I)) ≤ 1E-14

    Φ = transpose(rhsolve(transpose(G), 3000))
    @test norm(Φ.coefficients[end-50:end]) ≤ 10E-14

    @test Φ(2im) ≈ φ(2im)

    z = Fun(ℂ)          
    2im*(z*Φ[1,2])(Inf)≈ 2im*(z*φ[1,2])(Inf) 


    # make evaluation faster
    φ2 = Matrix(φ)
    φ̃ = z -> [φ2[1,1](z) φ2[1,2](z); φ2[2,1](z) φ2[2,2](z)]

    @time V = Fun(function(z)        
            z ∈ component(Γ,1) && return φ̃(z)*(inv(U(z))-I)
            z ∈ component(Γ,2) && return φ̃(z-10eps()im)L(z)-φ̃(z+10eps()im)inv(U(z))
            z ∈ component(Γ,3) && return φ̃(z)*(I-L(z))
            z ∈ component(Γ,4) && return φ̃(z)*(M(z)-I)
            z ∈ component(Γ,5) && return φ̃(z)*(I-inv(P(z)))
            error("Not in contour") 
        end, Γ, 1000)

    @test norm(V.coefficients[end-20:end]) ≤ 1E-15
    @test P(z₀+3exp(im*π/4)) ≈ [1 0; 0 1]
    Φ = I+cauchy(V)
    Φ̃ = function(z) 
            3π/4  < angle(z-z₀) < π     && return φ(z)inv(U(z))
            -π    < angle(z-z₀) < -3π/4 && return φ(z)L(z)
            -3π/4 < angle(z-z₀) < -π/4  && return φ(z)
            -π/4  < angle(z-z₀) <  0    && return φ(z)M(z)
            0    < angle(z-z₀) <  π/4  && return φ(z)inv(P(z))
            π/4  < angle(z-z₀) < 3π/4  && return φ(z)
            error("Not defined")
        end

    @test Φ̃(0.1+10eps()im) ≈ Φ̃(0.1-10eps()im)

    z = z₀+0.1exp(im*π/4); @test Φ(z-10eps())-Φ(z+10eps()) ≈ Φ̃(z-10eps())-Φ̃(z+10eps()) ≈ φ̃(z)*(I-inv(P(z)))
    z = z₀+0.1exp(3im*π/4); @test Φ(z-10eps())-Φ(z+10eps()) ≈ Φ̃(z-10eps())-Φ̃(z+10eps()) ≈ φ̃(z)*(inv(U(z))-I)
    z = z₀-0.1+0im; @test Φ(z-10im*eps())-Φ(z+10im*eps()) ≈ Φ̃(z-10im*eps())-Φ̃(z+10im*eps()) ≈ φ̃(z-10im*eps())*L(z)-φ̃(z+10im*eps())*inv(U(z))
    z = z₀+0.1exp(-3im*π/4); @test Φ(z+10eps())-Φ(z-10eps()) ≈ Φ̃(z+10eps())-Φ̃(z-10eps()) ≈ φ̃(z)*(I-L(z))
    z = z₀+0.1exp(-im*π/4); @test Φ(z+10eps())-Φ(z-10eps()) ≈ Φ̃(z+10eps())-Φ̃(z-10eps()) ≈ φ̃(z)*(M(z)-I)
    z = z₀+0.1+0im; @test Φ(z-10im*eps()) ≈ Φ(z+10im*eps()) ≈ Φ̃(z-10im*eps()) ≈ Φ̃(z+10im*eps()) 
end