using ContinuumArrays, RiemannHilbert, DomainSets, CairoMakie, Test


x = 0.1
s1,s2,s3 = 1+im,-2,1-im

# cyclic condition
@test s1 - s2 + s3 + s1*s2*s3 == 0

Θ(z) = 8/3*z^3+2*x*z
G = expand([1 0; s1*exp(im*Θ(z)) 1] for z in Segment(0, 2.5exp(im*π/6))) ⊎ 
    expand([1 s2*exp(-im*Θ(z)); 0 1] for z in Segment(0, 2.5exp(im*π/2))) ⊎ 
    expand([1 0; s3*exp(im*Θ(z)) 1] for z in Segment(0, 2.5exp(5im*π/6))) ⊎ 
    expand([1 -s1*exp(-im*Θ(z)); 0 1] for z in Segment(0, 2.5exp(-5im*π/6))) ⊎ 
    expand([1 0; -s2*exp(im*Θ(z)) 1] for z in Segment(0, 2.5exp(-im*π/2))) ⊎ 
    expand([1 -s3*exp(-im*Θ(z)); 0 1] for z in Segment(0, 2.5exp(-im*π/6))) 

    
# product condition
@test prod(first.(components(G))) ≈ I

n = 100
Φ = rhsolve(G, n)

z = 0.1exp(im*π/6)
@test Φ(z * ⁺) ≈ Φ(z * ⁻)G[z]