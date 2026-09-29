using ContinuumArrays, RiemannHilbert, DomainSets, CairoMakie, ComplexPhasePortrait, Test


x = 0.2
s₁,s₂,s₃ = 1+im,-2,1-im

# cyclic condition
@test s₁ - s₂ + s₃ + s₁*s₂*s₃ == 0

Θ(z) = 8/3*z^3+2*x*z
G = [[1 0; s₁*exp(im*Θ(z)) 1] for z in Segment(0, 2.5exp(im*π/6))] ⊎
    [[1 s₂*exp(-im*Θ(z)); 0 1] for z in Segment(0, 2.5exp(im*π/2))] ⊎
    [[1 0; s₃*exp(im*Θ(z)) 1] for z in Segment(0, 2.5exp(5im*π/6))] ⊎
    [[1 -s₁*exp(-im*Θ(z)); 0 1] for z in Segment(0, 2.5exp(-5im*π/6))] ⊎
    [[1 0; -s₂*exp(im*Θ(z)) 1] for z in Segment(0, 2.5exp(-im*π/2))] ⊎
    [[1 -s₃*exp(-im*Θ(z)); 0 1] for z in Segment(0, 2.5exp(-im*π/6))]


# product condition
@test prod(first.(components(G))) ≈ I

n = 100
Φ = rhsolve(G, n)

z = 0.1exp(im*π/6)
@test Φ(z * ⁺) ≈ Φ(z * ⁻)G[z]


# phase portraits of the entries of Φ, with the jump contour
xs = range(-1, 1; length=400)
Z = xs' .+ xs*im # portrait puts the last row at the top, so Im z increases with the row index
Φz = Φ.(Z)

fig = Figure(size=(800, 800))
for k = 1:2, j = 1:2
    ax = Axis(fig[k,j]; aspect=DataAspect(), title="Φ[$k,$j]")
    image!(ax, -1..1, -1..1, rotr90(portrait(getindex.(Φz, k, j))))
    plot!(ax, domain(G); color=:black)
    limits!(ax, -1, 1, -1, 1)
end
fig
