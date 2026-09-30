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


###
# Negative x
###
#
# For x < 0 the stationary points of Θ, where Θ'(z) = 8z^2 + 2x = 0, are the real points ±z₀ with z₀ = √(-x)/2,
# and the jumps on the rays at angles ±π/6 and ±5π/6 grow exponentially between the origin and ±z₀ as x → -∞.
# We deform these rays to start at ±z₀. Writing S₁,…,S₆ for the jumps on the rays in the order above, in the
# strip between the ray from 0 and the ray from ±z₀ we replace Φ by ΦS₁, ΦS₃⁻¹, ΦS₄ and ΦS₆⁻¹ respectively.
# The rays from 0 then have no jump, the rays from ±z₀ have the same jumps as before, and the strips introduce
# the jumps S₆S₁ on [0, z₀] and S₄⁻¹S₃⁻¹ on [-z₀, 0]. All jumps off the real axis now decay exponentially,
# while those on the real axis are bounded and oscillatory. The deformed solution equals Φ outside the strips,
# in particular as z → ∞, so it gives the same solution of Painlevé II, u(x) = 2 lim_{z → ∞} z Φ₁₂(z).
#
# We use Stokes multipliers with s₃ = conj(s₁) and s₂ real, which give a real solution, and |s₁| < 1, so that
# it decays in an oscillatory manner as x → -∞ with amplitude d(-x)^(-1/4), where d^2 = -log(1-|s₁|^2)/π.

s₁ = 0.3+0.4im
s₃ = conj(s₁)
s₂ = (s₁ + s₃)/(1 - s₁*s₃) # from the cyclic condition
@test s₁ - s₂ + s₃ + s₁*s₂*s₃ ≈ 0 atol=1E-15

θ(x, z) = 8/3*z^3 + 2x*z
L(s, x, z) = [1 0; s*exp(im*θ(x, z)) 1]
U(s, x, z) = [1 s*exp(-im*θ(x, z)); 0 1]

# the jumps S₁,…,S₆ on rays of length R from the origin, as above
painleveII_jump(x; R=2.5) =
    [L(s₁, x, z) for z in Segment(0, R*exp(im*π/6))] ⊎
    [U(s₂, x, z) for z in Segment(0, R*exp(im*π/2))] ⊎
    [L(s₃, x, z) for z in Segment(0, R*exp(5im*π/6))] ⊎
    [U(-s₁, x, z) for z in Segment(0, R*exp(-5im*π/6))] ⊎
    [L(-s₂, x, z) for z in Segment(0, R*exp(-im*π/2))] ⊎
    [U(-s₃, x, z) for z in Segment(0, R*exp(-im*π/6))]

# the deformed jumps, with the rays at angles ±π/6 and ±5π/6 starting at ±z₀
function deformed_painleveII_jump(x; R=2.5)
    z₀ = sqrt(-x)/2
    [L(s₁, x, z) for z in Segment(z₀, z₀ + R*exp(im*π/6))] ⊎
    [U(s₂, x, z) for z in Segment(0, R*exp(im*π/2))] ⊎
    [L(s₃, x, z) for z in Segment(-z₀, -z₀ + R*exp(5im*π/6))] ⊎
    [U(-s₁, x, z) for z in Segment(-z₀, -z₀ + R*exp(-5im*π/6))] ⊎
    [L(-s₂, x, z) for z in Segment(0, R*exp(-im*π/2))] ⊎
    [U(-s₃, x, z) for z in Segment(z₀, z₀ + R*exp(-im*π/6))] ⊎
    [U(-s₃, x, z)*L(s₁, x, z) for z in Segment(0.0, z₀)] ⊎
    [inv(U(-s₁, x, z))*inv(L(s₃, x, z)) for z in Segment(-z₀, 0.0)]
end

# Φ = I + 𝒞U where z𝒞U(z) → -∫U/(2πi) as z → ∞, so u(x) = 2 lim_{z → ∞} z Φ₁₂(z) = -∫U₁₂/(πi)
painleveII(G, n) = -sum(RiemannHilbert.rh_sie_solve(G, n)[1,2])/(π*im)

n = 100
x = -2.0
G = deformed_painleveII_jump(x)
Ψ = rhsolve(G, n)
@test Ψ(2im) ≈ rhsolve(painleveII_jump(x), n)(2im) # the solutions agree outside the strips
z = 0.3
@test Ψ(z * ⁺) ≈ Ψ(z * ⁻)G[z]

u = x -> real(painleveII(deformed_painleveII_jump(x), n))

# u satisfies Painlevé II, u'' = xu + 2u^3
x, h = -10.0, 0.01
@test (u(x+h) - 2u(x) + u(x-h))/h^2 ≈ x*u(x) + 2u(x)^3 rtol=1E-3

xs = range(-20, -0.1; length=200)
d = sqrt(-log(1-abs2(s₁))/π)
fig_u = Figure()
ax = Axis(fig_u[1,1]; xlabel="x", ylabel="u(x)")
lines!(ax, xs, u.(xs))
lines!(ax, xs, d*(-xs).^(-1/4); color=:gray, linestyle=:dash)
lines!(ax, xs, -d*(-xs).^(-1/4); color=:gray, linestyle=:dash)
ylims!(ax, -0.4, 0.4)
fig_u
