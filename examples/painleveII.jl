using ContinuumArrays, RiemannHilbert, DomainSets, CairoMakie, ComplexPhasePortrait, Test


# Ablowitz–Segur solution: s₂ = 0
x = 0.2
s₁,s₂,s₃ = im,0,-im

# cyclic condition
@test s₁ - s₂ + s₃ + s₁*s₂*s₃ == 0

Θ(z) = 8/3*z^3+2*x*z
G = [[1 0; s₁*exp(im*Θ(z)) 1] for z in Segment(0, 2.5exp(im*π/6))] ⊎
    [[1 0; s₃*exp(im*Θ(z)) 1] for z in Segment(0, 2.5exp(5im*π/6))] ⊎
    [[1 -s₁*exp(-im*Θ(z)); 0 1] for z in Segment(0, 2.5exp(-5im*π/6))] ⊎
    [[1 -s₃*exp(-im*Θ(z)); 0 1] for z in Segment(0, 2.5exp(-im*π/6))]

plot(domain(G))


# product condition, ensures that the solution is bounded near 0
h = 0.00000001
@test G[h*exp(im*π/6)] * G[h*exp(5im*π/6)] * G[h*exp(-5im*π/6)]* G[h*exp(-im*π/6)] ≈ I
@test prod(first.(components(G))) ≈ I

# without it Φ will normally have algebraic singularities,
#  you can use a parametrix to remove the singularity.

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

V = RiemannHilbert.rh_sie_solve(G, 200)

using SingularIntegrals
z = 1+2im
@test I + cauchy.(V, z) ≈ Φ(z)

cauchy(V[1,2],0.0000001) # bounded near origin
v_1 = components(V[1,2])[1] # ≈ 0 because no jump on (0,∞*exp(im*π/6))
v_2 = components(V[1,2])[2] # ≈ 0 because no jump on (0,∞*exp(5im*π/6))
v_3 = components(V[1,2])[3]
v_4 = components(V[1,2])[4]
p = plot(domain(v_1))
plot!(domain(v_2))
plot!(domain(v_3)); p
plot!(domain(v_4)); p
cauchy(v_1,0.0000001) 
cauchy(v_2,0.0000001)
cauchy(v_3,0.000000000001) # blows up! logarithmically
cauchy(v_4,0.000000000001) # blows up! logarithmically

cauchy(v_3,0.000000000001) + cauchy(v_4,0.000000000001) # blow up cancels


# product condition on G => sum condition on V => no blow up
# This is because the values actually cancel: we have a zero sum condition

@test v_1[0] + v_2[0] + v_3[0] + v_4[0] ≈ 0 atol=1E-10

using PowerNumbers, ClassicalOrthogonalPolynomials

z_1 = exp(im*π/6) * ϵ # like a dual number
z_2 = exp(5im*π/6) * ϵ # like a dual number
z_3 = exp(-5im*π/6) * ϵ # like a dual number
z_4 = exp(-im*π/6) * ϵ # like a dual number

P̃ = legendre(Segment(0,2.5exp(im*π/6))) # mapped Legendre
cauchy(P̃, z_2)[1] # behaviour of C[P_0 ∘ M^{-1}](z) near 0 from direction exp(im*5π/6)
cauchy(P̃, z_3)[1] # behaviour of C[P_0 ∘ M^{-1}](z) near 0 from direction exp(-im*5π/6)
cauchy(P̃, z_4)[1] # behaviour of C[P_0 ∘ M^{-1}](z) near 0 from direction exp(-im*5π/6)

cauchy(P̃, z_1*exp(im*h))[1]  # has a branch cut 
cauchy(P̃, z_1*exp(-im*h))[1] 



# log part is same regardless of direction, finite contributions vary
# automatic-differentiation-like implementation gives us values of each

# when we set up the collocation system, we ignore the logarithmic part for
# each collocation point associated with junction. The magic is that under
# broad conditions, the solution will satisfy the sum condition, justifying
# the collocation system.


# as x becomes large, the jump G (and V) become oscillatory:

x = -20
Θ(z) = 8/3*z^3+2*x*z

t = range(0,5,1000)
lines(t, @.(real(exp(im*Θ(t * exp(im*π/6))))))

# for integral representations of eg Airy we can deform along steepest descent
# curves. Can we do the same thing here?

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

p = plot(domain(painleveII_jump(-5)))
plot(domain(deformed_painleveII_jump(-10)))

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
