# load some packages we use
using ContinuumArrays, RiemannHilbert, DomainSets, CairoMakie, ComplexPhasePortrait, Test

# We want to find _a solution_ to Painleve II solves
# 
# u'' = x u + 2u^3 
#
# we want to evaluate at a point x. 
# There are of course many different solutions, eg defined by two initial conditions
#
# u(0), u'(0)
#
# But here, we define the solution by Stokes parameters s₁, s₂, s₃.
# We get the value at a given point x.

x = 0.1 # this gives us u(0)
s₁,s₂,s₃ = 1+im,-2,1-im # this is some Stokes parameters (happen to give real valued)

# cyclic condition is required to ensure well-posedness of the RHP
@test s₁ - s₂ + s₃ + s₁*s₂*s₃ == 0

Θ(z) = 8/3*z^3+2*x*z # x is a parameter 

# we can define G on the ray (0,∞*exp(im*π/6)), but truncated since
# exp(im*θ(z))

t = range(0,5,100)
# abs(G(z)[2,1]) for z in (0,∞*exp(im*π/6)
# We can see that by 2.5 * exp(im*π/6)
# it is numerically zero:
lines(t, @.(abs(exp(im*Θ(exp(im*π/6) * t )))))

# i.e. the jump itself is bassically I
[1 0; s₁*exp(im*Θ(2.5 * exp(im*π/6))) 1]

# i.e. Φ_+ = Φ_-, i.e., its continuous and there for analytic
# (to 16 digits)

# So we can replace the infinite rays with finite line segments 
# and still get high-accuracy solutions

# Here we represent this jump on a single segment using the following
# syntax:
G₁ = [[1 0; s₁*exp(im*Θ(z)) 1] for z in Segment(0, 2.5exp(im*π/6))]

# this is a "quasi-vector" which means we work with it like a vector
# whose axes are continuous
@test G₁[0.1exp(im*π/6)] ≈ [1 0; s₁*exp(im*Θ(0.1exp(im*π/6))) 1]

# behind the scenes it will make a Legendre expansion of the jump 
# on this line segment.
plot(domain(G₁))


# now we want to join jumps defined on multiple line segments:

G = [[1 0; s₁*exp(im*Θ(z)) 1] for z in Segment(0, 2.5exp(im*π/6))] ⊎
    [[1 s₂*exp(-im*Θ(z)); 0 1] for z in Segment(0, 2.5im)] ⊎
    [[1 0; s₃*exp(im*Θ(z)) 1] for z in Segment(0, 2.5exp(5im*π/6))] ⊎
    [[1 -s₁*exp(-im*Θ(z)); 0 1] for z in Segment(0, 2.5exp(-5im*π/6))] ⊎
    [[1 0; -s₂*exp(im*Θ(z)) 1] for z in Segment(0, -2.5im)] ⊎
    [[1 -s₃*exp(-im*Θ(z)); 0 1] for z in Segment(0, 2.5exp(-im*π/6))]

plot(domain(G))

# G is now defined on all 6 rays:

@test G[im] ≈ [1 s₂*exp(-im*Θ(im)); 0 1]


# product condition
@test prod(first.(components(G))) ≈ I


# Now we want to find Φ numerically. The following says
# use 100 collocation points (which I'll explain later)
# on each ray to numerically compute V and thereby recover Φ:

n = 100
@time Φ = rhsolve(G, n)


# It solves the right jump on each ray:

h = 0.0000000001
z = 0.9im # some point on segment (0,∞*im)
Φ₊ = Φ(z-h) # limit from the left
Φ₋ = Φ(z+h) # limit from the right
# actually first column is continuous but we see a jump
# in second column. That jump is given by G
@test Φ₊ ≈ Φ₋*G[z]

z = 0.1exp(im*π/6)
@test Φ(z * ⁺) ≈ Φ(z * ⁻)G[z]

# We have therefore matched the jump to very high accuracy:
@test norm(Φ(z * ⁺) - Φ(z * ⁻)G[z]) ≤ 3E-14

# Thus we expect this to be the "true" Φ to at least 13 digits.



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

# We can see the V from the Singular integral equation here:
V = RiemannHilbert.rh_sie_solve(G, 200)

# This is what we actually compute. It then tells Φ via the
# Cauchy transform:

using SingularIntegrals
z = 0.5+0.5im
@test Φ(z)[1,1] ≈ I + cauchy(V[1,1], z) # the 1,1 componento f Φ
@test Φ(z)[1,2] ≈ cauchy(V[1,2], z) # the 1,2 componento f Φ
# I.e. Φ(z) = I + cauchy(V, z)
# So any time we evaluate Φ we compute the Cauchy transform of V.
# Fortunately this is accurate even up to the jumps:

cauchy(V[1,2], im-eps()) # the jump works all the way up to the contour
cauchy(V[1,2], im+eps())

# the limit z*Φ(z) is equivalent to an integral of V
2*sum(V[1,2])/(-2π*im) # value for P_II(s₁,s₂,s₃; x)



# This is all open-source; You (I mean your favourite LLM) can make PRs on Github. I can review it, add features, add example RH problems, etc.

# Now I'm going to discuss how

@time cauchy(V[1,2], z);

# works.  Compare with just computing the Cauchy kernel:

z = 0.5+0.5im
ζ = randn(4800) + im*randn(4800) # pretend these are quadrature points

@time cauchy(V[1,2], z);
@time inv.(ζ .- z);
@time log.(ζ .- z); # just evaluating the kernel

using ClassicalOrthogonalPolynomials

z = 2.5 + 5.5im
# blue is legendre. Cauchy transform decays exponentation 
plt = scatter(log10.(abs.(cauchy(Legendre(), z)[1:50]) .+ 1E-200))

# yellow is Chebyshev. Cauchy transform has simple formula, but does not decay exponentially
#
scatter!(log10.(abs.((Legendre() \ Chebyshev())[1:50,1:50]' * cauchy(Legendre(), z)[1:50])))
plt

# So we can get much faster evaluation away from
# [-1,1] with Legendre since we can ignore coefficients where the cauchy of Legendre is below say 1E-16