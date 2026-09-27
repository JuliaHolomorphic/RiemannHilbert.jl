#
# Painlevé IV:  y'' = y'^2/(2y) + 3y^3/2 + 4x y^2 + 2(x^2 - α) y + β/y
#
# RH problem (Θ₀ = Θ∞ = 0 case): Φ(z) → I as z → ∞, with jumps on the four rays
#   Γₖ = {arg z = π/4 + (k-1)π/2},  k = 1,…,4 (oriented outwards),
#   Φ₊ = Φ₋ exp(θσ₃) Sₖ exp(-θσ₃),   θ(z) = z^2/2 + x z,
# where S₁ = [1 s₁; 0 1], S₂ = [1 0; s₂ 1], S₃ = [1 s₃; 0 1], S₄ = [1 0; s₄ 1]
# satisfy the cyclic relation S₁S₂S₃S₄ = I.
#
using ContinuumArrays, RiemannHilbert, LinearAlgebra, CairoMakie

# Stokes multipliers satisfying S₁S₂S₃S₄ = I
s₁ = 0.5im
s₃ = 0.3im
s₂ = -(s₁ + s₃) / (1 + s₁*s₃)   # from cyclic relation
θ = z -> z^2/2 + x*z
G = [[1 s₁; 0 1],
     [1 0; s₂ 1],
     [1 s₃; 0 1]]
# enforce S₄ exactly from S₄ = (S₁S₂S₃)⁻¹ (lower triangular when relation holds)
S[4] = inv(S[1]*S[2]*S[3])

const R = 6.0   # truncation radius (jumps are exponentially close to I beyond)
rays = [Segment(0, R*exp(im*(π/4 + (k-1)*π/2))) for k = 1:4]
Γ = ∪(rays...)

function painleveIV_jump(x)
    expand(begin
            k = findfirst(r -> z ∈ r, rays)
            E = Diagonal([exp(θ(z)), exp(-θ(z))])
            E * S[k] * inv(E)
        end for z in Γ)
end

# Solve the RHP and return the residue Y₁ in Φ(z) = I + Y₁/z + O(z⁻²)
function residue(x; n = 4*120)
    G = painleveIV_jump(x)
    U = transpose(rhsolve(transpose(G), n))   # Φ = I + 𝒞U
    -sum(U) / (2π*im)
end

# Recover y(x) from the Lax pair asymptotics:  y(x) = -2x - d/dx log((Y₁)₁₂(x))
function painleveIV(x; h = 1e-4)
    f = t -> log(residue(t)[1,2])
    -2x - (f(x + h) - f(x - h)) / (2h)
end

xs = range(-2, 2; length = 21)
ys = painleveIV.(xs)
