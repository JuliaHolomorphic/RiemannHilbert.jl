using SingularIntegrals, SemiclassicalOrthogonalPolynomials, ClassicalOrthogonalPolynomials

P = SemiclassicalJacobi(2, 1/2, 0, 1/2)

z = 1+im
W = Weighted(Jacobi(0, 1/2))
x = axes(W,1)
@test (inv.(z .- x') * W)[3] ≈ stieltjes(W, z)[3] ≈ sum(W[:,3] ./ (z .- x))

P = Jacobi(0, 1/2)
W = Weighted(P)
x = axes(W,1)
@test (inv.(z .- x') * W)[3] ≈ stieltjes(W, z)[3]

Y = (n,z) -> [Base.unsafe_getindex(P, z, n+1) stieltjes(W, z)[n+1]/(-2π*im)]

w = JacobiWeight(0, 1/2)
n = 5;  @test Y(n,0.1+0im) ≈  Y(n,0.1-0im) * [1 w[0.1]; 0 1]





