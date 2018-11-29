 using ApproxFun, SingularIntegralEquations, RiemannHilbert, Plots


struct ReflectionCoefficient{VM, VP} <: Function
    V₋::VM
    V₊::VP
end

function ReflectionCoefficient(V, a=-50.0, x₀=0.0, b=50.0)
    d₋, d₊ = a .. x₀ , x₀ .. b
    V₋,V₊ = Fun(V, d₋), Fun(V, d₊)
    ReflectionCoefficient{typeof(V₋),typeof(V₊)}(V₋,V₊)
end

# ψ'' + (V(x) + k^2) ψ = 0
function (R::ReflectionCoefficient)(k)
    k == 0 && return -one(ComplexF64)

    a,x₀ = endpoints(domain(R.V₋))
    b = rightendpoint(domain(R.V₊))
    D = Derivative()
    V₋,V₊ = R.V₋, R.V₊
    ψ = [ivp(); D^2  + (V₋ + k^2)] \ [exp(im*k*a), im*k*(exp(im*k*a)), 0.0]

    F = qr([rdirichlet(space(V₊)); rneumann(); D^2  + (V₊ + k^2)])
    ϕ₊ = F \ [exp(im*k*b), im*k*(exp(im*k*b)), 0.0]
    ϕ₋ = F \ [exp(-im*k*b), -im*k*(exp(-im*k*b)), 0.0]

    a,b = [ϕ₊(x₀)   ϕ₋(x₀);
           ϕ₊'(x₀)  ϕ₋'(x₀)] \ [ψ(x₀); ψ'(x₀)]
    b/a
end

# use multiple threads since reflection coefficient is slow
function tvalues(f, d, n)
    p = points(d, n)
    F = similar(p, ComplexF64)
    Threads.@threads for k=1:length(p)
        F[k] = R(p[k])
    end
    F
end

tFun(f, d::Space, n) = Fun(d, transform(d,tvalues(f,d,n)))

tFun(f, d, n) = tFun(f, Space(d), n)
V = x -> 0.1sech(x)
R = ReflectionCoefficient(V)
@time ρ = tFun(R, -5.0..5, 601)
plot(abs.(ρ.coefficients); yscale=:log10)
plot(ρ)

f = Fun(exp)
F = [f f; f 1.0]
F(0.1)

G = [1-abs2.(ρ) -conj.(ρ);
     ρ           1.0]


Φ = transpose(rhsolve(transpose(G), 2*4*200))


Φ(100.0im)*[1,1]

Φ(0.1+eps()im) - Φ(0.1-eps()im)*G(0.1)


D^2 + Fun(V, -10..10)


z = Fun(ℂ)
-2im*(z*Φ[2,1])(Inf)
-2im*1000Φ[2,1](1000)
0.1sech(0.0)

Φ(100.0im)

Φ(0.1-eps()im)*G(0.1)

domain(ρ)

v = tvalues(R, Chebyshev(-10..10), 100)


R(8.0)

plot(real.(v))
    plot!(imag.(v))

R(0.0)
R(-0.0000)



ρ(0.1)
R(0.1)

plot(ρ)

ρ.coefficients

ρ.coefficients


f = R
n = 100
d = Chebyshev(-10..10)



@time R(1.0)
ret = Vector{ComplexF64}(undef,100)
@time Threads.@threads for k=1:length(ret)
    ret[k] = R(k/10)
end

@time for k=1:100
    ret[k] = R(k/10)
end

@time pmap(R, 1.0:4)


ρ = Fun(R, -10..10, 200)

@time R(10.0)

ks = -10.0:0.1:10.0
Rks = R.(ks)

plot(ks, real.(Rks))
    plot!(ks, imag.(Rks))

R(0.1+0.1im)


#####
# Factor KdV
######
