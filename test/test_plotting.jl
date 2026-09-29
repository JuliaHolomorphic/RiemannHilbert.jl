using RiemannHilbert, RecipesBase, Test
import Makie

@testset "Plotting" begin
    d = Segment(0, 2exp(im*π/4))
    a, m, b = 0.0, exp(im*π/4), 2exp(im*π/4)

    @testset "RecipesBase" begin
        # a path to the midpoint with an arrow, then a path to the end
        first_half, second_half = RecipesBase.apply_recipe(Dict{Symbol,Any}(), d)
        @test all(first_half.args .≈ ([real(a), real(m)], [imag(a), imag(m)]))
        @test first_half.plotattributes[:arrow]
        @test all(second_half.args .≈ ([real(m), real(b)], [imag(m), imag(b)]))
        @test !second_half.plotattributes[:primary]

        first_half, second_half = RecipesBase.apply_recipe(Dict{Symbol,Any}(), Segment(-2, -1))
        @test first_half.args == ([-2, -1.5], [0, 0])
    end

    @testset "Makie" begin
        color(c) = Makie.to_color(c[])
        fig, ax, p = Makie.plot(d)
        lin, arrow = p.plots
        @test lin[1][] ≈ Makie.Point2f.([(real(a), imag(a)), (real(b), imag(b))])
        @test arrow[1][] ≈ [Makie.Point2f(real(m), imag(m))]
        @test arrow.rotation[] ≈ π/4
        @test color(lin.color) == color(arrow.color) == color(p.color)

        q = Makie.plot!(ax, Segment(1.0, -1.0); color=:red, linewidth=3, arrowsize=20)
        lin, arrow = q.plots
        @test arrow.rotation[] ≈ π
        @test color(lin.color) == color(arrow.color) == Makie.to_color(:red)
        @test lin.linewidth[] == 3
        @test all(==(20), arrow.markersize[])

        @test color(Makie.plot!(ax, Segment(0, 1+im)).color) ≠ color(p.color) # colors cycle
    end
end
