module RiemannHilbertRecipesBaseExt

using RiemannHilbert, RecipesBase
using RiemannHilbert: AbstractSegment, leftendpoint, rightendpoint

# a segment is plotted in the complex plane, with an arrow at its midpoint showing its orientation
@recipe function f(d::AbstractSegment)
    a, b = leftendpoint(d), rightendpoint(d)
    m = (a + b)/2
    @series begin
        seriestype := :path
        primary := true
        arrow := true
        [real(a), real(m)], [imag(a), imag(m)]
    end
    @series begin
        seriestype := :path
        primary := false
        [real(m), real(b)], [imag(m), imag(b)]
    end
end

end
