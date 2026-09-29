module RiemannHilbertMakieExt

using RiemannHilbert, Makie
using RiemannHilbert: AbstractSegment, leftendpoint, rightendpoint

"""
    plot(d::Segment; kwargs...)
    plot!(d::Segment; kwargs...)

plots the segment `d` in the complex plane with Makie, with an arrow at its midpoint showing its orientation.
"""
@recipe SegmentPlot (segment,) begin
    "Color of the segment and its arrow."
    color = @inherit linecolor
    "Width of the segment."
    linewidth = @inherit linewidth
    "Size of the arrow at the midpoint showing the orientation."
    arrowsize = 12
    cycle = [:color]
end

Makie.plottype(::AbstractSegment) = SegmentPlot

_point(z) = Point2f(real(z), imag(z))

function Makie.plot!(p::SegmentPlot)
    d = p[:segment]
    ends = lift(d -> [_point(leftendpoint(d)), _point(rightendpoint(d))], p, d)
    mid = lift(d -> [_point((leftendpoint(d) + rightendpoint(d))/2)], p, d)
    direction = lift(d -> Float32(angle(rightendpoint(d) - leftendpoint(d))), p, d)
    lines!(p, ends; color=p[:color], linewidth=p[:linewidth])
    # :rtriangle points in the positive real direction, so is rotated to point along the segment
    scatter!(p, mid; marker=:rtriangle, rotation=direction, markersize=p[:arrowsize], color=p[:color], strokewidth=0)
    p
end

end
