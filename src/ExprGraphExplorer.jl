# Copyright (c) 2026 Benoît Legat
# SPDX-License-Identifier: MIT

module ExprGraphExplorer

import LinearAlgebra
import Luxor
import Typstry

export ExprNode,
    ExprGraph,
    Frame,
    metadata,
    convert_value,
    metadata_rows,
    topological_order,
    forward_frames,
    capture_frame,
    render_svg,
    save_svg,
    save_png,
    save_eps

"""
    metadata(::Type{M}, value)

Create metadata for a node whose propagated value is `value`. By default this
calls `M(value)`; metadata types may overload this hook instead.
"""
metadata(::Type{M}, value) where {M} = M(value)

"""Convert an operand to the value type shared by an expression graph."""
convert_value(::Type{T}, value) where {T} = convert(T, value)

"""Rows displayed below a node by the generic visualizer."""
metadata_rows(::Any) = Pair{String,String}[]

mutable struct ExprNode{T,M}
    op::Union{Nothing,Symbol}
    args::Vector{ExprNode{T,M}}
    value::T
    metadata::M
end

function _value(::Type{T}, value) where {T}
    value isa T && return value
    return convert_value(T, value)
end

function ExprNode{T,M}(value) where {T,M}
    converted = _value(T, value)
    return ExprNode{T,M}(nothing, ExprNode{T,M}[], converted, metadata(M, converted))
end

function ExprNode{T,M}(op::Symbol, args::Vector{ExprNode{T,M}}, value) where {T,M}
    converted = _value(T, value)
    return ExprNode{T,M}(op, args, converted, metadata(M, converted))
end

_node_type(::ExprNode{T,M}) where {T,M} = ExprNode{T,M}
_constant(x::ExprNode, ::Type{<:ExprNode}) = x
_constant(x, ::Type{N}) where {N<:ExprNode} = N(x)

function _binary(op::Symbol, f, x::N, y) where {N<:ExprNode}
    ynode = _constant(y, N)
    return N(op, N[x, ynode], f(x.value, ynode.value))
end

function _binary(op::Symbol, f, x, y::N) where {N<:ExprNode}
    xnode = _constant(x, N)
    return N(op, N[xnode, y], f(xnode.value, y.value))
end

function _binary(op::Symbol, f, x::N, y::N) where {N<:ExprNode}
    return N(op, N[x, y], f(x.value, y.value))
end

function _unary(op::Symbol, f, x::N) where {N<:ExprNode}
    return N(op, N[x], f(x.value))
end

Base.:+(x::ExprNode, y) = _binary(:+, +, x, y)
Base.:+(x, y::ExprNode) = _binary(:+, +, x, y)
Base.:+(x::ExprNode, y::ExprNode) = _binary(:+, +, x, y)
Base.:-(x::ExprNode, y) = _binary(:-, -, x, y)
Base.:-(x, y::ExprNode) = _binary(:-, -, x, y)
Base.:-(x::ExprNode, y::ExprNode) = _binary(:-, -, x, y)
Base.:-(x::ExprNode) = _unary(:-, -, x)
Base.:*(x::ExprNode, y) = _binary(:*, *, x, y)
Base.:*(x, y::ExprNode) = _binary(:*, *, x, y)
Base.:*(x::ExprNode, y::ExprNode) = _binary(:*, *, x, y)
Base.:/(x::ExprNode, y) = _binary(:/, /, x, y)
Base.:/(x, y::ExprNode) = _binary(:/, /, x, y)
Base.:/(x::ExprNode, y::ExprNode) = _binary(:/, /, x, y)
Base.:^(x::ExprNode, y) = _binary(:^, ^, x, Float64(y))
Base.:^(x, y::ExprNode) = _binary(:^, ^, Float64(x), y)
Base.:^(x::ExprNode, y::ExprNode) = _binary(:^, ^, x, y)

for f in (:tanh, :exp, :log, :sqrt)
    @eval Base.$f(x::ExprNode) = _unary($(QuoteNode(f)), Base.$f, x)
end

_broadcast_symbol(op) = Symbol("broadcast_", nameof(op))
function Base.broadcasted(op::Function, x::N) where {N<:ExprNode}
    return _unary(_broadcast_symbol(op), value -> op.(value), x)
end
function Base.broadcasted(op::Function, x::N, y) where {N<:ExprNode}
    return _binary(_broadcast_symbol(op), (a, b) -> op.(a, b), x, y)
end
function Base.broadcasted(op::Function, x, y::N) where {N<:ExprNode}
    return _binary(_broadcast_symbol(op), (a, b) -> op.(a, b), x, y)
end
function Base.broadcasted(op::Function, x::N, y::N) where {N<:ExprNode}
    return _binary(_broadcast_symbol(op), (a, b) -> op.(a, b), x, y)
end
function Base.broadcasted(
    ::typeof(Base.literal_pow),
    ::typeof(^),
    x::ExprNode,
    ::Val{power},
) where {power}
    return Base.broadcasted(^, x, Float64(power))
end
Base.materialize(x::ExprNode) = x
Base.copy(x::ExprNode) = x

Base.zero(x::N) where {N<:ExprNode} = N(zero(x.value))
Base.length(x::ExprNode) = length(x.value)
Base.size(x::ExprNode) = size(x.value)
Base.size(x::ExprNode, dim) = size(x.value, dim)
Base.ndims(x::ExprNode) = ndims(x.value)
Base.eltype(x::ExprNode) = eltype(x.value)
Base.isless(x::ExprNode, y) = isless(x.value, y isa ExprNode ? y.value : y)
Base.isless(x, y::ExprNode) = isless(x isa ExprNode ? x.value : x, y.value)
Base.isless(x::ExprNode, y::ExprNode) = isless(x.value, y.value)

function Base.sum(x::N; dims = :) where {N<:ExprNode}
    if dims === Colon()
        return N(:sum, N[x], sum(x.value))
    end
    dim = N(Float64(only(dims isa Integer ? (dims,) : Tuple(dims))))
    return N(:sum_dims, N[x, dim], sum(x.value; dims = Int(dim.value)))
end

function Base.maximum(x::N; dims = :) where {N<:ExprNode}
    if dims === Colon()
        return N(:maximum, N[x], maximum(x.value))
    end
    dim = N(Float64(only(dims isa Integer ? (dims,) : Tuple(dims))))
    return N(:maximum_dims, N[x, dim], maximum(x.value; dims = Int(dim.value)))
end

function LinearAlgebra.adjoint(x::N) where {N<:ExprNode}
    return N(:adjoint, N[x], adjoint(x.value))
end

function Base.getindex(x::N, rows::AbstractVector{<:Integer}, ::Colon) where {N<:ExprNode}
    rownode = N(Float64.(rows))
    return N(:getindex_rows, N[x, rownode], x.value[rows, :])
end

function Base.reduce(::typeof(hcat), nodes::AbstractVector{N}) where {N<:ExprNode}
    return N(:hcat, collect(nodes), reduce(hcat, (node.value for node in nodes)))
end

function topological_order(output::N) where {N<:ExprNode}
    visited = Set{N}()
    order = N[]
    function visit(node)
        node in visited && return
        push!(visited, node)
        foreach(visit, node.args)
        push!(order, node)
    end
    visit(output)
    return order
end

struct ExprGraph{T,M}
    output::ExprNode{T,M}
    names::IdDict{ExprNode{T,M},String}
end

function ExprGraph(
    output::ExprNode{T,M};
    names = IdDict{ExprNode{T,M},String}(),
) where {T,M}
    order = topological_order(output)
    generated = IdDict{ExprNode{T,M},String}()
    for (index, node) in enumerate(order)
        generated[node] = get(names, node, "s$(index)")
    end
    return ExprGraph{T,M}(output, generated)
end

struct Frame{T,M}
    title::String
    active::Union{Nothing,ExprNode{T,M}}
    values::IdDict{ExprNode{T,M},Bool}
    metadata::IdDict{ExprNode{T,M},Vector{Pair{String,String}}}
end

function capture_frame(
    graph::ExprGraph{T,M},
    title::AbstractString;
    active = nothing,
    visible = Set(topological_order(graph.output)),
    show_metadata = true,
) where {T,M}
    values = IdDict(node => node in visible for node in topological_order(graph.output))
    rows = IdDict{ExprNode{T,M},Vector{Pair{String,String}}}()
    for node in topological_order(graph.output)
        rows[node] =
            show_metadata ? copy(metadata_rows(node.metadata)) : Pair{String,String}[]
    end
    return Frame{T,M}(String(title), active, values, rows)
end

function forward_frames(graph::ExprGraph)
    order = topological_order(graph.output)
    result = Frame[]
    visible = Set{eltype(order)}()
    push!(result, capture_frame(graph, "Expression graph"; visible, show_metadata = false))
    for node in order
        push!(visible, node)
        push!(
            result,
            capture_frame(
                graph,
                "Forward pass: evaluate $(graph.names[node])";
                active = node,
                visible,
                show_metadata = false,
            ),
        )
    end
    return result
end

_fmt(x::Number) = isinteger(x) ? string(Int(x)) : string(round(x; digits = 3))
_fmt(x::AbstractVector) = "[" * join(_fmt.(x), ", ") * "]"
function _fmt(x::AbstractMatrix)
    length(x) > 16 && return string(summary(x))
    rows = (join((_fmt(x[i, j]) for j in axes(x, 2)), " ") for i in axes(x, 1))
    return "[" * join(rows, "; ") * "]"
end
_fmt(x) = string(summary(x))

const _typst_preamble = Typstry.TypstString(
    Typstry.TypstText(
        "#set page(width: auto, height: auto, margin: 0pt, fill: none); #set text(size: 14pt);",
    ),
)
const _matrix_image_cache = Dict{String,Luxor.SVGimage}()
function _matrix_image(value::AbstractMatrix)
    key = _fmt(value)
    return get!(_matrix_image_cache, key) do
        matrix = Typstry.TypstString(value; mode = Typstry.markup, delim = "[")
        io = IOBuffer()
        show(IOContext(io, :preamble => _typst_preamble), "image/svg+xml", matrix)
        svg = String(take!(io))
        # Typst exports its page as an opaque white path even with `fill: none`.
        # The page is only a container here, so remove it before compositing.
        svg = replace(
            svg,
            r"\n\s*<path class=\"typst-shape\" fill=\"#ffffff\"[^>]*/>" => "";
            count = 1,
        )
        return Luxor.readsvg(svg)
    end
end

function _depth!(depth, node)
    haskey(depth, node) && return depth[node]
    depth[node] =
        isempty(node.args) ? 0 : 1 + maximum(_depth!(depth, arg) for arg in node.args)
end

function _positions(order; width = 1100, height = 560)
    depth = IdDict{eltype(order),Int}()
    foreach(node -> _depth!(depth, node), order)
    maxdepth = maximum(values(depth))
    positions = IdDict{eltype(order),Tuple{Float64,Float64}}()
    for d = 0:maxdepth
        level = [node for node in order if depth[node] == d]
        for (index, node) in enumerate(level)
            positions[node] = (
                100 + d * (width - 200) / max(maxdepth, 1),
                index * height / (length(level) + 1),
            )
        end
    end
    return positions, depth
end

function _draw_matrix_value(value, point)
    image = _matrix_image(value)
    Luxor.fontface("sans-serif")
    Luxor.fontsize(15)
    prefix = "value ="
    prefix_width = Luxor.textextents(prefix)[5]
    gap = 5
    total_width = prefix_width + gap + image.width
    left = point.x - total_width / 2
    Luxor.text(prefix, Luxor.Point(left, point.y); halign = :left, valign = :middle)
    Luxor.placeimage(
        image,
        Luxor.Point(left + prefix_width + gap, point.y - image.height / 2),
    )
end

function _draw_graph(
    graph::ExprGraph,
    frame::Frame;
    width = 1100,
    height = 620,
    exam = false,
)
    order = topological_order(graph.output)
    positions, depth = _positions(order; width, height = height - 60)
    Luxor.background("white")
    if !exam
        Luxor.sethue("black")
        Luxor.fontface("DejaVu Sans Bold")
        Luxor.fontsize(21)
        Luxor.text(
            frame.title,
            Luxor.Point(width / 2, 30);
            halign = :center,
            valign = :middle,
        )
    end
    Luxor.sethue("#667085")
    for node in order, arg in node.args
        x1, y1 = positions[arg]
        x2, y2 = positions[node]
        startx, endx = x1 + 68, x2 - 72
        start = Luxor.Point(startx, y1)
        finish = Luxor.Point(endx, y2)
        if depth[node] - depth[arg] > 1
            controlx = (startx + endx) / 2
            controly = y1 == y2 ? y1 + 120 : min(y1, y2) - 100
            control = Luxor.Point(controlx, controly)
            first_control = start + (control - start) * (2 / 3)
            second_control = finish + (control - finish) * (2 / 3)
            Luxor.arrow(
                start,
                first_control,
                second_control,
                finish;
                linewidth = 2.5,
                arrowheadlength = 10,
            )
        else
            Luxor.arrow(start, finish; linewidth = 2.5, arrowheadlength = 10)
        end
    end
    for node in order
        x, y = positions[node]
        center = Luxor.Point(x, y)
        active = node === frame.active
        fill = active ? "#fef0c7" : (isnothing(node.op) ? "#e0f2fe" : "#f2f4f7")
        stroke = active ? "#f79009" : "#344054"
        Luxor.sethue(fill)
        Luxor.box(center, 136, 84, 12, :fill)
        Luxor.sethue(stroke)
        Luxor.setline(active ? 4 : 2)
        Luxor.box(center, 136, 84, 12, :stroke)
        op = isnothing(node.op) ? "input" : string(node.op)
        Luxor.sethue("black")
        Luxor.fontface("DejaVu Sans Bold")
        Luxor.fontsize(19)
        Luxor.text(
            "$(graph.names[node])  [$op]",
            Luxor.Point(x, y - 15);
            halign = :center,
            valign = :middle,
        )
        if frame.values[node] && node.value isa AbstractMatrix && length(node.value) <= 16
            _draw_matrix_value(node.value, Luxor.Point(x, y + 13))
            value_y = y + 13
        else
            value = frame.values[node] ? _fmt(node.value) : "?"
            Luxor.fontface("sans-serif")
            Luxor.fontsize(17)
            Luxor.text(
                "value = $value",
                Luxor.Point(x, y + 13);
                halign = :center,
                valign = :middle,
            )
            value_y = y + 13
        end
        exam && continue
        for (index, row) in enumerate(frame.metadata[node])
            metadata_y = value_y + 21index
            Luxor.sethue("#175cd3")
            Luxor.fontsize(16)
            Luxor.text(
                "$(row.first) = $(row.second)",
                Luxor.Point(x, metadata_y);
                halign = :center,
                valign = :middle,
            )
        end
    end
end

function _render(
    graph,
    frame,
    surface;
    width = 1100,
    height = 620,
    exam = false,
    path = "",
    scale = 1.0,
)
    canvas_width = surface == :png ? round(Int, scale * width) : scale * width
    canvas_height = surface == :png ? round(Int, scale * height) : scale * height
    drawing = Luxor.Drawing(canvas_width, canvas_height, surface, path)
    Luxor.scale(scale)
    _draw_graph(graph, frame; width, height, exam)
    Luxor.finish()
    return drawing
end

function render_svg(
    graph::ExprGraph,
    frame::Frame;
    width = 1100,
    height = 620,
    exam = false,
    responsive = true,
)
    _render(graph, frame, :svg; width, height, exam)
    svg = Luxor.svgstring()
    if responsive
        root = "<svg xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" width=\"100%\" viewBox=\"0 0 $width $height\" preserveAspectRatio=\"xMidYMid meet\" style=\"display:block;height:auto\">"
        svg = replace(svg, r"<svg[^>]*>" => root; count = 1)
    end
    return svg
end

function save_svg(path, graph::ExprGraph, frame::Frame; responsive = false, kwargs...)
    if responsive
        write(path, render_svg(graph, frame; responsive, kwargs...))
    else
        _render(graph, frame, :svg; path, kwargs...)
    end
    return path
end

function save_png(
    path,
    graph::ExprGraph,
    frame::Frame;
    density = 180,
    width = 1600,
    height = 620,
    exam = false,
)
    scale = width / 1100
    _render(graph, frame, :png; width = 1100, height, exam, path, scale)
    return path
end

function save_eps(path, graph::ExprGraph, frame::Frame; kwargs...)
    _render(graph, frame, :eps; path, kwargs...)
    return path
end

end
