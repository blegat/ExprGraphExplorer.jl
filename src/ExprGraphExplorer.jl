module ExprGraphExplorer

import LinearAlgebra

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

function Base.sum(x::N; dims=:) where {N<:ExprNode}
    if dims === Colon()
        return N(:sum, N[x], sum(x.value))
    end
    dim = N(Float64(only(dims isa Integer ? (dims,) : Tuple(dims))))
    return N(:sum_dims, N[x, dim], sum(x.value; dims=Int(dim.value)))
end

function Base.maximum(x::N; dims=:) where {N<:ExprNode}
    if dims === Colon()
        return N(:maximum, N[x], maximum(x.value))
    end
    dim = N(Float64(only(dims isa Integer ? (dims,) : Tuple(dims))))
    return N(:maximum_dims, N[x, dim], maximum(x.value; dims=Int(dim.value)))
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

function ExprGraph(output::ExprNode{T,M}; names=IdDict{ExprNode{T,M},String}()) where {T,M}
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
    active=nothing,
    visible=Set(topological_order(graph.output)),
    show_metadata=true,
) where {T,M}
    values = IdDict(node => node in visible for node in topological_order(graph.output))
    rows = IdDict{ExprNode{T,M},Vector{Pair{String,String}}}()
    for node in topological_order(graph.output)
        rows[node] = show_metadata ? copy(metadata_rows(node.metadata)) : Pair{String,String}[]
    end
    return Frame{T,M}(String(title), active, values, rows)
end

function forward_frames(graph::ExprGraph)
    order = topological_order(graph.output)
    result = Frame[]
    visible = Set{eltype(order)}()
    push!(result, capture_frame(graph, "Expression graph"; visible, show_metadata=false))
    for node in order
        push!(visible, node)
        push!(result, capture_frame(
            graph,
            "Forward pass: evaluate $(graph.names[node])";
            active=node,
            visible,
            show_metadata=false,
        ))
    end
    return result
end

_fmt(x::Number) = isinteger(x) ? string(Int(x)) : string(round(x; digits=3))
_fmt(x) = string(summary(x))
_escape(s) = replace(string(s), "&" => "&amp;", "<" => "&lt;", ">" => "&gt;")

function _depth!(depth, node)
    haskey(depth, node) && return depth[node]
    depth[node] = isempty(node.args) ? 0 : 1 + maximum(_depth!(depth, arg) for arg in node.args)
end

function _positions(order; width=1100, height=560)
    depth = IdDict{eltype(order),Int}()
    foreach(node -> _depth!(depth, node), order)
    maxdepth = maximum(values(depth))
    positions = IdDict{eltype(order),Tuple{Float64,Float64}}()
    for d in 0:maxdepth
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

function render_svg(graph::ExprGraph, frame::Frame; width=1100, height=620, exam=false, responsive=true)
    order = topological_order(graph.output)
    positions, depth = _positions(order; width, height=height - 60)
    io = IOBuffer()
    dimensions = if responsive
        "width=\"100%\" viewBox=\"0 0 $width $height\" preserveAspectRatio=\"xMidYMid meet\" style=\"display:block;height:auto\""
    else
        "width=\"$width\" height=\"$height\" viewBox=\"0 0 $width $height\""
    end
    println(io, "<svg xmlns=\"http://www.w3.org/2000/svg\" $dimensions>")
    println(io, "<defs><marker id=\"arrow\" markerWidth=\"10\" markerHeight=\"10\" refX=\"9\" refY=\"3\" orient=\"auto\"><path d=\"M0,0 L0,6 L9,3 z\" fill=\"#667085\"/></marker></defs>")
    println(io, "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>")
    exam || println(io, "<text x=\"$(width / 2)\" y=\"30\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"21\" font-weight=\"bold\">$(_escape(frame.title))</text>")
    for node in order, arg in node.args
        x1, y1 = positions[arg]; x2, y2 = positions[node]
        startx, endx = x1 + 68, x2 - 72
        if depth[node] - depth[arg] > 1
            controlx = (startx + endx) / 2
            controly = y1 == y2 ? y1 + 120 : min(y1, y2) - 100
            println(io, "<path d=\"M $startx $y1 Q $controlx $controly $endx $y2\" fill=\"none\" stroke=\"#667085\" stroke-width=\"2.5\" marker-end=\"url(#arrow)\"/>")
        else
            println(io, "<line x1=\"$startx\" y1=\"$y1\" x2=\"$endx\" y2=\"$y2\" stroke=\"#667085\" stroke-width=\"2.5\" marker-end=\"url(#arrow)\"/>")
        end
    end
    for node in order
        x, y = positions[node]
        active = node === frame.active
        fill = active ? "#fef0c7" : (isnothing(node.op) ? "#e0f2fe" : "#f2f4f7")
        stroke = active ? "#f79009" : "#344054"
        println(io, "<rect x=\"$(x-68)\" y=\"$(y-42)\" width=\"136\" height=\"84\" rx=\"12\" fill=\"$fill\" stroke=\"$stroke\" stroke-width=\"$(active ? 4 : 2)\"/>")
        op = isnothing(node.op) ? "input" : string(node.op)
        println(io, "<text x=\"$x\" y=\"$(y-15)\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"19\" font-weight=\"bold\">$(_escape(graph.names[node]))  [$(_escape(op))]</text>")
        value = frame.values[node] ? _fmt(node.value) : "?"
        println(io, "<text x=\"$x\" y=\"$(y+13)\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"17\">value = $(_escape(value))</text>")
        exam && continue
        for (index, row) in enumerate(frame.metadata[node])
            println(io, "<text x=\"$x\" y=\"$(y+13+21index)\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"16\" fill=\"#175cd3\">$(_escape(row.first)) = $(_escape(row.second))</text>")
        end
    end
    println(io, "</svg>")
    return String(take!(io))
end

function save_svg(path, graph::ExprGraph, frame::Frame; responsive=false, kwargs...)
    open(path, "w") do io
        write(io, render_svg(graph, frame; responsive, kwargs...))
    end
    return path
end

function save_png(path, graph::ExprGraph, frame::Frame; density=180, width=1600, kwargs...)
    mktempdir() do directory
        source = save_svg(joinpath(directory, "graph.svg"), graph, frame; kwargs...)
        run(`magick -density $density -background white $source -resize $(width)x $path`)
    end
    return path
end

function save_eps(path, graph::ExprGraph, frame::Frame; kwargs...)
    mktempdir() do directory
        source = save_svg(joinpath(directory, "graph.svg"), graph, frame; kwargs...)
        run(`inkscape $source --export-type=eps --export-filename=$path`)
    end
    return path
end

end
