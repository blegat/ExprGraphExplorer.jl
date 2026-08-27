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
    backward!,
    pullback!,
    seed_metadata!,
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

"""
    pullback!(node::ExprNode)

Dispatch one node of a reverse pass to an operation-specific method such as
`pullback!(::typeof(+), output, x, y)`. Packages attaching derivative metadata
implement those methods. A missing derivative rule is therefore reported as
a `MethodError` with the corresponding Julia operator and node types.

Specialized metadata may overload this node-level method to bypass operation
dispatch, for example when local Jacobians were stored during the forward
pass.
"""
function pullback!(node::ExprNode)
    isnothing(node.op) && return node
    if node.op == :+
        if length(node.args) == 1
            pullback!(+, node, node.args[1])
        else
            pullback!(+, node, node.args[1], node.args[2])
        end
    elseif node.op == :-
        if length(node.args) == 1
            pullback!(-, node, node.args[1])
        else
            pullback!(-, node, node.args[1], node.args[2])
        end
    elseif node.op == :*
        if length(node.args) == 1
            pullback!(*, node, node.args[1])
        else
            pullback!(*, node, node.args[1], node.args[2])
        end
    elseif node.op == :/
        pullback!(/, node, node.args[1], node.args[2])
    elseif node.op == :^
        pullback!(^, node, node.args[1], node.args[2])
    elseif node.op == :tanh
        pullback!(tanh, node, node.args[1])
    elseif node.op == :exp
        pullback!(exp, node, node.args[1])
    elseif node.op == :log
        pullback!(log, node, node.args[1])
    elseif node.op == :sqrt
        pullback!(sqrt, node, node.args[1])
    elseif node.op == :sum
        pullback!(sum, node, node.args[1])
    elseif node.op == :sum_dims
        pullback!(sum, node, node.args[1], node.args[2])
    elseif node.op == :maximum
        pullback!(maximum, node, node.args[1])
    elseif node.op == :maximum_dims
        pullback!(maximum, node, node.args[1], node.args[2])
    elseif node.op == :adjoint
        pullback!(LinearAlgebra.adjoint, node, node.args[1])
    elseif node.op == :hcat
        pullback!(hcat, node, node.args...)
    elseif node.op == :getindex_rows
        pullback!(getindex, node, node.args[1], node.args[2])
    elseif node.op == _broadcast_symbol(+)
        _broadcast_pullback!(+, node)
    elseif node.op == _broadcast_symbol(-)
        _broadcast_pullback!(-, node)
    elseif node.op == _broadcast_symbol(*)
        _broadcast_pullback!(*, node)
    elseif node.op == _broadcast_symbol(/)
        _broadcast_pullback!(/, node)
    elseif node.op == _broadcast_symbol(^)
        _broadcast_pullback!(^, node)
    elseif node.op == _broadcast_symbol(tanh)
        _broadcast_pullback!(tanh, node)
    elseif node.op == _broadcast_symbol(exp)
        _broadcast_pullback!(exp, node)
    elseif node.op == _broadcast_symbol(log)
        _broadcast_pullback!(log, node)
    elseif node.op == _broadcast_symbol(sqrt)
        _broadcast_pullback!(sqrt, node)
    elseif node.op == _broadcast_symbol(min)
        _broadcast_pullback!(min, node)
    elseif node.op == _broadcast_symbol(max)
        _broadcast_pullback!(max, node)
    else
        pullback!(Val(node.op), node, node.args...)
    end
    return node
end

function _broadcast_pullback!(op, node)
    if length(node.args) == 1
        pullback!(Base.broadcasted, op, node, node.args[1])
    else
        pullback!(Base.broadcasted, op, node, node.args[1], node.args[2])
    end
    return node
end

"""Initialize user-defined metadata for a backward pass."""
function seed_metadata! end

"""
    backward!(output::ExprNode[, order])

Run a complete backward pass. Metadata types implement
`seed_metadata!(metadata, is_output)`, while operation-specific propagation is
provided through `pullback!` methods. Passing a precomputed topological order
makes the pass allocation-free when those methods are allocation-free.
"""
function backward!(output::ExprNode, order)
    for node in order
        seed_metadata!(node.metadata, node === output)
    end
    for index in Iterators.reverse(eachindex(order))
        pullback!(order[index])
    end
    return output
end

backward!(output::ExprNode) = backward!(output, topological_order(output))

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

include("draw.jl")

end
