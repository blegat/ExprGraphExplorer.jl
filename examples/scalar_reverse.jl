# Copyright (c) 2026 Benoît Legat
# SPDX-License-Identifier: MIT

module ScalarReverseExample

using ExprGraphExplorer

export ScalarReverseData, ScalarNode, example, frames, backward!

mutable struct ScalarReverseData
    derivative::Float64
end
ExprGraphExplorer.metadata(::Type{ScalarReverseData}, ::Float64) = ScalarReverseData(0.0)

function ExprGraphExplorer.metadata_rows(data::ScalarReverseData)
    value = iszero(data.derivative) ? "0" : string(data.derivative)
    return ["adjoint" => value]
end

const ScalarNode = ExprNode{Float64,ScalarReverseData}

function ExprGraphExplorer.seed_metadata!(data::ScalarReverseData, is_output::Bool)
    data.derivative = is_output ? 1.0 : 0.0
end

function example(; x = 2.0, y = 3.0)
    xnode = ScalarNode(x)
    ynode = ScalarNode(y)
    s1 = xnode * ynode
    s2 = s1 + xnode
    output = s1 * s2
    names = IdDict(xnode => "x", ynode => "y", s1 => "s₁", s2 => "s₂", output => "f")
    return ExprGraph(output; names)
end

function ExprGraphExplorer.pullback!(::typeof(+), node::ScalarNode, args::ScalarNode...)
    for arg in args
        arg.metadata.derivative += node.metadata.derivative
    end
end

function ExprGraphExplorer.pullback!(
    ::typeof(*),
    node::ScalarNode,
    x::ScalarNode,
    y::ScalarNode,
)
    x.metadata.derivative += node.metadata.derivative * y.value
    y.metadata.derivative += node.metadata.derivative * x.value
end

function frames(graph::ExprGraph)
    result = forward_frames(graph)
    order = topological_order(graph.output)
    for node in order
        node.metadata.derivative = 0.0
    end
    graph.output.metadata.derivative = 1.0
    push!(result, capture_frame(graph, "Reverse pass: seed f̄ = 1"; active = graph.output))
    for node in reverse(order)
        isempty(node.args) && continue
        ExprGraphExplorer.pullback!(node)
        push!(
            result,
            capture_frame(
                graph,
                "Reverse pass: propagate from $(graph.names[node])";
                active = node,
            ),
        )
    end
    return result
end

end
