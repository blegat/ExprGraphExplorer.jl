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

function example(; x=2.0, y=3.0)
    xnode = ScalarNode(x)
    ynode = ScalarNode(y)
    s1 = xnode * ynode
    s2 = s1 + xnode
    output = s1 * s2
    names = IdDict(xnode => "x", ynode => "y", s1 => "s₁", s2 => "s₂", output => "f")
    return ExprGraph(output; names)
end

function _backward!(node::ScalarNode)
    isnothing(node.op) && return
    derivative = node.metadata.derivative
    if node.op == :+
        for arg in node.args
            arg.metadata.derivative += derivative
        end
    elseif node.op == :- && length(node.args) == 2
        node.args[1].metadata.derivative += derivative
        node.args[2].metadata.derivative -= derivative
    elseif node.op == :- && length(node.args) == 1
        node.args[1].metadata.derivative -= derivative
    elseif node.op == :* && length(node.args) == 2
        node.args[1].metadata.derivative += derivative * node.args[2].value
        node.args[2].metadata.derivative += derivative * node.args[1].value
    elseif node.op == :/ && length(node.args) == 2
        node.args[1].metadata.derivative += derivative / node.args[2].value
        node.args[2].metadata.derivative -=
            derivative * node.args[1].value / node.args[2].value^2
    elseif node.op == :^ && length(node.args) == 2
        base, exponent = node.args
        base.metadata.derivative +=
            derivative * exponent.value * base.value^(exponent.value - 1)
    elseif node.op == :tanh
        arg = only(node.args)
        arg.metadata.derivative += derivative * (1 - tanh(arg.value)^2)
    elseif node.op == :exp
        arg = only(node.args)
        arg.metadata.derivative += derivative * exp(arg.value)
    elseif node.op == :log
        arg = only(node.args)
        arg.metadata.derivative += derivative / arg.value
    else
        error("Operator `$(node.op)` is not supported by the scalar reverse example")
    end
    return
end

function backward!(output::ScalarNode)
    order = topological_order(output)
    for node in order
        node.metadata.derivative = 0.0
    end
    output.metadata.derivative = 1.0
    for node in reverse(order)
        _backward!(node)
    end
    return output
end

function frames(graph::ExprGraph)
    result = forward_frames(graph)
    order = topological_order(graph.output)
    for node in order
        node.metadata.derivative = 0.0
    end
    graph.output.metadata.derivative = 1.0
    push!(result, capture_frame(
        graph,
        "Reverse pass: seed f̄ = 1";
        active=graph.output,
    ))
    for node in reverse(order)
        isempty(node.args) && continue
        _backward!(node)
        push!(result, capture_frame(
            graph,
            "Reverse pass: propagate from $(graph.names[node])";
            active=node,
        ))
    end
    return result
end

end
