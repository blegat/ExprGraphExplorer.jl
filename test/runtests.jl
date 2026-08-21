using ExprGraphExplorer
using LinearAlgebra
using Test

struct EmptyMetadata end
EmptyMetadata(::Any) = EmptyMetadata()

@testset "scalar expression graph" begin
    Node = ExprNode{Float64,EmptyMetadata}
    x, y = Node(2), Node(3)
    s1 = x * y
    output = s1 * (s1 + x)
    @test output.value == 48
    @test length(topological_order(output)) == 5
    graph = ExprGraph(output; names=IdDict(x => "x", y => "y"))
    states = forward_frames(graph)
    @test length(states) == 6
    @test occursin("<svg", render_svg(graph, states[end]))
end

@testset "small array value union" begin
    Value = Union{
        Float64,
        Vector{Float64},
        Matrix{Float64},
        Adjoint{Float64,Matrix{Float64}},
    }
    Node = ExprNode{Value,EmptyMetadata}
    x = Node(reshape(collect(1.0:6.0), 2, 3))
    w = Node(reshape(collect(1.0:6.0), 3, 2))
    product = x * w
    @test product.value == x.value * w.value
    @test sum(product; dims=2).value == sum(product.value; dims=2)
    @test w'.value isa Adjoint{Float64,Matrix{Float64}}
    @test (x' * x).value == x.value' * x.value
end

include(joinpath(@__DIR__, "..", "examples", "scalar_reverse.jl"))
using .ScalarReverseExample

@testset "scalar reverse example" begin
    graph = ScalarReverseExample.example()
    ScalarReverseExample.backward!(graph.output)
    derivatives = Dict(graph.names[node] => node.metadata.derivative for node in keys(graph.names))
    @test derivatives["x"] == 48
    @test derivatives["y"] == 28
    @test derivatives["s₁"] == 14
    @test derivatives["s₂"] == 6
end
