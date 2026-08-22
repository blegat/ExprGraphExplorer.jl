# Copyright (c) 2026 Benoît Legat
# SPDX-License-Identifier: MIT

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
    svg = render_svg(graph, states[end])
    @test occursin("<svg", svg)
    @test occursin("xmlns:xlink=\"http://www.w3.org/1999/xlink\"", svg)
    @test occursin("width=\"100%\"", svg)
    @test occursin("viewBox=\"0 0 1100 620\"", svg)
    @test occursin("width=\"1100\" height=\"620\"", render_svg(graph, states[end]; responsive=false))
    @test ExprGraphExplorer._fmt([1.0, 2.0]) == "[1, 2]"
    @test ExprGraphExplorer._fmt([1.0 2.0; 3.0 4.0]) == "[1 2; 3 4]"
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
