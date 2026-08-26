# ExprGraphExplorer.jl

`ExprGraphExplorer.jl` constructs expression graphs by operator overloading,
propagates primal Julia values through them, and provides step-by-step graph
visualization. Applications attach their own metadata to every node; optional
reverse-pass utilities route operations without prescribing an adjoint type.

## Pluto showcase

[`examples/autodiff_visualizer.jl`](examples/autodiff_visualizer.jl) is an
interactive Pluto notebook demonstrating how scalar reverse-mode automatic
differentiation can be implemented entirely as user-defined metadata and
reverse propagation rules. Its slider walks through every forward evaluation
and reverse adjoint update.

Launch it from the package directory with:

```sh
julia --project=examples -e 'using Pluto; Pluto.run(notebook="examples/autodiff_visualizer.jl")'
```

## Defining node metadata

```julia
using ExprGraphExplorer

mutable struct MyMetadata
    note::String
end
MyMetadata(value) = MyMetadata("value type: $(typeof(value))")

const MyNode = ExprNode{Float64,MyMetadata}
x = MyNode(2.0)
y = MyNode(3.0)
graph = ExprGraph(x * y; names=IdDict(x => "x", y => "y"))
```

## Defining reverse rules

`ExprGraphExplorer.pullback!(node)` translates the operation symbol stored in a
node into ordinary Julia multiple dispatch. For example, scalar reverse-mode
metadata can define

```julia
function ExprGraphExplorer.pullback!(::typeof(*), output::MyNode, x::MyNode, y::MyNode)
    # Propagate output metadata to x.metadata and y.metadata.
end
```

The package contains the small, explicit operation switch; applications only
provide the propagation rules. Calling `pullback!(node)` without a matching rule
throws a `MethodError` showing exactly which operator and node signature is
missing. Metadata that stores a pullback or local Jacobians may instead
specialize `ExprGraphExplorer.pullback!(node::MyNode)` and bypass the switch.

A complete reverse pass is provided by `backward!(output)`. Metadata defines
how adjoints are initialized:

```julia
function ExprGraphExplorer.seed_metadata!(data::MyMetadata, is_output::Bool)
    # Initialize the metadata, using a nonzero reverse seed iff `is_output`.
end
```

For repeated passes, `backward!(output, topological_order(output))` accepts a
prepared order and can be allocation-free.

Rendering and SVG, PNG, and EPS export use Luxor and its Cairo artifact. Small
matrix values are typeset by Typstry and its Typst artifact. No separately
installed graphics or typesetting executable is required.
