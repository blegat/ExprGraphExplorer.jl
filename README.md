# ExprGraphExplorer.jl

`ExprGraphExplorer.jl` constructs expression graphs by operator overloading,
propagates primal Julia values through them, and provides step-by-step graph
visualization. The package is deliberately independent of automatic
differentiation: applications attach their own metadata to every node.

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

The core SVG renderer has no external dependency. The optional `save_png` and
`save_eps` helpers call ImageMagick and Inkscape respectively.
