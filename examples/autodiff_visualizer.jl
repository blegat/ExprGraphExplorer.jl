### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 8cbb9e65-f693-4328-bf2e-b3f870e5369e
begin
    import Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
end

# ╔═╡ e1084400-e25c-4f9f-bdd7-c919cdf50ad8
using PlutoUI, ExprGraphExplorer

# ╔═╡ 6956a53c-58f2-4ee1-b9a5-c74678752eb5
include("scalar_reverse.jl")

# ╔═╡ bf1c5b69-b16c-4a44-a952-b91121536664
using .ScalarReverseExample

# ╔═╡ 93b3f711-14bf-4be6-9f81-44c5bedb3e56
md"""
# Expression graphs and reverse differentiation

`ExprGraphExplorer` builds and displays the expression graph while propagating
only primal values. This notebook adds scalar reverse-mode differentiation as
custom metadata on top of that generic graph.

The example represents
```math
s_1=xy,\qquad s_2=s_1+x,\qquad f=s_1s_2.
```
Move the slider to reveal the forward pass and then the accumulation of
adjoints during the reverse pass.
"""

# ╔═╡ 5b4aeff2-1c64-433b-98da-b10dcbd03290
begin
    graph = ScalarReverseExample.example(x=2, y=3)
    states = ScalarReverseExample.frames(graph)
end

# ╔═╡ ca498fc8-0917-4a97-bac0-1224ab2e2365
@bind step Slider(eachindex(states); default=1, show_value=true)

# ╔═╡ 5b273195-e5f2-4e82-846c-53db223652ac
HTML(render_svg(graph, states[step]))

# ╔═╡ Cell order:
# ╠═8cbb9e65-f693-4328-bf2e-b3f870e5369e
# ╠═e1084400-e25c-4f9f-bdd7-c919cdf50ad8
# ╠═6956a53c-58f2-4ee1-b9a5-c74678752eb5
# ╠═bf1c5b69-b16c-4a44-a952-b91121536664
# ╟─93b3f711-14bf-4be6-9f81-44c5bedb3e56
# ╠═5b4aeff2-1c64-433b-98da-b10dcbd03290
# ╠═ca498fc8-0917-4a97-bac0-1224ab2e2365
# ╠═5b273195-e5f2-4e82-846c-53db223652ac
