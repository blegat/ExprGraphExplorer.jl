### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 8cbb9e65-f693-4328-bf2e-b3f870e5369e
begin
    import Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
end

# ╔═╡ e1084400-e25c-4f9f-bdd7-c919cdf50ad8
using PlutoUI, ExprGraphExplorer, PlutoTeachingTools

# ╔═╡ 6956a53c-58f2-4ee1-b9a5-c74678752eb5
include("scalar_reverse.jl")

# ╔═╡ ff7556a4-0500-4a0c-b9a3-b10f1580dfe5
PlutoTeachingTools.ChooseDisplayMode()

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
    graph = ScalarReverseExample.example(x = 2, y = 3)
    states = ScalarReverseExample.frames(graph)
end

# ╔═╡ ca498fc8-0917-4a97-bac0-1224ab2e2365
@bind step Slider(eachindex(states); default = 1, show_value = true)

# ╔═╡ 5b273195-e5f2-4e82-846c-53db223652ac
HTML(render_svg(graph, states[step]))

# ╔═╡ Cell order:
# ╠═8cbb9e65-f693-4328-bf2e-b3f870e5369e
# ╠═e1084400-e25c-4f9f-bdd7-c919cdf50ad8
# ╟─ff7556a4-0500-4a0c-b9a3-b10f1580dfe5
# ╠═6956a53c-58f2-4ee1-b9a5-c74678752eb5
# ╟─93b3f711-14bf-4be6-9f81-44c5bedb3e56
# ╠═5b4aeff2-1c64-433b-98da-b10dcbd03290
# ╟─ca498fc8-0917-4a97-bac0-1224ab2e2365
# ╟─5b273195-e5f2-4e82-846c-53db223652ac
