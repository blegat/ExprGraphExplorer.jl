struct Frame{T,M}
    title::String
    active::Union{Nothing,ExprNode{T,M}}
    values::IdDict{ExprNode{T,M},Bool}
    metadata::IdDict{ExprNode{T,M},Vector{Pair{String,String}}}
end

function capture_frame(
    graph::ExprGraph{T,M},
    title::AbstractString;
    active = nothing,
    visible = Set(topological_order(graph.output)),
    show_metadata = true,
) where {T,M}
    values = IdDict(node => node in visible for node in topological_order(graph.output))
    rows = IdDict{ExprNode{T,M},Vector{Pair{String,String}}}()
    for node in topological_order(graph.output)
        rows[node] =
            show_metadata ? copy(metadata_rows(node.metadata)) : Pair{String,String}[]
    end
    return Frame{T,M}(String(title), active, values, rows)
end

function forward_frames(graph::ExprGraph)
    order = topological_order(graph.output)
    result = Frame[]
    visible = Set{eltype(order)}()
    push!(result, capture_frame(graph, "Expression graph"; visible, show_metadata = false))
    for node in order
        push!(visible, node)
        push!(
            result,
            capture_frame(
                graph,
                "Forward pass: evaluate $(graph.names[node])";
                active = node,
                visible,
                show_metadata = false,
            ),
        )
    end
    return result
end

_fmt(x::Number) = isinteger(x) ? string(Int(x)) : string(round(x; digits = 3))
_fmt(x::AbstractVector) = "[" * join(_fmt.(x), ", ") * "]"
function _fmt(x::AbstractMatrix)
    length(x) > 16 && return string(summary(x))
    rows = (join((_fmt(x[i, j]) for j in axes(x, 2)), " ") for i in axes(x, 1))
    return "[" * join(rows, "; ") * "]"
end
_fmt(x) = string(summary(x))

const _typst_preamble = Typstry.TypstString(
    Typstry.TypstText(
        "#set page(width: auto, height: auto, margin: 0pt, fill: none); #set text(size: 14pt);",
    ),
)
const _array_image_cache = Dict{String,Luxor.SVGimage}()
function _array_image(value::Union{AbstractVector,AbstractMatrix})
    key = string(typeof(value), ':', _fmt(value))
    return get!(_array_image_cache, key) do
        displayed = value isa AbstractVector ? reshape(value, :, 1) : value
        array = Typstry.TypstString(displayed; mode = Typstry.markup, delim = "[")
        io = IOBuffer()
        show(IOContext(io, :preamble => _typst_preamble), "image/svg+xml", array)
        svg = String(take!(io))
        # Typst exports its page as an opaque white path even with `fill: none`.
        # The page is only a container here, so remove it before compositing.
        svg = replace(
            svg,
            r"\n\s*<path class=\"typst-shape\" fill=\"#ffffff\"[^>]*/>" => "";
            count = 1,
        )
        return Luxor.readsvg(svg)
    end
end

function _depth!(depth, node)
    haskey(depth, node) && return depth[node]
    depth[node] =
        isempty(node.args) ? 0 : 1 + maximum(_depth!(depth, arg) for arg in node.args)
end

function _positions(order; width = 1100, height = 560, compact = false)
    depth = IdDict{eltype(order),Int}()
    foreach(node -> _depth!(depth, node), order)
    maxdepth = maximum(values(depth))
    positions = IdDict{eltype(order),Tuple{Float64,Float64}}()
    for d = 0:maxdepth
        level = [node for node in order if depth[node] == d]
        for (index, node) in enumerate(level)
            y = if compact
                height / 2 + (index - (length(level) + 1) / 2) * 110
            else
                index * height / (length(level) + 1)
            end
            positions[node] = (100 + d * (width - 200) / max(maxdepth, 1), y)
        end
    end
    return positions, depth
end

function _draw_array_value(value, point)
    image = _array_image(value)
    Luxor.fontface("sans-serif")
    Luxor.fontsize(15)
    prefix = "value ="
    prefix_width = Luxor.textextents(prefix)[5]
    gap = 5
    total_width = prefix_width + gap + image.width
    left = point.x - total_width / 2
    Luxor.text(prefix, Luxor.Point(left, point.y); halign = :left, valign = :middle)
    Luxor.placeimage(
        image,
        Luxor.Point(left + prefix_width + gap, point.y - image.height / 2),
    )
end

function _draw_graph(
    graph::ExprGraph,
    frame::Frame;
    width = 1100,
    height = 620,
    exam = false,
)
    order = topological_order(graph.output)
    positions, depth =
        _positions(order; width, height = exam ? height : height - 60, compact = exam)
    Luxor.background("white")
    if !exam
        Luxor.sethue("black")
        Luxor.fontface("DejaVu Sans Bold")
        Luxor.fontsize(21)
        Luxor.text(
            frame.title,
            Luxor.Point(width / 2, 30);
            halign = :center,
            valign = :middle,
        )
    end
    Luxor.sethue("#667085")
    for node in order, arg in node.args
        x1, y1 = positions[arg]
        x2, y2 = positions[node]
        startx, endx = x1 + 68, x2 - 72
        start = Luxor.Point(startx, y1)
        finish = Luxor.Point(endx, y2)
        Luxor.arrow(start, finish; linewidth = 2.5, arrowheadlength = 10)
    end
    for node in order
        x, y = positions[node]
        center = Luxor.Point(x, y)
        active = node === frame.active
        fill = active ? "#fef0c7" : (isnothing(node.op) ? "#e0f2fe" : "#f2f4f7")
        stroke = active ? "#f79009" : "#344054"
        Luxor.sethue(fill)
        Luxor.box(center, 136, 84, 12, :fill)
        Luxor.sethue(stroke)
        Luxor.setline(active ? 4 : 2)
        Luxor.box(center, 136, 84, 12, :stroke)
        op = isnothing(node.op) ? "input" : string(node.op)
        Luxor.sethue("black")
        Luxor.fontface("DejaVu Sans Bold")
        Luxor.fontsize(19)
        Luxor.text(
            "$(graph.names[node])  [$op]",
            Luxor.Point(x, y - 15);
            halign = :center,
            valign = :middle,
        )
        if frame.values[node] &&
           node.value isa Union{AbstractVector,AbstractMatrix} &&
           length(node.value) <= 16
            _draw_array_value(node.value, Luxor.Point(x, y + 13))
            value_y = y + 13
        else
            value = frame.values[node] ? _fmt(node.value) : "?"
            Luxor.fontface("sans-serif")
            Luxor.fontsize(17)
            Luxor.text(
                "value = $value",
                Luxor.Point(x, y + 13);
                halign = :center,
                valign = :middle,
            )
            value_y = y + 13
        end
        exam && continue
        for (index, row) in enumerate(frame.metadata[node])
            metadata_y = value_y + 21index
            Luxor.sethue("#175cd3")
            Luxor.fontsize(16)
            Luxor.text(
                "$(row.first) = $(row.second)",
                Luxor.Point(x, metadata_y);
                halign = :center,
                valign = :middle,
            )
        end
    end
end

function _default_height(graph, exam)
    exam || return 620
    order = topological_order(graph.output)
    depth = IdDict{eltype(order),Int}()
    foreach(node -> _depth!(depth, node), order)
    largest_level = maximum(d -> count(==(d), values(depth)), values(depth))
    return 40 + 110largest_level
end

function _render(
    graph,
    frame,
    surface;
    width = 1100,
    height = nothing,
    exam = false,
    path = "",
    scale = 1.0,
)
    height = isnothing(height) ? _default_height(graph, exam) : height
    canvas_width = surface == :png ? round(Int, scale * width) : scale * width
    canvas_height = surface == :png ? round(Int, scale * height) : scale * height
    drawing = Luxor.Drawing(canvas_width, canvas_height, surface, path)
    Luxor.scale(scale)
    _draw_graph(graph, frame; width, height, exam)
    Luxor.finish()
    return drawing
end

function render_svg(
    graph::ExprGraph,
    frame::Frame;
    width = 1100,
    height = nothing,
    exam = false,
    responsive = true,
)
    height = isnothing(height) ? _default_height(graph, exam) : height
    _render(graph, frame, :svg; width, height, exam)
    svg = Luxor.svgstring()
    if responsive
        root = "<svg xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" width=\"100%\" viewBox=\"0 0 $width $height\" preserveAspectRatio=\"xMidYMid meet\" style=\"display:block;height:auto\">"
        svg = replace(svg, r"<svg[^>]*>" => root; count = 1)
    end
    return svg
end

function save_svg(path, graph::ExprGraph, frame::Frame; responsive = false, kwargs...)
    if responsive
        write(path, render_svg(graph, frame; responsive, kwargs...))
    else
        _render(graph, frame, :svg; path, kwargs...)
    end
    return path
end

function save_png(
    path,
    graph::ExprGraph,
    frame::Frame;
    density = 180,
    width = 1600,
    height = nothing,
    exam = false,
)
    height = isnothing(height) ? _default_height(graph, exam) : height
    scale = width / 1100
    _render(graph, frame, :png; width = 1100, height, exam, path, scale)
    return path
end

function save_eps(path, graph::ExprGraph, frame::Frame; kwargs...)
    _render(graph, frame, :eps; path, kwargs...)
    return path
end
