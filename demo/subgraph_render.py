"""Matplotlib + networkx render of a joint subgraph for the P5 demo.

Exports `render_subgraph(nodes, edges) -> matplotlib.figure.Figure`. See
spec §3.2 for the type->color/shape table.
"""
from __future__ import annotations

from typing import Any

# Type -> (color, marker_shape) per spec §3.2.
TYPE_STYLE: dict[str, tuple[str, str]] = {
    "Concept":     ("#FFC107", "o"),  # amber, circle
    "Formula":     ("#2196F3", "s"),  # blue, square
    "Rule":        ("#9C27B0", "s"),  # purple, square
    "ValueMap":    ("#4CAF50", "D"),  # green, diamond
    "ColumnAlias": ("#FF5722", "h"),  # orange, hexagon
    "Column":      ("#9E9E9E", "o"),  # gray, small circle (L2)
}

# Edge style per kind.
EDGE_STYLE: dict[str, dict[str, Any]] = {
    "defined_by": {"style": "solid",  "label": "defines"},
    "depends_on": {"style": "dashed", "label": "depends"},
    "grounds":    {"style": "dotted", "label": "grounds"},
}


def _short_label(text: str, max_chars: int = 22) -> str:
    """Truncate long node labels so they don't overflow the node marker.

    e.g. 'frpm.Charter School (Y/N)' -> 'frpm.Charter School…'
    """
    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def render_subgraph(nodes: list[dict], edges: list[dict]):
    """Render a joint subgraph as a matplotlib Figure.

    `nodes`: list of dicts with at least `id`, `type`, `name`.
    `edges`: list of dicts with `src`, `dst`, `kind`.
    """
    # Heavyweight imports kept inside the function so the demo module can be
    # imported in headless test environments without matplotlib installed.
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    import networkx as nx

    g = nx.DiGraph()
    for n in nodes:
        g.add_node(n["id"], **n)
    for e in edges:
        if e["src"] in g and e["dst"] in g:
            g.add_edge(e["src"], e["dst"], kind=e["kind"])

    # Larger canvas — small graphs need room so labels don't pile up on edges.
    fig, ax = plt.subplots(figsize=(11, 8))

    if len(g) == 0:
        ax.text(0.5, 0.5, "(empty subgraph)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="#666")
        ax.axis("off")
        return fig

    # Layout — kamada_kawai gives much better spacing on small graphs (<20
    # nodes) than spring_layout. Falls back to spring if kamada fails
    # (e.g. disconnected graphs sometimes raise).
    try:
        pos = nx.kamada_kawai_layout(g, scale=1.0)
    except Exception:
        pos = nx.spring_layout(g, seed=42, k=2.0, iterations=80)

    # Push L2 Column nodes slightly outward — they're support nodes, not
    # the focus of the diagram. Helps visual hierarchy.
    for nid, p in list(pos.items()):
        n = g.nodes[nid]
        if n.get("type") == "Column":
            pos[nid] = (p[0] * 1.15, p[1] * 1.15)

    # Draw nodes per type so each type gets its own color + marker shape.
    for ntype, (color, shape) in TYPE_STYLE.items():
        ids = [n["id"] for n in nodes if n.get("type") == ntype]
        if not ids:
            continue
        size = 350 if ntype == "Column" else 1400
        nx.draw_networkx_nodes(
            g, pos, nodelist=ids, node_color=color,
            node_shape=shape, node_size=size,
            edgecolors="#222", linewidths=1.4, ax=ax,
        )

    # Edges per kind (drawn BEFORE labels so labels sit on top with bbox).
    edge_label_targets: dict[str, list[tuple[str, str]]] = {}
    for kind, style in EDGE_STYLE.items():
        edgelist = [(e["src"], e["dst"]) for e in edges
                    if e["kind"] == kind and e["src"] in g and e["dst"] in g]
        if not edgelist:
            continue
        nx.draw_networkx_edges(
            g, pos, edgelist=edgelist,
            style=style["style"], arrows=True, arrowsize=16,
            edge_color="#555", width=1.4,
            connectionstyle="arc3,rad=0.06",  # slight curve avoids overlap
            ax=ax,
        )
        edge_label_targets[kind] = edgelist

    # Node labels — with white bbox so they read clearly on top of any edge.
    labels = {n["id"]: _short_label(n.get("name") or n["id"]) for n in nodes}
    for nid, text in labels.items():
        if nid not in pos:
            continue
        x, y = pos[nid]
        ax.text(
            x, y, text,
            fontsize=9, ha="center", va="center",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="#999",
                linewidth=0.5,
                alpha=0.95,
            ),
            zorder=10,
        )

    # Edge kind labels — only for the "defines" / "depends" edges (grounds
    # is implicit from the dotted style; labelling all of them clutters).
    for kind in ("defined_by", "depends_on"):
        edgelist = edge_label_targets.get(kind, [])
        if not edgelist:
            continue
        elabels = {(s, d): EDGE_STYLE[kind]["label"] for (s, d) in edgelist}
        nx.draw_networkx_edge_labels(
            g, pos, edge_labels=elabels, font_size=7,
            font_color="#444",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="none", alpha=0.85),
            label_pos=0.5, ax=ax,
        )

    # Legend outside the axes — top right, in its own band.
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker=shape, linestyle="", color=color,
               markersize=11, label=t, markeredgecolor="#222")
        for t, (color, shape) in TYPE_STYLE.items()
        if any(n.get("type") == t for n in nodes)
    ]
    if handles:
        ax.legend(
            handles=handles, loc="upper left",
            bbox_to_anchor=(1.0, 1.0),
            fontsize=9, frameon=True, framealpha=0.95,
        )

    # Small margin so node labels at the edges don't get clipped.
    x_vals = [p[0] for p in pos.values()]
    y_vals = [p[1] for p in pos.values()]
    if x_vals and y_vals:
        x_pad = (max(x_vals) - min(x_vals)) * 0.18 + 0.1
        y_pad = (max(y_vals) - min(y_vals)) * 0.18 + 0.1
        ax.set_xlim(min(x_vals) - x_pad, max(x_vals) + x_pad)
        ax.set_ylim(min(y_vals) - y_pad, max(y_vals) + y_pad)

    ax.axis("off")
    fig.tight_layout()
    return fig
