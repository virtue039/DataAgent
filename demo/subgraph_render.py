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
        g.add_edge(e["src"], e["dst"], kind=e["kind"])

    fig, ax = plt.subplots(figsize=(8, 6))

    if len(g) == 0:
        ax.text(0.5, 0.5, "(empty subgraph)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="#666")
        ax.axis("off")
        return fig

    pos = nx.spring_layout(g, seed=42, k=1.2)

    # Draw nodes per type so each type gets its own color + marker shape.
    for ntype, (color, shape) in TYPE_STYLE.items():
        ids = [n["id"] for n in nodes if n.get("type") == ntype]
        if not ids:
            continue
        size = 250 if ntype == "Column" else 900
        nx.draw_networkx_nodes(
            g, pos, nodelist=ids, node_color=color,
            node_shape=shape, node_size=size,
            edgecolors="#333", linewidths=1.0, ax=ax,
        )

    # Node labels (use `name` when available, else id).
    labels = {n["id"]: (n.get("name") or n["id"]) for n in nodes}
    nx.draw_networkx_labels(g, pos, labels, font_size=8, ax=ax)

    # Edges per kind.
    for kind, style in EDGE_STYLE.items():
        edgelist = [(e["src"], e["dst"]) for e in edges if e["kind"] == kind]
        if not edgelist:
            continue
        nx.draw_networkx_edges(
            g, pos, edgelist=edgelist,
            style=style["style"], arrows=True, arrowsize=14,
            edge_color="#444", width=1.2, ax=ax,
        )
        # Edge labels (centered)
        elabels = {(s, d): style["label"] for (s, d) in edgelist}
        nx.draw_networkx_edge_labels(
            g, pos, edge_labels=elabels, font_size=7,
            font_color="#666", ax=ax,
        )

    # Legend.
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker=shape, linestyle="", color=color,
               markersize=10, label=t, markeredgecolor="#333")
        for t, (color, shape) in TYPE_STYLE.items()
        if any(n.get("type") == t for n in nodes)
    ]
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=7,
                  frameon=True, framealpha=0.9)

    ax.axis("off")
    fig.tight_layout()
    return fig
