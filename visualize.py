#!/usr/bin/env python3
"""
Graph visualization script for GraphML files.

Usage:
    python visualize.py output/math/math_primes.graphml
    python visualize.py output/math/math_primes.graphml --output graph.png
    python visualize.py output/math/math_primes.graphml --layout spring
"""

import argparse
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def visualize_graph(
    graphml_path: str,
    output_path: str = None,
    layout: str = "spring",
    figsize: tuple = (20, 15),
    node_size: int = 40,
    font_size: int = 5,
    show_edge_labels: bool = False,
    k: float = 5.0,
    iterations: int = 800,
):
    """
    Visualize a GraphML file with spread-out layout to prevent overlap.
    """
    # Load graph
    G = nx.read_graphml(graphml_path)

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    print(f"Loaded graph: {n_nodes} nodes, {n_edges} edges")

    # Print node info
    print("\nNodes:")
    for node_id, attrs in G.nodes(data=True):
        label = attrs.get('label', node_id)
        node_type = attrs.get('type', '?')
        print(f"  [{node_type}] {label}")

    # Print edge info
    print("\nEdges:")
    for u, v, attrs in G.edges(data=True):
        relation = attrs.get('relation', '?')
        print(f"  {u} -[{relation}]-> {v}")

    # Choose layout - use high k value to spread nodes apart
    if layout == "spring":
        pos = nx.spring_layout(G, seed=42, k=k, iterations=iterations)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "shell":
        pos = nx.shell_layout(G)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "spectral":
        pos = nx.spectral_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42, k=k, iterations=iterations)

    # Color nodes by type
    node_types = set(attrs.get('type', 'unknown') for _, attrs in G.nodes(data=True))
    colors = plt.cm.Set3(np.linspace(0, 1, max(len(node_types), 1)))
    color_map = {t: colors[i] for i, t in enumerate(sorted(node_types))}
    node_colors = [color_map.get(G.nodes[n].get('type', 'unknown'), 'lightgray') for n in G.nodes()]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    # Draw edges first (so they're behind nodes/labels)
    nx.draw_networkx_edges(
        G, pos,
        ax=ax,
        alpha=0.3,
        width=0.5,
        edge_color='gray',
        arrows=True,
        arrowsize=5,
        connectionstyle="arc3,rad=0.1",
    )

    # Draw edge labels along arcs
    if show_edge_labels:
        edge_labels = {(u, v): d.get('relation', '')[:12] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_size=font_size - 1,
            alpha=0.6,
            ax=ax,
        )

    # Draw nodes (small)
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_size,
        alpha=0.95,
        ax=ax,
        linewidths=0.3,
        edgecolors='black',
    )

    # Calculate span for label offset
    xs = [pos[n][0] for n in G.nodes()]
    ys = [pos[n][1] for n in G.nodes()]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)

    # Base offset for labels (in data coordinates)
    base_dy = 0.012 * span
    base_dx = 0.003 * span

    # Draw labels ABOVE nodes with jitter to reduce overlap
    for n in G.nodes():
        x, y = pos[n]
        label = G.nodes[n].get('label', n)[:20]

        # Deterministic jitter based on node hash
        h = (hash(n) % 360) * (math.pi / 180.0)
        jitter_r = 0.006 * span
        dx = base_dx * math.cos(h) + jitter_r * math.cos(2 * h)
        dy = base_dy + base_dx * math.sin(h) + jitter_r * math.sin(2 * h)

        ax.text(
            x + dx,
            y + dy,
            label,
            fontsize=font_size,
            ha='center',
            va='bottom',
            bbox=dict(
                boxstyle='round,pad=0.1',
                facecolor='white',
                edgecolor='none',
                alpha=0.7,
            ),
            zorder=10,
        )

    # Legend for node types
    from matplotlib.patches import Patch
    legend_handles = [Patch(color=color_map[t], label=t) for t in sorted(node_types)]
    ax.legend(handles=legend_handles, loc='upper left', title='Node Types', fontsize=font_size + 1)

    ax.set_title(f"Knowledge Graph: {n_nodes} nodes, {n_edges} edges", fontsize=10)
    ax.axis('off')
    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\nSaved to: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize GraphML knowledge graphs")
    parser.add_argument("graphml_path", help="Path to .graphml file")
    parser.add_argument("-o", "--output", help="Output image path (PNG/PDF/SVG)")
    parser.add_argument("-l", "--layout", default="spring",
                        choices=["spring", "circular", "shell", "kamada_kawai", "spectral"],
                        help="Layout algorithm")
    parser.add_argument("--edge-labels", action="store_true", help="Show edge labels")
    parser.add_argument("--figsize", type=int, nargs=2, default=[20, 15], help="Figure size")
    parser.add_argument("--node-size", type=int, default=40, help="Node size")
    parser.add_argument("--font-size", type=int, default=5, help="Font size")
    parser.add_argument("-k", type=float, default=5.0, help="Spring layout spacing (higher = more spread)")
    parser.add_argument("--iterations", type=int, default=800, help="Spring layout iterations")

    args = parser.parse_args()

    visualize_graph(
        graphml_path=args.graphml_path,
        output_path=args.output,
        layout=args.layout,
        figsize=tuple(args.figsize),
        node_size=args.node_size,
        font_size=args.font_size,
        show_edge_labels=args.edge_labels,
        k=args.k,
        iterations=args.iterations,
    )


if __name__ == "__main__":
    main()
