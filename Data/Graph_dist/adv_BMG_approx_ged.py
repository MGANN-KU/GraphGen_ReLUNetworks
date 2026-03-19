#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    Graph Edit Distance only edge deletion and insertion is allowed
    Compute symmetric distance

"""

from scipy.optimize import linear_sum_assignment
import numpy as np
import sys

__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat"

class AproximatedEditDistance:
    """
    Symmetric edge distance only.
    Graphs must have the same vertex set.
    Only edge insertion and deletion are allowed.
    """

    def __init__(self, edge_cost=1):
        self.edge_cost = edge_cost

    def ged(self, adj1, adj2):
        """
        Compute symmetric distance between two graphs.

        GED = |E1 Δ E2|
        """
        if adj1.shape != adj2.shape:
            raise ValueError("Graphs must have same number of vertices")

        n = adj1.shape[0]
        total_edge_edits = 0

        for i in range(n):
            for j in range(i + 1, n):
                if adj1[i, j] != adj2[i, j]:
                    total_edge_edits += self.edge_cost

        # Keep return format unchanged
        return total_edge_edits, None


# ============================================================================
# FILE READING FUNCTIONS (FIXED FOR EMPTY LINES)
# ============================================================================
def read_graphs(filename):
    """Read graphs from adjacency list file.
    Each graph starts with number of vertices n.
    Then exactly n lines of adjacency lists (some may be empty).
    Graphs are separated by exactly one empty line.
    """
    graphs = []
    
    with open(filename, 'r') as f:
        lines = [line.rstrip('\n') for line in f]
    
    i = 0
    while i < len(lines):
        # Skip empty lines (separators between graphs)
        if lines[i] == '':
            i += 1
            continue
        
        # First non-empty line should be number of vertices
        try:
            n = int(lines[i].strip())
        except ValueError:
            print(f"Warning: Expected number of vertices, got '{lines[i]}'")
            i += 1
            continue
        
        i += 1
        
        # Collect exactly n adjacency lines
        adjacency_lines = []
        lines_collected = 0
        
        while lines_collected < n and i < len(lines):
            # Empty lines within a graph are valid (isolated vertices)
            adjacency_lines.append(lines[i])
            i += 1
            lines_collected += 1
        
        # If we didn't get enough lines, pad with empty lines
        while lines_collected < n:
            adjacency_lines.append('')
            lines_collected += 1
        
        # Parse this graph
        graph_lines = [str(n)] + adjacency_lines
        graph = parse_graph(graph_lines)
        if graph is not None:
            graphs.append(graph)
    
    return graphs
    
def parse_graph(lines):
    """Parse adjacency list to numpy matrix."""
    if not lines:
        return None
    
    # First line: number of vertices
    try:
        n = int(lines[0].strip())
    except ValueError:
        print(f"Warning: First line must be integer. Got: '{lines[0]}'")
        return None
    
    if n == 0:
        return np.zeros((0, 0), dtype=int)
    
    # Check if we have enough lines
    if len(lines) < n + 1:
        print(f"Warning: Expected {n+1} lines, got {len(lines)}. Padding with empty lines.")
        # Pad with empty lines
        lines += [''] * (n + 1 - len(lines))
    
    adj = np.zeros((n, n), dtype=int)
    
    # Parse adjacency lists
    for i in range(1, n + 1):
        vertex = i - 1
        line = lines[i].strip()
        
        if line:
            parts = line.split()
            for part in parts:
                try:
                    nb = int(part)
                    if 0 <= nb < n and nb != vertex:
                        adj[vertex, nb] = 1
                        adj[nb, vertex] = 1
                except ValueError:
                    # Skip invalid entries
                    continue
        # Empty line = isolated vertex (no edges)
    
    return adj


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    if len(sys.argv) != 4:
        print("Usage: python AproximatedEditDistance.py fileA.txt fileB.txt d")
        print("\nRequirements:")
        print("1. Graphs must have SAME number of vertices")
        print("2. Only edge insertions/deletions (no vertex edits)")
        print("3. Edge edit cost = 1")
        print("4. Each edge counted only once")
        return
    
    fileA, fileB = sys.argv[1], sys.argv[2]
    d = int(sys.argv[3])
    
    # Read graphs
    graphsA = read_graphs(fileA)
    graphsB = read_graphs(fileB)
    
    if not graphsA:
        print(f"Error: No valid graphs found in {fileA}")
        return
    
    if not graphsB:
        print(f"Error: No valid graphs found in {fileB}")
        return
    
    graphA = graphsA[0]
    
    if graphA is None:
        print("Error: First graph in fileA could not be parsed")
        return
    
    # Get number of edges in graph A
    edges_A = np.sum(graphA) // 2
    vertices_A = graphA.shape[0]
    
    # Calculate edge thresholds
    min_edges_threshold = edges_A - d
    max_edges_threshold = edges_A + d
    
    # Print headers and graph A info
    print(f"Graph A: {vertices_A} vertices, {edges_A} edges")
    print(f"Edge thresholds: edges < {min_edges_threshold} or edges > {max_edges_threshold}")
    print()
    
    # Format headers with consistent column widths
    header_format = "{:<10}{:<10}{:<10}{:<15}{:<15}"
    row_format = "{:<10}{:<10}{:<10}{:<15}{:<15}"
    separator = "-" * 60
    
    print(header_format.format("Graph", "Nodes", "Edges", "GED", "EdgeStatus"))
    print(separator)
    
    # Initialize counters
    ged_at_most_d = 0           # GED ≤ d
    ged_above_d = 0             # GED > d
    edges_below_threshold = 0   # edges_B < edges_A - d
    edges_above_threshold = 0   # edges_B > edges_A + d
    edges_in_range = 0          # edges_A - d ≤ edges_B ≤ edges_A + d
    vertex_mismatch_count = 0   # Different vertex count
    invalid_graphs = 0          # Could not parse
    
    # Initialize algorithm
    ged_calculator = AproximatedEditDistance(edge_cost=1)
    
    # Process only first 500 graphs from file B
    max_graphs_to_process = min(500, len(graphsB))
    
    for i, graphB in enumerate(graphsB[:max_graphs_to_process], 1):
        # Skip None graphs
        if graphB is None:
            invalid_graphs += 1
            print(row_format.format(f"B{i}", "ERROR", "ERROR", "INVALID", "INVALID"))
            continue
        
        # Check if graphB has shape attribute
        if not hasattr(graphB, 'shape'):
            invalid_graphs += 1
            print(row_format.format(f"B{i}", "ERROR", "ERROR", "INVALID", "INVALID"))
            continue
        
        # Get edge count for graph B
        edges_B = np.sum(graphB) // 2
        nodes_B = graphB.shape[0]
        
        # Determine edge status
        if edges_B < min_edges_threshold:
            edge_status = "BELOW"
            edges_below_threshold += 1
        elif edges_B > max_edges_threshold:
            edge_status = "ABOVE"
            edges_above_threshold += 1
        else:
            edge_status = "IN_RANGE"
            edges_in_range += 1
        
        # Check vertex count
        if vertices_A != nodes_B:
            vertex_mismatch_count += 1
            print(row_format.format(f"B{i}", nodes_B, edges_B, "VERTEX_MISMATCH", edge_status))
            continue
        
        try:
            # Compute GED
            ged, _ = ged_calculator.ged(graphA, graphB)
            ged = int(ged)
        except Exception as e:
            error_msg = f"ERROR: {str(e)[:20]}"
            print(row_format.format(f"B{i}", nodes_B, edges_B, error_msg, edge_status))
            ged_above_d += 1
            continue
        
        # Output in tabular format
        print(row_format.format(f"B{i}", nodes_B, edges_B, ged, edge_status))
        
        # Count based on GED value
        if ged <= d:
            ged_at_most_d += 1
        else:
            ged_above_d += 1
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"STATISTICS (first {max_graphs_to_process} graphs from file B):")
    print(f"Graphs with GED <= {d}: {ged_at_most_d}")
    print(f"Graphs with GED > {d}: {ged_above_d}")
    print(f"Graphs with edges < {min_edges_threshold}: {edges_below_threshold}")
    print(f"Graphs with edges > {max_edges_threshold}: {edges_above_threshold}")
    print(f"Graphs with edges in range [{min_edges_threshold}, {max_edges_threshold}]: {edges_in_range}")
    print(f"Graphs with different vertex count: {vertex_mismatch_count}")
    if invalid_graphs > 0:
        print(f"Invalid graphs skipped: {invalid_graphs}")
    
    total_processed = (ged_at_most_d + ged_above_d + 
                      vertex_mismatch_count + invalid_graphs)
    print(f"Total graphs processed: {total_processed}")

if __name__ == "__main__":
    main()