"""
Knowledge Graph: Builds a graph where:
  - Nodes = entities (people, organizations, technologies, topics)
  - Edges = co-occurrence (two entities appeared in the same chunk)
  - Edge weight = how many chunks they co-occur in

Uses NetworkX for the graph, saves as JSON for the API/UI.

Example:
  "Deep Learning" ──(15 chunks)──> "Neural Networks"
  "Deep Learning" ──(8 chunks)───> "Medical Imaging"
  "Germany" ───────(5 chunks)───> "Automotive Industry"
"""

import os
import json
import networkx as nx
from collections import defaultdict
from entity_extractor import EntityExtractor
from config import CHUNKS_PATH, DATA_DIR


class KnowledgeGraph:
    """Build and query a knowledge graph from document chunks."""

    def __init__(self, load_existing=True):
        """
        Initialize the knowledge graph.
        If a saved graph exists, load it. Otherwise build from scratch.
        """
        self.graph = nx.Graph()
        self.entity_index = {}
        self.chunk_entities = {}

        graph_path = os.path.join(DATA_DIR, "knowledge_graph.json")

        if load_existing and os.path.exists(graph_path):
            self._load_graph(graph_path)
        else:
            self._build_graph()
            self._save_graph(graph_path)

    def _build_graph(self):
        """Build the graph by extracting entities and finding co-occurrences."""
        print("\n  Building knowledge graph...")

        # Step 1: Extract entities from all chunks
        extractor = EntityExtractor()
        self.entity_index, self.chunk_entities = extractor.extract_from_all_chunks()

        # Step 2: Add entity nodes
        for name, info in self.entity_index.items():
            self.graph.add_node(
                name,
                entity_type=info["type"],
                chunk_count=info["count"],
                chunk_ids=info["chunk_ids"],
            )

        # Step 3: Build edges from co-occurrence
        # Two entities in the same chunk = one connection
        print(f"\n  Building edges from co-occurrences...")
        edge_counts = defaultdict(int)
        edge_chunks = defaultdict(list)

        for chunk_id, entities in self.chunk_entities.items():
            # Filter to only entities in our index (2+ occurrences)
            valid_entities = [e for e in entities if e in self.entity_index]

            # Every pair of entities in the same chunk gets an edge
            for i in range(len(valid_entities)):
                for j in range(i + 1, len(valid_entities)):
                    e1, e2 = sorted([valid_entities[i], valid_entities[j]])
                    edge_key = (e1, e2)
                    edge_counts[edge_key] += 1
                    edge_chunks[edge_key].append(chunk_id)

        # Add edges with weight = co-occurrence count
        for (e1, e2), count in edge_counts.items():
            if count >= 2:  # Only keep edges with 2+ co-occurrences
                self.graph.add_edge(
                    e1, e2,
                    weight=count,
                    chunk_ids=edge_chunks[(e1, e2)][:10],  # Keep max 10 chunk refs
                )

        print(f"  Knowledge graph built:")
        print(f"    Nodes (entities): {self.graph.number_of_nodes()}")
        print(f"    Edges (connections): {self.graph.number_of_edges()}")

        # Show most connected entities
        if self.graph.number_of_nodes() > 0:
            degrees = sorted(self.graph.degree(), key=lambda x: x[1], reverse=True)[:10]
            print(f"\n  Most connected entities:")
            for name, degree in degrees:
                etype = self.graph.nodes[name].get("entity_type", "?")
                print(f"    {name} ({etype}): {degree} connections")

    def _save_graph(self, path):
        """Save graph as JSON."""
        data = {
            "nodes": [],
            "edges": [],
            "stats": {
                "num_nodes": self.graph.number_of_nodes(),
                "num_edges": self.graph.number_of_edges(),
            }
        }

        for node, attrs in self.graph.nodes(data=True):
            data["nodes"].append({
                "id": node,
                "type": attrs.get("entity_type", "UNKNOWN"),
                "chunk_count": attrs.get("chunk_count", 0),
                "connections": self.graph.degree(node),
            })

        for e1, e2, attrs in self.graph.edges(data=True):
            data["edges"].append({
                "source": e1,
                "target": e2,
                "weight": attrs.get("weight", 1),
                "chunk_ids": attrs.get("chunk_ids", []),
            })

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved graph to {path}")

        # Also save entity index
        entity_path = os.path.join(DATA_DIR, "entities.json")
        with open(entity_path, "w", encoding="utf-8") as f:
            json.dump(self.entity_index, f, indent=2, ensure_ascii=False, default=str)

    def _load_graph(self, path):
        """Load graph from JSON."""
        print(f"  Loading existing knowledge graph from {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for node in data["nodes"]:
            self.graph.add_node(
                node["id"],
                entity_type=node["type"],
                chunk_count=node.get("chunk_count", 0),
            )

        for edge in data["edges"]:
            self.graph.add_edge(
                edge["source"],
                edge["target"],
                weight=edge.get("weight", 1),
                chunk_ids=edge.get("chunk_ids", []),
            )

        print(f"    Nodes: {self.graph.number_of_nodes()}")
        print(f"    Edges: {self.graph.number_of_edges()}")

    def find_related_entities(self, entity_name, top_k=10):
        """
        Find entities most closely related to the given entity.
        Returns neighbors sorted by edge weight (co-occurrence count).
        """
        # Try exact match first
        if entity_name in self.graph:
            node = entity_name
        else:
            # Try case-insensitive match
            name_lower = entity_name.lower()
            node = None
            for n in self.graph.nodes():
                if n.lower() == name_lower:
                    node = n
                    break

        if node is None:
            return []

        neighbors = []
        for neighbor in self.graph.neighbors(node):
            edge_data = self.graph.edges[node, neighbor]
            neighbors.append({
                "entity": neighbor,
                "type": self.graph.nodes[neighbor].get("entity_type", "?"),
                "weight": edge_data.get("weight", 1),
                "chunk_ids": edge_data.get("chunk_ids", []),
            })

        neighbors.sort(key=lambda x: x["weight"], reverse=True)
        return neighbors[:top_k]

    def find_entities_in_query(self, query):
        """
        Find which graph entities are mentioned in a query.
        Returns list of matching entity names.
        """
        query_lower = query.lower()
        matches = []

        for node in self.graph.nodes():
            if node.lower() in query_lower:
                matches.append(node)
            elif len(node) > 4 and any(
                word in query_lower
                for word in node.lower().split()
                if len(word) > 3
            ):
                matches.append(node)

        return matches

    def get_graph_enhanced_chunks(self, query, chunk_ids_already_found):
        """
        Use the knowledge graph to find additional relevant chunk IDs
        that the regular search might have missed.

        1. Find entities mentioned in the query
        2. Find their neighbors in the graph
        3. Get chunks where those neighbors appear
        4. Return chunk IDs not already in the search results
        """
        # Find entities in the query
        query_entities = self.find_entities_in_query(query)

        if not query_entities:
            return []

        # Get neighbor entities and their chunks
        extra_chunk_ids = set()
        already_found = set(chunk_ids_already_found)

        for entity in query_entities:
            related = self.find_related_entities(entity, top_k=5)
            for rel in related:
                for cid in rel.get("chunk_ids", []):
                    if cid not in already_found:
                        extra_chunk_ids.add(cid)

        return list(extra_chunk_ids)[:5]  # Return max 5 extra chunks

    def get_stats(self):
        """Get graph statistics."""
        if self.graph.number_of_nodes() == 0:
            return {"nodes": 0, "edges": 0}

        degrees = dict(self.graph.degree())
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "avg_connections": round(sum(degrees.values()) / len(degrees), 2),
            "max_connections": max(degrees.values()),
            "most_connected": max(degrees, key=degrees.get),
        }

    def get_subgraph(self, entity_name, depth=1):
        """
        Get a subgraph centered on an entity, for visualization.
        Returns nodes and edges within 'depth' hops.
        """
        if entity_name not in self.graph:
            # Try case-insensitive
            for n in self.graph.nodes():
                if n.lower() == entity_name.lower():
                    entity_name = n
                    break
            else:
                return {"nodes": [], "edges": []}

        # BFS to get nodes within depth
        visited = {entity_name}
        current_level = {entity_name}

        for _ in range(depth):
            next_level = set()
            for node in current_level:
                for neighbor in self.graph.neighbors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_level.add(neighbor)
            current_level = next_level

        # Build subgraph data
        subgraph = self.graph.subgraph(visited)
        nodes = []
        for node in subgraph.nodes():
            nodes.append({
                "id": node,
                "type": self.graph.nodes[node].get("entity_type", "?"),
                "connections": subgraph.degree(node),
                "is_center": node == entity_name,
            })

        edges = []
        for e1, e2, data in subgraph.edges(data=True):
            edges.append({
                "source": e1,
                "target": e2,
                "weight": data.get("weight", 1),
            })

        return {"nodes": nodes, "edges": edges}


# ──────────────────────────────────────────────
# Test it
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("KNOWLEDGE GRAPH TEST")
    print("=" * 60)

    # Build graph (first run) or load existing
    kg = KnowledgeGraph(load_existing=False)

    # Test: find related entities
    test_entities = ["Deep Learning", "Machine Learning", "Neural Networks"]

    for entity in test_entities:
        print(f"\n  Related to '{entity}':")
        related = kg.find_related_entities(entity, top_k=5)
        if related:
            for r in related:
                print(f"    {r['entity']} ({r['type']}) — {r['weight']} shared chunks")
        else:
            print(f"    (not found in graph)")

    # Test: find entities in a query
    test_query = "How is deep learning used in medical imaging?"
    print(f"\n  Entities in query '{test_query}':")
    matches = kg.find_entities_in_query(test_query)
    for m in matches:
        print(f"    {m}")

    # Stats
    print(f"\n  Graph stats:")
    stats = kg.get_stats()
    for k, v in stats.items():
        print(f"    {k}: {v}")
