"""
Unified Retrieval Layer for RelationalDB → GraphRAG
----------------------------------------------------
Combines vector similarity, graph traversal, and logical filtering.
"""

import os
import json
import numpy as np
from fastapi import FastAPI, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from rich.console import Console
import uvicorn

load_dotenv()
console = Console()

# ===== Neo4j Config =====
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "neo4j1234")
DB_NAME = os.getenv("NEO4J_DB", "neo4j")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# ===== Embedding Model =====
model = SentenceTransformer("all-MiniLM-L6-v2")
console.print("[cyan]Loaded SentenceTransformer for retrieval.[/cyan]")

# ===== FastAPI App =====
app = FastAPI(title="GraphRAG Retrieval API", version="1.0")

# ===== Helper: Vector Search =====
def vector_search(query_text, top_k=5):
    q_emb = model.encode([query_text])[0]
    with driver.session(database=DB_NAME) as session:
        results = session.run("MATCH (n) WHERE n.embedding IS NOT NULL RETURN n.id AS id, n.name AS name, n.embedding AS emb LIMIT 500")
        scored = []
        for r in results:
            if not r["emb"]: continue
            emb = np.array(r["emb"])
            score = cosine_similarity([q_emb], [emb])[0][0]
            scored.append((r["id"], r["name"], float(score)))
        top = sorted(scored, key=lambda x: x[2], reverse=True)[:top_k]
    return [{"id": i, "name": n, "score": s} for i, n, s in top]

# ===== Helper: Graph Traversal =====
def graph_traversal(query_entity, hops=2):
    with driver.session(database=DB_NAME) as session:
        cypher = f"""
        MATCH (n)-[r*1..{hops}]-(m)
        WHERE n.name CONTAINS $query_entity OR n.id CONTAINS $query_entity
        RETURN DISTINCT labels(n) as n_labels, labels(m) as m_labels, n.name as n_name, m.name as m_name LIMIT 20
        """
        result = session.run(cypher, query_entity=query_entity)
        return [{"n": {"labels": record["n_labels"], "name": record["n_name"]}, 
                "m": {"labels": record["m_labels"], "name": record["m_name"]}} 
                for record in result]

# ===== Helper: Hybrid Retrieval =====
def hybrid_search(nl_query):
    """
    Use simple routing logic:
    - If query mentions 'similar' → vector
    - If query mentions 'related to' / 'connected to' → graph
    - Otherwise → hybrid (vector top-k + graph expand)
    """
    query = nl_query.lower()

    if "similar" in query:
        return {"mode": "vector", "results": vector_search(nl_query)}

    elif "related" in query or "connected" in query:
        # Extract entity name from query - simple approach
        words = query.split()
        entity = words[-1] if words else ""
        return {"mode": "graph", "results": graph_traversal(entity)}

    else:
        top_nodes = vector_search(nl_query, top_k=3)
        expanded = []
        for node in top_nodes:
            # Use the node name for graph traversal
            node_name = node.get("name", "")
            if node_name:
                expanded.extend(graph_traversal(node_name, hops=1))
        
        return {
            "mode": "hybrid", 
            "vector_hits": top_nodes, 
            "graph_context": expanded[:10]  # Limit context to avoid too much data
        }

# ===== Routes =====
class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def query_graph(req: QueryRequest):
    return hybrid_search(req.query)

@app.get("/")
def home():
    return {"message": "GraphRAG Unified Retrieval API running 🚀"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
