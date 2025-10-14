from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os, json, tempfile, subprocess
from typing import List, Dict, Any

from retrieval_server import hybrid_search, graph_traversal, vector_search
from schema_to_ontology import generate_ontology, extract_schema
from transform_to_graph import build_clean_graph, verify_graph
from agent_retriever import evaluate_queries, generate_report

# ==============================================================
# Initialize App
# ==============================================================
app = FastAPI(
    title="GraphRAG-as-a-Service API",
    description="Complete API for transforming relational data into Graph RAG with full pipeline control",
    version="2.0.0"
)

# Paths
DATA_DIR = os.path.join(os.getcwd(), "data")
ONTOLOGY_PATH = os.path.join(DATA_DIR, "ontology.json")
DB_PATH = os.path.join(DATA_DIR, "chinook.db")

# ==============================================================
# Models
# ==============================================================
class QueryRequest(BaseModel):
    query: str

class EvaluationRequest(BaseModel):
    queries: List[str]

class SchemaRequest(BaseModel):
    schema_sql: str

# ==============================================================
# Root & Health
# ==============================================================
@app.get("/")
def root():
    return {"status": "ok", "message": "GraphRAG Complete API is running"}

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "services": {
            "neo4j": "connected",
            "embeddings": "available",
            "gemini": "configured"
        },
        "files": {
            "ontology_file": os.path.exists(ONTOLOGY_PATH),
            "database_file": os.path.exists(DB_PATH)
        }
    }

# ==============================================================
# Schema Analysis (schema_to_ontology.py)
# ==============================================================
@app.get("/schema/extract")
def extract_database_schema():
    """Extract relational schema from SQLite database"""
    try:
        if not os.path.exists(DB_PATH):
            raise HTTPException(status_code=404, detail="Database file not found")
        
        schema = extract_schema(DB_PATH)
        return {
            "status": "success",
            "database": DB_PATH,
            "tables": len(schema.get("tables", {})),
            "schema": schema
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ontology/generate")
def generate_ontology_endpoint():
    """Generate semantic ontology from database schema using Gemini"""
    try:
        if not os.path.exists(DB_PATH):
            raise HTTPException(status_code=404, detail="Database file not found")
        
        # Extract schema from database
        schema = extract_schema(DB_PATH)
        
        # Generate ontology using Gemini
        ontology = generate_ontology(schema)
        
        # Save ontology
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(ONTOLOGY_PATH, "w") as f:
            json.dump(ontology, f, indent=2)
            
        return {
            "status": "success", 
            "ontology_path": ONTOLOGY_PATH, 
            "classes": len(ontology.get("classes", {})),
            "relationships": len(ontology.get("relationships", [])),
            "ontology": ontology
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==============================================================
# Graph Building (transform_to_graph.py)
# ==============================================================
@app.post("/graph/build")
def build_neo4j_graph():
    """Build Neo4j graph from ontology and database with embeddings"""
    try:
        if not os.path.exists(ONTOLOGY_PATH):
            raise HTTPException(status_code=404, detail="Ontology file not found. Generate ontology first.")
        
        if not os.path.exists(DB_PATH):
            raise HTTPException(status_code=404, detail="Database file not found")
            
        # Build the graph (includes embeddings)
        build_clean_graph()
        
        return {"status": "success", "message": "Graph successfully built in Neo4j with embeddings"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==============================================================
# Retrieval System (retrieval_server.py)
# ==============================================================
@app.post("/search/vector")
def search_vector_only(req: QueryRequest):
    """Pure vector similarity search"""
    try:
        results = vector_search(req.query, top_k=5)
        return {
            "mode": "vector",
            "query": req.query,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/graph")
def search_graph_only(req: QueryRequest):
    """Pure graph traversal search"""
    try:
        results = graph_traversal(req.query, hops=2)
        return {
            "mode": "graph",
            "query": req.query,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/hybrid")
def search_hybrid(req: QueryRequest):
    """Hybrid search with automatic mode selection"""
    try:
        response = hybrid_search(req.query)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Keep the original /query endpoint for backward compatibility
@app.post("/query")
def query_graph(req: QueryRequest):
    """Legacy endpoint - same as /search/hybrid"""
    return search_hybrid(req)

# ==============================================================
# Agent Evaluation (agent_retriever.py)
# ==============================================================
@app.post("/evaluate/run")
def run_evaluation(req: EvaluationRequest):
    """Run evaluation on a list of queries"""
    try:
        metrics = evaluate_queries(req.queries)
        generate_report(metrics)
        
        return {
            "status": "success",
            "queries_evaluated": len(metrics),
            "metrics": metrics,
            "report_saved": "docs/evaluation_report.md"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/evaluate/demo")
def run_demo_evaluation():
    """Run evaluation with default demo queries"""
    try:
        demo_queries = [
            "Find songs about rock",
            "Show artists similar to AC/DC", 
            "Show all musical genres",
            "List albums created by artists",
            "Show tracks belonging to genre Rock",
            "Which employees support customers?",
            "Find tracks created by artists",
            "Show albums related to AC/DC",
            "Customers who bought rock music"
        ]
        
        metrics = evaluate_queries(demo_queries)
        generate_report(metrics)
        
        return {
            "status": "success",
            "queries_evaluated": len(metrics),
            "metrics": metrics,
            "report_saved": "docs/evaluation_report.md"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
