import os
import json
import numpy as np
import torch
import sqlite3
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from rich.console import Console

# === Setup ===
console = Console()
load_dotenv()

ONTOLOGY_PATH = "data/ontology.json"
DB_PATH = "data/chinook.db"

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "neo4j1234")
DB_NAME = os.getenv("NEO4J_DB", "neo4j")

# Connect to Neo4j
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# Load ontology
with open(ONTOLOGY_PATH) as f:
    ontology = json.load(f)

classes = ontology.get("classes", ontology)
relations = ontology.get("relationships", [])

# Device setup
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
console.print(f"[cyan]Using device:[/cyan] {device}")

# === Parameters ===
DEDUP_THRESHOLD = 0.98
RELATION_SIM_THRESHOLD = 0.90
ALLOWED_RELATIONS = {"PURCHASED", "CONTAINS", "BELONGS_TO", "HAS", "INCLUDES", "CREATED_BY", "PLACED_BY", "REFERENCES", "CREATED", "SUPPORTS", "REPORTS_TO", "HAS_INVOICE", "IS_FOR", "HAS_TYPE", "HAS_GENRE"}

# === Utility ===
def cosine_sim(a, b):
    return cosine_similarity([a], [b])[0][0]

def get_embedding(text):
    """Get embedding and ensure it's a Python list of floats."""
    try:
        emb = model.encode([text], show_progress_bar=False)[0]
        # Convert numpy array to Python list of floats, filtering NaN values
        result = [float(x) for x in emb.tolist() if not np.isnan(x)]
        return result if result else [0.0] * len(emb)  # Fallback for all-NaN case
    except Exception as e:
        console.print(f"[red]Error generating embedding for text '{text[:50]}...': {e}[/red]")
        return [0.0] * 384  # Default embedding size for all-MiniLM-L6-v2

def create_node(tx, label, props):
    """Create a node with properties, handling embeddings properly."""
    # Separate embedding from other properties
    embedding = props.pop('embedding', None)
    
    # Create the basic properties query
    props_items = []
    for k, v in props.items():
        if k != 'id':  # id is used in MERGE clause
            props_items.append(f"n.{k} = ${k}")
    
    # Build the query
    if props_items:
        set_clause = ", ".join(props_items)
        query = f"MERGE (n:{label} {{id: $id}}) SET {set_clause}"
    else:
        query = f"MERGE (n:{label} {{id: $id}})"
    
    # Add embedding separately if it exists
    if embedding is not None:
        query += ", n.embedding = $embedding"
        props['embedding'] = embedding
    
    tx.run(query, **props)

def create_relation(tx, src_label, src_id, rel, dst_label, dst_id):
    rel_name = rel.upper().replace(' ', '_').replace('-', '_')
    query = (
        f"MATCH (a:{src_label} {{id: $src_id}}), (b:{dst_label} {{id: $dst_id}}) "
        f"MERGE (a)-[:{rel_name}]->(b)"
    )
    tx.run(query, src_id=src_id, dst_id=dst_id)

def get_sample_data(table_name: str, limit: int = 20):
    """Get sample data from SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit}")
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        return columns, rows
    except sqlite3.Error as e:
        console.print(f"[red]Error querying {table_name}: {e}[/red]")
        return [], []
    finally:
        conn.close()

def get_foreign_key_mappings():
    """Extract foreign key relationships from the database schema."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    fk_mappings = {}
    
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        for table in tables:
            cursor.execute(f"PRAGMA foreign_key_list({table})")
            fks = cursor.fetchall()
            
            for fk in fks:
                from_table = table
                to_table = fk[2]
                from_col = fk[3]
                to_col = fk[4]
                
                if from_table not in fk_mappings:
                    fk_mappings[from_table] = []
                
                fk_mappings[from_table].append({
                    'to_table': to_table,
                    'from_col': from_col,
                    'to_col': to_col
                })
    
    except sqlite3.Error as e:
        console.print(f"[red]Error getting foreign keys: {e}[/red]")
    finally:
        conn.close()
    
    return fk_mappings

# === Graph builder ===
def build_clean_graph():
    console.print("[bold blue] Cleaning database before rebuild...[/bold blue]")
    
    # Get foreign key mappings from actual database
    fk_mappings = get_foreign_key_mappings()
    
    with driver.session(database=DB_NAME) as session:
        # Clear existing data
        session.run("MATCH (n) DETACH DELETE n")
        console.print("[yellow]Cleared existing graph data[/yellow]")

    label_cache = {}
    node_mappings = {}  # Track created nodes for relationships

    with driver.session(database=DB_NAME) as session:
        # ---- Create nodes from actual database data ----
        for table, meta in tqdm(classes.items(), desc="Creating clean nodes"):
            label = meta.get("label", table)
            props = meta.get("properties", [])
            
            if label not in label_cache:
                label_cache[label] = []
            
            # Get actual data from database
            columns, rows = get_sample_data(table)
            
            if not columns or not rows:
                console.print(f"[yellow]No data found for table {table}[/yellow]")
                continue
            
            node_mappings[table] = []
            
            # Create nodes from actual database rows
            for i, row in enumerate(rows):
                # Create meaningful text for embedding
                text_parts = [f"{label} entity"]
                node_data = {"id": f"{table}_{i}"}
                
                # Add properties from ontology that exist in database
                for prop in props:
                    if prop in columns:
                        col_idx = columns.index(prop)
                        if col_idx < len(row) and row[col_idx] is not None:
                            prop_name = prop.lower().replace(' ', '_')
                            value = str(row[col_idx])
                            node_data[prop_name] = value
                            text_parts.append(f"{prop}: {value}")
                
                # Create rich descriptive text for better semantic meaning
                text = " ".join(text_parts)
                
                # Generate embedding and ensure it's Neo4j compatible
                vec = model.encode([text], show_progress_bar=False)[0]
                # Ensure it's a plain Python list of floats (Neo4j requirement)
                embedding = [float(x) for x in vec.tolist() if not np.isnan(x)]
                
                # Skip if embedding is empty or invalid
                if not embedding or len(embedding) == 0:
                    console.print(f"[red]Skipping node {table}_{i} - invalid embedding[/red]")
                    continue

                # Deduplication check
                existing_vecs = label_cache[label]
                if any(cosine_sim(embedding, v) >= DEDUP_THRESHOLD for v in existing_vecs):
                    continue
                
                label_cache[label].append(embedding)
                
                # Add embedding to node data
                node_data["embedding"] = embedding
                node_data["name"] = node_data.get("name", f"{table}_{i}")  # Ensure name exists
                
                # Store row data for foreign key relationships
                row_data = {}
                for j, col in enumerate(columns):
                    if j < len(row):
                        row_data[col] = row[j]
                
                node_mappings[table].append({
                    'id': node_data["id"],
                    'row_data': row_data,
                    'label': label
                })
                
                try:
                    session.execute_write(create_node, label, node_data)
                except Exception as e:
                    console.print(f"[red]Error creating node {node_data['id']}: {e}[/red]")

        # ---- Create ontology relationships ----
        console.print("[cyan]Creating ontology relationships...[/cyan]")
        for rel in tqdm(relations, desc="Creating ontology relationships"):
            src = rel.get("source")
            rel_type = rel.get("relation", "").upper()
            dst = rel.get("target")
            
            if not src or not dst or rel_type not in ALLOWED_RELATIONS:
                continue

            # Prevent overlinking similar classes (semantic pruning)
            if src in classes and dst in classes:
                src_label = classes[src].get("label", src)
                dst_label = classes[dst].get("label", dst)
                
                src_emb = get_embedding(src)
                dst_emb = get_embedding(dst)
                
                if cosine_sim(src_emb, dst_emb) < RELATION_SIM_THRESHOLD:
                    # Create relationships between sample nodes
                    src_nodes = node_mappings.get(src, [])[:3]
                    dst_nodes = node_mappings.get(dst, [])[:3]
                    
                    for src_node in src_nodes:
                        for dst_node in dst_nodes:
                            try:
                                session.execute_write(
                                    create_relation, 
                                    src_node['label'], src_node['id'],
                                    rel_type,
                                    dst_node['label'], dst_node['id']
                                )
                            except Exception as e:
                                console.print(f"[red]Error creating ontology relationship: {e}[/red]")

        # ---- Create foreign key relationships ----
        console.print("[cyan]Creating foreign key relationships...[/cyan]")
        for from_table, fk_list in fk_mappings.items():
            if from_table in node_mappings:
                for fk in fk_list:
                    to_table = fk['to_table']
                    from_col = fk['from_col']
                    to_col = fk['to_col']
                    
                    if to_table in node_mappings:
                        from_nodes = node_mappings[from_table]
                        to_nodes = node_mappings[to_table]
                        
                        for from_node in from_nodes:
                            fk_value = from_node['row_data'].get(from_col)
                            if fk_value is not None:
                                # Find matching target node
                                for to_node in to_nodes:
                                    if to_node['row_data'].get(to_col) == fk_value:
                                        try:
                                            session.execute_write(
                                                create_relation,
                                                from_node['label'], from_node['id'],
                                                'REFERENCES',
                                                to_node['label'], to_node['id']
                                            )
                                            break
                                        except Exception as e:
                                            console.print(f"[red]Error creating FK relationship: {e}[/red]")

    console.print("[green] Clean graph successfully constructed![/green]")

def verify_graph():
    """Verify the created graph."""
    with driver.session(database=DB_NAME) as session:
        # Count nodes
        result = session.run("MATCH (n) RETURN labels(n)[0] as label, count(n) as count ORDER BY count DESC")
        console.print("[blue]Node counts by label:[/blue]")
        total_nodes = 0
        for record in result:
            count = record['count']
            total_nodes += count
            console.print(f"  {record['label']}: {count}")
        
        # Count relationships
        result = session.run("MATCH ()-[r]->() RETURN type(r) as rel_type, count(r) as count ORDER BY count DESC")
        console.print("[blue]Relationship counts:[/blue]")
        total_rels = 0
        for record in result:
            count = record['count']
            total_rels += count
            console.print(f"  {record['rel_type']}: {count}")
        
        console.print(f"[green]Total: {total_nodes} nodes, {total_rels} relationships[/green]")
        
        # Test embedding storage
        result = session.run("MATCH (n) WHERE n.embedding IS NOT NULL RETURN count(n) as embedding_count")
        for record in result:
            console.print(f"[green]Nodes with embeddings: {record['embedding_count']}[/green]")
        
        # Show sample data
        result = session.run("MATCH (n) RETURN labels(n)[0] as label, n.id as id, n.name as name, size(n.embedding) as emb_size LIMIT 3")
        console.print("[blue]Sample nodes:[/blue]")
        for record in result:
            emb_info = f"embedding: {record['emb_size']} dims" if record['emb_size'] else "no embedding"
            console.print(f"  {record['label']} ({record['id']}): {record['name']}, {emb_info}")

if __name__ == "__main__":
    try:
        build_clean_graph()
        verify_graph()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
    finally:
        driver.close()
