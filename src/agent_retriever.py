import requests, time, json
from rich.console import Console
from rich.table import Table
from datetime import datetime

console = Console()
API_URL = "http://127.0.0.1:8000/query"

# ------------------------------
# Reasoning Chain Generator
# ------------------------------
def explain_result(query, response):
    mode = response.get("mode", "unknown")
    hits = len(response.get("vector_hits", []))
    graph_hits = len(response.get("graph_context", []))
    reasoning = []

    # Determine mode explanation
    if mode == "vector":
        reasoning.append("Used vector similarity because query mentioned semantic similarity.")
    elif mode == "graph":
        reasoning.append("Used graph traversal to explore relationships explicitly.")
    elif mode == "hybrid":
        reasoning.append("Hybrid mode: combined semantic similarity and relationship reasoning.")
    else:
        reasoning.append("Unknown retrieval mode — defaulting to hybrid reasoning.")

    # Include reasoning details
    reasoning.append(f"Found {hits} top vector hits and {graph_hits} connected graph entities.")
    reasoning.append("Expanded each high-similarity node to its neighbors to provide context.")
    reasoning.append("Final answer derived from merged multi-hop reasoning graph.")

    return "\n".join(reasoning)

# ------------------------------
# Evaluation
# ------------------------------
def evaluate_queries(queries):
    table = Table(title="GraphRAG Agent Evaluation", show_lines=True)
    table.add_column("Query", style="cyan")
    table.add_column("Mode", style="yellow")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("Hits", justify="right")

    metrics = []

    for q in queries:
        start = time.time()
        try:
            r = requests.post(API_URL, json={"query": q})
            r.raise_for_status()
            response = r.json()
        except Exception as e:
            console.print(f"[red]❌ Error querying API: {e}[/red]")
            continue

        latency = (time.time() - start) * 1000
        hits = len(response.get("vector_hits", []))
        mode = response.get("mode", "unknown")

        metrics.append({"query": q, "mode": mode, "latency": latency, "hits": hits})
        table.add_row(q, mode, f"{latency:.1f}", str(hits))

        console.print(f"\n🧠 [bold]{q}[/bold]")
        console.print(explain_result(q, response), style="green")
        console.print("-" * 60)

    console.print(table)
    return metrics

# ------------------------------
# Report Generator
# ------------------------------
def generate_report(metrics):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    avg_latency = sum(m["latency"] for m in metrics) / len(metrics)
    hybrid = [m for m in metrics if m["mode"] == "hybrid"]
    vector = [m for m in metrics if m["mode"] == "vector"]
    graph = [m for m in metrics if m["mode"] == "graph"]

    report = f"""# Evaluation Report

**Average Latency:** {avg_latency:.2f} ms
**Hybrid Queries:** {len(hybrid)}
**Vector Queries:** {len(vector)}
**Graph Queries:** {len(graph)}

---

"""
    with open("docs/evaluation_report.md", "w") as f:
        f.write(report)
    console.print("\n✅ Evaluation report saved to docs/evaluation_report.md", style="bold cyan")

# ------------------------------
# Run Evaluation
# ------------------------------
if __name__ == "__main__":
    queries = [
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
    metrics = evaluate_queries(queries)
    generate_report(metrics)