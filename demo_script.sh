#!/bin/bash
# ============================================================
# demo_script.sh
# Demonstration of GraphRAG Unified Retrieval API
# ============================================================

API_URL="http://127.0.0.1:8000/query"
HEADER="Content-Type: application/json"

echo "============================================================"
echo "🚀 VALIDATION CHECKS"
echo "============================================================"

echo -e "\n[1] Checking if API is live..."
curl -s http://127.0.0.1:8000/ | jq

echo -e "\n[2] Sending empty query (should handle gracefully)..."
curl -s -X POST $API_URL -H "$HEADER" -d '{"query": ""}' | jq

echo -e "\n[3] Sending gibberish query (edge case)..."
curl -s -X POST $API_URL -H "$HEADER" -d '{"query": "asdfgh"}' | jq


echo "============================================================"
echo "🎵 VECTOR-BASED SEMANTIC RETRIEVAL"
echo "============================================================"

echo -e "\n[4] Find songs about rock"
curl -s -X POST $API_URL -H "$HEADER" -d '{"query": "Find songs about rock"}' | jq '.vector_hits'

echo -e "\n[5] Show artists similar to AC/DC"
curl -s -X POST $API_URL -H "$HEADER" -d '{"query": "Show artists similar to AC/DC"}' | jq '.vector_hits'

echo -e "\n[6] Show all musical genres"
curl -s -X POST $API_URL -H "$HEADER" -d '{"query": "Show all musical genres"}' | jq '.vector_hits'


echo "============================================================"
echo "🕸 GRAPH TRAVERSAL RETRIEVAL"
echo "============================================================"

echo -e "\n[7] List albums created by artists"
curl -s -X POST $API_URL -H "$HEADER" -d '{"query": "List albums created by artists"}' | jq '.graph_context'

echo -e "\n[8] Show tracks belonging to genre Rock"
curl -s -X POST $API_URL -H "$HEADER" -d '{"query": "Show tracks belonging to genre Rock"}' | jq '.graph_context'

echo -e "\n[9] Which employees support customers?"
curl -s -X POST $API_URL -H "$HEADER" -d '{"query": "Which employees support customers?"}' | jq '.graph_context'


echo "============================================================"
echo "🔀 HYBRID RETRIEVAL (VECTOR + GRAPH)"
echo "============================================================"

echo -e "\n[10] Find tracks created by artists"
curl -s -X POST $API_URL -H "$HEADER" -d '{"query": "Find tracks created by artists"}' | jq

echo -e "\n[11] Show albums related to AC/DC"
curl -s -X POST $API_URL -H "$HEADER" -d '{"query": "Show albums related to AC/DC"}' | jq

echo -e "\n[12] Customers who bought rock music"
curl -s -X POST $API_URL -H "$HEADER" -d '{"query": "Customers who bought rock music"}' | jq


echo "============================================================"
echo "✅ DEMO COMPLETED"
echo "============================================================"
