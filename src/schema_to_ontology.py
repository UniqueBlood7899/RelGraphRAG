import os
import json
import re
import sqlalchemy
from sqlalchemy import create_engine, inspect
import google.generativeai as genai
from dotenv import load_dotenv
from rich.console import Console

console = Console()
load_dotenv()

DB_PATH = "data/chinook.db"
OUTPUT_PATH = "data/ontology.json"

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
MODEL = "gemini-2.5-pro"


def extract_schema(db_path: str):
    """Extract relational schema using SQLAlchemy inspector."""
    engine = create_engine(f"sqlite:///{db_path}")
    inspector = inspect(engine)
    schema = {"tables": {}}

    for table in inspector.get_table_names():
        columns = [
            {"name": col["name"], "type": str(col["type"])}
            for col in inspector.get_columns(table)
        ]
        fks = [
            {
                "constrained_column": fk["constrained_columns"][0],
                "referred_table": fk["referred_table"],
            }
            for fk in inspector.get_foreign_keys(table)
        ]
        schema["tables"][table] = {"columns": columns, "foreign_keys": fks}
    return schema


def extract_json_from_response(text: str) -> str:
    """Extract JSON from markdown code blocks or clean text."""
    text = text.strip()
    
    # Remove markdown code blocks
    if text.startswith('```') and text.endswith('```'):
        lines = text.split('\n')
        # Remove first line (```json or ```) and last line (```)
        if len(lines) > 2:
            text = '\n'.join(lines[1:-1])
    
    # Alternative: use regex to extract JSON from code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    
    return text.strip()


def generate_ontology(schema_dict: dict):
    """Use Gemini to generate ontology."""
    # Simplified prompt to reduce complexity
    prompt = f"""Convert this database schema to a JSON ontology.

For each table, create an entry with:
- class_label: semantic meaning (Person, Product, etc.)
- properties: list of column names that are attributes
- relationships: list of connections to other tables

Return ONLY the JSON object without any markdown formatting.

Schema: {json.dumps(schema_dict, indent=2)}

Output format:
{{
  "TableName": {{
    "class_label": "SemanticLabel",
    "properties": ["col1", "col2"],
    "relationships": ["relates to OtherTable"]
  }}
}}
"""

    console.print("[cyan]Generating ontology...[/cyan]")
    model = genai.GenerativeModel(MODEL)
    
    # Increased token limit and adjusted parameters
    generation_config = genai.types.GenerationConfig(
        temperature=0.2,
        top_p=0.9,
        top_k=50,
        max_output_tokens=8192,  # Increased from 2048
    )
    
    try:
        response = model.generate_content(prompt, generation_config=generation_config)
        text = response.text.strip()
        
        console.print(f"[yellow]Raw response preview:[/yellow] {text[:200]}...")
        
        # Extract JSON from the response
        json_text = extract_json_from_response(text)
        
        # Try to parse the JSON
        parsed_json = json.loads(json_text)
        console.print("[green]✓ Successfully parsed JSON response[/green]")
        return parsed_json
        
    except json.JSONDecodeError as e:
        console.print(f"[red]JSON parsing error: {e}[/red]")
        console.print(f"[yellow]Attempting to fix truncated JSON...[/yellow]")
        
        # Try to fix common JSON truncation issues
        json_text = extract_json_from_response(text)
        fixed_json = fix_truncated_json(json_text)
        
        try:
            parsed_json = json.loads(fixed_json)
            console.print("[green]✓ Successfully fixed and parsed JSON[/green]")
            return parsed_json
        except json.JSONDecodeError:
            console.print(f"[red]Could not fix JSON. Saving raw output.[/red]")
            return {"error": "Invalid JSON", "raw_output": text}
    
    except Exception as e:
        console.print(f"[red]API Error: {e}[/red]")
        return {"error": str(e)}


def fix_truncated_json(json_text: str) -> str:
    """Attempt to fix truncated JSON by closing open structures."""
    # Count open braces and brackets
    open_braces = json_text.count('{') - json_text.count('}')
    open_brackets = json_text.count('[') - json_text.count(']')
    
    # Remove trailing comma if exists
    json_text = json_text.rstrip().rstrip(',')
    
    # Close open structures
    if open_brackets > 0:
        json_text += ']' * open_brackets
    if open_braces > 0:
        json_text += '}' * open_braces
    
    return json_text


def save_ontology(ontology, output_path):
    """Save ontology to JSON file."""
    with open(output_path, "w") as f:
        json.dump(ontology, f, indent=2)
    console.print(f"[green]✓ Ontology saved to {output_path}[/green]")


if __name__ == "__main__":
    console.print("[bold blue]Extracting relational schema...[/bold blue]")
    schema = extract_schema(DB_PATH)
    console.print_json(data=schema)

    console.print("\n[bold blue]Generating ontology...[/bold blue]")
    ontology = generate_ontology(schema)
    
    if "error" not in ontology:
        console.print(f"[green]Generated ontology for {len(ontology)} tables[/green]")
    
    save_ontology(ontology, OUTPUT_PATH)
