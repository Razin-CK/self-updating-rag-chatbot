import os
import sys
import time

# Add base directory to system path so imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vector_store import VectorStore
from updater import get_updater
import config

def run_test():
    print("=============================================================")
    print("   DYNAMIC KNOWLEDGE BASE CHATBOT - AUTOMATED TEST PIPELINE  ")
    print("=============================================================\n")

    print("[STEP 1] Checking current database state...")
    db = VectorStore()
    stats = db.get_stats()
    print(f"  Total Indexed Chunks: {stats['total_chunks']}")
    print(f"  Total Indexed Files:  {stats['total_files']}")
    print(f"  Currently Indexed:    {stats['files']}\n")
    
    # 1. Adding a new text file
    new_file_name = "project_beta.txt"
    new_file_path = os.path.join(config.DOCS_DIR, new_file_name)
    
    content = """ACME CORP SECRET PROJECT INFO
Code Name: Project Beta
Objective: Designing an antigravity propulsion engine for heavy-duty payload transport.
Key Personnel: Miles Dyson (Chief Scientist), Marcus Wright (Field Engineer).
Target Release Date: July 2028.
Security Clearance: Level 5 or higher.
Description: Project Beta utilizes dark matter accelerators to generate localized antigravity fields, enabling frictionless high-speed travel.
"""
    
    print(f"[STEP 2] Adding new text file: 'docs/{new_file_name}'...")
    with open(new_file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  File created successfully at: {new_file_path}\n")
    
    # 2. Running database update (simulating updater/scheduler scan)
    print("[STEP 3] Running database synchronization...")
    updater = get_updater(db)
    results = updater.scan_and_update()
    
    print("\n  Sync Action Summary:")
    print(f"    Added:   {results['added']}")
    print(f"    Updated: {results['updated']}")
    print(f"    Deleted: {results['deleted']}\n")
    
    # 3. Asking questions about new content
    query = "What is the objective and release date of Project Beta?"
    print(f"[STEP 4] Querying database for new content: '{query}'...")
    
    # Retrieve matching chunks using the Two-Stage Re-ranking engine
    hits = db.query(query, use_reranking=True, n_results=1)
    
    print("\n[STEP 5] Verifying chatbot retrieved and learned new information:")
    if hits:
        best_hit = hits[0]
        print(f"  SUCCESS: Match found in document '{best_hit['metadata']['source']}'")
        print(f"  Bi-Encoder Cosine Distance:   {best_hit['distance']:.4f} (lower is closer)")
        print(f"  Cross-Encoder Re-rank Score:  {best_hit['rerank_score']:.4f} (higher is more relevant)")
        print("  --- Extracted Content Snippet ---")
        print(best_hit['text'].strip())
        print("  ---------------------------------")
    else:
        print("  FAILURE: No matching information was found in the database.")

    print("\n=============================================================")
    print("  Test complete! You can also check this file in the GUI:     ")
    print("  1. Open the Streamlit chatbot in your browser.             ")
    print("  2. Enter your query: 'Who is the chief scientist for Beta?'")
    print("=============================================================")

if __name__ == "__main__":
    run_test()
