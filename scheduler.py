import os
import sys
import time

# Ensure the project directory is in the import path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vector_store import VectorStore
from updater import get_updater
import config

def main():
    print("\n=============================================")
    print("Starting Knowledge Base Periodic Scheduler")
    print(f"Monitoring directory: '{config.DOCS_DIR}'")
    print("Checking for changes every 1 minute (60 seconds)...")
    print("Press Ctrl+C in this terminal to exit.")
    print("=============================================\n")
    
    # Initialize connection to ChromaDB and model loading
    print("[INIT] Connecting to vector database...")
    try:
        db = VectorStore()
        updater = get_updater(db)
        print("[INIT] Database initialized and ready.\n")
    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to initialize database: {e}")
        sys.exit(1)
        
    # Run the scheduler loop
    try:
        # Perform an initial scan immediately on startup
        print(f"[{time.strftime('%H:%M:%S')}] Running startup database scan...")
        updater.scan_and_update()
        print("Sleeping for 1 minute before next scan...\n")
        time.sleep(60)
        
        while True:
            current_time = time.strftime('%H:%M:%S')
            print(f"[{current_time}] Running scheduled directory scan...")
            
            # scan_and_update internally compares docs folder with manifest to identify 
            # new, modified, and deleted files to avoid double-processing.
            results = updater.scan_and_update()
            
            # Print a clean summary to the terminal if updates occurred
            if results["added"] or results["updated"] or results["deleted"]:
                print("---------------------------------------------")
                print("Sync Action Summary:")
                if results["added"]:
                    print(f"  Added:   {', '.join(results['added'])}")
                if results["updated"]:
                    print(f"  Updated: {', '.join(results['updated'])}")
                if results["deleted"]:
                    print(f"  Deleted: {', '.join(results['deleted'])}")
                print("---------------------------------------------")
            else:
                print("No changes detected in docs/ directory.")
                
            print("Sleeping for 1 minute before next scan...\n")
            time.sleep(60)
            
    except KeyboardInterrupt:
        print("\nScheduler loop stopped by user (Ctrl+C). Exiting safely.")

if __name__ == "__main__":
    main()
