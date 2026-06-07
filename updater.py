import os
import json
import time
import logging
import threading
from typing import Dict, Any, List
from vector_store import VectorStore
import config

# Setup standard logging to both file and console
log_file = os.path.join(config.DB_DIR, "updater.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("updater")

class KnowledgeBaseUpdater:
    """
    Scans the docs directory for new, updated, or deleted files,
    indexing/deleting them incrementally, and maintaining a state manifest.
    """
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.manifest_path = config.TRACKING_FILE
        self.is_running = False
        self._thread = None
        self._lock = threading.Lock()

    def load_manifest(self) -> Dict[str, Any]:
        """
        Loads the indexed files manifest from disk.
        """
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading manifest: {e}")
                return {}
        return {}

    def save_manifest(self, manifest: Dict[str, Any]):
        """
        Saves the indexed files manifest to disk.
        """
        try:
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=4)
        except Exception as e:
            logger.error(f"Error writing manifest: {e}")

    def scan_and_update(self) -> Dict[str, List[str]]:
        """
        Main scanner logic:
        1. Compares current files in `docs/` with the manifest.
        2. Detects additions, modifications, and deletions.
        3. Invokes ChromaDB add/delete operations.
        4. Updates the manifest file.
        """
        with self._lock:
            manifest = self.load_manifest()
            current_files = {}
            supported_extensions = {".txt", ".md", ".pdf"}

            # Ensure the docs directory exists
            if not os.path.exists(config.DOCS_DIR):
                os.makedirs(config.DOCS_DIR, exist_ok=True)
                
            for filename in os.listdir(config.DOCS_DIR):
                file_path = os.path.join(config.DOCS_DIR, filename)
                if os.path.isfile(file_path):
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in supported_extensions:
                        stat = os.stat(file_path)
                        current_files[filename] = {
                            "path": file_path,
                            "mtime": stat.st_mtime,
                            "size": stat.st_size
                        }

            added_files = []
            updated_files = []
            deleted_files = []

            # 1. Detect New and Modified Files
            for filename, info in current_files.items():
                is_new = filename not in manifest
                is_modified = False
                
                if not is_new:
                    old_info = manifest[filename]
                    is_modified = (
                        abs(info["mtime"] - old_info.get("mtime", 0)) > 0.1 or 
                        info["size"] != old_info.get("size", 0)
                    )

                if is_new or is_modified:
                    action_str = "NEW" if is_new else "MODIFIED"
                    print(f"[CHANGE] [{action_str}] Change detected in: {filename}")
                    
                    # Index in vector store (converts text to embeddings and saves to ChromaDB)
                    result = self.vector_store.add_document(info["path"])
                    
                    if result["status"] == "success":
                        manifest[filename] = {
                            "mtime": info["mtime"],
                            "size": info["size"],
                            "last_indexed": result["timestamp"],
                            "chunks": result["chunks_added"]
                        }
                        if is_new:
                            added_files.append(filename)
                        else:
                            updated_files.append(filename)
                        print(f"[SUCCESS] Indexed: {filename} ({result['chunks_added']} chunks added)")
                    elif result["status"] == "warning":
                        print(f"[WARNING] Warning indexing {filename}: {result['message']}")
                    else:
                        print(f"[ERROR] Error indexing {filename}: {result.get('message')}")

            # 2. Detect Deleted Files
            for filename in list(manifest.keys()):
                if filename not in current_files:
                    print(f"[DELETED] Removing from database: {filename}")
                    # Remove from database
                    self.vector_store.delete_document(filename)
                    # Remove from manifest
                    manifest.pop(filename, None)
                    deleted_files.append(filename)

            # 3. Save manifest if any actions were taken
            if added_files or updated_files or deleted_files:
                self.save_manifest(manifest)
                logger.info(
                    f"Scan complete. Added: {len(added_files)}, "
                    f"Updated: {len(updated_files)}, "
                    f"Deleted: {len(deleted_files)}"
                )
            else:
                print("[INFO] Database is already up to date. No changes detected.")
            
            return {
                "added": added_files,
                "updated": updated_files,
                "deleted": deleted_files
            }

    def _loop(self):
        """
        Internal loop run by background thread.
        """
        logger.info(f"Background updater thread started. Interval: {config.SCAN_INTERVAL_SECONDS}s")
        while self.is_running:
            try:
                self.scan_and_update()
            except Exception as e:
                logger.error(f"Error in background update loop: {e}")
            
            # Sleep in tiny steps to allow rapid thread stopping
            for _ in range(config.SCAN_INTERVAL_SECONDS):
                if not self.is_running:
                    break
                time.sleep(1)
                
        logger.info("Background updater thread stopped.")

    def start(self):
        """
        Starts the scanning updater on a background thread.
        """
        if self.is_running:
            logger.warning("Background updater is already running.")
            return

        self.is_running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        """
        Stops the background thread loop.
        """
        if not self.is_running:
            return

        logger.info("Stopping background updater...")
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

# Single instance helper
_updater_instance = None
_instance_lock = threading.Lock()

def get_updater(vector_store: VectorStore) -> KnowledgeBaseUpdater:
    """
    Singleton-like helper to retrieve a single updater instance.
    """
    global _updater_instance
    with _instance_lock:
        if _updater_instance is None:
            _updater_instance = KnowledgeBaseUpdater(vector_store)
        return _updater_instance

# Standalone execution entry point
if __name__ == "__main__":
    print("\n=============================================")
    print("Running Standalone Knowledge Base Sync Scanner")
    print("=============================================\n")
    
    # Initialize the database connection
    db_store = VectorStore()
    
    # Initialize the updater
    kb_updater = get_updater(db_store)
    
    # Execute scan and print summary
    print(f"Scanning directory: '{config.DOCS_DIR}'...")
    results = kb_updater.scan_and_update()
    
    print("\n--- Summary of Database Actions ---")
    print(f"Added Files:   {len(results['added'])}")
    print(f"Updated Files: {len(results['updated'])}")
    print(f"Deleted Files: {len(results['deleted'])}")
    print("=============================================\n")
