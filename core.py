import time
import uuid
import math
import faiss
import numpy as np
import json
import os
from typing import List, Dict, Optional, Any, Callable, Tuple
from Cryptodome.Cipher import AES
from Cryptodome.Protocol.KDF import PBKDF2
from sentence_transformers import SentenceTransformer
import atexit


# ----------------- EncryptionManager: Handles secure data storage -----------------
class EncryptionManager:
    """Handles authenticated encryption and decryption using AES GCM and PBKDF2."""

    def __init__(self, password: str, salt: Optional[bytes] = None):
        """Initializes the encryption manager with a password and salt."""
        self.salt = salt if salt else os.urandom(16)
        self.key = PBKDF2(password.encode('utf-8'), self.salt, dkLen=32, count=100000)

    def encrypt(self, plaintext: bytes) -> tuple[bytes, bytes]:
        """Encrypts plaintext and returns nonce and ciphertext."""
        cipher = AES.new(self.key, AES.MODE_GCM)
        nonce = cipher.nonce
        ciphertext, tag = cipher.encrypt_and_digest(plaintext)
        return nonce, ciphertext + tag

    def decrypt(self, nonce: bytes, ciphertext_with_tag: bytes) -> bytes:
        """Decrypts ciphertext using nonce and verifies the tag."""
        ciphertext = ciphertext_with_tag[:-16]
        tag = ciphertext_with_tag[-16:]
        cipher = AES.new(self.key, AES.MODE_GCM, nonce=nonce)
        return cipher.decrypt_and_verify(ciphertext, tag)


# ----------------- MemoryItem: Enhanced with emotion and associations -----------------
class MemoryItem:
    """Represents a memory unit with content, embedding, and meta info."""

    def __init__(self,
                 embedding: List[float],
                 text: str,
                 timestamp: Optional[float] = None,
                 recall_count: int = 0,
                 importance: float = 1.0,
                 difficulty_factor: float = 1.0,
                 parent_ids: Optional[List[str]] = None,
                 item_id: Optional[str] = None,
                 emotional_intensity: float = 0.0,
                 associated_memories: Optional[List[str]] = None):
        self.id = item_id if item_id else str(uuid.uuid4())
        self.embedding = np.array(embedding).astype(np.float32)
        self.text = text
        self.creation_timestamp = timestamp if timestamp else time.time()
        self.last_recall_time = self.creation_timestamp
        self.recall_count = recall_count
        self.importance = importance
        self.difficulty_factor = difficulty_factor
        self.parent_ids = parent_ids if parent_ids else []
        self.emotional_intensity = emotional_intensity
        self.associated_memories = associated_memories if associated_memories else []

    def get_retention_factor(self, elapsed_time: float) -> float:
        """
        Calculates the retention factor using an enhanced model based on the Ebbinghaus forgetting curve.
        Incorporates non-linear decay and emotional influence.
        """
        base_stability = 1000
        emotional_influence = 5000 * self.emotional_intensity
        recall_influence = 500 * self.recall_count
        stability = (base_stability + recall_influence + emotional_influence) / self.difficulty_factor

        # Non-linear forgetting: Memories forgotten for a long time decay faster
        decay_rate_modifier = 1.0 + (elapsed_time / (stability * 50))
        retention = math.exp(-elapsed_time / (stability * decay_rate_modifier))
        return retention

    def get_decayed_score(self, current_time: float) -> float:
        """Calculates the current decayed score of the memory, combining initial importance and retention."""
        elapsed_since_recall = current_time - self.last_recall_time
        retention_factor = self.get_retention_factor(elapsed_since_recall)
        return self.importance * retention_factor

    def spaced_repetition_score(self, current_time: float):
        """Main score for retrieval, combining importance and retention."""
        return self.get_decayed_score(current_time)

    def update_on_recall(self, emotional_boost: float = 0.0):
        """
        Updates the memory item's state after a successful recall.
        emotional_boost simulates 'multimodal' trigger's emotional impact.
        """
        self.recall_count += 1
        self.last_recall_time = time.time()
        self.importance = min(self.importance + 0.1, 10.0)
        self.difficulty_factor = max(self.difficulty_factor * 0.95, 0.1)
        self.emotional_intensity = min(self.emotional_intensity + emotional_boost, 1.0)

    def associability_score(self, query_embedding: np.ndarray) -> float:
        """Cosine similarity between memory and query."""
        norm_self = np.linalg.norm(self.embedding)
        norm_query = np.linalg.norm(query_embedding)
        if norm_self == 0 or norm_query == 0:
            return 0.0
        return float(np.dot(self.embedding, query_embedding) / (norm_self * norm_query))

    def to_dict(self) -> Dict:
        """Convert the MemoryItem object to a dictionary for serialization."""
        return {
            'id': self.id,
            'embedding': self.embedding.tolist(),
            'text': self.text,
            'creation_timestamp': self.creation_timestamp,
            'last_recall_time': self.last_recall_time,
            'recall_count': self.recall_count,
            'importance': self.importance,
            'difficulty_factor': self.difficulty_factor,
            'parent_ids': self.parent_ids,
            'emotional_intensity': self.emotional_intensity,
            'associated_memories': self.associated_memories
        }

    @staticmethod
    def from_dict(data: Dict):
        """Create a MemoryItem object from a dictionary, with backward compatibility."""
        item = MemoryItem(
            embedding=data['embedding'],
            text=data['text'],
            timestamp=data['creation_timestamp'],
            recall_count=data['recall_count'],
            importance=data['importance'],
            difficulty_factor=data['difficulty_factor'],
            parent_ids=data['parent_ids'],
            item_id=data['id'],
            emotional_intensity=data.get('emotional_intensity', 0.0),
            associated_memories=data.get('associated_memories', [])
        )
        item.last_recall_time = data['last_recall_time']
        return item


# ----------------- MemoryDB: Optimized with custom JSON persistence, dynamic I/O, and caching -----------------
class MemoryDB:
    """Memory storage and retrieval system with custom JSON persistence, dynamic I/O, and caching."""

    def __init__(self, dim: int, db_dir: str = 'memories_db', password: str = 'my_secret_password',
                 embedding_func: Callable[[str], List[float]] = None, train_threshold: int = 1000,
                 cache_size: int = 5000, chunk_size: int = 1000):
        self.dim = dim
        self.db_dir = db_dir
        self.password = password
        self.embedding_func = embedding_func
        self.train_threshold = train_threshold
        self.cache_size = cache_size
        self.chunk_size = chunk_size

        self.faiss_index_path = os.path.join(self.db_dir, 'faiss.bin')
        self.faiss_id_map_path = os.path.join(self.db_dir, 'faiss_id_map.json')
        self.salt_path = os.path.join(self.db_dir, 'salt.bin')
        self.memory_index_path = os.path.join(self.db_dir, 'memory_index.json')

        self.index = None  # Will be initialized later
        self.faiss_id_map: Dict[int, str] = {}
        self.rev_faiss_id_map: Dict[str, int] = {}

        # In-memory cache for frequently accessed items (LRU)
        self.cache: Dict[str, MemoryItem] = {}
        self.cache_access_time: Dict[str, float] = {}

        # In-memory index mapping memory ID to its location on disk
        # Structure: { memory_id: { 'chunk_id': int, 'offset': int, 'length': int, 'dirty': bool } }
        self.memory_map: Dict[str, Dict[str, Any]] = {}

        self.enc_manager = None
        self.current_chunk_id = 0
        self.current_chunk_offset = 0

        self._init_db_directory()
        self._init_encryption()
        self._load_memory_index()
        self._load_faiss_index()

        atexit.register(self.save_and_close)

    def _init_db_directory(self):
        """Creates the database directory if it doesn't exist."""
        os.makedirs(self.db_dir, exist_ok=True)
        print(f"-> Database directory initialized at: {self.db_dir}")

    def _init_encryption(self):
        """Initializes the encryption manager and loads/saves the salt from a file."""
        if os.path.exists(self.salt_path):
            with open(self.salt_path, 'rb') as f:
                salt = f.read()
            self.enc_manager = EncryptionManager(self.password, salt)
            print("-> Found existing salt file. Initializing EncryptionManager...")
        else:
            salt = os.urandom(16)
            with open(self.salt_path, 'wb') as f:
                f.write(salt)
            self.enc_manager = EncryptionManager(self.password, salt)
            print("-> No salt file found. Generating and saving a new one. Initializing EncryptionManager...")

    def _load_memory_index(self):
        """Loads the in-memory memory map from a JSON index file."""
        if os.path.exists(self.memory_index_path):
            print(f"--- Loading memory index from file: {self.memory_index_path} ---")
            try:
                with open(self.memory_index_path, 'r') as f:
                    self.memory_map = json.load(f)

                if self.memory_map:
                    # Determine the last used chunk ID and offset
                    max_chunk_id = 0
                    for item_info in self.memory_map.values():
                        if item_info['chunk_id'] > max_chunk_id:
                            max_chunk_id = item_info['chunk_id']

                    self.current_chunk_id = max_chunk_id
                    # To be safe, find the size of the last chunk file
                    last_chunk_path = os.path.join(self.db_dir, f'chunk_{self.current_chunk_id}.json.enc')
                    if os.path.exists(last_chunk_path):
                        self.current_chunk_offset = os.path.getsize(last_chunk_path)

                print(f"-> Loaded {len(self.memory_map)} memory index entries. Current chunk: {self.current_chunk_id}")
            except (IOError, json.JSONDecodeError) as e:
                print(f"-> ERROR: Failed to load memory index ({e}). Starting with a new index.")
                self.memory_map = {}
        else:
            print("-> No memory index file found. Starting with a new index.")
            self.memory_map = {}
            self.current_chunk_id = 0
            self.current_chunk_offset = 0

    def _save_memory_index(self):
        """Saves the in-memory memory map to a JSON index file."""
        print(f"--- Saving memory index to file: {self.memory_index_path} ---")
        try:
            with open(self.memory_index_path, 'w') as f:
                # We need to save the dirty flag and other info
                json.dump(self.memory_map, f, indent=2)
            print("-> Memory index saved.")
        except Exception as e:
            print(f"-> ERROR: Failed to save memory index ({e}).")

    def _load_faiss_index(self):
        """Loads the FAISS index and ID map from disk if they exist, or creates a new one."""
        if os.path.exists(self.faiss_index_path) and os.path.exists(self.faiss_id_map_path):
            print("-> Found trained FAISS index. Loading from disk...")
            try:
                self.index = faiss.read_index(self.faiss_index_path)
                with open(self.faiss_id_map_path, 'r') as f:
                    self.faiss_id_map = {int(k): v for k, v in json.load(f).items()}
                    self.rev_faiss_id_map = {v: k for k, v in self.faiss_id_map.items()}
                print(f"-> FAISS index loaded. Total vectors: {self.index.ntotal}")

                # Check consistency
                if self.index.ntotal != len(self.memory_map):
                    print("-> WARNING: Index vector count does not match memory entries. Rebuilding index.")
                    self._build_index_from_memories()
            except Exception as e:
                print(f"-> ERROR: Failed to load FAISS index ({e}). Rebuilding from scratch.")
                self._build_index_from_memories()
        else:
            print("-> No trained FAISS index found. Building from scratch...")
            self._build_index_from_memories()

    def _build_index_from_memories(self):
        """Rebuilds a new FAISS index from all embeddings found in the memory map."""
        # Check if index is already initialized or not
        is_trained = self.index and self.index.is_trained

        if len(self.memory_map) < self.train_threshold and not is_trained:
            print(f"-> Not enough vectors ({len(self.memory_map)}) to train FAISS IVFPQ. Using a FlatL2 index.")
            self.index = faiss.IndexFlatL2(self.dim)
        elif len(self.memory_map) >= self.train_threshold and not is_trained:
            print(f"--- Training FAISS IVFPQ index with {len(self.memory_map)} vectors ---")
            quantizer = faiss.IndexFlatL2(self.dim)
            nlist = min(100, len(self.memory_map) // 39)
            nlist = max(1, nlist)
            m_pq = 16

            self.index = faiss.IndexIVFPQ(quantizer, self.dim, nlist, m_pq, 8)
            self.index.nprobe = max(1, nlist // 10)

            # Load all embeddings from disk to train the index
            embeddings = []
            for mem_id in self.memory_map:
                mem_item = self._read_memory_from_disk(mem_id)
                if mem_item:
                    embeddings.append(mem_item.embedding)

            if embeddings:
                embeddings_np = np.array(embeddings).astype('float32')
                self.index.train(embeddings_np)
                print("-> FAISS index trained successfully.")
            else:
                print("-> No embeddings found to train the index.")
                self.index = faiss.IndexFlatL2(self.dim)  # Fallback to FlatL2

        # Re-populate the index
        if self.index:
            self.index.reset()
            self.faiss_id_map = {}
            self.rev_faiss_id_map = {}

            print("--- Adding vectors to the index ---")
            for mem_id in self.memory_map:
                mem_item = self._read_memory_from_disk(mem_id)
                if mem_item:
                    faiss_idx = self.index.ntotal
                    self.index.add(np.expand_dims(mem_item.embedding, axis=0))
                    self.faiss_id_map[faiss_idx] = mem_id
                    self.rev_faiss_id_map[mem_id] = faiss_idx
            print(f"-> FAISS index populated with {self.index.ntotal} vectors.")
        else:
            print("-> FAISS index is not initialized. Cannot add vectors.")

    def _read_memory_from_disk(self, item_id: str) -> Optional[MemoryItem]:
        """Reads a single memory from a chunk file based on its index entry."""
        if item_id in self.cache:
            self.cache_access_time[item_id] = time.time()
            return self.cache[item_id]

        if item_id not in self.memory_map:
            return None

        mem_info = self.memory_map[item_id]
        chunk_path = os.path.join(self.db_dir, f"chunk_{mem_info['chunk_id']}.json.enc")

        if not os.path.exists(chunk_path):
            print(f"WARNING: Chunk file {chunk_path} not found for memory {item_id}.")
            return None

        try:
            # Read only the specific portion of the file
            with open(chunk_path, 'rb') as f:
                f.seek(mem_info['offset'])
                encrypted_data = f.read(mem_info['length'])

            # Decrypt and parse
            nonce_len = 16
            nonce = encrypted_data[:nonce_len]
            encrypted_json_with_tag = encrypted_data[nonce_len:]

            json_str = self.enc_manager.decrypt(nonce, encrypted_json_with_tag).decode('utf-8')
            data = json.loads(json_str)
            item = MemoryItem.from_dict(data)

            # Add to cache (and perform LRU eviction if needed)
            if len(self.cache) >= self.cache_size:
                lru_id = min(self.cache_access_time, key=self.cache_access_time.get)
                # Before evicting, save dirty items
                if self.memory_map.get(lru_id, {}).get('dirty', False):
                    self._write_memory_to_disk(self.cache[lru_id])
                del self.cache[lru_id]
                del self.cache_access_time[lru_id]

            self.cache[item.id] = item
            self.cache_access_time[item.id] = time.time()

            return item
        except Exception as e:
            print(f"ERROR: Failed to read/decrypt memory {item_id} from disk: {e}")
            return None

    def _write_memory_to_disk(self, item: MemoryItem):
        """Writes a single memory to the current chunk file using append-only mode."""
        chunk_path = os.path.join(self.db_dir, f"chunk_{self.current_chunk_id}.json.enc")

        # Serialize and encrypt the data
        json_data = json.dumps(item.to_dict()).encode('utf-8')
        nonce, encrypted_data_with_tag = self.enc_manager.encrypt(json_data)

        data_to_write = nonce + encrypted_data_with_tag

        with open(chunk_path, 'ab') as f:  # Use 'ab' for append in binary mode
            offset = f.tell()  # Get the current position before writing
            f.write(data_to_write)
            length = len(data_to_write)

        # Update the memory map with the new location and mark as clean
        self.memory_map[item.id] = {
            'chunk_id': self.current_chunk_id,
            'offset': offset,
            'length': length,
            'dirty': False
        }

        self.current_chunk_offset = offset + length

        # If the chunk is full, move to the next one
        if self.current_chunk_offset >= 1024 * 1024 * 10:  # ~10 MB per chunk
            self.current_chunk_id += 1
            self.current_chunk_offset = 0
            print(f"-> Chunk {self.current_chunk_id - 1} is full. Starting new chunk {self.current_chunk_id}.")

    def add_memory(self, item: MemoryItem):
        """Adds or updates a memory. Writes to disk immediately."""
        self.cache[item.id] = item  # Update cache
        self.cache_access_time[item.id] = time.time()

        # Write to disk and update the memory map
        self._write_memory_to_disk(item)

        # Update FAISS index
        if self.index:
            if item.id not in self.rev_faiss_id_map:
                faiss_idx = self.index.ntotal
                self.index.add(np.expand_dims(item.embedding, axis=0))
                self.faiss_id_map[faiss_idx] = item.id
                self.rev_faiss_id_map[item.id] = faiss_idx

            # If we have enough memories, train the index (if not already trained)
            if not self.index.is_trained and self.index.ntotal >= self.train_threshold:
                self._build_index_from_memories()

    def find_memory_by_text(self, text: str, similarity_threshold: float = 0.9) -> Optional[MemoryItem]:
        """Finds a memory by its text content using semantic similarity, checking cache first."""
        if not self.embedding_func:
            print("WARNING: No embedding function provided. Semantic search is not possible.")
            return None

        query_embedding = np.array(self.embedding_func(text)).astype(np.float32)

        # Check in-memory cache first for exact text match or high similarity
        for item in self.cache.values():
            if item.text == text:  # Exact match is prioritized
                self.cache_access_time[item.id] = time.time()
                return item

        # If not in cache, perform a semantic search on the index
        if self.index and self.index.ntotal > 0:
            results = self.semantic_search(query_embedding, top_k=5)
            if results:
                for item in results:
                    if item.text == text or item.associability_score(query_embedding) > similarity_threshold:
                        return item
        return None

    def add_or_review_memory(self, item: MemoryItem, similarity_threshold: float = 0.9):
        """
        Adds a new memory or reviews an existing one based on text or semantic similarity.
        """
        existing_item = self.find_memory_by_text(item.text, similarity_threshold)

        if existing_item:
            print(f"-> Existing memory found: \"{existing_item.text}\". Treating as review.")
            existing_item.update_on_recall(emotional_boost=item.emotional_intensity)
            self.add_memory(existing_item)  # This will write the updated item to disk
            return

        print("-> No existing memory found. Treating as new memory.")
        self.add_memory(item)

    def delete_memory(self, item_id: str):
        """Deletes a memory by marking it as 'deleted' in the index and removing from cache."""
        if item_id in self.memory_map:
            # For this simple implementation, we just remove it from the map.
            # A more robust system would mark it as 'invalid' and run a garbage collection later.
            del self.memory_map[item_id]
            if item_id in self.cache:
                del self.cache[item_id]
                del self.cache_access_time[item_id]

            # Rebuild FAISS index to remove the embedding
            # This is slow! A better way is to use a FAISS index that supports deletion.
            print(f"-> Memory {item_id} deleted from map. Rebuilding FAISS index...")
            self._build_index_from_memories()
        else:
            print(f"-> Memory {item_id} not found.")

    def semantic_search(self, embedding: List[float], top_k: int = 5, emotional_boost: float = 0.0) -> List[MemoryItem]:
        """
        Performs semantic search with emotional boost.
        Upon recall, this method triggers the 'reconstruction' of associated memories.
        """
        query = np.array(embedding).astype(np.float32).reshape(1, -1)
        if self.index is None or self.index.ntotal == 0:
            print("-> FAISS index is empty or not initialized.")
            return []

        distances, indices = self.index.search(query, top_k)
        results = []

        recalled_ids = []
        for i in indices[0]:
            if i in self.faiss_id_map:
                item_id = self.faiss_id_map[i]
                item = self._read_memory_from_disk(item_id)  # Read from disk or cache
                if item:
                    item.update_on_recall(emotional_boost)
                    # Write the updated memory back to disk
                    self._write_memory_to_disk(item)
                    results.append(item)
                    recalled_ids.append(item_id)

        # Recursively update associated memories with a weaker emotional boost
        for recalled_id in recalled_ids:
            recalled_item = self._read_memory_from_disk(recalled_id)
            if recalled_item:
                for associated_id in recalled_item.associated_memories:
                    associated_item = self._read_memory_from_disk(associated_id)
                    if associated_item:
                        associated_item.update_on_recall(emotional_boost * 0.5)
                        self._write_memory_to_disk(associated_item)

        current_time = time.time()
        return sorted(results, key=lambda x: x.spaced_repetition_score(current_time), reverse=True)

    def _save_faiss_index(self):
        """Saves the FAISS index and ID map to disk."""
        if self.index and self.index.is_trained:
            print("--- Saving FAISS index and ID map ---")
            faiss.write_index(self.index, self.faiss_index_path)
            with open(self.faiss_id_map_path, 'w') as f:
                json.dump(self.faiss_id_map, f)
            print("-> FAISS index saved.")
        else:
            print("-> FAISS index is not trained. Not saving.")

    def get_all_memories(self):
        """Retrieves all memories for display/debugging (expensive!)."""
        all_memories = []
        for item_id in self.memory_map.keys():
            item = self._read_memory_from_disk(item_id)
            if item:
                all_memories.append(item)
        return all_memories

    def save_and_close(self):
        """Saves all in-memory states to disk and performs cleanup."""
        print("\n--- Program is exiting. Saving all in-memory states... ---")
        self._save_memory_index()
        self._save_faiss_index()  # Corrected call
        print("-> All states saved. Cleanup complete.")

    def __del__(self):
        pass  # atexit handles cleanup


class MemorySimulator:
    """Simulates learning, recall, association and decay with a real embedding model."""

    def __init__(self, db_dir: str = 'memories_db_dynamic', password: str = 'my_secret_password',
                 train_threshold: int = 1000):
        print("--- Loading Sentence-Transformer model... ---")
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"--- Model loaded successfully. Embedding dimension: {self.embedding_dim} ---")
            self.get_embedding = lambda text: self.model.encode(text).tolist()
        except Exception as e:
            print(f"WARNING: Could not load Sentence-Transformer model. Using mock embeddings. Error: {e}")
            self.embedding_dim = 384
            self.get_embedding = lambda text: np.random.rand(self.embedding_dim).tolist()

        # Initialize MemoryDB with the real embedding function
        self.memory_db = MemoryDB(self.embedding_dim, db_dir=db_dir, password=password,
                                  embedding_func=self.get_embedding, train_threshold=train_threshold)

    def learn_or_review_memory(self, text: str, importance: float = 1.0, difficulty_factor: float = 1.0,
                               emotional_intensity: float = 0.0, associated_texts: Optional[List[str]] = None):
        """
        Adds a new memory or reviews an existing one, with support for associations and emotions.
        """
        print(f"\n--- Processing memory: \"{text}\" ---")
        embedding = self.get_embedding(text)

        associated_ids = []
        if associated_texts:
            for assoc_text in associated_texts:
                existing_assoc = self.memory_db.find_memory_by_text(assoc_text)
                if existing_assoc:
                    associated_ids.append(existing_assoc.id)
                else:
                    temp_mem = MemoryItem(embedding=self.get_embedding(assoc_text), text=assoc_text)
                    self.memory_db.add_memory(temp_mem)
                    associated_ids.append(temp_mem.id)

        mem_item = MemoryItem(
            embedding=embedding,
            text=text,
            importance=importance,
            difficulty_factor=difficulty_factor,
            emotional_intensity=emotional_intensity,
            associated_memories=associated_ids
        )

        self.memory_db.add_or_review_memory(mem_item)

    def recall_memory(self, query: str, emotional_boost: float = 0.0):
        print(f"\n--- Recalling memory for query: \"{query}\" with emotional boost: {emotional_boost} ---")
        query_embedding = self.get_embedding(query)
        results = self.memory_db.semantic_search(query_embedding, top_k=5, emotional_boost=emotional_boost)
        if not results:
            print("-> No relevant memories found. 未找到相关记忆。")
        else:
            print(f"-> Found {len(results)} relevant memories: 找到{len(results)}条相关记忆：")
            for i, item in enumerate(results):
                current_time = time.time()
                print(
                    f"   {i + 1}. Text: \"{item.text}\" | Score: {item.spaced_repetition_score(current_time):.4f} | Recalls: {item.recall_count} | Emotion: {item.emotional_intensity:.2f}")
        return results

    def simulate_time_passage(self, seconds: int):
        print(f"\n--- Simulating time passage: {seconds} seconds ---")
        time.sleep(seconds)
        # Note: Decay all is now expensive as it needs to read all memories.
        # A more advanced system would track decay in the index.
        # self.memory_db.decay_all()
        # print("-> Memory importance has decayed. 记忆重要性已衰减。")

    def get_all_memories(self):
        """Retrieves all memories for display/debugging."""
        return self.memory_db.get_all_memories()

    def __del__(self):
        pass  # atexit handles cleanup


def cleanup_db_dir(db_dir: str):
    """Helper function to clean up the database directory."""
    if os.path.exists(db_dir):
        import shutil
        shutil.rmtree(db_dir)
        print(f"--- Deleted database directory: {db_dir} ---")


if __name__ == '__main__':
    DB_DIR = 'memories_db_dynamic'
    PASSWORD = 'a_new_super_secure_password!'

    # --- First Run: Learning and Saving with the new dynamic I/O system ---
    cleanup_db_dir(DB_DIR)

    print("\n===== First Run: Learning and Saving a diverse set of memories =====")
    # Set a lower training threshold for testing purposes.
    simulator_instance_1 = MemorySimulator(db_dir=DB_DIR, password=PASSWORD, train_threshold=10)

    memories_to_add = [
        # Computer Hardware
        ("CPU是计算机的中央处理器。", 1.0, 0.0, []),
        ("GPU是图形处理器，擅长并行计算。", 1.0, 0.0, []),
        ("RAM是随机存取存储器，用于临时存储数据。", 1.0, 0.0, []),
        ("SSD是固态硬盘，读写速度比HDD快。", 1.0, 0.0, []),
        # Programming Concepts
        ("Python是一种高级编程语言，以其简洁著称。", 1.2, 0.0, []),
        ("C++是一种支持面向对象和泛型编程的语言。", 1.1, 0.0, []),
        ("Java是一种跨平台的面向对象编程语言。", 1.0, 0.0, []),
        ("编程中的变量是存储数据的容器。", 0.9, 0.0, []),
        # Operating Systems
        ("Linux是一个开源的操作系统内核。", 1.0, 0.0, []),
        ("Windows是微软开发的流行操作系统。", 1.0, 0.0, []),
        # Specific Technical Knowledge with associations
        ("AES 加密是一种对称加密算法。", 1.3, 0.0, []),
        ("PBKDF2 用于从密码派生密钥。", 1.3, 0.0, []),
        ("我学会在Python中用PyCryptodome进行加密。", 1.5, 0.4,
         ["AES 加密是一种对称加密算法.", "PBKDF2 用于从密码派生密钥."]),
        # Emotional Memory
        ("我最喜欢的游戏在2023年停止了服务。", 1.8, 0.7, []),
        ("在海边看日落让我感到平静。", 1.5, 0.8, []),
        ("听到那首歌，我回忆起了学生时代。", 1.6, 0.9, []),
        ("机器学习是人工智能的一个分支。", 1.4, 0.0, []),
        ("数据结构是组织和存储数据的方式。", 1.2, 0.0, []),
    ]

    for text, importance, emotion, associated_texts in memories_to_add:
        simulator_instance_1.learn_or_review_memory(
            text, importance=importance, emotional_intensity=emotion, associated_texts=associated_texts
        )

    print("\n" + "=" * 40 + "\n")

    print("===== Step 2: Performing semantic recall (First Run) =====")

    print("\n--- Query 1: What is the brain of a computer? ---")
    simulator_instance_1.recall_memory("什么是计算机的大脑？")

    print("\n--- Query 2: I need a programming language for simple scripting. ---")
    simulator_instance_1.recall_memory("我需要一个用于简单脚本的编程语言。")

    print("\n--- Query 3: What is the fastest storage? ---")
    simulator_instance_1.recall_memory("最快的存储设备是什么？")

    print("\n--- Query 4: Tell me about emotional experiences. ---")
    simulator_instance_1.recall_memory("告诉我关于情感体验的事情。")

    print("\n" + "=" * 40 + "\n")

    print("===== Step 3: Recalling with emotional trigger and checking associations =====")
    simulator_instance_1.recall_memory("我最喜欢的游戏停止了。", emotional_boost=0.8)

    print("\n--- Recalling a memory that has associations to check for boosts ---")
    simulator_instance_1.recall_memory("如何使用PyCryptodome？")

    print("\n" + "=" * 40 + "\n")
    print("--- Simulating time passage: 5 seconds ---")
    time.sleep(5)

    # Let atexit handle saving on exit
    del simulator_instance_1  # Manually trigger __del__ to ensure atexit is called

    print("\n" + "=" * 40 + "\n")

    print("===== Second Run: Loading and Recalling from saved state =====")
    # The files are saved in the DB_DIR.
    simulator_instance_2 = MemorySimulator(db_dir=DB_DIR, password=PASSWORD)

    print("\n--- Query 1: Recalling the emotional memory after some time ---")
    simulator_instance_2.recall_memory("在海边看日落让我感到平静。")

    print("\n--- Query 2: Let's query about encryption again and see the score change ---")
    simulator_instance_2.recall_memory("对称加密是什么？")

    # Note: Getting all memories is now an expensive operation, as it reads from disk.
    print("\n--- Final memory states after the second run (expensive to retrieve all) ---")
    now = time.time()
    for item in simulator_instance_2.get_all_memories():
        print(
            f"Text: \"{item.text}\" | Score: {item.get_decayed_score(now):.4f} | Recalls: {item.recall_count} | Emotion: {item.emotional_intensity:.2f}")

    # Let atexit handle saving on exit
    del simulator_instance_2  # Manually trigger __del__ to ensure atexit is called

    print("\nProcess finished with exit code 0")