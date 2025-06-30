import time
import uuid
import math
import faiss
import numpy as np
import json
import sqlite3
import os
from typing import List, Dict, Optional, Any, Callable
from Cryptodome.Cipher import AES
from Cryptodome.Protocol.KDF import PBKDF2
from sentence_transformers import SentenceTransformer


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


# ----------------- MemoryDB: Optimized with SQLite and trained FAISS index -----------------
class MemoryDB:
    """Memory storage and retrieval system with SQLite and a trained FAISS index."""

    def __init__(self, dim: int, db_path: str = 'memory.db', password: str = 'my_secret_password',
                 embedding_func: Callable[[str], List[float]] = None, train_threshold: int = 1000):
        self.dim = dim
        self.db_path = db_path
        self.password = password
        self.embedding_func = embedding_func
        self.train_threshold = train_threshold  # Threshold to start training the FAISS index
        self.faiss_index_path = f"{db_path.split('.')[0]}_faiss.bin"
        self.faiss_id_map_path = f"{db_path.split('.')[0]}_faiss_id_map.json"
        self.salt_path = f"{db_path.split('.')[0]}_salt.bin"

        self.items: Dict[str, MemoryItem] = {}
        self.id_map: Dict[int, str] = {}
        self.rev_id_map: Dict[str, int] = {}
        self.next_index_id = 0
        self.conn = None
        self.enc_manager = None
        self._train_buffer = []
        self._train_ids = []

        self._init_db()
        self._load_memories_from_db()

        self.index = None
        self._load_faiss_index()

    def _init_db(self):
        """Initializes the database schema and encryption salt."""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()

        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS crypto_params
                       (
                           param_name
                           TEXT
                           PRIMARY
                           KEY,
                           param_value
                           BLOB
                           NOT
                           NULL
                       )
                       ''')

        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS memories
                       (
                           id
                           TEXT
                           PRIMARY
                           KEY,
                           nonce
                           BLOB
                           NOT
                           NULL,
                           encrypted_data
                           BLOB
                           NOT
                           NULL
                       )
                       ''')
        self.conn.commit()

        if os.path.exists(self.salt_path):
            with open(self.salt_path, 'rb') as f:
                salt = f.read()
            self.enc_manager = EncryptionManager(self.password, salt)
            print("-> Found existing salt. Initializing EncryptionManager...")
        else:
            salt = os.urandom(16)
            with open(self.salt_path, 'wb') as f:
                f.write(salt)
            self.enc_manager = EncryptionManager(self.password, salt)
            print("-> No salt found. Generating and saving a new one. Initializing EncryptionManager...")

    def _load_memories_from_db(self):
        """Loads all memories from the SQLite DB into memory."""
        print(f"--- Loading memories from DB: {self.db_path} ---")
        cursor = self.conn.cursor()
        cursor.execute("SELECT nonce, encrypted_data FROM memories")

        for row in cursor.fetchall():
            nonce = bytes(row[0])
            encrypted_data_with_tag = bytes(row[1])
            try:
                decrypted_data = self.enc_manager.decrypt(nonce, encrypted_data_with_tag)
                data = json.loads(decrypted_data.decode('utf-8'))
                item = MemoryItem.from_dict(data)
                self.items[item.id] = item
            except ValueError:
                print(
                    "-> WARNING: Failed to decrypt a record, possibly due to tampering or incorrect password. Skipping.")
                continue
        print(f"-> Loaded {len(self.items)} memories.")

    def _load_faiss_index(self):
        """Loads the FAISS index and ID map from disk if they exist, or creates a new one."""
        if os.path.exists(self.faiss_index_path) and os.path.exists(self.faiss_id_map_path):
            print("-> Found trained FAISS index. Loading from disk...")
            try:
                self.index = faiss.read_index(self.faiss_index_path)
                with open(self.faiss_id_map_path, 'r') as f:
                    self.id_map = {int(k): v for k, v in json.load(f).items()}
                    self.rev_id_map = {v: k for k, v in self.id_map.items()}
                print(f"-> FAISS index loaded. Total vectors: {self.index.ntotal}")

                # Verify that all loaded memories are in the index
                if self.index.ntotal != len(self.items):
                    print("-> WARNING: Index count does not match memories in DB. Rebuilding index.")
                    self._build_index_from_memories()
            except Exception as e:
                print(f"-> ERROR: Failed to load FAISS index ({e}). Rebuilding from scratch.")
                self._build_index_from_memories()
        else:
            print("-> No trained FAISS index found. Building from scratch...")
            self._build_index_from_memories()

    def _build_index_from_memories(self):
        """Builds a new FAISS index from all memories loaded in self.items."""
        # Use a dummy FlatL2 for initial training
        quantizer = faiss.IndexFlatL2(self.dim)
        nlist = min(100, len(self.items) // 39)  # Rule of thumb: nlist = sqrt(ntotal)
        nlist = max(1, nlist)
        m_pq = 16

        self.index = faiss.IndexIVFPQ(quantizer, self.dim, nlist, m_pq, 8)
        self.index.nprobe = max(1, nlist // 10)  # Set search parameter

        self.id_map = {}
        self.rev_id_map = {}
        self.next_index_id = 0

        if len(self.items) > self.train_threshold:
            print(f"--- Training FAISS index with {len(self.items)} vectors ---")
            embeddings = np.array([item.embedding for item in self.items.values()]).astype('float32')
            self.index.train(embeddings)

            print("--- Adding vectors to the trained index ---")
            for item in self.items.values():
                faiss_idx = self.index.ntotal
                self.index.add(np.expand_dims(item.embedding, axis=0))
                self.id_map[faiss_idx] = item.id
                self.rev_id_map[item.id] = faiss_idx
            self.next_index_id = self.index.ntotal
            print(f"-> FAISS index trained and populated with {self.index.ntotal} vectors.")
        else:
            print(f"-> Not enough vectors ({len(self.items)}) to train FAISS IVFPQ. Using a FlatL2 index.")
            self.index = faiss.IndexFlatL2(self.dim)
            for item in self.items.values():
                faiss_idx = self.index.ntotal
                self.index.add(np.expand_dims(item.embedding, axis=0))
                self.id_map[faiss_idx] = item.id
                self.rev_id_map[item.id] = faiss_idx
            self.next_index_id = self.index.ntotal
            print(f"-> FAISS FlatL2 index populated with {self.index.ntotal} vectors.")

    def _save_faiss_index(self):
        """Saves the FAISS index and ID map to disk."""
        if self.index and self.index.ntotal > 0:
            print("--- Saving FAISS index and ID map ---")
            faiss.write_index(self.index, self.faiss_index_path)
            with open(self.faiss_id_map_path, 'w') as f:
                json.dump(self.id_map, f)
            print("-> FAISS index saved.")

    def add_memory(self, item: MemoryItem):
        """Adds a new memory to the database and in-memory structures, or updates an existing one."""
        # Add to in-memory dictionary
        if item.id not in self.items:
            # If it's a new item, add it to the FAISS index
            faiss_idx = self.index.ntotal
            self.index.add(np.expand_dims(item.embedding, axis=0))
            self.id_map[faiss_idx] = item.id
            self.rev_id_map[item.id] = faiss_idx
            self.next_index_id = self.index.ntotal

        self.items[item.id] = item

        # Save to SQLite database
        cursor = self.conn.cursor()
        data_json = json.dumps(item.to_dict())
        nonce, encrypted_data_with_tag = self.enc_manager.encrypt(data_json.encode('utf-8'))

        # Use INSERT OR REPLACE to handle both adding new items and updating existing ones
        cursor.execute("INSERT OR REPLACE INTO memories (id, nonce, encrypted_data) VALUES (?, ?, ?)",
                       (item.id, sqlite3.Binary(nonce), sqlite3.Binary(encrypted_data_with_tag)))
        self.conn.commit()

    def find_memory_by_text(self, text: str, similarity_threshold: float = 0.9) -> Optional[MemoryItem]:
        """Finds a memory by its text content using semantic similarity."""
        if not self.embedding_func:
            print("WARNING: No embedding function provided. Semantic search is not possible.")
            return None

        query_embedding = np.array(self.embedding_func(text)).astype(np.float32)

        for item in self.items.values():
            if item.text == text:  # Exact match is prioritized
                return item

            # Use associability score for semantic similarity check
            if item.associability_score(query_embedding) > similarity_threshold:
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
            self.add_memory(existing_item)
            return

        print("-> No existing memory found. Treating as new memory.")
        self.add_memory(item)

    def delete_memory(self, item_id: str):
        """Deletes a memory from the database and in-memory structures."""
        if item_id in self.items:
            del self.items[item_id]
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM memories WHERE id = ?", (item_id,))
            self.conn.commit()
            print(f"-> Memory {item_id} deleted from DB.")
            # Rebuild the FAISS index to ensure it's clean
            self._build_index_from_memories()

    def semantic_search(self, embedding: List[float], top_k: int = 5, emotional_boost: float = 0.0) -> List[MemoryItem]:
        """
        Performs semantic search with emotional boost.
        Upon recall, this method triggers the 'reconstruction' of associated memories.
        """
        query = np.array(embedding).astype(np.float32).reshape(1, -1)
        if self.index.ntotal == 0:
            return []

        distances, indices = self.index.search(query, top_k)
        results = []

        recalled_ids = []
        for i in indices[0]:
            if i in self.id_map:
                item_id = self.id_map[i]
                item = self.items[item_id]
                item.update_on_recall(emotional_boost)
                results.append(item)
                recalled_ids.append(item_id)

        # Recursively update associated memories with a weaker emotional boost
        for recalled_id in recalled_ids:
            recalled_item = self.items[recalled_id]
            for associated_id in recalled_item.associated_memories:
                if associated_id in self.items:
                    self.items[associated_id].update_on_recall(emotional_boost * 0.5)

        current_time = time.time()
        # Sort by the final decayed score to present the most "important" memories first
        return sorted(results, key=lambda x: x.spaced_repetition_score(current_time), reverse=True)

    def decay_all(self):
        """Decay all memory scores and delete if below threshold."""
        current_time = time.time()
        to_delete = []
        for item in self.items.values():
            if item.get_decayed_score(current_time) < 0.01:
                to_delete.append(item.id)
        for item_id in to_delete:
            self.delete_memory(item_id)

    def get_all_memories(self):
        """Retrieves all memories for display/debugging."""
        return self.items.values()

    def __del__(self):
        """Saves all in-memory memory states to DB, saves FAISS index, and closes the connection."""
        self._save_faiss_index()
        if self.conn:
            print("\n--- Saving all memory states before closing the database ---")
            for item in self.items.values():
                self.add_memory(item)
            self.conn.close()
            print("-> Database connection closed.")


class MemorySimulator:
    """Simulates learning, recall, association and decay with a real embedding model."""

    def __init__(self, db_path: str = 'enhanced_memories_db.db', password: str = 'my_secret_password',
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
        self.memory_db = MemoryDB(self.embedding_dim, db_path=db_path, password=password,
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
        self.memory_db.decay_all()
        print("-> Memory importance has decayed. 记忆重要性已衰减。")

    def get_all_memories(self):
        """Retrieves all memories for display/debugging."""
        return self.memory_db.get_all_memories()

    def __del__(self):
        del self.memory_db


if __name__ == '__main__':
    DB_PATH = 'enhanced_memories_db.db'
    PASSWORD = 'a_new_super_secure_password!'

    # --- First Run: Learning and Saving with New Features ---
    # Delete the old DB file and associated FAISS files to start fresh
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"--- Deleted old database file: {DB_PATH} ---")

    faiss_index_path = f"{DB_PATH.split('.')[0]}_faiss.bin"
    faiss_id_map_path = f"{DB_PATH.split('.')[0]}_faiss_id_map.json"
    salt_path = f"{DB_PATH.split('.')[0]}_salt.bin"
    if os.path.exists(faiss_index_path):
        os.remove(faiss_index_path)
        print(f"--- Deleted old FAISS index file: {faiss_index_path} ---")
    if os.path.exists(faiss_id_map_path):
        os.remove(faiss_id_map_path)
        print(f"--- Deleted old FAISS ID map file: {faiss_id_map_path} ---")
    if os.path.exists(salt_path):
        os.remove(salt_path)
        print(f"--- Deleted old salt file: {salt_path} ---")

    print("\n===== First Run: Learning and Saving a diverse set of memories =====")
    # Set a lower training threshold for testing purposes.
    # For a production system, use a larger number (e.g., 10000).
    simulator_instance_1 = MemorySimulator(db_path=DB_PATH, password=PASSWORD, train_threshold=10)

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
         ["AES 加密是一种对称加密算法。", "PBKDF2 用于从密码派生密钥。"]),
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
    simulator_instance_1.memory_db.decay_all()
    print("-> Memory importance has decayed. 记忆重要性已衰减。")

    del simulator_instance_1  # This will trigger saving the FAISS index and closing the DB

    print("\n" + "=" * 40 + "\n")

    print("===== Second Run: Loading and Recalling from saved state =====")
    # Now, all memories are saved to the SQLite file and the FAISS index is saved to disk.
    simulator_instance_2 = MemorySimulator(db_path=DB_PATH, password=PASSWORD)

    print("\n--- Query 1: Recalling the emotional memory after some time ---")
    simulator_instance_2.recall_memory("在海边看日落让我感到平静。")

    print("\n--- Query 2: Let's query about encryption again and see the score change ---")
    simulator_instance_2.recall_memory("对称加密是什么？")

    print("\n--- Final memory states after the second run ---")
    now = time.time()
    for item in simulator_instance_2.get_all_memories():
        print(
            f"Text: \"{item.text}\" | Score: {item.get_decayed_score(now):.4f} | Recalls: {item.recall_count} | Emotion: {item.emotional_intensity:.2f}")

    del simulator_instance_2

    print("\nProcess finished with exit code 0")