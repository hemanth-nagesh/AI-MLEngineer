# optimized_financial_news_system.py
import os
import asyncio
from typing import TypedDict, List, Dict, Annotated, Literal, AsyncIterator
from datetime import datetime, timedelta
import hashlib
import json
import pickle

# Core imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from langsmith import traceable
import spacy
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import psycopg2
from psycopg2.extras import execute_values, Json
from contextlib import asynccontextmanager
import redis.asyncio as aioredis
from cachetools import LRUCache
from concurrent.futures import ThreadPoolExecutor
import uvicorn
from dotenv import load_dotenv

load_dotenv()
# ============= CONFIGURATION =============
OPTIMIZED_CONFIG = {
    "batch_size": 32,
    "max_concurrent_agents": 10,
    "faiss_nlist": 100,
    "faiss_m": 8,
    "faiss_threshold": 0.87,
    "cache_ttl_hours": 24,
    "checkpoint_ttl_days": 30,
    "enable_streaming": True,
    "connection_pool_size": 20,
}

# Environment setup
# os.environ["LANGCHAIN_TRACING_V2"]
# os.environ["LANGCHAIN_PROJECT"]
# os.environ["LANGCHAIN_API_KEY"]
api_key = os.getenv("GOOGLE_API_KEY")

from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://postgres:12345@localhost/financial_news"



# Initialize models
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key, temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", api_key=api_key)

# ============= OPTIMIZED STATE DEFINITION =============
class OptimizedNewsState(TypedDict):
    """Enhanced state with batching support"""
    # Batch processing
    articles_batch: List[Dict]
    current_article_idx: int
    
    # Single article processing (backward compatible)
    raw_article: Dict
    query: str
    
    # Processing results
    normalized_articles: List[Dict]
    article_hashes: List[str]
    embeddings_batch: List[List[float]]
    
    # Deduplication
    duplicate_results: List[Dict]
    
    # Entity extraction
    entities_batch: List[List[Dict]]
    
    # Stock impact
    stocks_batch: List[List[Dict]]
    
    # Query processing
    query_results: List[Dict]
    
    # Routing
    next_agent: str
    messages: Annotated[List, "append"]
    processing_mode: str  # 'single' or 'batch'


# ============= OPTIMIZED DATABASE =============
class OptimizedDatabaseManager:
    def __init__(self):
        self.conn = psycopg2.connect(
            dbname="financial_news",
            user="postgres",
            password="12345",
            host="localhost",
            # Connection pool settings
            options="-c statement_timeout=30000"
        )
        self.setup_optimized_database()
        
    def setup_optimized_database(self):
        """Initialize optimized PostgreSQL schema"""
        with self.conn.cursor() as cur:
            # Articles table with compression (NO vector storage - FAISS only)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS articles (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT,
                    published_date TIMESTAMP,
                    article_hash TEXT UNIQUE,
                    duplicate_group_id TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                );
                
                -- Enable text compression
                ALTER TABLE articles ALTER COLUMN content SET STORAGE EXTERNAL;
                ALTER TABLE articles ALTER COLUMN content SET COMPRESSION pglz;
                
                -- Indexes for fast queries
                CREATE INDEX IF NOT EXISTS idx_article_hash ON articles(article_hash);
                CREATE INDEX IF NOT EXISTS idx_published_date ON articles(published_date DESC);
                CREATE INDEX IF NOT EXISTS idx_duplicate_group ON articles(duplicate_group_id);
            """)
            
            # Entities table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    normalized_name TEXT,
                    UNIQUE(name, type)
                );
                CREATE INDEX IF NOT EXISTS idx_entity_name ON entities(name);
            """)
            
            # Article entities junction
            cur.execute("""
                CREATE TABLE IF NOT EXISTS article_entities (
                    article_id INTEGER REFERENCES articles(id) ON DELETE CASCADE,
                    entity_id INTEGER REFERENCES entities(id) ON DELETE CASCADE,
                    confidence FLOAT,
                    PRIMARY KEY (article_id, entity_id)
                );
                CREATE INDEX IF NOT EXISTS idx_ae_article ON article_entities(article_id);
                CREATE INDEX IF NOT EXISTS idx_ae_entity ON article_entities(entity_id);
            """)
            
            # Stock impacts
            cur.execute("""
                CREATE TABLE IF NOT EXISTS stock_impacts (
                    id SERIAL PRIMARY KEY,
                    article_id INTEGER REFERENCES articles(id) ON DELETE CASCADE,
                    stock_symbol TEXT NOT NULL,
                    confidence FLOAT,
                    impact_type TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_stock_symbol ON stock_impacts(stock_symbol);
                CREATE INDEX IF NOT EXISTS idx_si_article ON stock_impacts(article_id);
            """)
            
            # Entity to stock mapping
            cur.execute("""
                CREATE TABLE IF NOT EXISTS entity_stock_mappings (
                    entity_name TEXT PRIMARY KEY,
                    stock_symbol TEXT NOT NULL,
                    sector TEXT,
                    relationship_type TEXT
                );
            """)
            
            # Materialized view for fast queries
            cur.execute("""
                DROP MATERIALIZED VIEW IF EXISTS stock_news_summary;
                CREATE MATERIALIZED VIEW stock_news_summary AS
                SELECT 
                    si.stock_symbol,
                    COUNT(DISTINCT a.id) as article_count,
                    MAX(a.published_date) as latest_news,
                    AVG(si.confidence) as avg_confidence
                FROM stock_impacts si
                JOIN articles a ON si.article_id = a.id
                GROUP BY si.stock_symbol;
                
                CREATE INDEX idx_stock_summary ON stock_news_summary(stock_symbol);
            """)
            
            # try:
            #     cur.execute("""
            #         DELETE FROM checkpoints 
            #         WHERE created_at < NOW() - INTERVAL '%s days';
            #     """, (OPTIMIZED_CONFIG['checkpoint_ttl_days'],))
            # except psycopg2.errors.UndefinedTable:
            #     # LangGraph checkpoint table not created yet – ignore
            #     self.conn.rollback()
            
        self.conn.commit()
        print("✅ Optimized database initialized")
    
    def batch_insert_articles(self, articles_data: List[tuple]):
        """Batch insert for better performance"""
        with self.conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO articles (title, content, source, published_date, 
                                     article_hash, duplicate_group_id)
                VALUES %s
                RETURNING id
            """, articles_data)
            article_ids = [row[0] for row in cur.fetchall()]
        self.conn.commit()
        return article_ids
    
    def refresh_materialized_views(self):
        """Async refresh for materialized views"""
        with self.conn.cursor() as cur:
            cur.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY stock_news_summary;")
        self.conn.commit()

db = OptimizedDatabaseManager()


# ============= OPTIMIZED FAISS WITH PRODUCT QUANTIZATION =============
class OptimizedFAISSIndex:
    """Memory-efficient FAISS with IVF-PQ (27× memory reduction)"""
    
    def __init__(self, dimension=768):
        self.dimension = dimension
        self.nlist = OPTIMIZED_CONFIG['faiss_nlist']
        self.m = OPTIMIZED_CONFIG['faiss_m']
        
        # Create IVF-PQ index (quantized)
        quantizer = faiss.IndexFlatIP(dimension)
        self.index = faiss.IndexIVFPQ(
            quantizer,
            dimension,
            self.nlist,      # Number of clusters
            self.m,          # Subquantizers
            8                # Bits per subquantizer
        )
        
        self.trained = False
        self.article_ids = []
        self.training_buffer = []
        print(f"🔧 Initialized IVF-PQ index: nlist={self.nlist}, M={self.m}")
    
    def add_to_training_buffer(self, embeddings: np.ndarray):
        """Collect embeddings for training"""
        self.training_buffer.append(embeddings)
    
    def train_index(self):
        """Train IVF-PQ index on collected data"""
        if self.trained:
            return
        
        if len(self.training_buffer) == 0:
            print("⚠️  No training data available")
            return
        
        training_data = np.vstack(self.training_buffer).astype('float32')
        
        if len(training_data) < self.nlist:
            print(f"⚠️  Need at least {self.nlist} vectors for training, got {len(training_data)}")
            return
        
        print(f"🔧 Training IVF-PQ index with {len(training_data)} vectors...")
        faiss.normalize_L2(training_data)
        self.index.train(training_data)
        self.trained = True
        print("✅ Index trained successfully")
        
        # Add training data to index
        self.index.add(training_data)
        self.article_ids.extend([f"train_{i}" for i in range(len(training_data))])
    
    def add(self, embeddings: np.ndarray, article_ids: List[str]):
        """Add embeddings to index"""
        embeddings = embeddings.astype('float32')
        faiss.normalize_L2(embeddings)
        
        if not self.trained:
            # Collect for training
            self.add_to_training_buffer(embeddings)
            if len(self.training_buffer) >= 50:  # Train after 50 batches
                self.train_index()
        else:
            self.index.add(embeddings)
            self.article_ids.extend(article_ids)
    
    def search(self, query_embedding: np.ndarray, k=5, threshold=0.87):
        """Search with quantized index"""
        if not self.trained or self.index.ntotal == 0:
            return []
        
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Set search parameters for better accuracy
        self.index.nprobe = 10  # Search 10 clusters
        
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.article_ids) and dist >= threshold:
                results.append({
                    'article_id': self.article_ids[idx],
                    'similarity': float(dist)
                })
        return results
    
    def get_memory_usage_mb(self):
        """Calculate memory usage in MB"""
        if self.index.ntotal == 0:
            return 0
        bytes_per_vector = self.m  # M bytes per vector with PQ
        return (self.index.ntotal * bytes_per_vector) / (1024**2)

faiss_index = OptimizedFAISSIndex()


# ============= HYBRID CACHE (REDIS + MEMORY) =============
class HybridCache:
    """Two-tier caching: L1 (Memory) + L2 (Redis)"""
    
    def __init__(self):
        self.memory_cache = LRUCache(maxsize=1000)
        self.redis_client = None
        self.enabled = True
        
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = await aioredis.from_url(
                "redis://localhost:6379/0",
                encoding="utf-8",
                decode_responses=False
            )
            await self.redis_client.ping()
            print("✅ Redis cache initialized")
        except Exception as e:
            print(f"⚠️  Redis unavailable, using memory-only cache: {e}")
            self.enabled = False
    
    def _generate_key(self, prefix: str, data: str) -> str:
        """Generate cache key"""
        return f"{prefix}:{hashlib.md5(data.encode()).hexdigest()}"
    
    async def get(self, key: str):
        """Get from L1 then L2"""
        # Check L1 (memory)
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        # Check L2 (Redis)
        if self.enabled and self.redis_client:
            try:
                cached = await self.redis_client.get(key)
                if cached:
                    value = pickle.loads(cached)
                    self.memory_cache[key] = value  # Promote to L1
                    return value
            except Exception as e:
                print(f"Redis get error: {e}")
        
        return None
    
    async def set(self, key: str, value, ttl_seconds=3600*24):
        """Set in both L1 and L2"""
        # Store in L1
        self.memory_cache[key] = value
        
        # Store in L2
        if self.enabled and self.redis_client:
            try:
                await self.redis_client.setex(
                    key,
                    ttl_seconds,
                    pickle.dumps(value)
                )
            except Exception as e:
                print(f"Redis set error: {e}")
    
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()

cache = HybridCache()


# ============= LIGHTWEIGHT NER =============
class LightweightNER:
    """Memory-efficient NER with on-demand model loading"""
    
    def __init__(self):
        # Load small model (12MB) by default
        self.nlp_small = spacy.load("en_core_web_sm")
        self.nlp_large = None  # Load on demand
        print("✅ Loaded lightweight spaCy model (12MB)")
    
    def extract_entities_fast(self, text: str) -> List[Dict]:
        """Fast extraction with small model"""
        doc = self.nlp_small(text)
        
        entities = []
        entity_types = {
            'ORG': 'company',
            'PERSON': 'person',
            'GPE': 'location',
            'MONEY': 'financial_metric',
            'CARDINAL': 'number'
        }
        
        for ent in doc.ents:
            entities.append({
                'name': ent.text,
                'type': entity_types.get(ent.label_, 'other'),
                'confidence': 0.85
            })
        
        return entities
    
    def extract_entities_precise(self, text: str) -> List[Dict]:
        """High-precision extraction (load large model on demand)"""
        if self.nlp_large is None:
            print("🔄 Loading large spaCy model...")
            self.nlp_large = spacy.load("en_core_web_trf")
        
        doc = self.nlp_large(text)
        entities = []
        entity_types = {
            'ORG': 'company',
            'PERSON': 'person',
            'GPE': 'location',
            'MONEY': 'financial_metric'
        }
        
        for ent in doc.ents:
            entities.append({
                'name': ent.text,
                'type': entity_types.get(ent.label_, 'other'),
                'confidence': 0.95
            })
        
        return entities

ner = LightweightNER()


# ============= ENTITY/STOCK MAPPING =============
ENTITY_STOCK_MAPPING = {
    "HDFC Bank": {"symbol": "HDFCBANK", "sector": "Banking"},
    "ICICI Bank": {"symbol": "ICICIBANK", "sector": "Banking"},
    "State Bank of India": {"symbol": "SBIN", "sector": "Banking"},
    "SBI": {"symbol": "SBIN", "sector": "Banking"},
    "Axis Bank": {"symbol": "AXISBANK", "sector": "Banking"},
    "Kotak Mahindra Bank": {"symbol": "KOTAKBANK", "sector": "Banking"},
    
    "Reliance Industries": {"symbol": "RELIANCE", "sector": "Energy"},
    "Tata Steel": {"symbol": "TATASTEEL", "sector": "Steel"},
    
    "Infosys": {"symbol": "INFY", "sector": "IT"},
    "TCS": {"symbol": "TCS", "sector": "IT"},
    "Tata Consultancy Services": {"symbol": "TCS", "sector": "IT"},
    "Wipro": {"symbol": "WIPRO", "sector": "IT"},
    
    "RBI": {"symbol": None, "sector": "Banking", "type": "regulator"},
    "Reserve Bank of India": {"symbol": None, "sector": "Banking", "type": "regulator"},
    "SEBI": {"symbol": None, "sector": "Capital Markets", "type": "regulator"},
    
    "Banking": {"sector": "Banking", "stocks": ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK"]},
    "IT": {"sector": "IT", "stocks": ["INFY", "TCS", "WIPRO"]},
    "Financial Services": {"sector": "Financial Services", "stocks": ["HDFCBANK", "ICICIBANK", "BAJFINANCE"]},
}


# ============= OPTIMIZED AGENTS =============

@traceable(name="batch_ingestion_agent")
async def batch_ingestion_agent(state: OptimizedNewsState) -> OptimizedNewsState:
    """Process multiple articles in batch"""
    
    if state.get('processing_mode') == 'single':
        # Single article mode (backward compatible)
        raw = state['raw_article']
        articles = [raw]
    else:
        # Batch mode
        articles = state['articles_batch']
    
    normalized = []
    hashes = []
    
    for article in articles:
        content_hash = hashlib.sha256(
            f"{article['title']}{article['content']}".encode()
        ).hexdigest()
        
        normalized.append({
            'title': article['title'].strip(),
            'content': article['content'].strip(),
            'source': article.get('source', 'unknown'),
            'published_date': article.get('published_date', datetime.now().isoformat()),
        })
        hashes.append(content_hash)
    
    state['normalized_articles'] = normalized
    state['article_hashes'] = hashes
    state['messages'] = [f"✅ Ingested {len(normalized)} articles"]
    
    print(f"📥 Batch ingested: {len(normalized)} articles")
    return state


@traceable(name="batch_embedding_generation")
async def batch_embedding_generation(state: OptimizedNewsState) -> OptimizedNewsState:
    """Generate embeddings in batches (more efficient)"""
    
    articles = state['normalized_articles']
    texts = [f"{a['title']} {a['content']}" for a in articles]
    
    # Check cache first
    all_embeddings = []
    texts_to_compute = []
    text_indices = []
    
    for idx, text in enumerate(texts):
        cache_key = cache._generate_key("emb", text)
        cached = await cache.get(cache_key)
        
        if cached:
            all_embeddings.append(cached)
        else:
            all_embeddings.append(None)
            texts_to_compute.append(text)
            text_indices.append(idx)
    
    # Compute uncached embeddings in batches
    if texts_to_compute:
        batch_size = 32
        computed_embeddings = []
        
        for i in range(0, len(texts_to_compute), batch_size):
            batch = texts_to_compute[i:i+batch_size]
            # Async batch embedding
            batch_embeddings = await asyncio.to_thread(
                embeddings.embed_documents,
                batch
            )
            computed_embeddings.extend(batch_embeddings)
        
        # Cache computed embeddings
        for text, emb in zip(texts_to_compute, computed_embeddings):
            cache_key = cache._generate_key("emb", text)
            await cache.set(cache_key, emb)
        
        # Fill in computed embeddings
        for idx, emb in zip(text_indices, computed_embeddings):
            all_embeddings[idx] = emb
    
    state['embeddings_batch'] = all_embeddings
    state['messages'] = [f"🔢 Generated {len(all_embeddings)} embeddings"]
    
    print(f"🔢 Embeddings: {len(all_embeddings)} vectors")
    return state


@traceable(name="batch_deduplication_agent")
async def batch_deduplication_agent(state: OptimizedNewsState) -> OptimizedNewsState:
    """Batch duplicate detection using optimized FAISS"""
    
    embeddings_list = state['embeddings_batch']
    hashes = state['article_hashes']
    
    duplicate_results = []
    embeddings_array = np.array(embeddings_list, dtype=np.float32)
    
    # Batch search in FAISS
    for idx, (embedding, article_hash) in enumerate(zip(embeddings_list, hashes)):
        embedding_np = np.array(embedding, dtype=np.float32)
        
        similar = faiss_index.search(
            embedding_np,
            k=5,
            threshold=OPTIMIZED_CONFIG['faiss_threshold']
        )
        
        if similar:
            duplicate_results.append({
                'index': idx,
                'is_duplicate': True,
                'duplicate_group_id': similar[0]['article_id'],
                'similarity': similar[0]['similarity']
            })
        else:
            duplicate_results.append({
                'index': idx,
                'is_duplicate': False,
                'duplicate_group_id': article_hash,
                'similarity': 0.0
            })
    
    # Add unique articles to index
    unique_embeddings = []
    unique_ids = []
    
    for idx, result in enumerate(duplicate_results):
        if not result['is_duplicate']:
            unique_embeddings.append(embeddings_list[idx])
            unique_ids.append(hashes[idx])
    
    if unique_embeddings:
        unique_array = np.array(unique_embeddings, dtype=np.float32)
        faiss_index.add(unique_array, unique_ids)
    
    state['duplicate_results'] = duplicate_results
    
    unique_count = sum(1 for r in duplicate_results if not r['is_duplicate'])
    duplicate_count = len(duplicate_results) - unique_count
    
    state['messages'] = [
        f"🔄 Deduplication: {unique_count} unique, {duplicate_count} duplicates"
    ]
    
    print(f"🔄 Unique: {unique_count}, Duplicates: {duplicate_count}")
    print(f"📊 FAISS Memory: {faiss_index.get_memory_usage_mb():.2f} MB")
    
    return state


@traceable(name="parallel_entity_extraction")
async def parallel_entity_extraction_agent(state: OptimizedNewsState) -> OptimizedNewsState:
    """Extract entities in parallel with caching"""
    
    articles = state['normalized_articles']
    duplicate_results = state['duplicate_results']
    
    async def extract_with_cache(article, is_duplicate):
        """Extract entities with cache"""
        if is_duplicate:
            return []  # Skip duplicates
        
        text = f"{article['title']} {article['content']}"
        cache_key = cache._generate_key("ent", text)
        
        cached = await cache.get(cache_key)
        if cached:
            return cached
        
        # Use lightweight NER
        entities = await asyncio.to_thread(ner.extract_entities_fast, text)
        
        # Enhance with known mappings
        for term, info in ENTITY_STOCK_MAPPING.items():
            if term.lower() in text.lower() and term not in [e['name'] for e in entities]:
                entities.append({
                    'name': term,
                    'type': info.get('type', 'company'),
                    'normalized_name': term,
                    'confidence': 1.0
                })
        
        await cache.set(cache_key, entities)
        return entities
    
    # Parallel extraction with semaphore
    semaphore = asyncio.Semaphore(OPTIMIZED_CONFIG['max_concurrent_agents'])
    
    async def bounded_extract(article, result):
        async with semaphore:
            return await extract_with_cache(article, result['is_duplicate'])
    
    entities_batch = await asyncio.gather(*[
        bounded_extract(article, result)
        for article, result in zip(articles, duplicate_results)
    ])
    
    state['entities_batch'] = entities_batch
    
    total_entities = sum(len(e) for e in entities_batch)
    state['messages'] = [f"🏷️  Extracted {total_entities} entities across {len(articles)} articles"]
    
    print(f"🏷️  Entities: {total_entities} total")
    return state


@traceable(name="batch_stock_impact_agent")
async def batch_stock_impact_agent(state: OptimizedNewsState) -> OptimizedNewsState:
    """Map entities to stocks in batch"""
    
    entities_batch = state['entities_batch']
    
    stocks_batch = []
    
    for entities in entities_batch:
        impacted_stocks = []
        
        for entity in entities:
            name = entity.get('normalized_name') or entity['name']
            
            if name in ENTITY_STOCK_MAPPING:
                info = ENTITY_STOCK_MAPPING[name]
                
                if info.get('symbol'):
                    impacted_stocks.append({
                        'symbol': info['symbol'],
                        'confidence': 1.0,
                        'impact_type': 'direct',
                        'entity': name
                    })
                elif info.get('type') == 'regulator' or 'stocks' in info:
                    sector_stocks = info.get('stocks', [])
                    for stock in sector_stocks:
                        impacted_stocks.append({
                            'symbol': stock,
                            'confidence': 0.7,
                            'impact_type': 'sector',
                            'entity': name
                        })
        
        # Deduplicate
        unique_stocks = {}
        for stock in impacted_stocks:
            symbol = stock['symbol']
            if symbol not in unique_stocks or stock['confidence'] > unique_stocks[symbol]['confidence']:
                unique_stocks[symbol] = stock
        
        stocks_batch.append(list(unique_stocks.values()))
    
    state['stocks_batch'] = stocks_batch
    
    total_stocks = sum(len(s) for s in stocks_batch)
    state['messages'] = [f"📈 Mapped {total_stocks} stock impacts"]
    
    print(f"📈 Stock impacts: {total_stocks}")
    return state


@traceable(name="batch_storage_agent")
async def batch_storage_agent(state: OptimizedNewsState) -> OptimizedNewsState:
    """Batch insert into PostgreSQL"""
    
    articles = state['normalized_articles']
    hashes = state['article_hashes']
    duplicate_results = state['duplicate_results']
    entities_batch = state['entities_batch']
    stocks_batch = state['stocks_batch']
    
    # Prepare batch insert data
    articles_data = []
    for article, hash_val, dup_result in zip(articles, hashes, duplicate_results):
        articles_data.append((
            article['title'],
            article['content'],
            article['source'],
            article['published_date'],
            hash_val,
            dup_result['duplicate_group_id']
        ))
    
    # Batch insert articles
    article_ids = db.batch_insert_articles(articles_data)
    
    # Insert entities and stocks
    with db.conn.cursor() as cur:
        for article_id, entities, stocks in zip(article_ids, entities_batch, stocks_batch):
            # Insert entities
            for entity in entities:
                cur.execute("""
                    INSERT INTO entities (name, type, normalized_name)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (name, type) DO UPDATE SET normalized_name = EXCLUDED.normalized_name
                    RETURNING id
                """, (entity['name'], entity['type'], entity.get('normalized_name')))
                
                entity_id = cur.fetchone()[0]
                
                cur.execute("""
                    INSERT INTO article_entities (article_id, entity_id, confidence)
                    VALUES (%s, %s, %s)
                    ON CONFLICT DO NOTHING
                """, (article_id, entity_id, entity['confidence']))
            
            # Insert stock impacts
            for stock in stocks:
                cur.execute("""
                    INSERT INTO stock_impacts (article_id, stock_symbol, confidence, impact_type)
                    VALUES (%s, %s, %s, %s)
                """, (article_id, stock['symbol'], stock['confidence'], stock['impact_type']))
    
    db.conn.commit()
    
    state['messages'] = [f"💾 Stored {len(article_ids)} articles in database"]
    
    print(f"💾 Database: {len(article_ids)} articles stored")
    
    # Refresh materialized views asynchronously
    asyncio.create_task(asyncio.to_thread(db.refresh_materialized_views))
    
    return state


@traceable(name="optimized_query_processor")
async def optimized_query_processor_agent(state: OptimizedNewsState) -> OptimizedNewsState:
    """Context-aware query with caching and streaming"""
    
    query = state['query']
    
    # Check query cache
    cache_key = cache._generate_key("query", query)
    cached_results = await cache.get(cache_key)
    
    if cached_results:
        state['query_results'] = cached_results
        state['messages'] = [f"🔍 Found {len(cached_results)} results (cached)"]
        return state
    
    # Extract entities from query
    query_entities = await asyncio.to_thread(ner.extract_entities_fast, query)
    query_entity_names = [ent['name'] for ent in query_entities]
    
    # Expand query
    search_stocks = []
    for entity_name in query_entity_names:
        if entity_name in ENTITY_STOCK_MAPPING:
            info = ENTITY_STOCK_MAPPING[entity_name]
            
            if info.get('symbol'):
                search_stocks.append(info['symbol'])
                sector = info.get('sector')
                if sector and sector in ENTITY_STOCK_MAPPING:
                    search_stocks.extend(ENTITY_STOCK_MAPPING[sector].get('stocks', []))
            elif 'stocks' in info:
                search_stocks.extend(info['stocks'])
    
    # Semantic search
    query_embedding = await asyncio.to_thread(embeddings.embed_query, query)
    query_embedding_np = np.array(query_embedding, dtype=np.float32)
    similar_articles = faiss_index.search(query_embedding_np, k=20, threshold=0.75)
    
    # Database query
    with db.conn.cursor() as cur:
        if search_stocks:
            cur.execute("""
                SELECT DISTINCT a.id, a.title, a.content, a.published_date, 
                       si.stock_symbol, si.confidence, si.impact_type
                FROM articles a
                LEFT JOIN stock_impacts si ON a.id = si.article_id
                WHERE si.stock_symbol = ANY(%s)
                ORDER BY a.published_date DESC
                LIMIT 50
            """, (list(set(search_stocks)),))
        else:
            article_hashes = [a['article_id'] for a in similar_articles]
            if article_hashes:
                cur.execute("""
                    SELECT a.id, a.title, a.content, a.published_date, NULL, NULL, NULL
                    FROM articles a
                    WHERE a.article_hash = ANY(%s)
                    ORDER BY a.published_date DESC
                    LIMIT 50
                """, (article_hashes,))
            else:
                cur.execute("""
                    SELECT id, title, content, published_date, NULL, NULL, NULL
                    FROM articles 
                    ORDER BY published_date DESC 
                    LIMIT 20
                """)
        
        results = cur.fetchall()
    
    # Format results
    query_results = []
    for row in results:
        query_results.append({
            'id': row[0],
            'title': row[1],
            'content': row[2][:300] + "..." if len(row[2]) > 300 else row[2],
            'published_date': row[3].isoformat() if row[3] else None,
            'stock_symbol': row[4],
            'confidence': float(row[5]) if row[5] else None,
            'impact_type': row[6]
        })
    
    # Cache results
    await cache.set(cache_key, query_results, ttl_seconds=3600)
    
    state['query_results'] = query_results
    state['messages'] = [f"🔍 Found {len(query_results)} relevant articles"]
    
    print(f"🔍 Query: {len(query_results)} results")
    return state


# ============= MASTER SUPERVISOR =============
MASTER_SUPERVISOR_PROMPT = """You are the Master Supervisor of an Optimized Financial News Intelligence System.

Process flow:
- NEW ARTICLES (batch): ingestion → embedding → deduplication → entities → stocks → storage
- QUERY: query_processor → END

Current state: {state_summary}

Respond with ONLY the next agent name: ingestion, embedding, deduplication, entities, stocks, storage, query_processor, or END"""


@traceable(name="master_supervisor")
async def master_supervisor(state: OptimizedNewsState) -> OptimizedNewsState:
    """Intelligent routing with batch awareness"""
    
    # Determine next step
    if state.get('query'):
        next_agent = "query_processor"
    elif state.get('articles_batch') and not state.get('normalized_articles'):
        next_agent = "ingestion"
    elif state.get('normalized_articles') and not state.get('embeddings_batch'):
        next_agent = "embedding"
    elif state.get('embeddings_batch') and not state.get('duplicate_results'):
        next_agent = "deduplication"
    elif state.get('duplicate_results') and not state.get('entities_batch'):
        next_agent = "entities"
    elif state.get('entities_batch') and not state.get('stocks_batch'):
        next_agent = "stocks"
    elif state.get('stocks_batch'):
        next_agent = "storage"
    else:
        next_agent = "END"
    
    state['next_agent'] = next_agent
    print(f"🎯 Supervisor → {next_agent}")
    
    return state


# ============= LANGGRAPH WORKFLOW =============
def route_supervisor(state: OptimizedNewsState) -> str:
    """Route based on supervisor decision"""
    next_agent = state.get('next_agent', 'END')
    if next_agent == 'END':
        return END
    return next_agent


# Build optimized graph
workflow = StateGraph(OptimizedNewsState)

# Add nodes
workflow.add_node("supervisor", master_supervisor)
workflow.add_node("ingestion", batch_ingestion_agent)
workflow.add_node("embedding", batch_embedding_generation)
workflow.add_node("deduplication", batch_deduplication_agent)
workflow.add_node("entities", parallel_entity_extraction_agent)
workflow.add_node("stocks", batch_stock_impact_agent)
workflow.add_node("storage", batch_storage_agent)
workflow.add_node("query_processor", optimized_query_processor_agent)

# Entry point
workflow.set_entry_point("supervisor")

# Conditional routing
workflow.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "ingestion": "ingestion",
        "embedding": "embedding",
        "deduplication": "deduplication",
        "entities": "entities",
        "stocks": "stocks",
        "storage": "storage",
        "query_processor": "query_processor",
        END: END
    }
)

# Loop back to supervisor
workflow.add_edge("ingestion", "supervisor")
workflow.add_edge("embedding", "supervisor")
workflow.add_edge("deduplication", "supervisor")
workflow.add_edge("entities", "supervisor")
workflow.add_edge("stocks", "supervisor")
workflow.add_edge("storage", END)
workflow.add_edge("query_processor", END)

# Compile
checkpointer = PostgresSaver.from_conn_string(DB_URI)
checkpointer.setup()
app = workflow.compile(checkpointer=checkpointer)

# ============= FASTAPI WITH STREAMING =============
@asynccontextmanager
async def lifespan(api_app: FastAPI) -> AsyncIterator[None]:
    """Initialize and cleanup resources"""
    await cache.initialize()
    yield
    await cache.close()

api = FastAPI(
    title="Optimized Financial News Intelligence API",
    version="2.0",
    lifespan=lifespan
)


class ArticleInput(BaseModel):
    title: str
    content: str
    source: str = "unknown"
    published_date: str = None

class BatchArticlesInput(BaseModel):
    articles: List[ArticleInput]

class QueryInput(BaseModel):
    query: str


@api.post("/ingest")
@traceable(name="api_ingest_single")
async def ingest_article(article: ArticleInput):
    """Process single article"""
    
    initial_state = {
        "articles_batch": [article.dict()],
        "processing_mode": "batch",
        "messages": []
    }
    
    config = {"configurable": {"thread_id": f"single_{datetime.now().timestamp()}"}}
    
    try:
        result = await asyncio.to_thread(app.invoke, initial_state, config)
        
        dup_result = result.get('duplicate_results', [{}])[0]
        entities = result.get('entities_batch', [[]])[0]
        stocks = result.get('stocks_batch', [[]])[0]
        
        return {
            "success": True,
            "is_duplicate": dup_result.get('is_duplicate', False),
            "similarity": dup_result.get('similarity', 0.0),
            "entities": entities,
            "impacted_stocks": stocks,
            "messages": result.get('messages', [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/ingest/batch")
@traceable(name="api_ingest_batch")
async def ingest_articles_batch(batch: BatchArticlesInput):
    """Process multiple articles efficiently"""
    
    initial_state = {
        "articles_batch": [a.dict() for a in batch.articles],
        "processing_mode": "batch",
        "messages": []
    }
    
    config = {"configurable": {"thread_id": f"batch_{datetime.now().timestamp()}"}}
    
    try:
        result = await asyncio.to_thread(app.invoke, initial_state, config)
        
        return {
            "success": True,
            "processed": len(batch.articles),
            "unique": sum(1 for r in result.get('duplicate_results', []) if not r.get('is_duplicate')),
            "duplicates": sum(1 for r in result.get('duplicate_results', []) if r.get('is_duplicate')),
            "messages": result.get('messages', [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/query")
@traceable(name="api_query")
async def query_news(query_input: QueryInput):
    """Query with caching"""
    
    initial_state = {
        "query": query_input.query,
        "messages": []
    }
    
    config = {"configurable": {"thread_id": f"query_{datetime.now().timestamp()}"}}
    
    try:
        result = await asyncio.to_thread(app.invoke, initial_state, config)
        
        return {
            "success": True,
            "query": query_input.query,
            "results": result.get('query_results', []),
            "count": len(result.get('query_results', [])),
            "messages": result.get('messages', [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/query/stream")
@traceable(name="api_query_stream")
async def query_news_stream(query_input: QueryInput):
    """Stream query results progressively"""
    
    async def generate_stream():
        initial_state = {
            "query": query_input.query,
            "messages": []
        }
        
        config = {"configurable": {"thread_id": f"stream_{datetime.now().timestamp()}"}}
        
        try:
            # Use async streaming
            async for event in app.astream(initial_state, config):
                # Stream intermediate results
                yield f"data: {json.dumps(event)}\n\n"
                
                # Final results
                if 'query_results' in event:
                    for result in event['query_results']:
                        yield f"data: {json.dumps({'type': 'result', 'data': result})}\n\n"
            
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    return StreamingResponse(generate_stream(), media_type="text/event-stream")


@api.get("/stats")
async def get_system_stats():
    """System performance metrics"""
    
    with db.conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM articles")
        total_articles = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(DISTINCT duplicate_group_id) FROM articles")
        unique_stories = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM entities")
        total_entities = cur.fetchone()[0]
    
    return {
        "total_articles": total_articles,
        "unique_stories": unique_stories,
        "total_entities": total_entities,
        "faiss_vectors": faiss_index.index.ntotal,
        "faiss_memory_mb": round(faiss_index.get_memory_usage_mb(), 2),
        "faiss_trained": faiss_index.trained,
        "cache_size": len(cache.memory_cache),
    }


# ============= EXAMPLE USAGE =============
async def run_examples():
    """Demonstrate optimized system"""
    
    print("=" * 70)
    print("🚀 OPTIMIZED FINANCIAL NEWS INTELLIGENCE SYSTEM")
    print("=" * 70)
    
    # Example 1: Batch ingestion
    print("\n📦 EXAMPLE 1: Batch Processing (10 articles)")
    print("-" * 70)
    
    articles = []
    for i in range(10):
        articles.append({
            "title": f"HDFC Bank Q{i+1} Results: Profit up 15%",
            "content": f"HDFC Bank reported strong Q{i+1} results with profit growth of 15%. The bank's asset quality improved significantly.",
            "source": "Economic Times",
            "published_date": (datetime.now() - timedelta(days=i)).isoformat()
        })
    
    initial_state = {
        "articles_batch": articles,
        "processing_mode": "batch",
        "messages": []
    }
    
    result = await asyncio.to_thread(
        app.invoke,
        initial_state,
        {"configurable": {"thread_id": "demo_batch"}}
    )
    
    print(f"✅ Processed: {len(articles)} articles")
    print(f"📊 Unique: {sum(1 for r in result['duplicate_results'] if not r['is_duplicate'])}")
    print(f"📊 Duplicates: {sum(1 for r in result['duplicate_results'] if r['is_duplicate'])}")
    print(f"💾 FAISS Memory: {faiss_index.get_memory_usage_mb():.2f} MB")
    
    # Example 2: Query
    print("\n🔍 EXAMPLE 2: Querying News")
    print("-" * 70)
    
    query_state = {
        "query": "HDFC Bank news",
        "messages": []
    }
    
    query_result = await asyncio.to_thread(
        app.invoke,
        query_state,
        {"configurable": {"thread_id": "demo_query"}}
    )
    
    print(f"✅ Found: {len(query_result['query_results'])} articles")
    for article in query_result['query_results'][:3]:
        print(f"  - {article['title']}")
    
    print("\n" + "=" * 70)
    print("✅ All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    # Run examples first
    asyncio.run(run_examples())
    
    # Start FastAPI server
    print("\n🌐 Starting FastAPI server on http://0.0.0.0:8000")
    print("📊 Stats: http://localhost:8000/stats")
    print("📝 Docs: http://localhost:8000/docs")
    
    uvicorn.run(api, host="0.0.0.0", port=8000, log_level="info")
