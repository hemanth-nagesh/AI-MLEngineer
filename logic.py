# optimized_financial_news_system.py
import os
import asyncio
from typing import TypedDict, List, Dict, Annotated, Literal, AsyncIterator, Optional, Union
from datetime import datetime, timedelta
import hashlib
import json
import pickle
import logging
import traceback
import time
from functools import wraps

# Core imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from langsmith import traceable
import spacy
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import jinja2
from pydantic import BaseModel, validator
import psycopg2
from psycopg2.extras import execute_values, Json
from psycopg2 import pool
from contextlib import asynccontextmanager
import redis.asyncio as aioredis
from cachetools import LRUCache
from concurrent.futures import ThreadPoolExecutor
import uvicorn
from dotenv import load_dotenv

load_dotenv()

# ============= LOGGING CONFIGURATION =============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('financial_news_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============= ERROR HANDLING UTILITIES =============
class SystemError(Exception):
    """Base exception for system errors"""
    pass

class DatabaseError(SystemError):
    """Database related errors"""
    pass

class CacheError(SystemError):
    """Cache related errors"""
    pass

class EmbeddingError(SystemError):
    """Embedding generation errors"""
    pass

def retry_on_failure(max_retries=3, delay=1, backoff=2, exceptions=(Exception,)):
    """Retry decorator for handling transient failures"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            retries = 0
            current_delay = delay
            last_exception = None
            
            while retries < max_retries:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    retries += 1
                    if retries >= max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries} retries: {str(e)}")
                        raise
                    
                    logger.warning(f"Function {func.__name__} failed (attempt {retries}/{max_retries}): {str(e)}. Retrying in {current_delay}s...")
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
            
            raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            retries = 0
            current_delay = delay
            last_exception = None
            
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    retries += 1
                    if retries >= max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries} retries: {str(e)}")
                        raise
                    
                    logger.warning(f"Function {func.__name__} failed (attempt {retries}/{max_retries}): {str(e)}. Retrying in {current_delay}s...")
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            raise last_exception
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

def handle_exceptions(func):
    """Decorator to handle exceptions and log them properly"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Exception in {func.__name__}: {str(e)}")
            logger.debug(traceback.format_exc())
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Exception in {func.__name__}: {str(e)}")
            logger.debug(traceback.format_exc())
            raise
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

# ============= CONFIGURATION =============
class Config:
    """Centralized configuration management"""
    
    def __init__(self):
        self.batch_size = int(os.getenv("BATCH_SIZE", "32"))
        self.max_concurrent_agents = int(os.getenv("MAX_CONCURRENT_AGENTS", "10"))
        self.faiss_nlist = int(os.getenv("FAISS_NLIST", "100"))
        self.faiss_m = int(os.getenv("FAISS_M", "8"))
        self.faiss_threshold = float(os.getenv("FAISS_THRESHOLD", "0.87"))
        self.cache_ttl_hours = int(os.getenv("CACHE_TTL_HOURS", "24"))
        self.checkpoint_ttl_days = int(os.getenv("CHECKPOINT_TTL_DAYS", "30"))
        self.enable_streaming = os.getenv("ENABLE_STREAMING", "true").lower() == "true"
        self.connection_pool_size = int(os.getenv("CONNECTION_POOL_SIZE", "20"))
        
        # Database configuration
        self.db_host = os.getenv("DB_HOST", "localhost")
        self.db_port = int(os.getenv("DB_PORT", "5432"))
        self.db_name = os.getenv("DB_NAME", "financial_news")
        self.db_user = os.getenv("DB_USER", "postgres")
        self.db_password = os.getenv("DB_PASSWORD", "12345")
        
        # Redis configuration
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", "6379"))
        self.redis_db = int(os.getenv("REDIS_DB", "0"))
        
        # API configuration
        self.api_host = os.getenv("API_HOST", "0.0.0.0")
        self.api_port = int(os.getenv("API_PORT", "8000"))
        
        # Security
        self.secret_key = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
        
        # External services
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        
        self.validate()
    
    def validate(self):
        """Validate configuration parameters"""
        if self.batch_size <= 0:
            raise ValueError("BATCH_SIZE must be positive")
        if self.max_concurrent_agents <= 0:
            raise ValueError("MAX_CONCURRENT_AGENTS must be positive")
        if self.faiss_threshold < 0 or self.faiss_threshold > 1:
            raise ValueError("FAISS_THRESHOLD must be between 0 and 1")
        if self.connection_pool_size <= 0:
            raise ValueError("CONNECTION_POOL_SIZE must be positive")
    
    @property
    def db_uri(self):
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
    
    @property
    def redis_uri(self):
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

config = Config()

# Legacy compatibility
OPTIMIZED_CONFIG = {
    "batch_size": config.batch_size,
    "max_concurrent_agents": config.max_concurrent_agents,
    "faiss_nlist": config.faiss_nlist,
    "faiss_m": config.faiss_m,
    "faiss_threshold": config.faiss_threshold,
    "cache_ttl_hours": config.cache_ttl_hours,
    "checkpoint_ttl_days": config.checkpoint_ttl_days,
    "enable_streaming": config.enable_streaming,
    "connection_pool_size": config.connection_pool_size,
}

# Environment setup
# os.environ["LANGCHAIN_TRACING_V2"]
# os.environ["LANGCHAIN_PROJECT"]
# os.environ["LANGCHAIN_API_KEY"]
api_key = config.google_api_key

from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = config.db_uri



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
        self.connection_pool = None
        self.initialize_connection_pool()
        self.setup_optimized_database()
        
    @handle_exceptions
    @retry_on_failure(max_retries=3, delay=1)
    def initialize_connection_pool(self):
        """Initialize PostgreSQL connection pool"""
        try:
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=config.connection_pool_size,
                host=config.db_host,
                database=config.db_name,
                user=config.db_user,
                password=config.db_password,
                port=config.db_port,
                options="-c statement_timeout=30000"
            )
            logger.info("✅ Database connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database connection pool: {str(e)}")
            raise DatabaseError(f"Connection pool initialization failed: {str(e)}")
    
    @handle_exceptions
    def get_connection(self):
        """Get a connection from the pool"""
        try:
            return self.connection_pool.getconn()
        except Exception as e:
            logger.error(f"Failed to get database connection: {str(e)}")
            raise DatabaseError(f"Failed to get connection: {str(e)}")
    
    @handle_exceptions
    def release_connection(self, conn):
        """Release a connection back to the pool"""
        try:
            self.connection_pool.putconn(conn)
        except Exception as e:
            logger.error(f"Failed to release database connection: {str(e)}")
    
    @handle_exceptions
    def setup_optimized_database(self):
        """Initialize optimized PostgreSQL schema"""
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor() as cur:
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
                
                # Clean old checkpoints
                try:
                    cur.execute(f"""
                        DELETE FROM checkpoints
                        WHERE created_at < NOW() - INTERVAL '{config.checkpoint_ttl_days} days';
                    """)
                except psycopg2.errors.UndefinedTable:
                    # LangGraph checkpoint table not created yet – ignore
                    conn.rollback()
                
            conn.commit()
            logger.info("✅ Optimized database initialized")
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database setup failed: {str(e)}")
            raise DatabaseError(f"Database setup failed: {str(e)}")
        finally:
            if conn:
                self.release_connection(conn)
    
    @handle_exceptions
    @retry_on_failure(max_retries=3, delay=0.5)
    def batch_insert_articles(self, articles_data: List[tuple]):
        """Batch insert for better performance"""
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor() as cur:
                execute_values(cur, """
                    INSERT INTO articles (title, content, source, published_date,
                                         article_hash, duplicate_group_id)
                    VALUES %s
                    RETURNING id
                """, articles_data)
                article_ids = [row[0] for row in cur.fetchall()]
            conn.commit()
            return article_ids
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Batch insert failed: {str(e)}")
            raise DatabaseError(f"Batch insert failed: {str(e)}")
        finally:
            if conn:
                self.release_connection(conn)
    
    @handle_exceptions
    @retry_on_failure(max_retries=2, delay=1)
    def refresh_materialized_views(self):
        """Async refresh for materialized views"""
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor() as cur:
                cur.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY stock_news_summary;")
            conn.commit()
            logger.info("Materialized views refreshed")
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Failed to refresh materialized views: {str(e)}")
            raise DatabaseError(f"Failed to refresh materialized views: {str(e)}")
        finally:
            if conn:
                self.release_connection(conn)
    
    def close_all_connections(self):
        """Close all connections in the pool"""
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("All database connections closed")

# Initialize database manager
db = OptimizedDatabaseManager()


# ============= OPTIMIZED FAISS WITH PRODUCT QUANTIZATION =============
class OptimizedFAISSIndex:
    """Memory-efficient FAISS with IVF-PQ (27× memory reduction)"""
    
    def __init__(self, dimension=768):
        self.dimension = dimension
        self.nlist = config.faiss_nlist
        self.m = config.faiss_m
        
        try:
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
            logger.info(f"🔧 Initialized IVF-PQ index: nlist={self.nlist}, M={self.m}")
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {str(e)}")
            raise SystemError(f"FAISS initialization failed: {str(e)}")
    
    @handle_exceptions
    def add_to_training_buffer(self, embeddings: np.ndarray):
        """Collect embeddings for training"""
        self.training_buffer.append(embeddings)
    
    @handle_exceptions
    def train_index(self):
        """Train IVF-PQ index on collected data"""
        if self.trained:
            return
        
        if len(self.training_buffer) == 0:
            logger.warning("⚠️  No training data available")
            return
        
        try:
            training_data = np.vstack(self.training_buffer).astype('float32')
            
            if len(training_data) < self.nlist:
                logger.warning(f"⚠️  Need at least {self.nlist} vectors for training, got {len(training_data)}")
                return
            
            logger.info(f"🔧 Training IVF-PQ index with {len(training_data)} vectors...")
            faiss.normalize_L2(training_data)
            self.index.train(training_data)
            self.trained = True
            logger.info("✅ Index trained successfully")
            
            # Add training data to index
            self.index.add(training_data)
            self.article_ids.extend([f"train_{i}" for i in range(len(training_data))])
        except Exception as e:
            logger.error(f"Index training failed: {str(e)}")
            raise SystemError(f"Index training failed: {str(e)}")
    
    @handle_exceptions
    def add(self, embeddings: np.ndarray, article_ids: List[str]):
        """Add embeddings to index"""
        try:
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
        except Exception as e:
            logger.error(f"Failed to add embeddings to index: {str(e)}")
            raise SystemError(f"Failed to add embeddings: {str(e)}")
    
    @handle_exceptions
    def search(self, query_embedding: np.ndarray, k=5, threshold=None):
        """Search with quantized index"""
        if threshold is None:
            threshold = config.faiss_threshold
            
        if not self.trained or self.index.ntotal == 0:
            return []
        
        try:
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
        except Exception as e:
            logger.error(f"FAISS search failed: {str(e)}")
            raise SystemError(f"FAISS search failed: {str(e)}")
    
    @handle_exceptions
    def get_memory_usage_mb(self):
        """Calculate memory usage in MB"""
        try:
            if self.index.ntotal == 0:
                return 0
            bytes_per_vector = self.m  # M bytes per vector with PQ
            return (self.index.ntotal * bytes_per_vector) / (1024**2)
        except Exception as e:
            logger.error(f"Failed to calculate memory usage: {str(e)}")
            return 0

faiss_index = OptimizedFAISSIndex()


# ============= HYBRID CACHE (REDIS + MEMORY) =============
class HybridCache:
    """Two-tier caching: L1 (Memory) + L2 (Redis)"""
    
    def __init__(self):
        self.memory_cache = LRUCache(maxsize=1000)
        self.redis_client = None
        self.enabled = True
        
    @handle_exceptions
    @retry_on_failure(max_retries=3, delay=1)
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = await aioredis.from_url(
                config.redis_uri,
                encoding="utf-8",
                decode_responses=False,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
            await self.redis_client.ping()
            logger.info("✅ Redis cache initialized")
        except Exception as e:
            logger.warning(f"⚠️  Redis unavailable, using memory-only cache: {e}")
            self.enabled = False
            raise CacheError(f"Redis initialization failed: {str(e)}")
    
    @handle_exceptions
    def _generate_key(self, prefix: str, data: str) -> str:
        """Generate cache key"""
        return f"{prefix}:{hashlib.md5(data.encode()).hexdigest()}"
    
    @handle_exceptions
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
                logger.error(f"Redis get error: {e}")
                raise CacheError(f"Redis get failed: {str(e)}")
        
        return None
    
    @handle_exceptions
    async def set(self, key: str, value, ttl_seconds=None):
        """Set in both L1 and L2"""
        if ttl_seconds is None:
            ttl_seconds = config.cache_ttl_hours * 3600
            
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
                logger.error(f"Redis set error: {e}")
                raise CacheError(f"Redis set failed: {str(e)}")
    
    @handle_exceptions
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Redis connection closed")

cache = HybridCache()


# ============= LIGHTWEIGHT NER =============
class LightweightNER:
    """Memory-efficient NER with on-demand model loading"""
    
    def __init__(self):
        self.nlp_small = None
        self.nlp_large = None  # Load on demand
        self.load_small_model()
    
    @handle_exceptions
    def load_small_model(self):
        """Load small spaCy model"""
        try:
            self.nlp_small = spacy.load("en_core_web_sm")
            logger.info("✅ Loaded lightweight spaCy model (12MB)")
        except Exception as e:
            logger.error(f"Failed to load spaCy small model: {str(e)}")
            raise SystemError(f"Failed to load NER model: {str(e)}")
    
    @handle_exceptions
    def extract_entities_fast(self, text: str) -> List[Dict]:
        """Fast extraction with small model"""
        if not self.nlp_small:
            self.load_small_model()
            
        if not text or not text.strip():
            return []
            
        try:
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
                    'name': ent.text.strip(),
                    'type': entity_types.get(ent.label_, 'other'),
                    'confidence': 0.85
                })
            
            return entities
        except Exception as e:
            logger.error(f"Entity extraction failed: {str(e)}")
            return []
    
    @handle_exceptions
    def extract_entities_precise(self, text: str) -> List[Dict]:
        """High-precision extraction (load large model on demand)"""
        if not text or not text.strip():
            return []
            
        try:
            if self.nlp_large is None:
                logger.info("🔄 Loading large spaCy model...")
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
                    'name': ent.text.strip(),
                    'type': entity_types.get(ent.label_, 'other'),
                    'confidence': 0.95
                })
            
            return entities
        except Exception as e:
            logger.error(f"Precise entity extraction failed: {str(e)}")
            # Fallback to fast extraction
            return self.extract_entities_fast(text)

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
@handle_exceptions
async def batch_ingestion_agent(state: OptimizedNewsState) -> OptimizedNewsState:
    """Process multiple articles in batch"""
    
    try:
        # Update agent status
        manager.update_agent_status("ingestion", "running")
        await manager.broadcast(json.dumps({
            "type": "agent_status",
            "agent": "ingestion",
            "status": "running",
            "message": "Starting article ingestion..."
        }))
        
        if state.get('processing_mode') == 'single':
            # Single article mode (backward compatible)
            raw = state['raw_article']
            articles = [raw]
        else:
            # Batch mode
            articles = state['articles_batch']
        
        # Input validation
        if not articles:
            raise ValueError("No articles provided for ingestion")
        
        normalized = []
        hashes = []
        
        for i, article in enumerate(articles):
            # Validate article structure
            if not isinstance(article, dict):
                logger.warning(f"Skipping invalid article at index {i}: not a dictionary")
                continue
                
            if 'title' not in article or 'content' not in article:
                logger.warning(f"Skipping article at index {i}: missing title or content")
                continue
                
            if not article['title'] or not article['content']:
                logger.warning(f"Skipping article at index {i}: empty title or content")
                continue
            
            # Sanitize input
            title = str(article['title']).strip()
            content = str(article['content']).strip()
            
            if len(title) > 1000 or len(content) > 100000:
                logger.warning(f"Article at index {i} exceeds length limits, truncating")
                title = title[:1000]
                content = content[:100000]
            
            content_hash = hashlib.sha256(
                f"{title}{content}".encode()
            ).hexdigest()
            
            normalized.append({
                'title': title,
                'content': content,
                'source': str(article.get('source', 'unknown')).strip()[:100],
                'published_date': article.get('published_date', datetime.now().isoformat()),
            })
            hashes.append(content_hash)
            
            # Broadcast progress
            if i % 5 == 0:  # Update every 5 articles
                await manager.broadcast(json.dumps({
                    "type": "agent_progress",
                    "agent": "ingestion",
                    "progress": round((i + 1) / len(articles) * 100),
                    "message": f"Processed {i + 1}/{len(articles)} articles"
                }))
        
        if not normalized:
            raise ValueError("No valid articles found after validation")
        
        state['normalized_articles'] = normalized
        state['article_hashes'] = hashes
        state['messages'] = [f"✅ Ingested {len(normalized)} articles"]
        
        # Update final status
        manager.update_agent_status("ingestion", "completed", len(normalized))
        await manager.broadcast(json.dumps({
            "type": "agent_status",
            "agent": "ingestion",
            "status": "completed",
            "message": f"Successfully ingested {len(normalized)} articles"
        }))
        
        logger.info(f"📥 Batch ingested: {len(normalized)} articles")
        return state
    except Exception as e:
        manager.update_agent_status("ingestion", "error")
        await manager.broadcast(json.dumps({
            "type": "agent_status",
            "agent": "ingestion",
            "status": "error",
            "message": f"Ingestion failed: {str(e)}"
        }))
        logger.error(f"Batch ingestion failed: {str(e)}")
        raise


@traceable(name="batch_embedding_generation")
@handle_exceptions
@retry_on_failure(max_retries=3, delay=1, exceptions=(EmbeddingError,))
async def batch_embedding_generation(state: OptimizedNewsState) -> OptimizedNewsState:
    """Generate embeddings in batches (more efficient)"""
    
    try:
        articles = state['normalized_articles']
        if not articles:
            raise ValueError("No articles provided for embedding generation")
            
        texts = [f"{a['title']} {a['content']}" for a in articles]
        
        # Check cache first
        all_embeddings = []
        texts_to_compute = []
        text_indices = []
        
        for idx, text in enumerate(texts):
            if not text or not text.strip():
                logger.warning(f"Empty text at index {idx}, skipping")
                all_embeddings.append([])
                continue
                
            cache_key = cache._generate_key("emb", text)
            try:
                cached = await cache.get(cache_key)
                
                if cached:
                    all_embeddings.append(cached)
                else:
                    all_embeddings.append(None)
                    texts_to_compute.append(text)
                    text_indices.append(idx)
            except Exception as e:
                logger.warning(f"Cache get failed for text {idx}: {str(e)}")
                all_embeddings.append(None)
                texts_to_compute.append(text)
                text_indices.append(idx)
        
        # Compute uncached embeddings in batches
        if texts_to_compute:
            batch_size = config.batch_size
            computed_embeddings = []
            
            for i in range(0, len(texts_to_compute), batch_size):
                batch = texts_to_compute[i:i+batch_size]
                try:
                    # Async batch embedding with retry
                    batch_embeddings = await asyncio.to_thread(
                        embeddings.embed_documents,
                        batch
                    )
                    computed_embeddings.extend(batch_embeddings)
                except Exception as e:
                    logger.error(f"Embedding generation failed for batch {i//batch_size}: {str(e)}")
                    # Add empty embeddings for failed batch
                    computed_embeddings.extend([[0.0] * 768] * len(batch))
            
            # Cache computed embeddings
            for text, emb in zip(texts_to_compute, computed_embeddings):
                try:
                    cache_key = cache._generate_key("emb", text)
                    await cache.set(cache_key, emb)
                except Exception as e:
                    logger.warning(f"Cache set failed for embedding: {str(e)}")
            
            # Fill in computed embeddings
            for idx, emb in zip(text_indices, computed_embeddings):
                all_embeddings[idx] = emb
        
        # Validate embeddings
        valid_embeddings = []
        for i, emb in enumerate(all_embeddings):
            if emb and len(emb) > 0:
                valid_embeddings.append(emb)
            else:
                logger.warning(f"Invalid or empty embedding at index {i}")
                # Use zero embedding as fallback
                valid_embeddings.append([0.0] * 768)
        
        state['embeddings_batch'] = valid_embeddings
        state['messages'] = [f"🔢 Generated {len(valid_embeddings)} embeddings"]
        
        logger.info(f"🔢 Embeddings: {len(valid_embeddings)} vectors")
        return state
    except Exception as e:
        logger.error(f"Batch embedding generation failed: {str(e)}")
        raise EmbeddingError(f"Embedding generation failed: {str(e)}")


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
@handle_exceptions
@retry_on_failure(max_retries=3, delay=1, exceptions=(DatabaseError,))
async def batch_storage_agent(state: OptimizedNewsState) -> OptimizedNewsState:
    """Batch insert into PostgreSQL"""
    
    conn = None
    try:
        articles = state['normalized_articles']
        hashes = state['article_hashes']
        duplicate_results = state['duplicate_results']
        entities_batch = state['entities_batch']
        stocks_batch = state['stocks_batch']
        
        # Input validation
        if not articles or not hashes or not duplicate_results:
            raise ValueError("Missing required data for storage")
            
        if len(articles) != len(hashes) or len(articles) != len(duplicate_results):
            raise ValueError("Data length mismatch in storage agent")
        
        # Prepare batch insert data
        articles_data = []
        for article, hash_val, dup_result in zip(articles, hashes, duplicate_results):
            # Validate data
            if not article or not hash_val or dup_result is None:
                logger.warning("Skipping invalid article data in batch")
                continue
                
            articles_data.append((
                article['title'][:1000],  # Limit title length
                article['content'][:100000],  # Limit content length
                article.get('source', 'unknown')[:100],  # Limit source length
                article.get('published_date'),
                hash_val,
                dup_result.get('duplicate_group_id', hash_val)
            ))
        
        if not articles_data:
            raise ValueError("No valid articles to store")
        
        # Batch insert articles
        article_ids = db.batch_insert_articles(articles_data)
        
        # Get database connection for entities and stocks
        conn = db.get_connection()
        
        try:
            with conn.cursor() as cur:
                for article_id, entities, stocks in zip(article_ids, entities_batch, stocks_batch):
                    # Insert entities
                    if entities:
                        for entity in entities:
                            if not isinstance(entity, dict) or 'name' not in entity:
                                continue
                                
                            try:
                                cur.execute("""
                                    INSERT INTO entities (name, type, normalized_name)
                                    VALUES (%s, %s, %s)
                                    ON CONFLICT (name, type) DO UPDATE SET normalized_name = EXCLUDED.normalized_name
                                    RETURNING id
                                """, (
                                    entity['name'][:200],  # Limit name length
                                    entity.get('type', 'other')[:50],  # Limit type length
                                    entity.get('normalized_name', entity['name'])[:200]
                                ))
                                
                                entity_id = cur.fetchone()[0]
                                
                                cur.execute("""
                                    INSERT INTO article_entities (article_id, entity_id, confidence)
                                    VALUES (%s, %s, %s)
                                    ON CONFLICT DO NOTHING
                                """, (article_id, entity_id, min(float(entity.get('confidence', 0.5)), 1.0)))
                            except Exception as e:
                                logger.warning(f"Failed to insert entity {entity.get('name', 'unknown')}: {str(e)}")
                    
                    # Insert stock impacts
                    if stocks:
                        for stock in stocks:
                            if not isinstance(stock, dict) or 'symbol' not in stock:
                                continue
                                
                            try:
                                cur.execute("""
                                    INSERT INTO stock_impacts (article_id, stock_symbol, confidence, impact_type)
                                    VALUES (%s, %s, %s, %s)
                                """, (
                                    article_id,
                                    stock['symbol'][:10],  # Limit symbol length
                                    min(float(stock.get('confidence', 0.5)), 1.0),
                                    stock.get('impact_type', 'unknown')[:50]  # Limit impact type length
                                ))
                            except Exception as e:
                                logger.warning(f"Failed to insert stock impact {stock.get('symbol', 'unknown')}: {str(e)}")
                
                conn.commit()
                
        except Exception as e:
            if conn:
                conn.rollback()
            raise DatabaseError(f"Failed to store entities and stocks: {str(e)}")
        finally:
            if conn:
                db.release_connection(conn)
        
        state['messages'] = [f"💾 Stored {len(article_ids)} articles in database"]
        
        logger.info(f"💾 Database: {len(article_ids)} articles stored")
        
        # Refresh materialized views asynchronously
        try:
            asyncio.create_task(asyncio.to_thread(db.refresh_materialized_views))
        except Exception as e:
            logger.warning(f"Failed to schedule materialized view refresh: {str(e)}")
        
        return state
    except Exception as e:
        if conn:
            db.release_connection(conn)
        logger.error(f"Batch storage failed: {str(e)}")
        raise DatabaseError(f"Batch storage failed: {str(e)}")


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

# ============= REAL-TIME MONITORING =============
class ConnectionManager:
    """WebSocket connection manager for real-time updates"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.agent_status = {
            "ingestion": {"status": "idle", "processed": 0},
            "embedding": {"status": "idle", "processed": 0},
            "deduplication": {"status": "idle", "processed": 0},
            "entities": {"status": "idle", "processed": 0},
            "stocks": {"status": "idle", "processed": 0},
            "storage": {"status": "idle", "processed": 0},
            "query_processor": {"status": "idle", "processed": 0}
        }
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {str(e)}")
    
    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.warning(f"Failed to send to WebSocket client: {str(e)}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)
    
    def update_agent_status(self, agent_name: str, status: str, processed: int = None):
        """Update agent status for monitoring"""
        if agent_name in self.agent_status:
            self.agent_status[agent_name]["status"] = status
            if processed is not None:
                self.agent_status[agent_name]["processed"] = processed

manager = ConnectionManager()

# Create templates directory and dashboard template
templates = Jinja2Templates(directory="templates")

api = FastAPI(
    title="Optimized Financial News Intelligence API",
    version="2.0",
    lifespan=lifespan
)

# Mount static files for dashboard
api.mount("/static", StaticFiles(directory="static"), name="static")


class ArticleInput(BaseModel):
    title: str
    content: str
    source: str = "unknown"
    published_date: str = None
    
    @validator('title')
    def validate_title(cls, v):
        if not v or not v.strip():
            raise ValueError('Title cannot be empty')
        if len(v) > 1000:
            raise ValueError('Title cannot exceed 1000 characters')
        return v.strip()
    
    @validator('content')
    def validate_content(cls, v):
        if not v or not v.strip():
            raise ValueError('Content cannot be empty')
        if len(v) > 100000:
            raise ValueError('Content cannot exceed 100000 characters')
        return v.strip()
    
    @validator('source')
    def validate_source(cls, v):
        if v and len(v) > 100:
            raise ValueError('Source cannot exceed 100 characters')
        return v.strip() if v else "unknown"

class BatchArticlesInput(BaseModel):
    articles: List[ArticleInput]
    
    @validator('articles')
    def validate_articles(cls, v):
        if not v:
            raise ValueError('Articles list cannot be empty')
        if len(v) > 100:
            raise ValueError('Cannot process more than 100 articles at once')
        return v

class QueryInput(BaseModel):
    query: str
    
    @validator('query')
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError('Query cannot be empty')
        if len(v) > 1000:
            raise ValueError('Query cannot exceed 1000 characters')
        return v.strip()


@api.post("/ingest")
@traceable(name="api_ingest_single")
@handle_exceptions
async def ingest_article(article: ArticleInput):
    """Process single article"""
    
    try:
        # Input validation is handled by Pydantic validators
        initial_state = {
            "articles_batch": [article.dict()],
            "processing_mode": "batch",
            "messages": []
        }
        
        config = {"configurable": {"thread_id": f"single_{datetime.now().timestamp()}"}}
        
        result = await asyncio.to_thread(app.invoke, initial_state, config)
        
        dup_result = result.get('duplicate_results', [{}])[0]
        entities = result.get('entities_batch', [[]])[0]
        stocks = result.get('stocks_batch', [[]])[0]
        
        logger.info(f"Successfully processed single article: {article.title[:50]}...")
        
        return {
            "success": True,
            "is_duplicate": dup_result.get('is_duplicate', False),
            "similarity": dup_result.get('similarity', 0.0),
            "entities": entities,
            "impacted_stocks": stocks,
            "messages": result.get('messages', [])
        }
    except ValueError as e:
        # Input validation errors
        logger.warning(f"Invalid input for single article: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to process single article: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@api.post("/ingest/batch")
@traceable(name="api_ingest_batch")
@handle_exceptions
async def ingest_articles_batch(batch: BatchArticlesInput):
    """Process multiple articles efficiently"""
    
    try:
        # Input validation is handled by Pydantic validators
        initial_state = {
            "articles_batch": [a.dict() for a in batch.articles],
            "processing_mode": "batch",
            "messages": []
        }
        
        config = {"configurable": {"thread_id": f"batch_{datetime.now().timestamp()}"}}
        
        result = await asyncio.to_thread(app.invoke, initial_state, config)
        
        unique_count = sum(1 for r in result.get('duplicate_results', []) if not r.get('is_duplicate'))
        duplicate_count = sum(1 for r in result.get('duplicate_results', []) if r.get('is_duplicate'))
        
        logger.info(f"Successfully processed batch: {len(batch.articles)} articles, {unique_count} unique, {duplicate_count} duplicates")
        
        return {
            "success": True,
            "processed": len(batch.articles),
            "unique": unique_count,
            "duplicates": duplicate_count,
            "messages": result.get('messages', [])
        }
    except ValueError as e:
        # Input validation errors
        logger.warning(f"Invalid input for batch articles: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to process batch articles: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@api.post("/query")
@traceable(name="api_query")
@handle_exceptions
async def query_news(query_input: QueryInput):
    """Query with caching"""
    
    try:
        # Input validation is handled by Pydantic validators
        initial_state = {
            "query": query_input.query,
            "messages": []
        }
        
        config = {"configurable": {"thread_id": f"query_{datetime.now().timestamp()}"}}
        
        result = await asyncio.to_thread(app.invoke, initial_state, config)
        query_results = result.get('query_results', [])
        
        logger.info(f"Query processed: '{query_input.query[:50]}...', found {len(query_results)} results")
        
        return {
            "success": True,
            "query": query_input.query,
            "results": query_results,
            "count": len(query_results),
            "messages": result.get('messages', [])
        }
    except ValueError as e:
        # Input validation errors
        logger.warning(f"Invalid query input: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to process query: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@api.post("/query/stream")
@traceable(name="api_query_stream")
@handle_exceptions
async def query_news_stream(query_input: QueryInput):
    """Stream query results progressively"""
    
    async def generate_stream():
        try:
            # Input validation is handled by Pydantic validators
            initial_state = {
                "query": query_input.query,
                "messages": []
            }
            
            config = {"configurable": {"thread_id": f"stream_{datetime.now().timestamp()}"}}
            
            # Use async streaming
            async for event in app.astream(initial_state, config):
                # Stream intermediate results
                yield f"data: {json.dumps(event)}\n\n"
                
                # Final results
                if 'query_results' in event:
                    for result in event['query_results']:
                        yield f"data: {json.dumps({'type': 'result', 'data': result})}\n\n"
            
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
        except ValueError as e:
            # Input validation errors
            logger.warning(f"Invalid stream query input: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        except Exception as e:
            logger.error(f"Failed to stream query results: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'error': 'Internal server error'})}\n\n"
    
    return StreamingResponse(generate_stream(), media_type="text/event-stream")


@api.get("/health")
@handle_exceptions
async def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0",
        "checks": {}
    }
    
    # Check database
    conn = None
    try:
        conn = db.get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            health_status["checks"]["database"] = "healthy"
    except Exception as e:
        logger.warning(f"Database health check failed: {str(e)}")
        health_status["checks"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
    finally:
        if conn:
            db.release_connection(conn)
    
    # Check cache
    try:
        test_key = "health_check"
        await cache.set(test_key, "ok", ttl_seconds=10)
        cached = await cache.get(test_key)
        if cached == "ok":
            health_status["checks"]["cache"] = "healthy"
        else:
            health_status["checks"]["cache"] = "unhealthy: cache test failed"
            health_status["status"] = "unhealthy"
    except Exception as e:
        logger.warning(f"Cache health check failed: {str(e)}")
        health_status["checks"]["cache"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
    
    # Check FAISS index
    try:
        if faiss_index.trained and faiss_index.index.ntotal >= 0:
            health_status["checks"]["faiss"] = "healthy"
        else:
            health_status["checks"]["faiss"] = "unhealthy: index not trained"
            health_status["status"] = "degraded"
    except Exception as e:
        logger.warning(f"FAISS health check failed: {str(e)}")
        health_status["checks"]["faiss"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
    
    # Check NER
    try:
        test_text = "Test health check"
        entities = ner.extract_entities_fast(test_text)
        health_status["checks"]["ner"] = "healthy"
    except Exception as e:
        logger.warning(f"NER health check failed: {str(e)}")
        health_status["checks"]["ner"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
    
    # Return appropriate status code
    status_code = 200 if health_status["status"] == "healthy" else 503
    
    return JSONResponse(
        content=health_status,
        status_code=status_code
    )


@api.websocket("/ws")
@handle_exceptions
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time monitoring"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and send periodic updates
            await asyncio.sleep(1)
            
            # Send current status
            await manager.send_personal_message(
                json.dumps({
                    "type": "status_update",
                    "timestamp": datetime.now().isoformat(),
                    "agents": manager.agent_status
                }),
                websocket
            )
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket client disconnected")


@api.get("/dashboard")
@handle_exceptions
async def dashboard():
    """Serve the monitoring dashboard"""
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Financial News Intelligence System - Dashboard</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
            color: white;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        .header p {
            margin: 10px 0 0 0;
            opacity: 0.8;
        }
        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.15);
        }
        .card h3 {
            margin-top: 0;
            color: #333;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .status {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: bold;
            text-transform: uppercase;
        }
        .status.idle {
            background-color: #f8f9fa;
            color: #6c757d;
        }
        .status.running {
            background-color: #fff3cd;
            color: #856404;
        }
        .status.completed {
            background-color: #d4edda;
            color: #155724;
        }
        .status.error {
            background-color: #f8d7da;
            color: #721c24;
        }
        .metrics {
            margin-top: 15px;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }
        .metric:last-child {
            border-bottom: none;
        }
        .metric-value {
            font-weight: bold;
            color: #333;
        }
        .progress-bar {
            width: 100%;
            height: 8px;
            background-color: #e9ecef;
            border-radius: 4px;
            margin-top: 10px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #28a745, #20c997);
            transition: width 0.3s ease;
        }
        .connection-status {
            position: fixed;
            top: 20px;
            right: 20px;
            background: white;
            padding: 10px 15px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            display: flex;
            align-items: center;
        }
        .connection-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .connected {
            background-color: #28a745;
        }
        .disconnected {
            background-color: #dc3545;
        }
        @media (max-width: 768px) {
            .dashboard {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📈 Financial News Intelligence System</h1>
            <p>Real-time Processing Dashboard</p>
        </div>
        
        <div class="connection-status">
            <div class="connection-indicator" id="connectionIndicator"></div>
            <span id="connectionText">Connecting...</span>
        </div>
        
        <div class="dashboard" id="dashboard">
            <!-- Agent cards will be populated by JavaScript -->
        </div>
    </div>

    <script>
        const ws = new WebSocket('ws://localhost:8000/ws');
        const connectionIndicator = document.getElementById('connectionIndicator');
        const connectionText = document.getElementById('connectionText');
        const dashboard = document.getElementById('dashboard');
        
        ws.onopen = function() {
            connectionIndicator.className = 'connection-indicator connected';
            connectionText.textContent = 'Connected';
        };
        
        ws.onclose = function() {
            connectionIndicator.className = 'connection-indicator disconnected';
            connectionText.textContent = 'Disconnected';
        };
        
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            
            if (data.type === 'status_update') {
                updateDashboard(data.agents);
            } else if (data.type === 'agent_status') {
                updateAgentCard(data.agent, data);
            } else if (data.type === 'agent_progress') {
                updateAgentProgress(data.agent, data.progress, data.message);
            }
        };
        
        function updateDashboard(agents) {
            for (const [agentName, agentData] of Object.entries(agents)) {
                updateAgentCard(agentName, agentData);
            }
        }
        
        function updateAgentCard(agentName, data) {
            let card = document.getElementById(`card-${agentName}`);
            
            if (!card) {
                card = createAgentCard(agentName, data);
                dashboard.appendChild(card);
            } else {
                updateCardStatus(card, data);
            }
        }
        
        function createAgentCard(agentName, data) {
            const card = document.createElement('div');
            card.className = 'card';
            card.id = `card-${agentName}`;
            
            const displayName = agentName.replace('_', ' ').replace(/\\b\\w/g, ' ');
            
            card.innerHTML = `
                <h3>
                    ${displayName}
                    <span class="status ${data.status || 'idle'}" id="status-${agentName}">${data.status || 'idle'}</span>
                </h3>
                <div class="metrics">
                    <div class="metric">
                        <span>Status:</span>
                        <span class="metric-value" id="status-text-${agentName}">${data.status || 'idle'}</span>
                    </div>
                    <div class="metric">
                        <span>Processed:</span>
                        <span class="metric-value" id="processed-${agentName}">${data.processed || 0}</span>
                    </div>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" id="progress-${agentName}" style="width: 0%"></div>
                </div>
            `;
            
            return card;
        }
        
        function updateCardStatus(card, data) {
            const statusElement = document.getElementById(`status-${card.id.replace('card-', '')}`);
            const statusTextElement = document.getElementById(`status-text-${card.id.replace('card-', '')}`);
            const processedElement = document.getElementById(`processed-${card.id.replace('card-', '')}`);
            const progressElement = document.getElementById(`progress-${card.id.replace('card-', '')}`);
            
            statusElement.className = `status ${data.status}`;
            statusElement.textContent = data.status;
            
            if (data.processed !== undefined) {
                processedElement.textContent = data.processed;
            }
            
            if (data.message) {
                statusTextElement.textContent = data.message;
            }
        }
        
        function updateAgentProgress(agentName, progress, message) {
            const progressElement = document.getElementById(`progress-${agentName}`);
            if (progressElement) {
                progressElement.style.width = `${progress}%`;
            }
            
            const statusTextElement = document.getElementById(`status-text-${agentName}`);
            if (statusTextElement && message) {
                statusTextElement.textContent = message;
            }
        }
        
        // Initialize dashboard with empty cards
        const initialAgents = ['ingestion', 'embedding', 'deduplication', 'entities', 'stocks', 'storage', 'query_processor'];
        initialAgents.forEach(agent => {
            const card = createAgentCard(agent, {});
            dashboard.appendChild(card);
        });
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


@api.get("/stats")
@handle_exceptions
async def get_system_stats():
    """System performance metrics"""
    
    conn = None
    try:
        conn = db.get_connection()
        with conn.cursor() as cur:
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
    except Exception as e:
        logger.error(f"Failed to get system stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system stats")
    finally:
        if conn:
            db.release_connection(conn)


# ============= GRACEFUL SHUTDOWN =============
import signal
import sys

class GracefulShutdown:
    def __init__(self):
        self.shutdown = False
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)
    
    def handle_signal(self, signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown = True
    
    async def cleanup_resources(self):
        """Cleanup all resources"""
        try:
            logger.info("Closing cache connections...")
            await cache.close()
            
            logger.info("Closing database connections...")
            db.close_all_connections()
            
            logger.info("Resources cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

shutdown_handler = GracefulShutdown()

# ============= EXAMPLE USAGE =============
async def run_examples():
    """Demonstrate optimized system"""
    
    try:
        logger.info("=" * 70)
        logger.info("🚀 OPTIMIZED FINANCIAL NEWS INTELLIGENCE SYSTEM")
        logger.info("=" * 70)
        
        # Example 1: Batch ingestion
        logger.info("\n📦 EXAMPLE 1: Batch Processing (10 articles)")
        logger.info("-" * 70)
        
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
        
        logger.info(f"✅ Processed: {len(articles)} articles")
        logger.info(f"📊 Unique: {sum(1 for r in result['duplicate_results'] if not r['is_duplicate'])}")
        logger.info(f"📊 Duplicates: {sum(1 for r in result['duplicate_results'] if r['is_duplicate'])}")
        logger.info(f"💾 FAISS Memory: {faiss_index.get_memory_usage_mb():.2f} MB")
        
        # Example 2: Query
        logger.info("\n🔍 EXAMPLE 2: Querying News")
        logger.info("-" * 70)
        
        query_state = {
            "query": "HDFC Bank news",
            "messages": []
        }
        
        query_result = await asyncio.to_thread(
            app.invoke,
            query_state,
            {"configurable": {"thread_id": "demo_query"}}
        )
        
        logger.info(f"✅ Found: {len(query_result['query_results'])} articles")
        for article in query_result['query_results'][:3]:
            logger.info(f"  - {article['title']}")
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ All examples completed!")
        logger.info("=" * 70)
    except Exception as e:
        logger.error(f"Example execution failed: {str(e)}")


if __name__ == "__main__":
    try:
        # Run examples first
        if len(sys.argv) > 1 and sys.argv[1] == "--examples":
            asyncio.run(run_examples())
        else:
            # Initialize cache before starting server
            asyncio.run(cache.initialize())
            
            # Start FastAPI server
            logger.info("\n🌐 Starting FastAPI server on http://0.0.0.0:8000")
            logger.info("📊 Stats: http://localhost:8000/stats")
            logger.info("📝 Docs: http://localhost:8000/docs")
            
            config = uvicorn.Config(
                api,
                host=config.api_host,
                port=config.api_port,
                log_level="info"
            )
            server = uvicorn.Server(config)
            
            # Run server with graceful shutdown support
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def serve():
                await server.serve()
                
            task = loop.create_task(serve())
            
            def run_server():
                try:
                    while not shutdown_handler.shutdown:
                        time.sleep(0.1)
                except KeyboardInterrupt:
                    pass
                finally:
                    logger.info("Shutting down server...")
                    task.cancel()
                    
                    # Create a new event loop for cleanup
                    cleanup_loop = asyncio.new_event_loop()
                    cleanup_loop.run_until_complete(shutdown_handler.cleanup_resources())
                    cleanup_loop.run_until_complete(server.shutdown())
                    cleanup_loop.close()
                    
                    loop.close()
            
            run_server()
    except Exception as e:
        logger.error(f"Application startup failed: {str(e)}")
        sys.exit(1)
