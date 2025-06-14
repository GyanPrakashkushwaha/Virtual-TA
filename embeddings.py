# preprocess.py
import json
import aiohttp
import asyncio
import logging
import os
import sqlite3
from dotenv import load_dotenv


DB_PATH = "knowledge_base.db"
load_dotenv()


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Chunking parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Get API key from environment variable
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    logger.error("API_KEY environment variable not set. Please set it before running.")

# Create a connection to the SQLite database
def create_connection():
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        logger.info(f"Connected to SQLite database at {DB_PATH}")
        return conn
    except sqlite3.Error as e:
        logger.error(f"Error connecting to database: {e}")
        return None

conn = create_connection()


# Function to create embeddings using aipipe proxy with improved error handling and retries
async def create_embeddings(api_key):
    if not api_key:
        logger.error("API_KEY environment variable not set. Cannot create embeddings.")
        return
        
    conn = create_connection()
    cursor = conn.cursor()
    
    # Get discourse chunks without embeddings
    cursor.execute("SELECT id, content FROM discourse_chunks WHERE embedding IS NULL")
    discourse_chunks = cursor.fetchall()
    logger.info(f"Found {len(discourse_chunks)} discourse chunks to embed")
    
    # Get markdown chunks without embeddings
    cursor.execute("SELECT id, content FROM markdown_chunks WHERE embedding IS NULL")
    markdown_chunks = cursor.fetchall()
    logger.info(f"Found {len(markdown_chunks)} markdown chunks to embed")
    
    # Function to handle long texts by breaking them into multiple embeddings
    async def handle_long_text(session, text, record_id, is_discourse=True, max_retries=3):
        # **ADD: Check if already processed**
        if is_discourse:
            cursor.execute("SELECT embedding FROM discourse_chunks WHERE id = ?", (record_id,))
        else:
            cursor.execute("SELECT embedding FROM markdown_chunks WHERE id = ?", (record_id,))
        
        current_embedding = cursor.fetchone()
        if current_embedding and current_embedding[0] is not None:
            logger.info(f"Record {record_id} already processed, skipping")
            return True
        
        max_chars = 8000
        
        if len(text) <= max_chars:
            return await embed_text(session, text, record_id, is_discourse, max_retries)
        
        logger.info(f"Text exceeds embedding limit for {record_id}: {len(text)} chars. Creating multiple embeddings.")
        
        overlap = 200
        subchunks = []
        
        for i in range(0, len(text), max_chars - overlap):
            end = min(i + max_chars, len(text))
            subchunk = text[i:end]
            if subchunk:
                subchunks.append(subchunk)
        
        logger.info(f"Split into {len(subchunks)} subchunks for embedding")
        
        # Create embeddings for all subchunks
        success_count = 0
        for i, subchunk in enumerate(subchunks):
            logger.info(f"Embedding subchunk {i+1}/{len(subchunks)} for {record_id}")
            success = await embed_text(
                session, 
                subchunk, 
                record_id,
                is_discourse, 
                max_retries,
                f"part_{i+1}_of_{len(subchunks)}"
            )
            if success:
                success_count += 1
            else:
                logger.error(f"Failed to embed subchunk {i+1}/{len(subchunks)} for {record_id}")
        
        # **CRITICAL FIX: Mark original record as processed**
        if success_count > 0:
            placeholder_embedding = json.dumps(["PROCESSED_AS_MULTIPART"]).encode()
            
            if is_discourse:
                cursor.execute(
                    "UPDATE discourse_chunks SET embedding = ? WHERE id = ?",
                    (placeholder_embedding, record_id)
                )
            else:
                cursor.execute(
                    "UPDATE markdown_chunks SET embedding = ? WHERE id = ?",
                    (placeholder_embedding, record_id)
                )
            conn.commit()
            logger.info(f"Marked original record {record_id} as processed")
        
        return success_count > 0

    
    # Function to embed a single text with retry mechanism
    async def embed_text(session, text, record_id, is_discourse=True, max_retries=3, part_id=None):
        retries = 0
        while retries < max_retries:
            try:
                # Call the embedding API through aipipe proxy
                url = "https://aipipe.org/openai/v1/embeddings"
                headers = {
                    "Authorization": api_key,
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "text-embedding-3-small",
                    "input": text
                }
                
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        embedding = result["data"][0]["embedding"]
                        
                        # Convert embedding to binary blob
                        embedding_blob = json.dumps(embedding).encode()
                        
                        # Update the database - handle multi-part embeddings differently
                        if part_id:
                            # For multi-part embeddings, we create additional records
                            if is_discourse:
                                # First, get the original chunk data to duplicate
                                cursor.execute("""
                                SELECT post_id, topic_id, topic_title, post_number, author, created_at, 
                                       likes, chunk_index, content, url FROM discourse_chunks 
                                WHERE id = ?
                                """, (record_id,))
                                original = cursor.fetchone()
                                
                                if original:
                                    # Create a new record with the subchunk and its embedding
                                    cursor.execute("""
                                    INSERT INTO discourse_chunks 
                                    (post_id, topic_id, topic_title, post_number, author, created_at, 
                                     likes, chunk_index, content, url, embedding)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                    """, (
                                        original["post_id"], 
                                        original["topic_id"], 
                                        original["topic_title"], 
                                        original["post_number"],
                                        original["author"], 
                                        original["created_at"], 
                                        original["likes"], 
                                        f"{original['chunk_index']}_{part_id}",  # Append part_id to chunk_index
                                        text, 
                                        original["url"], 
                                        embedding_blob
                                    ))
                            else:
                                # Handle markdown chunks similarly
                                cursor.execute("""
                                SELECT doc_title, original_url, downloaded_at, chunk_index FROM markdown_chunks 
                                WHERE id = ?
                                """, (record_id,))
                                original = cursor.fetchone()
                                
                                if original:
                                    cursor.execute("""
                                    INSERT INTO markdown_chunks 
                                    (doc_title, original_url, downloaded_at, chunk_index, content, embedding)
                                    VALUES (?, ?, ?, ?, ?, ?)
                                    """, (
                                        original["doc_title"],
                                        original["original_url"],
                                        original["downloaded_at"],
                                        f"{original['chunk_index']}_{part_id}",  # Append part_id to chunk_index
                                        text,
                                        embedding_blob
                                    ))
                        else:
                            # For regular embeddings, just update the existing record
                            if is_discourse:
                                cursor.execute(
                                    "UPDATE discourse_chunks SET embedding = ? WHERE id = ?",
                                    (embedding_blob, record_id)
                                )
                            else:
                                cursor.execute(
                                    "UPDATE markdown_chunks SET embedding = ? WHERE id = ?",
                                    (embedding_blob, record_id)
                                )
                        
                        conn.commit()
                        return True
                    elif response.status == 429:  # Rate limit error
                        error_text = await response.text()
                        logger.warning(f"Rate limit reached, retrying after delay (retry {retries+1}): {error_text}")
                        await asyncio.sleep(5 * (retries + 1))  # Exponential backoff
                        retries += 1
                    else:
                        error_text = await response.text()
                        logger.error(f"Error embedding text (status {response.status}): {error_text}")
                        return False
            except Exception as e:
                logger.error(f"Exception embedding text: {e}")
                retries += 1
                await asyncio.sleep(3 * retries)  # Wait before retry
        
        logger.error(f"Failed to embed text after {max_retries} retries")
        return False
    
    # Log how many records already have embeddings (for transparency)
    cursor.execute("SELECT COUNT(*) FROM discourse_chunks WHERE embedding IS NOT NULL")
    existing_discourse = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM markdown_chunks WHERE embedding IS NOT NULL")
    existing_markdown = cursor.fetchone()[0]
    logger.info(f"Skipping {existing_discourse} discourse chunks and {existing_markdown} markdown chunks that already have embeddings")
    
    # Process in smaller batches to avoid rate limits
    batch_size = 10
    async with aiohttp.ClientSession() as session:
        # Process discourse chunks
        for i in range(0, len(discourse_chunks), batch_size):
            batch = discourse_chunks[i:i+batch_size]
            tasks = [handle_long_text(session, text, record_id, True) for record_id, text in batch]
            results = await asyncio.gather(*tasks)
            logger.info(f"Embedded discourse batch {i//batch_size + 1}/{(len(discourse_chunks) + batch_size - 1)//batch_size}: {sum(results)}/{len(batch)} successful")
            
            # Sleep to avoid rate limits
            if i + batch_size < len(discourse_chunks):
                await asyncio.sleep(2)
            
            
            
        
        # Process markdown chunks
        for i in range(0, len(markdown_chunks), batch_size):
            batch = markdown_chunks[i:i+batch_size]
            tasks = [handle_long_text(session, text, record_id, False) for record_id, text in batch]
            results = await asyncio.gather(*tasks)
            logger.info(f"Embedded markdown batch {i//batch_size + 1}/{(len(markdown_chunks) + batch_size - 1)//batch_size}: {sum(results)}/{len(batch)} successful")
            
            # Sleep to avoid rate limits
            if i + batch_size < len(markdown_chunks):
                await asyncio.sleep(2)
            
            
            
    
    conn.close()
    logger.info("Finished creating embeddings")


if __name__ == "__main__":
    conn = create_connection()
    # if conn is None:
    #     break
     # Create embeddings
    asyncio.run(create_embeddings(API_KEY))
    
    # Close connection
    conn.close()
    logger.info("Preprocessing complete")

