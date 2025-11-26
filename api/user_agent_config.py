"""
User Agent Configuration Functions
Handles user-specific agent configurations including Pinecone index creation and website scraping
"""
import os
import httpx
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Import MongoDB client
from config.mongo_con import client

# Import Pinecone
try:
    from pinecone.grpc import PineconeGRPC as Pinecone
except ImportError:
    from pinecone import Pinecone
    
from pinecone import ServerlessSpec

load_dotenv()

# MongoDB setup
collection = client["chat-bot"]  
user_configs_coll = collection["Configuration"]  


async def setup_user_agent_and_scrape(
    user_id: str,
    index_name: str,
) -> Dict[str, Any]:
    """
    Complete setup for user agent:
    1. Fetch user details from MongoDB
    2. Create Pinecone index for the user
    3. Fetch website URL from DB
    4. Call scrapedata API to scrape the website
    
    Args:
        user_id: User ID from authentication
        index_name: Name of the Pinecone index to create
        dimension: Dimension of the vectors (default 384 for all-MiniLM-L6-v2)
        metric: Distance metric (cosine, euclidean, dotproduct)
        cloud: Cloud provider (aws, gcp, azure)
        region: Cloud region
        
    Returns:
        Dict with setup and scrape status
    """
    try:
        # Step 1: Fetch user details from MongoDB
        user_config = user_configs_coll.find_one({
                                        'user_id': user_id,
                                         "index_name": index_name})
        
        if not user_config:
            return {
                "success": False,
                "error": f"User configuration not found for user_id: {user_id}",
                "user_id": user_id
            }
        
        # Extract website URL from user config
        root_url = user_config.get('root_url')
        
        if not root_url:
            return {
                "success": False,
                "error": "No root URL found in user configuration",
                "user_id": user_id
            }
        
        # Step 2: Create Pinecone index
        api_key = os.getenv("PINECONE_API_KEY")
        
        if not api_key:
            return {
                "success": False,
                "error": "Pinecone API key not found in environment variables",
                "user_id": user_id
            }
        
        # Initialize Pinecone client
        pc = Pinecone(api_key=api_key)
        
        # Check if index already exists
        existing_indexes = [idx["name"] for idx in pc.list_indexes().indexes]
        metric = user_config.get("vector_data", {}).get("metric")
        dimension = user_config.get("vector_data", {}).get("dimension")
        cloud = user_config.get("vector_data", {}).get("cloud")
        region = user_config.get("vector_data", {}).get("region")


        index_created = False
        if index_name not in existing_indexes:
            # Validate metric
            valid_metrics = ["cosine", "euclidean", "dotproduct"]
            if metric not in valid_metrics:
                return {
                    "success": False,
                    "error": f"Invalid metric. Must be one of: {', '.join(valid_metrics)}",
                    "user_id": user_id
                }
            
            # Create the index
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud=cloud,
                    region=region
                )
            )
            index_created = True
        
        # Step 3: Call scrapedata API
        # Get the base URL for the scraper API (assuming it's running on the same server)
        base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
        scraper_url = f"{base_url}/scraper/scrapedata"
        
        # Prepare scrape request payload
        scrape_payload = {
            "index_name": index_name,
            "user_id": user_id,
            "url": root_url,
            "scroll": False,
            "format": "json",
            "fast_mode": False,
            "sitemap_mode": True,
            "max_pages": None,
            "complete_scraping": True,
            "spa_mode": None,
            "use_script_generation": None
        }
        
        # Make async HTTP request to scraper API
        async with httpx.AsyncClient(timeout=None) as http_client:
            scrape_response = await http_client.post(scraper_url, json=scrape_payload)
            
            if scrape_response.status_code != 200:
                return {
                    "success": False,
                    "error": f"Scraping failed: {scrape_response.text}",
                    "user_id": user_id,
                    "index_name": index_name
                }
            
            scrape_result = scrape_response.json()
        
        # Step 4: Update user configuration in MongoDB with scrape results
        update_data = {
            'index_name': index_name,
            'index_created': index_created,
            'last_scraped': scrape_result.get('data', {}).get('bedrock_products_count', 0),
            'scrape_status': 'completed'
        }
        
        user_configs_coll.update_one(
            {'user_id': user_id,  
             'index_name': index_name },
            {'$set': update_data}
        )
        
        # Extract products count from scrape result
        products_count = scrape_result.get('data', {}).get('bedrock_products_count', 0)
        
        return {
            "success": True,
            "message": f"User agent setup completed successfully. Index {'created' if index_created else 'already exists'}, website scraped.",
            "user_id": user_id,
            "index_name": index_name,
            "scrape_status": "completed",
            "products_count": products_count,
            "details": {
                "index_created": index_created,
                "website_url": root_url,
                "dimension": dimension,
                "metric": metric,
                "cloud": cloud,
                "region": region
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error setting up user agent: {str(e)}",
            "user_id": user_id,
            "index_name": index_name
        }
