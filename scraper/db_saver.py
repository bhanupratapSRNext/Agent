"""
Database saver for scraped products.
Saves product data extracted from Bedrock to PostgreSQL database.
"""
import os
import json
import logging
from typing import List, Dict, Any
from datetime import datetime
import psycopg2
from psycopg2.extras import execute_batch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)


class ProductDBSaver:
    """Handles saving scraped products to PostgreSQL database."""
    
    def __init__(self):
        """Initialize database connection using environment variables."""
        self.db_config = {
            'host': os.getenv('PG_HOST', 'localhost'),
            'port': int(os.getenv('PG_PORT', 5432)),
            'database': os.getenv('PG_DB', 'agent'),
            'user': os.getenv('PG_USER', 'postgres'),
            'password': os.getenv('PG_PASSWORD', 'postgres')
        }
        self.connection = None
        self.cursor = None
    
    def connect(self):
        """Establish database connection."""
        try:
            self.connection = psycopg2.connect(**self.db_config)
            self.cursor = self.connection.cursor()
            logger.info(f"✓ Connected to PostgreSQL database: {self.db_config['database']}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {e}")
            return False
    
    def disconnect(self):
        """Close database connection."""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
            logger.info("✓ Database connection closed")
    
    def create_table_if_not_exists(self):
        """Create products table if it doesn't exist."""
        create_table_query = """
        CREATE TABLE IF NOT EXISTS products (
            id SERIAL PRIMARY KEY,
            source_url TEXT,
            scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            product_data JSONB,
            title TEXT,
            price TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_products_source_url ON products(source_url);
        CREATE INDEX IF NOT EXISTS idx_products_scraped_at ON products(scraped_at);
        CREATE INDEX IF NOT EXISTS idx_products_title ON products(title);
        """
        
        try:
            self.cursor.execute(create_table_query)
            self.connection.commit()
            logger.info("✓ Table 'products' is ready")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create table: {e}")
            self.connection.rollback()
            return False
    
    def save_products(self, products: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Save products to database.
        
        Args:
            products: List of product dictionaries
            
        Returns:
            Dictionary with save statistics
        """
        if not products:
            logger.warning("⚠️  No products to save")
            return {
                'success': True,
                'saved_count': 0,
                'failed_count': 0,
                'message': 'No products to save'
            }
        
        # Connect to database
        if not self.connect():
            return {
                'success': False,
                'saved_count': 0,
                'failed_count': len(products),
                'error': 'Failed to connect to database'
            }
        
        # Create table if needed
        if not self.create_table_if_not_exists():
            self.disconnect()
            return {
                'success': False,
                'saved_count': 0,
                'failed_count': len(products),
                'error': 'Failed to create table'
            }
        
        # Prepare insert query
        insert_query = """
        INSERT INTO products (source_url, product_data, title, price, scraped_at)
        VALUES (%s, %s, %s, %s, %s)
        """
        
        saved_count = 0
        failed_count = 0
        
        try:
            # Prepare data for batch insert
            data_to_insert = []
            for product in products:
                try:
                    source_url = product.get('source_url', '')
                    title = product.get('title', product.get('name', ''))
                    price = product.get('price', product.get('regular_price', ''))
                    
                    # Store entire product as JSONB
                    product_json = json.dumps(product)
                    
                    data_to_insert.append((
                        source_url,
                        product_json,
                        title,
                        price,
                        datetime.now()
                    ))
                except Exception as e:
                    logger.error(f"❌ Failed to prepare product for insert: {e}")
                    failed_count += 1
            
            # Batch insert
            if data_to_insert:
                execute_batch(self.cursor, insert_query, data_to_insert, page_size=100)
                self.connection.commit()
                saved_count = len(data_to_insert)
                logger.info(f"✅ Successfully saved {saved_count} products to database")
            
        except Exception as e:
            logger.error(f"❌ Failed to save products: {e}")
            self.connection.rollback()
            failed_count = len(products)
        finally:
            self.disconnect()
        
        return {
            'success': saved_count > 0,
            'saved_count': saved_count,
            'failed_count': failed_count,
            'message': f'Saved {saved_count} products, {failed_count} failed'
        }


async def save_bedrock_products_to_db(products: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Async wrapper to save Bedrock extracted products to PostgreSQL.
    
    Args:
        products: List of product dictionaries from Bedrock extraction
        
    Returns:
        Dictionary with save statistics
    """
    logger.info(f"\n💾 Saving {len(products)} products to PostgreSQL database...")
    
    saver = ProductDBSaver()
    result = saver.save_products(products)
    
    if result['success']:
        logger.info(f"✅ Database save complete: {result['message']}")
    else:
        logger.error(f"❌ Database save failed: {result.get('error', 'Unknown error')}")
    
    return result
