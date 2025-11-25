"""
Scheduler for running scraper jobs
Monitors MongoDB Configuration collection and processes pending scrape tasks
"""
import asyncio
import logging
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

# Import MongoDB client
from config.mongo_con import client

# Import the setup function
from api.user_agent_config import setup_user_agent_and_scrape

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# MongoDB setup
collection = client["chat-bot"]
indexes_coll = collection["Configuration"]


async def process_pending_configurations():
    """
    Check MongoDB for unprocessed configurations and process them.
    Looks for documents with index_name, url, and processed=False
    """
    try:
        logger.info("🔍 Checking for pending configurations to process...")
        
        # Find documents that have index_name, url, and processed=False
        pending_configs = indexes_coll.find({
            "index_name": {"$exists": True, "$ne": None},
            "root_url": {"$exists": True, "$ne": None},
            "processed": False,
            "progress": "pending"
        })
        
        # Convert cursor to list
        pending_list = list(pending_configs)
        
        if not pending_list:
            logger.info("✓ No pending configurations found")
            return
    
        
        # Process each pending configuration
        for config in pending_list:
            try:
                config_id = config.get("_id")
                user_id = config.get("user_id", "unknown")
                index_name = config.get("index_name")
                root_url = config.get("root_url")

                
                logger.info(f"🚀 Processing configuration for user: {user_id}")
                logger.info(f"   Index: {index_name}")
                logger.info(f"   Root URL: {root_url}")
        
                indexes_coll.update_one(
                        {"_id": config_id},
                        {
                            "$set": {
                                "progress": "processing",
                            }
                        }
                    )
                # Call the setup function
                result = await setup_user_agent_and_scrape(
                    user_id=user_id,
                    index_name=index_name,
                    root_url=root_url,
                )
                
                # Update based on result
                if result.get("success"):
                    logger.info(f"✅ Successfully processed configuration for user: {user_id}")
                    
                    # Update MongoDB - mark as processed
                    indexes_coll.update_one(
                        {"_id": config_id},
                        {
                            "$set": {
                                "processed": True,
                                "processing_completed_at": datetime.utcnow(),
                                "status": "completed",
                                "last_result": result,
                                "progress": "completed"
                            }
                        }
                    )
                else:
                    logger.error(f"❌ Failed to process configuration for user: {user_id}")
                    logger.error(f"   Error: {result.get('error', 'Unknown error')}")
                    
                    # Update MongoDB - mark as failed but keep processed=False for retry
                    indexes_coll.update_one(
                        {"_id": config_id},
                        {
                            "$set": {
                                "processing_completed_at": datetime.utcnow(),
                                "status": "failed",
                                "last_error": result.get("error", "Unknown error"),
                                "progress": "failed"
                            }
                        }
                    )
                
            except Exception as e:
                logger.error(f"❌ Error processing configuration {config_id}: {e}")
                
                # Update MongoDB - mark as failed
                try:
                    indexes_coll.update_one(
                        {"_id": config_id},
                        {
                            "$set": {
                                "processing_completed_at": datetime.utcnow(),
                                "status": "failed",
                                "last_error": str(e),
                                "progress": "failed"
                            }
                        }
                    )
                except Exception as update_error:
                    logger.error(f"Failed to update error status: {update_error}")
                
                continue
        
        logger.info(f"✓ Completed processing {len(pending_list)} configuration(s)")
        
    except Exception as e:
        logger.error(f"❌ Error in process_pending_configurations: {e}")
        import traceback
        traceback.print_exc()


# Create scheduler instance
scheduler = AsyncIOScheduler()


def start_scheduler():
    """
    Start the scheduler with cron job
    Runs every minute to check for pending configurations
    """
    try:
        # Add job to run every minute
        scheduler.add_job(
            process_pending_configurations,
            trigger=CronTrigger(minute='*'), 
            id='process_pending_configs',
            name='Process Pending Configurations',
            replace_existing=True
        )
        
        # Start the scheduler
        scheduler.start()
        logger.info("✓ Scheduler started - checking for pending configurations every minute")
        
    except Exception as e:
        logger.error(f"❌ Error starting scheduler: {e}")
        raise


def stop_scheduler():
    """Stop the scheduler gracefully"""
    try:
        scheduler.shutdown()
        logger.info("✓ Scheduler stopped")
    except Exception as e:
        logger.error(f"❌ Error stopping scheduler: {e}")

