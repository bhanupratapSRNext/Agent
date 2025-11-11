from fastapi import APIRouter, HTTPException
from typing import Optional, List, Dict
from pydantic import BaseModel

from scraper.sitemap_analyzer import SitemapAnalyzer
from scraper.dom_fetcher_async import AsyncDOMFetcher
from scraper.utils import create_response, logger, log_step
from scraper.aws_bedrock import process_urls_with_bedrock
from scraper.db_saver import save_bedrock_products_to_db

# Create router for scraper endpoints
router = APIRouter(prefix="/scraper", tags=["scraper"])


sitemap_analyzer = SitemapAnalyzer()
fetcher = AsyncDOMFetcher()

# Request/Response Models
class FullScrapeRequest(BaseModel):
    url: str
    scroll: Optional[bool] = False
    format: Optional[str] = "json"
    fast_mode: Optional[bool] = False
    sitemap_mode: Optional[bool] = True  # Enable sitemap analysis by default
    max_pages: Optional[int] = None  # None = scrape ALL pages, number = limit pages
    complete_scraping: Optional[bool] = True  # NEW: Enable complete website scraping
    spa_mode: Optional[bool] = None  # NEW: Force SPA mode (auto-detected if None)
    use_script_generation: Optional[bool] = None  # NEW: Use script generation (auto-detected if None)

    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://books.toscrape.com/",
                "scroll": False,
                "format": "json",
                "fast_mode": False,
                "sitemap_mode": True,
                "max_pages": None,
                "complete_scraping": True,
                "spa_mode": None,
                "use_script_generation": None
            }
        }



@router.post("/scrapedata")
async def full_scrape(request: FullScrapeRequest):
    """
    Complete pipeline: analyze website and scrape products in one call.
    
    This endpoint supports multiple scraping modes:
    - COMPLETE SCRAPING MODE (default): Discovers ALL pages via sitemap, scrapes ENTIRE website
    - LIMITED SCRAPING MODE: Scrapes up to max_pages for testing/sampling
    - SINGLE PAGE MODE: Traditional single-page scraping
    
    Complete mode provides maximum product coverage by scraping all product pages.
    Warning: Large sites may take hours and consume significant API tokens.
    """
    try:
        log_step(f"Starting {'sitemap-enhanced' if request.sitemap_mode else 'single-page'} scrape for {request.url}")
        
        # Log configuration
        if request.fast_mode:
            logger.info("⚡ Fast mode enabled - using aggressive optimizations")
        if request.sitemap_mode:
            if request.complete_scraping and request.max_pages is None:
                logger.info("🌐 COMPLETE SCRAPING MODE - Will scrape ALL pages from entire website")
            elif request.max_pages:
                logger.info(f"🗺️ Sitemap mode enabled - analyzing up to {request.max_pages} pages")
            else:
                logger.info("🗺️ Sitemap mode enabled - using smart page calculation")
        
        all_products = []
        all_selectors = {}
        bedrock_products = []  # Track bedrock extracted products
        scrape_stats = {
            'pages_attempted': 0,
            'pages_successful': 0,
            'pages_failed': 0,
        }
        
        if request.sitemap_mode:
            # SITEMAP MODE: Multi-page scraping
            logger.info("🔍 Step 0/6: Analyzing sitemap...")
            
            # Determine max_pages for analysis
            if request.complete_scraping and request.max_pages is None:
                # Complete scraping mode - no limit
                analysis_max_pages = None
            else:
                # Limited scraping mode
                analysis_max_pages = request.max_pages
            
            sitemap_result = await sitemap_analyzer.analyze_website(request.url, analysis_max_pages)
            
            if not sitemap_result['sitemap_found']:
                logger.warning("No sitemap found, falling back to single-page mode")
                pages_to_scrape = [request.url]
            else:
                pages_to_scrape = sitemap_result['categorized_urls']['product_listings']
                logger.info(f"✓ Found {len(pages_to_scrape)} pages to scrape")
                
                # Log what types of pages we found
                categorized = sitemap_result['categorized_urls']
                logger.info("📊 Page breakdown:")
                for category, urls in categorized.items():
                    if urls:
                        logger.info(f"  {category}: {len(urls)} pages")
                
            # NEW: Determine if script generation should be used
            domain = request.url.split('/')[2] if '/' in request.url else request.url
            use_script_gen = True
            
            # Process pages based on selected approach
            if use_script_gen and len(pages_to_scrape) >= 3:
                # NEW: SCRIPT GENERATION APPROACH
                try:
                    logger.info("🚀 Using SCRIPT GENERATION approach")
                    
                    # Fetch HTML for all pages first
                    logger.info("📥 Fetching HTML for all pages...")
                    urls_and_html = []
                    
                    for i, page_url in enumerate(pages_to_scrape, 1):
                        # if i > 3:  # Limit for testing
                        #     break
                            
                        try:
                            scrape_stats['pages_attempted'] += 1
                            logger.info(f"Fetching page {i}/{len(pages_to_scrape)}: {page_url}")
                            
                            # Determine SPA mode
                            spa_mode = request.spa_mode if request.spa_mode is not None else True
                            
                            # Fetch DOM
                            if request.scroll:
                                dom_result = await fetcher.fetch_with_scroll(page_url, spa_mode=spa_mode)
                            else:
                                dom_result = await fetcher.fetch(page_url, fast_mode=request.fast_mode, spa_mode=spa_mode)
                            # print(dom_result)
                            urls_and_html.append((page_url, dom_result['html']))
                            
                        except Exception as e:
                            logger.error(f"Failed to fetch {page_url}: {e}")
                            scrape_stats['pages_failed'] += 1
                            continue
                    
                    # Use script generation to process all pages
                    if urls_and_html:
                        #Bedrock approach
                        bedrock_products = await process_urls_with_bedrock(urls_and_html)
                        logger.info(f"🧪 Bedrock test extracted: {len(bedrock_products)} products")
                        
                        # Save extracted products to PostgreSQL database
                        db_save_result = await save_bedrock_products_to_db(bedrock_products)
                        logger.info(f"💾 Database save result: {db_save_result['message']}")
                    
                except Exception as e:
                    logger.error(f"Script generation failed: {e}")
                    logger.info("🔄 Falling back to traditional LLM analysis")
                    use_script_gen = False
            
            logger.info(f"🎉 {'COMPLETE WEBSITE SCRAPE' if request.complete_scraping else 'SITEMAP SCRAPE'} FINISHED!")
            logger.info(f"📊 FINAL STATISTICS:")
            logger.info(f"   🌐 Website: {request.url}")
            logger.info(f"   📄 Pages attempted: {scrape_stats['pages_attempted']}")
            logger.info(f"   ❌ Pages failed: {scrape_stats['pages_failed']}")
            logger.info(f"   🧪 Bedrock products extracted: {len(bedrock_products)}")
    
            
            result = {
                'url': request.url,
                'mode': 'complete_scraping' if request.complete_scraping else 'sitemap',
                'sitemap_analysis': sitemap_result,
                'scrape_stats': scrape_stats,
                'bedrock_products_count': len(bedrock_products),
            }
        
        response_data = create_response(
            True,
            f"Scrape complete: {len(bedrock_products)} products extracted and saved to database",
            result
        )
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error in scrape endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Scrape failed: {str(e)}")
