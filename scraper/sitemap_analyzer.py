"""
Sitemap Analyzer Module
Discovers, parses, and analyzes website sitemaps to identify all product pages for comprehensive scraping.
"""
import xml.etree.ElementTree as ET
import httpx
import asyncio
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Optional, Set
from scraper.utils import logger, log_step, handle_errors


class SitemapAnalyzer:
    """Analyzes website sitemaps to discover all scrapable pages."""
    
    def __init__(self):
        """Initialize the sitemap analyzer."""
        self.timeout = 30
        
        # Common sitemap locations
        self.sitemap_paths = [
            '/sitemap.xml',
            '/sitemap_index.xml',
            '/sitemaps/sitemap.xml',
            '/sitemap/sitemap.xml',
            '/robots.txt'  # Often contains sitemap reference
        ]
        
        # Product page indicators
        self.product_indicators = [
            'product', 'item', 'collection', 'category', 'shop',
            'catalog', 'store', 'buy', 'goods', 'merchandise'
        ]
        
        # Exclude patterns
        self.exclude_patterns = [
            'account', 'login', 'register', 'checkout', 'cart',
            'about', 'contact', 'privacy', 'terms', 'policy',
            'blog', 'news', 'support', 'help', 'faq',
            'search', 'wishlist', 'compare', 'reviews'
        ]
    
    @handle_errors
    async def analyze_website(self, base_url: str, max_pages: Optional[int] = None) -> Dict:
        """
        Complete sitemap analysis pipeline.
        
        Args:
            base_url: The website base URL
            max_pages: Maximum number of pages to analyze (if None, will calculate optimal)
            
        Returns:
            Dictionary containing discovered URLs and analysis
        """
        log_step(f"Analyzing sitemap for {base_url}")
        
        # Step 1: Discover sitemaps
        sitemap_urls = await self.discover_sitemaps(base_url)
        
        if not sitemap_urls:
            logger.warning("No sitemaps found, will use homepage only")
            return {
                'sitemap_found': False,
                'total_urls': 1,
                'priority_pages': [base_url],
                'categorized_urls': {'homepage': [base_url]},
                'optimal_pages': 1,
                'analysis': 'No sitemap found, falling back to homepage scraping'
            }
        
        # Step 2: Parse all sitemaps
        all_urls = []
        for sitemap_url in sitemap_urls:
            urls = await self.parse_sitemap(sitemap_url)
            all_urls.extend(urls)
        
        logger.info(f"Found {len(all_urls)} total URLs in sitemaps")
        
        # Step 3: Categorize URLs
        categorized = self.categorize_urls(all_urls, base_url)
        
        # Step 4: Calculate optimal pages if not specified
        if max_pages is None:
            max_pages = self.calculate_optimal_max_pages(categorized)
        
        # Step 5: Prioritize pages
        priority_pages = self.prioritize_pages(categorized, max_pages)
        
        logger.info(f"✓ Sitemap analysis complete: {len(priority_pages)} priority pages selected")
        
        return {
            'sitemap_found': True,
            'total_urls': len(all_urls),
            'priority_pages': priority_pages,
            'categorized_urls': categorized,
            'optimal_pages': max_pages,
            'analysis': f'Found {len(all_urls)} URLs, selected {len(priority_pages)} high-priority pages'
        }
    
    async def discover_sitemaps(self, base_url: str) -> List[str]:
        """
        Discover sitemap URLs for a website.
        
        Args:
            base_url: The website base URL
            
        Returns:
            List of discovered sitemap URLs
        """
        logger.info("Discovering sitemaps...")
        discovered = []
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Try common sitemap locations
            for path in self.sitemap_paths:
                sitemap_url = urljoin(base_url, path)
                
                try:
                    if path == '/robots.txt':
                        # Parse robots.txt for sitemap references
                        response = await client.get(sitemap_url)
                        if response.status_code == 200:
                            for line in response.text.split('\n'):
                                if line.lower().startswith('sitemap:'):
                                    sitemap_ref = line.split(':', 1)[1].strip()
                                    discovered.append(sitemap_ref)
                                    logger.info(f"Found sitemap in robots.txt: {sitemap_ref}")
                    else:
                        # Check if sitemap exists
                        response = await client.head(sitemap_url)
                        if response.status_code == 200:
                            discovered.append(sitemap_url)
                            logger.info(f"Found sitemap: {sitemap_url}")
                            
                except Exception as e:
                    logger.debug(f"No sitemap at {sitemap_url}: {e}")
                    continue
        
        # Remove duplicates while preserving order
        unique_sitemaps = list(dict.fromkeys(discovered))
        
        if unique_sitemaps:
            logger.info(f"✓ Discovered {len(unique_sitemaps)} sitemap(s)")
        else:
            logger.warning("No sitemaps discovered")
            
        return unique_sitemaps
    
    async def parse_sitemap(self, sitemap_url: str) -> List[str]:
        """
        Parse a sitemap XML file and extract URLs.
        
        Args:
            sitemap_url: URL of the sitemap to parse
            
        Returns:
            List of URLs found in the sitemap
        """
        logger.info(f"Parsing sitemap: {sitemap_url}")
        urls = []
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(sitemap_url)
                response.raise_for_status()
                
                # Parse XML
                root = ET.fromstring(response.content)
                
                # Handle different sitemap formats
                if 'sitemapindex' in root.tag:
                    # This is a sitemap index, get individual sitemaps
                    logger.info("Found sitemap index, parsing nested sitemaps...")
                    nested_sitemaps = []
                    
                    for sitemap in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap'):
                        loc_elem = sitemap.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                        if loc_elem is not None:
                            nested_sitemaps.append(loc_elem.text)
                    
                    # Parse nested sitemaps (limit to prevent infinite recursion)
                    for nested_url in nested_sitemaps[:10]:  # Limit to 10 nested sitemaps
                        nested_urls = await self.parse_sitemap(nested_url)
                        urls.extend(nested_urls)
                        
                else:
                    # Regular sitemap with URLs
                    for url_elem in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url'):
                        loc_elem = url_elem.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                        if loc_elem is not None:
                            urls.append(loc_elem.text)
                
                logger.info(f"✓ Parsed {len(urls)} URLs from {sitemap_url}")
                
        except Exception as e:
            logger.error(f"Error parsing sitemap {sitemap_url}: {e}")
        
        return urls
    
    def categorize_urls(self, urls: List[str], base_url: str) -> Dict[str, List[str]]:
        """
        Categorize URLs into different types.
        
        Args:
            urls: List of URLs to categorize
            base_url: Base URL to filter by domain
            
        Returns:
            Dictionary with categorized URLs
        """
        logger.info("Categorizing URLs...")
        
        categories = {
            'product_listings': [],
            'product_pages': [],
            'category_pages': [],
            'collection_pages': [],
            'pagination_pages': [],
            'other_pages': []
        }
        
        base_domain = urlparse(base_url).netloc
        
        for url in urls:
            try:
                # Only process URLs from the same domain
                if urlparse(url).netloc != base_domain:
                    continue
                
                url_lower = url.lower()
                path = urlparse(url).path.lower()
                
                # Skip excluded patterns
                if any(pattern in url_lower for pattern in self.exclude_patterns):
                    continue
                
                # Categorize based on URL patterns
                if any(pattern in path for pattern in ['/product/', '/item/', '/p/']):
                    categories['product_pages'].append(url)
                elif any(pattern in path for pattern in ['/collection/', '/collections/']):
                    categories['collection_pages'].append(url)
                elif any(pattern in path for pattern in ['/category/', '/categories/', '/cat/']):
                    categories['category_pages'].append(url)
                elif any(pattern in url_lower for pattern in ['page=', '/page/', 'offset=', 'skip=']):
                    categories['pagination_pages'].append(url)
                elif any(pattern in path for pattern in self.product_indicators):
                    categories['product_listings'].append(url)
                else:
                    categories['other_pages'].append(url)
                    
            except Exception as e:
                logger.debug(f"Error categorizing URL {url}: {e}")
                continue
        
        # Log categorization results
        total_categorized = sum(len(urls) for urls in categories.values())
        logger.info(f"✓ Categorized {total_categorized} URLs:")
        for category, category_urls in categories.items():
            if category_urls:
                logger.info(f"  {category}: {len(category_urls)} URLs")
        
        return categories
    
    def prioritize_pages(self, categorized_urls: Dict[str, List[str]], max_pages: int) -> List[str]:
        """
        Return ALL pages for complete scraping, ordered by priority.
        
        Args:
            categorized_urls: Dictionary of categorized URLs
            max_pages: Total number of pages (not used as limit in complete mode)
            
        Returns:
            List of ALL prioritized URLs for complete coverage
        """
        logger.info(f"Preparing complete scraping of ALL pages...")
        
        # Priority order for complete scraping
        priority_order = [
            'collection_pages',     # Most efficient - many products per page
            'category_pages',       # Good efficiency - category listings
            'product_listings',     # General product listings
            'product_pages'         # Individual products - get everything not covered above
        ]
        
        all_prioritized = []
        
        for category in priority_order:
            category_urls = categorized_urls.get(category, [])
            
            if category_urls:
                logger.info(f"Adding ALL {len(category_urls)} {category} to scraping queue")
                all_prioritized.extend(category_urls)
        
        # Remove duplicates while preserving order
        seen = set()
        deduplicated = []
        for url in all_prioritized:
            if url not in seen:
                seen.add(url)
                deduplicated.append(url)
        
        logger.info(f"✓ COMPLETE SCRAPING: {len(deduplicated)} total pages queued")
        logger.info(f"  This will extract ALL products from the entire website")
        
        return deduplicated
    
    def get_pagination_pattern(self, categorized_urls: Dict[str, List[str]]) -> Optional[str]:
        """
        Identify pagination patterns from URLs.
        
        Args:
            categorized_urls: Dictionary of categorized URLs
            
        Returns:
            Pagination pattern if found, None otherwise
        """
        pagination_urls = categorized_urls.get('pagination_pages', [])
        
        if not pagination_urls:
            return None
        
        # Look for common pagination patterns
        for url in pagination_urls[:5]:  # Check first few
            if 'page=' in url:
                return 'query_param_page'
            elif '/page/' in url:
                return 'path_page'
            elif 'offset=' in url:
                return 'query_param_offset'
        
        return 'unknown'
    
    def calculate_optimal_max_pages(self, categorized_urls: Dict[str, List[str]]) -> int:
        """
        Calculate total number of pages to scrape for COMPLETE coverage.
        
        Args:
            categorized_urls: Dictionary of categorized URLs
            
        Returns:
            Total number of pages to scrape (ALL relevant pages)
        """
        collection_count = len(categorized_urls.get('collection_pages', []))
        category_count = len(categorized_urls.get('category_pages', []))
        listing_count = len(categorized_urls.get('product_listings', []))
        product_count = len(categorized_urls.get('product_pages', []))
        
        # COMPLETE SCRAPING MODE: Include ALL product-related pages
        total_pages = collection_count + category_count + listing_count + product_count
        
        logger.info(f"📊 Complete sitemap analysis:")
        logger.info(f"  Collection pages: {collection_count}")
        logger.info(f"  Category pages: {category_count}")
        logger.info(f"  Product listing pages: {listing_count}")
        logger.info(f"  Individual product pages: {product_count}")
        logger.info(f"🎯 COMPLETE SCRAPING: {total_pages} total pages selected")
        logger.info(f"⚠️  Note: This will scrape ALL products from the entire website")
        
        return max(1, total_pages)  # Scrape ALL pages, minimum 1
