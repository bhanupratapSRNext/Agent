"""
Async DOM Fetcher Module
Fetches HTML content from URLs using Playwright's Async API for FastAPI compatibility.
"""
import random
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
from typing import Optional, Dict
from scraper.config import Config
from scraper.utils import logger

class AsyncDOMFetcher:
    """Fetches DOM content from websites using Playwright's Async API."""
    
    def __init__(self):
        """Initialize the DOM fetcher."""
        self.timeout = Config.REQUEST_TIMEOUT * 1000  # Convert to milliseconds
        # self.timeout = 1000 
    async def fetch(self, url: str, wait_for_selector: Optional[str] = None, fast_mode: bool = False, spa_mode: bool = False) -> Dict[str, any]:
        """
        Fetch the DOM content from a URL asynchronously with adaptive timeout strategy.
        
        Args:
            url: The URL to fetch
            wait_for_selector: Optional CSS selector to wait for before capturing DOM
            fast_mode: If True, use aggressive optimizations for speed (recommended for simple sites)
            spa_mode: If True, optimized for Single Page Applications (Vue, React, Angular)
            
        Returns:
            Dictionary containing HTML content and metadata
            
        Raises:
            Exception: If fetching fails
        """
        logger.info(f"Fetch parameters: {url}")

        # Adaptive timeout based on fast_mode
        timeout = self.timeout if not fast_mode else min(self.timeout, 45000)
        
        try:
            async with async_playwright() as playwright:
                # Launch browser with performance optimizations
                browser = await playwright.chromium.launch(
                    headless=True,
                    args=[
                        '--disable-blink-features=AutomationControlled',
                        '--disable-dev-shm-usage',
                        '--no-sandbox',
                        '--disable-gpu',
                        '--disable-software-rasterizer',
                        '--disable-extensions',
                        '--disable-setuid-sandbox',
                        '--no-first-run',
                        '--no-default-browser-check',
                        '--disable-background-networking',
                        '--disable-sync',
                        '--disable-translate',
                        '--metrics-recording-only',
                        '--disable-default-apps',
                        '--mute-audio',
                        '--no-zygote',
                        '--disable-accelerated-2d-canvas'
                    ]
                )
                
                # Create context with random user agent and optimizations
                user_agent = random.choice(Config.USER_AGENTS)
                context = await browser.new_context(
                    user_agent=user_agent,
                    viewport={'width': 1920, 'height': 1080},
                    java_script_enabled=True,
                    ignore_https_errors=True,
                )
                
                # Adaptive resource blocking based on mode
                async def block_resources(route):
                    resource_type = route.request.resource_type
                    url_lower = route.request.url.lower()
                    
                    if spa_mode:
                        # SPA mode: Only block heavy media, allow JS/CSS for proper rendering
                        if resource_type in ['image', 'media']:
                            await route.abort()
                        else:
                            await route.continue_()
                    elif fast_mode:
                        # Fast mode: Block images, fonts, stylesheets, and media
                        if resource_type in ['image', 'stylesheet', 'font', 'media']:
                            await route.abort()
                        # Block analytics and ads
                        elif any(analytics in url_lower for analytics in ['google-analytics', 'facebook', 'doubleclick', 'ads', 'analytics']):
                            await route.abort()  
                        else:
                            await route.continue_()
                    else:
                        # Normal mode: Allow most resources
                        if resource_type in ['image'] or any(analytics in url_lower for analytics in ['google-analytics', 'doubleclick']):
                            await route.abort()
                        else:
                            await route.continue_()
                
                await context.route("**/*", block_resources)
                
                # Create new page
                page = await context.new_page()
                
                # Set extra headers to look more like a real browser
                await page.set_extra_http_headers({
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1'
                })
                
                try:
                    # Navigate to URL with adaptive timeout strategy
                    logger.info(f"Navigating to {url} (timeout: {timeout/1000}s, fast_mode: {fast_mode})")
                    
                    html = None
                    strategy_used = None
                    
                    # STRATEGY 1: Try domcontentloaded (fastest, works for most sites)
                    try:
                        await page.goto(url, timeout=timeout, wait_until='domcontentloaded')
                        strategy_used = "domcontentloaded"
                        logger.info("✓ Strategy 1 SUCCESS: domcontentloaded")
                    except PlaywrightTimeout:
                        logger.warning("⚠ Strategy 1 TIMEOUT: domcontentloaded failed, trying 'load'...")
                        
                        # STRATEGY 2: Try load (more lenient, waits for all resources)
                        try:
                            await page.goto(url, timeout=timeout, wait_until='load')
                            strategy_used = "load"
                            logger.info("✓ Strategy 2 SUCCESS: load")
                        except PlaywrightTimeout:
                            logger.warning("⚠ Strategy 2 TIMEOUT: load failed, trying partial capture...")
                            strategy_used = "partial"
                    
                    # Wait for content based on strategy and user preference
                    if wait_for_selector and strategy_used != "partial":
                        try:
                            logger.info(f"Waiting for selector: {wait_for_selector}")
                            await page.wait_for_selector(wait_for_selector, timeout=10000)
                            logger.info(f"✓ Selector found: {wait_for_selector}")
                        except PlaywrightTimeout:
                            logger.warning(f"⚠ Selector '{wait_for_selector}' not found, continuing anyway...")
                    else:
                        # Smart wait strategy based on mode
                        if spa_mode:
                            # SPA mode: Wait longer for JavaScript frameworks to render
                            logger.info("SPA mode: waiting for JavaScript framework to render...")
                            try:
                                # Wait for network idle (better for SPAs)
                                await page.wait_for_load_state('networkidle', timeout=12000)
                                logger.info("✓ Network idle achieved for SPA")
                                # Additional wait for Vue/React rendering
                                await page.wait_for_timeout(3000)
                                logger.info("✓ Additional SPA rendering wait completed")
                            except PlaywrightTimeout:
                                logger.warning("⚠ SPA network idle timeout, waiting 5s for dynamic content...")
                                await page.wait_for_timeout(5000)
                        elif fast_mode:
                            # Fast mode: minimal wait for speed
                            logger.info("Fast mode: minimal 1s wait")
                            await page.wait_for_timeout(1000)
                        elif strategy_used == "partial":
                            # Partial mode: give a chance for JS to execute
                            logger.info("Partial mode: waiting 3s for dynamic content")
                            await page.wait_for_timeout(3000)
                        else:
                            # Normal mode: try network idle with fallback
                            try:
                                await page.wait_for_load_state('networkidle', timeout=8000)
                                logger.info("✓ Network idle achieved")
                            except PlaywrightTimeout:
                                logger.warning("⚠ Network idle timeout, waiting 2s for dynamic content...")
                                await page.wait_for_timeout(2000)
                    
                    # Get HTML content
                    html = await page.content()
                    
                    # Validate we got meaningful content
                    if len(html) < 500:
                        logger.warning(f"⚠ Very small HTML content: {len(html)} bytes")
                        raise Exception(f"Received insufficient content from {url}")
                    
                    logger.info(f"✓ Successfully fetched DOM ({len(html):,} characters) using '{strategy_used}' strategy")
                    
                    return {
                        'html': html,
                        'url': url,
                        'status': 'success' if strategy_used != 'partial' else 'partial',
                        'strategy': strategy_used,
                        'size': len(html)
                    }
                    
                except PlaywrightTimeout as e:
                    logger.error(f"❌ TIMEOUT while fetching {url}: {str(e)}")
                    
                    # STRATEGY 3: Last resort - try to get whatever content we have
                    try:
                        html = await page.content()
                        if html and len(html) > 1000:  # If we got some meaningful content
                            logger.warning(f"⚠ Returning partial content ({len(html):,} characters)")
                            return {
                                'html': html,
                                'url': url,
                                'status': 'partial',
                                'strategy': 'timeout_fallback',
                                'size': len(html)
                            }
                    except Exception as fallback_error:
                        logger.error(f"❌ Fallback content retrieval failed: {fallback_error}")
                    
                    # Provide helpful error message
                    suggestions = [
                        f"Try using fast_mode=True for faster sites",
                        f"Increase REQUEST_TIMEOUT in .env (current: {timeout/1000}s)",
                        f"Use specific category pages instead of homepage",
                        f"Check if site has bot protection (Cloudflare, Akamai)"
                    ]
                    
                    error_msg = (
                        f"Timeout: Could not load {url} within {timeout/1000} seconds.\n"
                        f"Suggestions:\n" + "\n".join(f"  - {s}" for s in suggestions)
                    )
                    raise Exception(error_msg)
                    
                except Exception as e:
                    logger.error(f"❌ Error fetching page: {str(e)}")
                    raise
                    
                finally:
                    try:
                        await context.close()
                        await browser.close()
                    except Exception as cleanup_error:
                        logger.warning(f"⚠ Cleanup error (non-critical): {cleanup_error}")
                    
        except Exception as e:
            logger.error(f"Error in fetch: {str(e)}")
            raise Exception(f"Failed to fetch DOM: {str(e)}")
    
    async def fetch_with_scroll(self, url: str, scrolls: int = 3, spa_mode: bool = False) -> Dict[str, any]:
        """
        Fetch DOM with scrolling to load lazy-loaded content.
        
        Args:
            url: The URL to fetch
            scrolls: Number of times to scroll down
            spa_mode: Enable SPA optimizations for JavaScript frameworks
            
        Returns:
            Dictionary containing HTML content and metadata
        """
        logger.info(f"Fetching DOM with scroll from {url} (spa_mode: {spa_mode})")
        
        try:
            async with async_playwright() as playwright:
                # Launch browser
                browser = await playwright.chromium.launch(headless=True)
                
                # Create context
                user_agent = random.choice(Config.USER_AGENTS)
                context = await browser.new_context(
                    user_agent=user_agent,
                    viewport={'width': 1920, 'height': 1080}
                )
                
                page = await context.new_page()
                
                try:
                    # Navigate to URL
                    logger.info(f"Navigating to {url}")
                    await page.goto(url, timeout=self.timeout, wait_until='domcontentloaded')
                    
                    # Wait for initial content
                    await page.wait_for_load_state('networkidle', timeout=10000)
                    
                    # Scroll down multiple times
                    logger.info(f"Scrolling {scrolls} times to load content...")
                    for i in range(scrolls):
                        await page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
                        await page.wait_for_timeout(2000)  # Wait 2 seconds between scrolls
                        logger.info(f"Scroll {i+1}/{scrolls} completed")
                    
                    # Get final HTML
                    html = await page.content()
                    
                    logger.info(f"✓ Successfully fetched DOM with scrolling ({len(html)} characters)")
                    
                    return {
                        'html': html,
                        'url': url,
                        'status': 'success',
                        'scrolls': scrolls
                    }
                    
                except PlaywrightTimeout:
                    logger.error(f"Timeout while fetching {url}")
                    raise Exception(f"Timeout: Could not load {url} within {self.timeout/1000} seconds")
                    
                except Exception as e:
                    logger.error(f"Error during scroll fetch: {str(e)}")
                    raise
                    
                finally:
                    await context.close()
                    await browser.close()
                    
        except Exception as e:
            logger.error(f"Error in fetch_with_scroll: {str(e)}")
            raise Exception(f"Failed to fetch DOM with scroll: {str(e)}")
