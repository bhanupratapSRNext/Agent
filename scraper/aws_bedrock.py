"""
Test AWS Bedrock approach for dynamic HTML parsing
Two-step process:
1. Extract product details from raw HTML using Bedrock LLM
2. Generate a Python parser function using Bedrock LLM based on the extracted details
"""
import json
import re
import os
from typing import Dict, List, Optional, Any
import asyncio
from dotenv import load_dotenv
from scraper.utils import logger

# Load environment variables from .env file
load_dotenv()


import boto3
from botocore.config import Config as BotoConfig
BEDROCK_AVAILABLE = True

import os                      
os.environ['AWS_BEARER_TOKEN_BEDROCK'] = os.getenv('AWS_BEARER_TOKEN_BEDROCK')

class BedrockParserGenerator:
    """Uses AWS Bedrock to extract product data and generate parser functions."""
    
    def __init__(self):
        """Initialize AWS Bedrock client."""
        if not BEDROCK_AVAILABLE:
            raise ImportError("boto3 is required for Bedrock integration")
        
        # Configure AWS Bedrock client
        # You can set AWS credentials via environment variables:
        # AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
        self.bedrock_config = BotoConfig(
            region_name=os.getenv('AWS_REGION'),
            retries={'max_attempts': 3, 'mode': 'adaptive'}
        )
        
        # Create Bedrock Runtime client
        self.bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            config=self.bedrock_config
        )
        
        # Model configuration
        # Using Claude 3 Sonnet - good balance of speed and capability
        self.model_id = os.getenv('BEDROCK_MODEL_ID')

        logger.info(f"✓ AWS Bedrock client initialized")
        logger.info(f"  Region: {self.bedrock_config.region_name}")
        logger.info(f"  Model: {self.model_id}")

    def _invoke_bedrock(self, system_prompt: str, user_prompt: str, max_tokens: int = 4000) -> str:
        """
        Invoke AWS Bedrock with given prompts.
        
        Args:
            system_prompt: System prompt for context
            user_prompt: User prompt with the task
            max_tokens: Maximum tokens in response
            
        Returns:
            Response text from the model
        """
        # Prepare request body for Claude
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        }
        
        # Invoke the model
        response = self.bedrock_runtime.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(request_body)
        )
        
        # Parse response
        response_body = json.loads(response['body'].read())
        
        # Extract text from Claude response
        if 'content' in response_body and len(response_body['content']) > 0:
            return response_body['content'][0]['text']
        
        logger.error(" No response content from Bedrock model")
        raise Exception("No response from Bedrock model")
    
    def extract_details_from_html(self, raw_html: str) -> List[Dict[str, Any]]:
        """
        Step 1: Use Bedrock LLM to extract product details from raw HTML.
        
        Args:
            raw_html: Raw HTML content
            
        Returns:
            List of extracted product dictionaries
        """
        
        # Prepare HTML sample (truncate if too large)
        raw_html = raw_html[:60000] if len(raw_html) > 60000 else raw_html
        
        system_prompt = """You are an expert e-commerce data extraction AI. 
Your task is to analyze raw HTML and extract product information.
You must return ONLY valid JSON, no markdown, no explanations."""

        user_prompt = f"""Analyze this raw HTML and extract ALL products you can find.

Look for product data in:
1. Escaped JSON in HTML (like {{\\"id\\":\\"123\\",\\"title\\":\\"Product\\"...}})
2. JSON in <script> tags
3. Traditional HTML elements with product information

For EACH product found, extract ALL available information including but not limited to:
- Product identifiers (id, sku, etc.)
- Names and titles
- Pricing information (price, sale price, currency, etc.)
- Images (all image URLs, thumbnails, etc.)
- Links and URLs
- Descriptions (short, long, features, etc.)
- Availability (stock status, quantity, etc.)
- Categories and tags
- Ratings and reviews
- Variants (colors, sizes, etc.)
- Any other relevant product data

Return flexible JSON format where each product is an object with whatever fields are available:
[
  {{{{
    "title": "Product Name",
    "price": "$99.99",
    "sale_price": "$79.99",
    "sku": "ABC123",
    "stock_status": "in_stock",
    "rating": "4.5",
    "review_count": "123"
  }}}}
]

If you find escaped JSON, parse it properly. If a field is missing, use empty string "".

HTML to analyze:
{raw_html}

Respond with ONLY the JSON array, no markdown code blocks, no explanations."""

        try:
            response_text = self._invoke_bedrock(system_prompt, user_prompt)
            
            # Parse JSON response
            products = self._parse_json_response(response_text)
            
            logger.info(f"✓ Extracted {len(products)} products from HTML")
            if products:
                logger.info(f"  Sample product: {products[0].get('title', 'N/A')[:50]}...")
            
            return products
            
        except Exception as e:
            logger.error(f"❌ Error extracting details: {e}")
            raise
    
    def generate_parser_function(self, raw_html: str, extracted_details: List[Dict[str, Any]]) -> str:
        """
        Step 2: Use Bedrock LLM to generate a Python parser function based on 
        the raw HTML and the extracted details.
        
        Args:
            raw_html: Raw HTML content (the input)
            extracted_details: Product details extracted by Step 1 (the expected output)
            
        Returns:
            Generated Python function code as string
        """
        
        html_preview = raw_html[:20000] if len(raw_html) > 20000 else raw_html
        
        details_sample = extracted_details[:1]

        system_prompt = """You are a code-generating assistant that specializes in web scraping and HTML parsing. Your task is to create a single parse_products(html) Python function that uses BeautifulSoup to extract structured data from raw HTML pages.

Input:
A raw HTML page (as a string), representing a specific webpage layout.
A list of extracted details (as key-value pairs) that were manually retrieved from the same page.
Objective:
Generate a robust Python parsing function that:
import the dependencies inside the function
Uses BeautifulSoup to parse the HTML.
Extracts all the required fields as shown in the provided list of extracted details.
Can generalize to parse similar pages with the same structure.
JSON-LD may be either a dict or a list. Always check:
    * if it's a list → iterate and pick the first dict with "@type": "Product"
    * never call .get() on a list
    * only call .get() after confirming `isinstance(obj, dict)`

Handles edge cases gracefully (e.g., missing fields, minor layout changes)."""

        user_prompt = f"""Below is the raw HTML of a webpage and a list of dictionarie of manually extracted details. Use this information to generate a Python function that parses similar HTML pages and extracts the same type of details.

### Raw HTML:
{html_preview}

### Extracted Details:
{json.dumps(details_sample, indent=2)}"""
        try:
            response_text = self._invoke_bedrock(system_prompt, user_prompt)
            
            # Clean up the response (remove markdown if present)
            function_code = self._clean_function_code(response_text)
            
            logger.info(f"✓ Generated parser function ({len(function_code)} chars)")
            
            return function_code
            
        except Exception as e:
            logger.error(f"❌ Error generating parser function: {e}")
            raise
    
    def validate_parser_function(self, function_code: str, test_html: str, 
                                expected_products: List[Dict]) -> Dict[str, Any]:
        """
        Validate the generated parser function by testing it.
        
        Args:
            function_code: Generated Python function code
            test_html: HTML to test with
            expected_products: Products we expect to extract
            
        Returns:
            Validation results dictionary
        """
        logger.info("STEP 3: Validating generated parser function")
        
        try:
            # Execute the function code
            exec_globals = {
                '__builtins__': __builtins__,
            }
            exec_locals = {}
            
            exec(function_code, exec_globals, exec_locals)
            
            # Get the parse_products function
            parse_function = exec_locals.get('parse_products')
            if not parse_function:
                return {
                    'success': False,
                    'error': 'No parse_products function found in generated code',
                    'extracted_count': 0
                }
            
            # Test the function
            extracted_products = parse_function(test_html)
            
            # Validate results
            if not isinstance(extracted_products, list):
                return {
                    'success': False,
                    'error': 'Function did not return a list',
                    'extracted_count': 0
                }
            
            logger.info(f"✓ Function executed successfully")
            logger.info(f"  Expected: {len(expected_products)} products")
            logger.info(f"  Extracted: {len(extracted_products)} products")

            # Compare with expected
            match_ratio = self._compare_products(expected_products, extracted_products)
            
            success = len(extracted_products) > 0 and match_ratio > 0.5
            
            return {
                'success': success,
                'extracted_count': len(extracted_products),
                'expected_count': len(expected_products),
                'match_ratio': match_ratio,
                'sample_product': extracted_products[0] if extracted_products else None
            }
            
        except Exception as e:
            logger.error(f"❌ Validation error: {e}")
            return {
                'success': False,
                'error': str(e),
                'extracted_count': 0
            }
    
    def _parse_json_response(self, text: str) -> List[Dict]:
        """Parse JSON from LLM response, handling markdown code blocks."""
        # Remove markdown code blocks if present
        cleaned = re.sub(r'```json\s*', '', text)
        cleaned = re.sub(r'```\s*', '', cleaned)
        cleaned = cleaned.strip()
        
        try:
            # Try direct parse
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to find JSON array in the text
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise Exception("Could not parse JSON from response")
    
    def _clean_function_code(self, text: str) -> str:
        """Clean function code from LLM response."""
        # Remove markdown code blocks
        cleaned = re.sub(r'```python\s*', '', text)
        cleaned = re.sub(r'```\s*', '', cleaned)
        cleaned = cleaned.strip()
        
        # Ensure it starts with def parse_products
        if not cleaned.startswith('def parse_products'):
            # Try to find the function in the text
            match = re.search(r'def parse_products.*', cleaned, re.DOTALL)
            if match:
                cleaned = match.group(0)
        
        return cleaned
    
    def _compare_products(self, expected: List[Dict], extracted: List[Dict]) -> float:
        """Compare expected vs extracted products, return match ratio."""
        if not expected or not extracted:
            return 0.0
        
        def normalize_title(p: Dict) -> str:
            title = (p.get('title') or '').strip().lower()
            return re.sub(r'[^a-z0-9 ]', '', title)
        
        expected_titles = {normalize_title(p) for p in expected if normalize_title(p)}
        extracted_titles = {normalize_title(p) for p in extracted if normalize_title(p)}
        
        if not expected_titles or not extracted_titles:
            return 0.0
        
        intersection = len(expected_titles & extracted_titles)
        union = len(expected_titles | extracted_titles)
        
        return intersection / union if union > 0 else 0.0


async def process_urls_with_bedrock_and_generate_parser(dom_result: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """
    Process multiple URLs with their HTML using Bedrock approach.
    This function integrates with the app.py scraper workflow.
    
    Args:
        urls_and_html: List of (url, html_content) tuples
        
    Returns:
        List of all extracted products from all pages
    """
    logger.info("PROCESSING PAGES WITH AWS BEDROCK")
    
    try:
        payload = {
                "status": False,
                "parse_function": None,
        }
        if not dom_result:
            logger.warning("❌ No pages to process")
            return payload
        
        # Initialize generator
        generator = BedrockParserGenerator()
        
        all_sample_details = []
        sample_page_details = []  # Store details per page for validation

        parse_function = None
        html = dom_result.get("html")

        page_details = generator.extract_details_from_html(html)
        
        sample_page_details.append({
            'url': dom_result.get("url"),
            'html': html,
            'products': page_details
        })
        
        all_sample_details.extend(page_details)

        
        retry_attempt = 0
        
        if page_details:
            while parse_function is None:
                retry_attempt += 1
                logger.info(f"  Attempt {retry_attempt}: Generating parser function...")
                parser_function_code = generator.generate_parser_function(html, all_sample_details)                
                
                exec_globals = {'__builtins__': __builtins__}
                exec_locals = {}
                exec(parser_function_code, exec_globals, exec_locals)
                test_parse_function = exec_locals.get('parse_products')
                
                if not test_parse_function:
                    logger.warning("  ❌ Parser function 'parse_products' not found in generated code")
                    continue
                
                # Test on first sample page
                test_products = test_parse_function(html)
                
                if not isinstance(test_products, list):
                    logger.warning(f"  ❌ Parser returned {type(test_products).__name__} instead of list")
                    continue

                logger.info(f"  ✅ Parser function works! Extracted {len(test_products)} products from test page")
                if len(test_products)==0:
                    parse_function=None
                    break

                parse_function = parser_function_code  
                break   # Success! Exit loop
    
        if not all_sample_details:
            logger.warning(" No products extracted from any sample page")
            return payload
        
        return {
            "status": True,
            "parse_function": parse_function
        }
    
    except Exception as e:
        logger.error(f"❌ Error processing with Bedrock: {e}")
        return payload


async def process_urls_with_parser(raw_html: str, parser_code: str) -> List[Dict]:
    """
    Process multiple URLs with their HTML using a provided parser function.
    
    Args:
        urls_and_html: List of (url, html_content) tuples
        parser_code: String containing the parser function code (must define parse_products function)
        
    Returns:
        List of all extracted products from all pages
    """
    logger.info("PROCESSING URLs WITH PROVIDED PARSER")
    
    try:
        if not raw_html:
            logger.warning("❌ No pages to process")
            return []
        
        if not parser_code:
            logger.warning("❌ No parser code provided")
            return []
        
        # Execute the parser code to get the parse_products function
        exec_globals = {'__builtins__': __builtins__}
        exec_locals = {}
        
        try:
            exec(parser_code, exec_globals, exec_locals)
        except Exception as e:
            logger.error(f"❌ Error executing parser code: {e}")
            return []
        
        # Get the parse_products function
        parse_function = exec_locals.get('parse_products')
        if not parse_function:
            logger.warning("❌ No 'parse_products' function found in provided parser code")
            return []
        
        validation_passed = True
        
        if validation_passed:
                product_detail = parse_function(raw_html)

        return product_detail
        
    except Exception as e:
        logger.error(f"❌ Error in process_urls_with_parser: {e}")
        import traceback
        traceback.print_exc()
        return []
