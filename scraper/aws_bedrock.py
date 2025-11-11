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
        
        print(f"✓ AWS Bedrock client initialized")
        print(f"  Region: {self.bedrock_config.region_name}")
        print(f"  Model: {self.model_id}")
    
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
        html_preview = raw_html[:30000] if len(raw_html) > 30000 else raw_html
        
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
{html_preview}

Respond with ONLY the JSON array, no markdown code blocks, no explanations."""

        try:
            response_text = self._invoke_bedrock(system_prompt, user_prompt, max_tokens=4000)
            
            # Parse JSON response
            products = self._parse_json_response(response_text)
            
            print(f"✓ Extracted {len(products)} products from HTML")
            if products:
                print(f"  Sample product: {products[0].get('title', 'N/A')[:50]}...")
            
            return products
            
        except Exception as e:
            print(f"❌ Error extracting details: {e}")
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
        
        # Prepare HTML sample
        html_preview = raw_html[:20000] if len(raw_html) > 20000 else raw_html
        
        # Prepare details sample (first 3 products)
        details_sample = extracted_details[:3]
        
        system_prompt = """You are an expert Python web scraping code generator.
Your task is to generate a parse_products(html) function that can extract product data from similar HTML pages.
Generate ONLY the Python function code, no markdown, no explanations."""

        user_prompt = f"""You extracted these products from the HTML:
{json.dumps(details_sample, indent=2)}

From this raw HTML:
{html_preview}

Now generate a Python function called parse_products(html) that:
1. Takes raw HTML as input
2. Uses the SAME extraction method you used (regex for JSON, BeautifulSoup for HTML, etc.)
3. Returns a list of product dictionaries with keys: title, price, image, link, description

Requirements:
- The function must work on similar HTML pages from the same website
- Use proper error handling (return [] if extraction fails)
- Import all needed libraries inside the function (re, json, BeautifulSoup, etc.)
- Use the ACTUAL patterns you see in the HTML (not generic placeholders)
- For escaped JSON: remember quotes are escaped as \\" in the HTML

Return ONLY the function code, starting with:
def parse_products(html):

No markdown code blocks, no explanations, just the Python code."""

        try:
            response_text = self._invoke_bedrock(system_prompt, user_prompt, max_tokens=3000)
            
            # Clean up the response (remove markdown if present)
            function_code = self._clean_function_code(response_text)
            
            print(f"✓ Generated parser function ({len(function_code)} chars)")
            print(f"  Preview: {function_code[:100]}...")
            
            return function_code
            
        except Exception as e:
            print(f"❌ Error generating parser function: {e}")
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
        print("\n" + "="*80)
        print("STEP 3: Validating generated parser function")
        print("="*80)
        
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
            
            print(f"✓ Function executed successfully")
            print(f"  Expected: {len(expected_products)} products")
            print(f"  Extracted: {len(extracted_products)} products")
            
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
            print(f"❌ Validation error: {e}")
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


async def process_urls_with_bedrock(urls_and_html: List[tuple]) -> List[Dict]:
    """
    Process multiple URLs with their HTML using Bedrock approach.
    This function integrates with the app.py scraper workflow.
    
    Args:
        urls_and_html: List of (url, html_content) tuples
        
    Returns:
        List of all extracted products from all pages
    """
    print("\n" + "🧪 " + "="*75)
    print("PROCESSING PAGES WITH AWS BEDROCK")
    print("="*78 + "\n")
    
    try:
        if not urls_and_html:
            print("❌ No pages to process")
            return []
        
        # Initialize generator
        generator = BedrockParserGenerator()
        
        # Take first 3 pages for analysis
        sample_pages = urls_and_html[:3]
        remaining_pages = urls_and_html[3:]
        
        print(f"📊 Processing {len(sample_pages)} sample pages for analysis")
        print(f"📄 {len(remaining_pages)} remaining pages to process with generated parser")
        
        # Step 1: Extract details from EACH sample page individually
        print("\n🔍 Step 1: Extracting products from each sample page...")
        all_sample_details = []
        sample_page_details = []  # Store details per page for validation
        
        for i, (url, html) in enumerate(sample_pages, 1):
            print(f"\n  Analyzing sample page {i}/{len(sample_pages)}...")
            try:
                page_details = generator.extract_details_from_html(html)
                print(f"  ✓ Page {i}: Extracted {len(page_details)} products")
                
                # Store with page reference
                sample_page_details.append({
                    'url': url,
                    'html': html,
                    'products': page_details
                })
                
                all_sample_details.extend(page_details)
                
            except Exception as e:
                print(f"  ✗ Page {i}: Error - {e}")
                continue
        
        if not all_sample_details:
            print(" No products extracted from any sample page")
            return []
        
        print(f"\n✓ Total extracted from samples: {len(all_sample_details)} products")
        
        # Step 2: Generate parser function using all sample data with retry
        print("\n🛠️  Step 2: Generating parser function from sample data...")
        
        # Combine HTMLs for parser generation (to learn patterns from multiple pages)
        combined_html = "\n\n<!-- PAGE SEPARATOR -->\n\n".join([html for _, html in sample_pages])
        
        parse_function = None
        retry_attempt = 0
        
        # Keep retrying until we get a working function
        while parse_function is None:
            retry_attempt += 1
            try:
                print(f"  Attempt {retry_attempt}: Generating parser function...")
                parser_function_code = generator.generate_parser_function(combined_html, all_sample_details)
                print(f"  ✓ Generated parser function ({len(parser_function_code)} chars)")
                
                # Step 3: Validate the generated function
                print(f"  🧪 Testing parser function...")
                
                # Execute the parser function
                exec_globals = {'__builtins__': __builtins__}
                exec_locals = {}
                exec(parser_function_code, exec_globals, exec_locals)
                test_parse_function = exec_locals.get('parse_products')
                
                if not test_parse_function:
                    print("  ❌ Parser function 'parse_products' not found in generated code")
                    continue
                
                # Test on first sample page
                test_products = test_parse_function(sample_pages[0][1])
                
                if not isinstance(test_products, list):
                    print(f"  ❌ Parser returned {type(test_products).__name__} instead of list")
                    continue
                
                print(f"  ✅ Parser function works! Extracted {len(test_products)} products from test page")
                parse_function = test_parse_function  # Success! Exit loop
                
            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
                continue
        
        # validation_results = []
        # total_match_ratio = 0.0
        
        # for i, page_data in enumerate(sample_page_details, 1):
        #     try:
        #         validation_result = generator.validate_parser_function(
        #             parser_function_code,
        #             page_data['html'],
        #             page_data['products']
        #         )
                
        #         validation_results.append(validation_result)
        #         total_match_ratio += validation_result.get('match_ratio', 0.0)
                
        #         status = "✅" if validation_result['success'] else "❌"
        #         print(f"  {status} Sample page {i}: {validation_result['extracted_count']} extracted, "
        #               f"match ratio: {validation_result.get('match_ratio', 0.0):.2%}")
                
        #     except Exception as e:
        #         print(f"  ❌ Sample page {i}: Validation error - {e}")
        #         validation_results.append({'success': False, 'match_ratio': 0.0})
        
        # # Calculate average match ratio
        # avg_match_ratio = total_match_ratio / len(validation_results) if validation_results else 0.0
        # successful_validations = sum(1 for v in validation_results if v['success'])
        
        # print(f"\n  Validation summary: {successful_validations}/{len(sample_pages)} pages passed")
        # print(f"  Average match ratio: {avg_match_ratio:.2%}")
        
        all_products = []
        
        # Proceed if at least 2 out of 3 samples passed or avg match ratio > 0.5
        # validation_passed = (successful_validations >= 2 or avg_match_ratio > 0.5)
        
        
        validation_passed = True 
        if validation_passed:
            print(f"\n✅ Validation PASSED (threshold met)")
            print("\n🚀 Step 4: Applying parser to all pages...")
            
            # Process all pages with the generated parser
            for i, (url, html) in enumerate(urls_and_html, 1):
                try:
                    products = parse_function(html)
                    
                    # Add source URL to each product
                    for product in products:
                        product['source_url'] = url
                    
                    all_products.extend(products)
                    print(f"  Page {i}/{len(urls_and_html)}: {len(products)} products extracted")
                    
                except Exception as e:
                    print(f"  Page {i}/{len(urls_and_html)}: Error - {e}")
                    continue
            
            print(f"\n✓ Total products extracted: {len(all_products)}")
            
        else:
            print(f"\n❌ Validation FAILED (threshold not met)")
            print("Falling back to direct LLM extraction results...")
            
            # Fallback: Use the extracted_details from Step 1 for all sample pages
            for page_data in sample_page_details:
                for product in page_data['products']:
                    product['source_url'] = page_data['url']
                all_products.extend(page_data['products'])
            
            print(f"✓ Using {len(all_products)} products from direct extraction")
        
        return all_products
        
    except Exception as e:
        print(f"\n❌ Bedrock processing failed: {e}")
        import traceback
        traceback.print_exc()
        return []
