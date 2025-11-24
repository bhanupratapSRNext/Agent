from bs4 import BeautifulSoup
import json
import re
from typing import Optional, Dict, List

async def formate_raw_html(
    html: str,
    preserve_json_ld: bool = True,
    preserve_meta_tags: bool = True,
    preserve_microdata: bool = True,
    remove_comments: bool = True,
    remove_hidden_elements: bool = True,
    extract_tables: bool = True,
    verbose: bool = True
) -> str:
    """
    Normalize HTML from e-commerce pages for LLM processing.
    
    Args:
        html: Raw HTML string
        preserve_json_ld: Keep JSON-LD structured data scripts
        preserve_meta_tags: Keep meta tags (Open Graph, Twitter, etc.)
        preserve_microdata: Keep microdata attributes (itemprop, itemtype, etc.)
        remove_comments: Remove HTML comments
        remove_hidden_elements: Remove elements with display:none or visibility:hidden
        extract_tables: Keep specification/feature tables
        verbose: Print cleaning statistics
    
    Returns:
        Normalized HTML string
    """
    if verbose:
        print(f"Original HTML length: {len(html):,} characters")
    
    soup = BeautifulSoup(html, 'html.parser')
    
    # ===== STEP 1: Remove noise elements =====
    noise_tags = ['style', 'nav', 'header', 'footer', 'aside', 'iframe', 'noscript']
    for tag_name in noise_tags:
        for tag in soup.find_all(tag_name):
            tag.decompose()
    
    # ===== STEP 2: Handle script tags selectively =====
    for script in soup.find_all('script'):
        # Preserve JSON-LD structured data (often contains product info)
        if preserve_json_ld and script.get('type') == 'application/ld+json':
            # Clean up and validate JSON
            try:
                json_content = json.loads(script.string)
                # Keep only if it's product-related
                if isinstance(json_content, dict):
                    json_type = json_content.get('@type', '')
                    if any(t in json_type for t in ['Product', 'Offer', 'Review', 'Rating', 'AggregateRating']):
                        continue
                elif isinstance(json_content, list):
                    # Check if list contains product data
                    if any(item.get('@type', '').find('Product') >= 0 for item in json_content if isinstance(item, dict)):
                        continue
            except (json.JSONDecodeError, AttributeError):
                pass
        
        # Remove all other scripts
        script.decompose()
    
    # ===== STEP 3: Remove CSS-related elements =====
    # Remove link tags (stylesheets, preload CSS, etc.)
    for link in soup.find_all('link'):
        rel = link.get('rel', [])
        if isinstance(rel, list):
            rel = ' '.join(rel)
        if any(r in rel for r in ['stylesheet', 'preload', 'prefetch']) or '.css' in link.get('href', ''):
            link.decompose()
    
    # Remove inline style attributes
    for tag in soup.find_all(style=True):
        del tag['style']
    
    # ===== STEP 4: Preserve or remove meta tags =====
    if not preserve_meta_tags:
        for meta in soup.find_all('meta'):
            meta.decompose()
    else:
        # Keep only useful meta tags
        useful_meta_properties = [
            'og:', 'twitter:', 'product:', 'price', 'availability',
            'description', 'title', 'image', 'brand', 'category'
        ]
        for meta in soup.find_all('meta'):
            property_val = meta.get('property', '') or meta.get('name', '')
            if not any(prop in property_val.lower() for prop in useful_meta_properties):
                meta.decompose()
    
    # ===== STEP 5: Handle microdata attributes =====
    if not preserve_microdata:
        # Remove microdata attributes
        microdata_attrs = ['itemprop', 'itemtype', 'itemscope', 'itemid']
        for tag in soup.find_all():
            for attr in microdata_attrs:
                if tag.has_attr(attr):
                    del tag[attr]
    
    # ===== STEP 6: Remove hidden elements =====
    if remove_hidden_elements:
        # Remove elements with display:none or visibility:hidden in inline styles
        # Note: Can't catch CSS-based hiding after removing <style> tags
        for tag in soup.find_all():
            if tag.get('hidden') is not None or tag.get('aria-hidden') == 'true':
                tag.decompose()
    
    # ===== STEP 7: Remove common useless attributes =====
    useless_attrs = [
        'class', 'id', 'data-*', 'aria-*', 'role', 'tabindex',
        'onclick', 'onload', 'onerror', 'onfocus', 'onblur',
        'style'  # in case any remain
    ]
    
    for tag in soup.find_all():
        attrs_to_remove = []
        for attr in tag.attrs:
            # Remove data-* and aria-* attributes
            if attr.startswith('data-') or attr.startswith('aria-') or attr.startswith('on'):
                attrs_to_remove.append(attr)
            # Remove specific useless attributes
            elif attr in ['class', 'id', 'role', 'tabindex', 'style']:
                # Keep class/id if they contain meaningful info
                if attr in ['class', 'id']:
                    value = tag.get(attr, '')
                    # Keep if contains product-related keywords
                    if not any(keyword in str(value).lower() for keyword in 
                              ['product', 'price', 'description', 'rating', 'review', 'spec', 'feature']):
                        attrs_to_remove.append(attr)
                else:
                    attrs_to_remove.append(attr)
        
        for attr in attrs_to_remove:
            del tag[attr]
    
    # ===== STEP 8: Remove HTML comments =====
    if remove_comments:
        for comment in soup.find_all(string=lambda text: isinstance(text, type(soup.Comment))):
            comment.extract()
    
    # ===== STEP 9: Remove empty tags and whitespace =====
    for tag in soup.find_all():
        # Remove tags with no content and no useful attributes
        if not tag.get_text(strip=True) and not tag.find_all() and not tag.get('src') and not tag.get('href'):
            # Preserve meta tags and JSON-LD scripts even if "empty"
            if tag.name not in ['meta', 'script', 'link', 'img']:
                tag.decompose()
    
    # ===== STEP 10: Normalize whitespace =====
    cleaned_html = str(soup)
    # Remove excessive newlines (more than 2 consecutive)
    cleaned_html = re.sub(r'\\n{3,}', '\\n\\n', cleaned_html)
    # Remove spaces before newlines
    cleaned_html = re.sub(r' +\\n', '\\n', cleaned_html)
    
    if verbose:
        reduction_pct = ((len(html) - len(cleaned_html)) / len(html)) * 100
        print(f"Cleaned HTML length: {len(cleaned_html):,} characters")
        print(f"Reduction: {reduction_pct:.1f}%")
        
        # Count preserved structured data
        final_soup = BeautifulSoup(cleaned_html, 'html.parser')
        json_ld_count = len(final_soup.find_all('script', type='application/ld+json'))
        meta_count = len(final_soup.find_all('meta'))
        print(f"Preserved: {json_ld_count} JSON-LD scripts, {meta_count} meta tags")
    
    return cleaned_html


def extract_structured_data(html: str) -> Dict:
    """
    Extract structured data from HTML for analysis.
    
    Returns a dictionary containing:
    - json_ld: List of parsed JSON-LD objects
    - meta_tags: Dictionary of meta tag properties and content
    - microdata: List of elements with microdata
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    structured_data = {
        'json_ld': [],
        'meta_tags': {},
        'microdata': []
    }
    
    # Extract JSON-LD
    for script in soup.find_all('script', type='application/ld+json'):
        try:
            data = json.loads(script.string)
            structured_data['json_ld'].append(data)
        except json.JSONDecodeError:
            pass
    
    # Extract meta tags
    for meta in soup.find_all('meta'):
        property_name = meta.get('property') or meta.get('name')
        content = meta.get('content')
        if property_name and content:
            structured_data['meta_tags'][property_name] = content
    
    # Extract microdata
    for tag in soup.find_all(attrs={'itemprop': True}):
        structured_data['microdata'].append({
            'tag': tag.name,
            'itemprop': tag.get('itemprop'),
            'content': tag.get('content') or tag.get_text(strip=True),
            'itemtype': tag.find_parent(attrs={'itemtype': True}).get('itemtype') if tag.find_parent(attrs={'itemtype': True}) else None
        })
    
    return structured_data

