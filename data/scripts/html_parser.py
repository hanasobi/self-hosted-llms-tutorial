"""
html_parser.py - HTML Document Parser with Heading Hierarchy

Extracts structured content from AWS documentation in HTML format.
Preserves heading hierarchy, filters irrelevant content, and outputs
clean text for the chunking stage.
"""

from bs4 import BeautifulSoup
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ContentSection:
    """
    A logical content section with its hierarchical context.
    
    Represents a contiguous text block together with its position
    in the document heading hierarchy.
    """
    text: str  # The actual content
    heading_hierarchy: List[str]  # e.g. ["S3 Storage", "Glacier", "Retrieval Options"]
    heading_level: int  # 1 for H1, 2 for H2, etc.
    
    def get_full_context(self) -> str:
        """
        Build full context string including heading hierarchy.
        
        Used during chunking to ensure each chunk retains
        its hierarchical context.
        
        Returns:
            String like "S3 Storage > Glacier > Retrieval Options: [text]"
        """
        if self.heading_hierarchy:
            hierarchy_str = " > ".join(self.heading_hierarchy)
            return f"{hierarchy_str}\n\n{self.text}"
        return self.text


class HTMLParser:
    """
    Parser for AWS documentation in HTML format.
    
    Specialized for AWS docs but can also be used
    for other technical documentation.
    """
    
    # HTML elements to ignore
    IGNORED_TAGS = {
        'script', 'style', 'nav', 'footer', 'header',
        'aside', 'iframe', 'noscript'
    }
    
    # CSS classes that typically contain navigation or ads
    IGNORED_CLASSES = {
        'navigation', 'nav', 'sidebar', 'menu', 'footer',
        'header', 'advertisement', 'ad', 'breadcrumb'
    }
    
    def __init__(self):
        """Initialize HTML Parser."""
        print("HTMLParser initialized")
    
    def parse(self, html_path: Path) -> List[ContentSection]:
        """
        Parse an HTML file and extract structured content.
        """
        print(f"\nParsing HTML: {html_path.name}")
        
        # Load HTML
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # DEBUG: Count headings BEFORE cleanup
        headings_before = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        print(f"  DEBUG: Headings BEFORE cleanup: {len(headings_before)}")
        if headings_before:
            for i, h in enumerate(headings_before[:5]):
                text = h.get_text(strip=True)
                print(f"    {h.name}: '{text[:60]}'")
        
        # Remove irrelevant elements
        self._remove_irrelevant_elements(soup)
        
        # DEBUG: Count headings AFTER cleanup
        headings_after = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        print(f"  DEBUG: Headings AFTER cleanup: {len(headings_after)}")
        if headings_after:
            for i, h in enumerate(headings_after[:5]):
                text = h.get_text(strip=True)
                print(f"    {h.name}: '{text[:60]}'")
        
        # Extract content sections
        sections = self._extract_sections(soup)
        
        print(f"  Extracted {len(sections)} content sections")
        
        # Debugging: Show first few sections
        if sections:
            print(f"\n  Example sections:")
            for i, section in enumerate(sections[:3]):
                hierarchy_str = " > ".join(section.heading_hierarchy) if section.heading_hierarchy else "No hierarchy"
                text_preview = section.text[:80].replace('\n', ' ')
                print(f"    {i+1}. [{hierarchy_str}]")
                print(f"       {text_preview}...")
        
        return sections
    
    def _remove_irrelevant_elements(self, soup: BeautifulSoup):
        """
        Remove navigation, ads, scripts, etc.
        
        UPDATED: Much more conservative for the AWS Rigel Design System.
        We only remove clearly irrelevant tags, not entire containers.
        """
        # Remove only these specific tags (not containers!)
        for tag in ['script', 'style', 'noscript', 'iframe']:
            for element in soup.find_all(tag):
                element.decompose()
        
        # Remove specific navigation elements via data-testid
        # (AWS uses these for UI components)
        for testid in ['subnav-desktop', 'subnav-mobile', 'breadcrumb']:
            for element in soup.find_all(attrs={'data-testid': testid}):
                element.decompose()
        
    
    def _extract_sections(self, soup: BeautifulSoup) -> List[ContentSection]:
        """
        Extract content sections with heading hierarchy.
        
        UPDATED: Supports both regular headings (data-rg-n="HeadingText")
        and FAQ questions (data-rg-n="UtilityText" in h3 tags).
        """
        sections = []
        heading_stack = []
        page_title = None
        
        print("  Debugging: Extracting sections...")
        found_headings = 0
        found_content = 0
        
        all_elements = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'span', 'div', 'p', 'ul'])
        
        for element in all_elements:
            # IS IT A HEADING?
            if element.name.startswith('h'):
                level = int(element.name[1])
                heading_text = element.get_text(strip=True)
                
                # Strict whitespace check: Ignore truly empty headings
                # strip() removes whitespace, then check if string is empty
                if not heading_text or heading_text.isspace() or len(heading_text) < 2:
                    continue
                
                # Is this the first heading? -> Page Title
                if page_title is None:
                    page_title = (level, heading_text)
                    heading_stack = [page_title]
                    found_headings += 1
                    print(f"    Page Title (H{level}): {heading_text}")
                    continue
                
                # For all subsequent headings:
                # Remove headings at same or deeper level (but never the Page Title)
                heading_stack = [
                    (l, h) for l, h in heading_stack 
                    if l < level or (l, h) == page_title
                ]
                
                # Add new heading
                heading_stack.append((level, heading_text))
                found_headings += 1
                
                current_hierarchy = [h for _, h in heading_stack]
                print(f"    Found H{level}: {heading_text[:60]}")
                print(f"      -> Hierarchy: {' > '.join(current_hierarchy)}")
            
            # IS IT CONTENT with data-rg-n="BodyText"?
            elif element.get('data-rg-n') == 'BodyText':
                text = element.get_text(separator='\n', strip=True)
                
                if len(text) < 20:
                    continue
                
                current_hierarchy = [h for _, h in heading_stack]
                current_level = heading_stack[-1][0] if heading_stack else 0
                
                section = ContentSection(
                    text=text,
                    heading_hierarchy=current_hierarchy,
                    heading_level=current_level
                )
                sections.append(section)
                found_content += 1
            
            # FALLBACK: Regular p, ul, ol tags (for non-AWS docs)
            elif element.name in ['p', 'ul', 'ol'] and not element.find_parent(attrs={'data-rg-n': 'BodyText'}):
                text = element.get_text(separator='\n', strip=True)
                
                if len(text) < 20:
                    continue
                
                current_hierarchy = [h for _, h in heading_stack]
                current_level = heading_stack[-1][0] if heading_stack else 0
                
                section = ContentSection(
                    text=text,
                    heading_hierarchy=current_hierarchy,
                    heading_level=current_level
                )
                sections.append(section)
                found_content += 1
        
        print(f"  Debug Summary:")
        print(f"    Page Title: {page_title[1] if page_title else 'None'}")
        print(f"    Headings found: {found_headings}")
        print(f"    Content sections found: {found_content}")
        
        return sections


def test_html_parser():
    """
    Test function for the HTML parser.
    
    Run this to test the parser on one of your downloaded documents.
    """
    print("=" * 80)
    print("HTML PARSER TEST")
    print("=" * 80)
    
    # Find first HTML document in data/raw/
    from pathlib import Path
    
    # Calculate path to data/raw/ relative to this script
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    raw_dir = project_root / "data" / "raw"
    
    print(f"\nProject Root: {project_root}")
    print(f"Raw Data Dir: {raw_dir}")
    print(f"Exists: {raw_dir.exists()}")
    
    if not raw_dir.exists():
        print(f"\nERROR: Directory {raw_dir} not found!")
        print(f"Make sure you have run ./sync_docs.sh")
        return
    
    # Find HTML files
    html_files = list(raw_dir.glob("*/*.html"))
    
    if not html_files:
        print("No HTML files found in data/raw/")
        print("Make sure you have run ./sync_docs.sh")
        return
    
    # Use first HTML as test
    test_file = html_files[100]
    print(f"\nTest file: {test_file}")
    
    # Parse
    parser = HTMLParser()
    sections = parser.parse(test_file)
    
    print(f"\n{'='*80}")
    print(f"RESULT")
    print(f"{'='*80}")
    print(f"Total Sections: {len(sections)}")
    
    # Show detailed stats
    total_chars = sum(len(s.text) for s in sections)
    print(f"Total Characters: {total_chars:,}")
    print(f"Average Section Length: {total_chars // len(sections) if sections else 0} chars")
    
    # Show hierarchy distribution
    from collections import Counter
    hierarchy_depths = Counter(len(s.heading_hierarchy) for s in sections)
    print(f"\nHierarchy Depth Distribution:")
    for depth in sorted(hierarchy_depths.keys()):
        print(f"  Depth {depth}: {hierarchy_depths[depth]} sections")
    
    # Show a few complete sections as examples
    print(f"\n{'='*80}")
    print("EXAMPLE SECTIONS (first 3)")
    print(f"{'='*80}")
    
    for i, section in enumerate(sections[:3]):
        print(f"\nSection {i+1}:")
        print(f"  Hierarchy: {' > '.join(section.heading_hierarchy) if section.heading_hierarchy else 'None'}")
        print(f"  Level: H{section.heading_level}")
        print(f"  Text length: {len(section.text)} chars")
        print(f"  Text preview:")
        print(f"    {section.text[:200]}...")


if __name__ == "__main__":
    test_html_parser()