"""
pdf_parser.py - PDF Document Parser with Heading Detection

NOTE: This parser is included for completeness but is NOT actively used
in this tutorial. We process only HTML documents because the HTML FAQ pages
provide better structure and more balanced service coverage than multi-hundred-
page PDF guides (see Blog Post 4 for the full reasoning).

Extracts structured content from AWS User Guides and Developer Guides
in PDF format. Uses font properties (size, weight) to identify headings
and build a heading hierarchy.
"""

import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from collections import Counter, defaultdict
import re


@dataclass
class TextBlock:
    """
    A text block with layout information.
    
    Represents a contiguous text segment with its visual
    properties extracted from the PDF.
    """
    text: str
    font_size: float
    font_name: str
    is_bold: bool
    page_num: int
    y_position: float  # Vertical position on the page
    
    def is_likely_heading(self, body_font_size: float) -> bool:
        """
        Heuristic: Is this block likely a heading?
        
        A block is likely a heading if:
        - Font size is significantly larger than body text (>= 1.2x)
        - The font is bold
        - The text is not too long (< 200 characters)
        """
        size_threshold = body_font_size * 1.2
        is_larger = self.font_size >= size_threshold
        is_short = len(self.text) < 200
        
        return (is_larger or self.is_bold) and is_short
    
    def estimate_heading_level(self, font_sizes: List[float]) -> int:
        """
        Estimate heading level based on font size.
        
        Larger fonts = higher levels (H1, H2, etc.)
        We sort all unique font sizes and assign levels accordingly.
        """
        # Sort font sizes descending
        sorted_sizes = sorted(set(font_sizes), reverse=True)
        
        try:
            # Find position of this font size in the list
            level = sorted_sizes.index(self.font_size) + 1
            # Limit to H1-H6
            return min(level, 6)
        except ValueError:
            return 3  # Default H3


# Re-use ContentSection from html_parser
from html_parser import ContentSection


class PDFParser:
    """
    Parser for AWS documentation in PDF format.
    
    Uses font properties to detect document structure.
    Specialized for technical documentation.
    
    NOTE: Not actively used in this tutorial — see module docstring.
    """
    
    # Patterns for header/footer detection
    HEADER_FOOTER_PATTERNS = [
        r'^AWS\s+.+\s+(User|Developer)\s+Guide$',  # "AWS S3 User Guide"
        r'^\d+$',  # Page numbers only
        r'^Page\s+\d+',  # "Page 123"
        r'^Chapter\s+\d+',  # "Chapter 5"
        r'^\s*\d+\s*$',  # Whitespace + number
    ]
    
    def __init__(self):
        """Initialize PDF Parser."""
        print("PDFParser initialized")
    
    def parse(self, pdf_path: Path) -> List[ContentSection]:
        """
        Parse a PDF file and extract structured content.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of ContentSection objects
        """
        print(f"\nParsing PDF: {pdf_path.name}")
        
        # Open PDF
        doc = fitz.open(pdf_path)
        print(f"  Pages: {len(doc)}")
        
        # Extract all text blocks with font information
        all_blocks = self._extract_text_blocks(doc)
        print(f"  Extracted {len(all_blocks)} text blocks")
        
        # Analyze font sizes to identify body text
        body_font_size = self._detect_body_font_size(all_blocks)
        print(f"  Body font size: {body_font_size}pt")
        
        # Filter headers and footers
        content_blocks = self._filter_headers_footers(all_blocks)
        print(f"  After header/footer filtering: {len(content_blocks)} blocks")
        
        # Classify blocks as headings or content
        heading_blocks, text_blocks = self._classify_blocks(
            content_blocks, 
            body_font_size
        )
        print(f"  Headings: {len(heading_blocks)}, Content: {len(text_blocks)}")
        
        # Build heading hierarchy
        heading_font_sizes = [b.font_size for b in heading_blocks]
        
        # Create ContentSections
        sections = self._build_sections(
            heading_blocks, 
            text_blocks,
            heading_font_sizes
        )
        
        print(f"  Created {len(sections)} content sections")
        
        # Debugging: Show first few sections
        if sections:
            print(f"\n  Example sections:")
            for i, section in enumerate(sections[:3]):
                hierarchy_str = " > ".join(section.heading_hierarchy) if section.heading_hierarchy else "No hierarchy"
                text_preview = section.text[:80].replace('\n', ' ')
                print(f"    {i+1}. [{hierarchy_str}]")
                print(f"       {text_preview}...")
        
        doc.close()
        return sections
    
    def _extract_text_blocks(self, doc: fitz.Document) -> List[TextBlock]:
        """
        Extract all text blocks from the PDF with font information.
        
        PyMuPDF provides font details for each text span.
        """
        blocks = []
        
        for page_num, page in enumerate(doc):
            # Extract text with font information
            # dict gives us blocks with font details
            text_dict = page.get_text("dict")
            
            for block in text_dict.get("blocks", []):
                # Text blocks only, no images
                if block.get("type") != 0:
                    continue
                
                for line in block.get("lines", []):
                    # Combine all spans in a line
                    line_text = ""
                    avg_font_size = 0
                    font_name = ""
                    span_count = 0
                    
                    for span in line.get("spans", []):
                        line_text += span.get("text", "")
                        avg_font_size += span.get("size", 0)
                        font_name = span.get("font", "")
                        span_count += 1
                    
                    if not line_text.strip() or span_count == 0:
                        continue
                    
                    avg_font_size /= span_count
                    
                    # Detect bold font
                    is_bold = "bold" in font_name.lower() or "bd" in font_name.lower()
                    
                    # Y position (for header/footer detection)
                    y_pos = line.get("bbox", [0, 0, 0, 0])[1]
                    
                    blocks.append(TextBlock(
                        text=line_text.strip(),
                        font_size=round(avg_font_size, 1),
                        font_name=font_name,
                        is_bold=is_bold,
                        page_num=page_num,
                        y_position=y_pos
                    ))
        
        return blocks
    
    def _detect_body_font_size(self, blocks: List[TextBlock]) -> float:
        """
        Detect the most frequent font size (= body text).
        
        The most common font size is most likely the regular body text.
        """
        if not blocks:
            return 12.0  # Default
        
        # Count font sizes
        size_counter = Counter(b.font_size for b in blocks)
        
        # Most common size
        most_common_size = size_counter.most_common(1)[0][0]
        
        return most_common_size
    
    def _filter_headers_footers(self, blocks: List[TextBlock]) -> List[TextBlock]:
        """
        Filter out headers and footers.
        
        Strategy:
        1. Text that repeats on every page at the same Y position
        2. Text that matches our header/footer patterns
        """
        if not blocks:
            return blocks
        
        # Find text that repeats (on multiple pages at the same position)
        position_text = defaultdict(list)
        for block in blocks:
            # Round Y position to 10px for tolerance
            y_rounded = round(block.y_position / 10) * 10
            key = (y_rounded, block.text)
            position_text[key].append(block.page_num)
        
        # If text appears on >50% of pages, it's likely a header/footer
        total_pages = max(b.page_num for b in blocks) + 1
        repeated_texts = {
            text for (y, text), pages in position_text.items()
            if len(set(pages)) > total_pages * 0.5
        }
        
        # Filter
        filtered = []
        for block in blocks:
            # Skip if text repeats across pages
            if block.text in repeated_texts:
                continue
            
            # Skip if pattern matches
            is_header_footer = any(
                re.match(pattern, block.text.strip())
                for pattern in self.HEADER_FOOTER_PATTERNS
            )
            
            if is_header_footer:
                continue
            
            filtered.append(block)
        
        return filtered
    
    def _classify_blocks(
        self, 
        blocks: List[TextBlock], 
        body_font_size: float
    ) -> Tuple[List[TextBlock], List[TextBlock]]:
        """
        Classify blocks as headings or content.
        
        Returns:
            Tuple of (heading_blocks, content_blocks)
        """
        heading_blocks = []
        content_blocks = []
        
        for block in blocks:
            if block.is_likely_heading(body_font_size):
                heading_blocks.append(block)
            else:
                content_blocks.append(block)
        
        return heading_blocks, content_blocks
    
    def _build_sections(
        self,
        heading_blocks: List[TextBlock],
        content_blocks: List[TextBlock],
        heading_font_sizes: List[float]
    ) -> List[ContentSection]:
        """
        Build ContentSections with heading hierarchy.
        
        Strategy:
        - Sort all blocks by (page_num, y_position)
        - Build heading stack like in the HTML parser
        - Assign content to the current heading
        """
        # Combine all blocks and sort by position
        all_blocks = heading_blocks + content_blocks
        all_blocks.sort(key=lambda b: (b.page_num, b.y_position))
        
        # Create lookup for fast heading detection
        heading_texts = {b.text for b in heading_blocks}
        
        sections = []
        heading_stack = []
        current_content = []
        
        for block in all_blocks:
            if block.text in heading_texts:
                # It's a heading
                
                # Save previous content as section
                if current_content and heading_stack:
                    combined_text = "\n".join(current_content)
                    if len(combined_text) >= 20:  # Min. length
                        section = ContentSection(
                            text=combined_text,
                            heading_hierarchy=[h for _, h in heading_stack],
                            heading_level=heading_stack[-1][0] if heading_stack else 0
                        )
                        sections.append(section)
                
                # Reset content
                current_content = []
                
                # Update heading stack
                level = block.estimate_heading_level(heading_font_sizes)
                
                # Remove headings at same or deeper level
                heading_stack = [(l, h) for l, h in heading_stack if l < level]
                
                # Add new heading
                heading_stack.append((level, block.text))
                
            else:
                # It's content
                current_content.append(block.text)
        
        # Save last content block
        if current_content and heading_stack:
            combined_text = "\n".join(current_content)
            if len(combined_text) >= 20:
                section = ContentSection(
                    text=combined_text,
                    heading_hierarchy=[h for _, h in heading_stack],
                    heading_level=heading_stack[-1][0] if heading_stack else 0
                )
                sections.append(section)
        
        return sections


def test_pdf_parser():
    """
    Test function for the PDF parser.
    """
    print("=" * 80)
    print("PDF PARSER TEST")
    print("=" * 80)
    
    # Find first PDF document in data/raw/
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
    
    pdf_files = list(raw_dir.glob("*/*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in data/raw/")
        print("Make sure you have run ./sync_docs.sh")
        return
    
    # Use first PDF as test
    test_file = pdf_files[0]
    print(f"\nTest file: {test_file}")
    
    # Parse
    parser = PDFParser()
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
    print("EXAMPLE SECTIONS (first 5)")
    print(f"{'='*80}")
    
    for i, section in enumerate(sections[:5]):
        print(f"\nSection {i+1}:")
        print(f"  Hierarchy: {' > '.join(section.heading_hierarchy) if section.heading_hierarchy else 'None'}")
        print(f"  Level: H{section.heading_level}")
        print(f"  Text length: {len(section.text)} chars")
        print(f"  Text preview:")
        print(f"    {section.text[:200]}...")


if __name__ == "__main__":
    test_pdf_parser()