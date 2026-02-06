"""
chunk_documents.py - Main Script for Document Chunking Pipeline

UPDATED: Uses TokenRecursiveChunker for longer, coherent chunks
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import argparse
import json

from html_parser import HTMLParser, ContentSection
from pdf_parser import PDFParser  # Included for completeness; not used with HTML-only input
from document_loader import DocumentMetadata
from token_recursive_chunker import TokenRecursiveChunker
from transformers import AutoTokenizer


# ============================================================================
# Helper Functions for Full-Text Chunking
# ============================================================================

def concatenate_sections_with_markers(sections: List[ContentSection]) -> Tuple[str, List[Dict]]:
    """Concatenate all sections into a single text with position markers.
    
    Returns:
        Tuple of (full_text, section_markers)
        
        section_markers is a list of dicts with:
        - start_pos: Character position in full_text
        - end_pos: Character position in full_text
        - hierarchy: Heading hierarchy for this section
        - level: Heading level
    """
    full_text = ""
    section_markers = []
    
    for section in sections:
        start_pos = len(full_text)
        
        # Concatenate section text (with double newline as separator)
        section_text = section.text
        full_text += section_text + "\n\n"
        
        end_pos = len(full_text)
        
        # Store marker
        section_markers.append({
            'start_pos': start_pos,
            'end_pos': end_pos,
            'hierarchy': section.heading_hierarchy,
            'level': section.heading_level
        })
    
    return full_text, section_markers


def find_hierarchy_for_position(position: int, section_markers: List[Dict]) -> List[str]:
    """Find heading hierarchy for a character position in the text.
    
    Args:
        position: Character position in the concatenated text
        section_markers: List of section markers
    
    Returns:
        Heading hierarchy (list of strings)
    """
    for marker in section_markers:
        if marker['start_pos'] <= position < marker['end_pos']:
            return marker['hierarchy']
    
    # Fallback: Last hierarchy (if position is at the end of text)
    if section_markers:
        return section_markers[-1]['hierarchy']
    
    return []


def create_chunks_from_full_text(
    full_text: str,
    section_markers: List[Dict],
    service: str,
    doc_type: str,
    source_file: str,
    chunker: TokenRecursiveChunker
) -> List[Dict]:
    """Create chunks from the entire document text.
    
    Args:
        full_text: Concatenated text of all sections
        section_markers: Position markers for hierarchy tracking
        service: AWS service name
        doc_type: Document type
        source_file: Original filename
        chunker: TokenRecursiveChunker instance
    
    Returns:
        List of chunk dicts with content, token count, and metadata
    """
    # Base metadata
    base_metadata = {
        'service': service,
        'doc_type': doc_type,
        'source_file': source_file
    }
    
    # Chunk the full text
    chunks = chunker.chunk(full_text, metadata=base_metadata)

    source_base = source_file.replace('.html', '').replace('.pdf', '')
    
    # Add hierarchy information to each chunk
    # Strategy: Use hierarchy of the FIRST section that overlaps with chunk
    for idx, chunk in enumerate(chunks):

        # Override chunk_id with unique composite ID
        chunk['metadata']['chunk_id'] = f"{service.lower()}-{source_base}-{idx}"

        # Find first character position of this chunk in full_text
        chunk_start_char = full_text.find(chunk['content'][:50])  # Use first 50 chars to find position
        
        if chunk_start_char >= 0:
            hierarchy = find_hierarchy_for_position(chunk_start_char, section_markers)
            chunk['metadata']['heading_hierarchy'] = hierarchy
        else:
            chunk['metadata']['heading_hierarchy'] = []
    
    return chunks


# ============================================================================
# Main Processing Functions
# ============================================================================

def process_document(
    doc_path: Path,
    metadata: DocumentMetadata,
    chunker: TokenRecursiveChunker
) -> List[Dict]:
    """Process a single document (HTML or PDF).
    
    UPDATED: Uses full-text chunking instead of section-by-section
    
    Returns:
        List of chunk dicts
    """
    print(f"\n{'='*80}")
    print(f"Processing: {doc_path.name}")
    print(f"  Service: {metadata.service}")
    print(f"  Type: {metadata.doc_type}")
    print(f"{'='*80}")
    
    # Parse document
    if doc_path.suffix == '.html':
        parser = HTMLParser()
        sections = parser.parse(doc_path)
    elif doc_path.suffix == '.pdf':
        parser = PDFParser()
        sections = parser.parse(doc_path)
    else:
        print(f"  ⚠ Unsupported file type: {doc_path.suffix}")
        return []
    
    print(f"  Parsed {len(sections)} sections")
    
    # Concatenate sections to full text
    full_text, section_markers = concatenate_sections_with_markers(sections)
    print(f"  Full text: {len(full_text):,} characters")
    
    # Chunk the full text (not section-by-section!)
    chunks = create_chunks_from_full_text(
        full_text=full_text,
        section_markers=section_markers,
        service=metadata.service,
        doc_type=metadata.doc_type,
        source_file=doc_path.name,
        chunker=chunker
    )
    
    print(f"  ✓ Created {len(chunks)} chunks")
    
    # Statistics
    token_counts = [c['token_count'] for c in chunks]
    if token_counts:
        print(f"    Token stats: min={min(token_counts)}, max={max(token_counts)}, avg={sum(token_counts)/len(token_counts):.0f}")
    
    return chunks


def main(
    chunk_size: int = 512,
    chunk_overlap: int = 0,
    output_filename: str = "chunks_token_based.jsonl",
    limit: Optional[int] = None
):
    """Main pipeline for token-based document chunking.
    
    Args:
        chunk_size: Target chunk size in TOKENS (default: 512)
        chunk_overlap: Overlap in TOKENS (default: 0)
        output_filename: Output JSONL filename
    """
    print("="*80)
    print("TOKEN-BASED DOCUMENT CHUNKING PIPELINE")
    print("="*80)
    print(f"Chunk size: {chunk_size} tokens")
    print(f"Chunk overlap: {chunk_overlap} tokens")
    
    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    raw_dir = project_root / "data" / "raw"
    output_dir = project_root / "data" / "processed"
    
    print(f"\nPaths:")
    print(f"  Raw data: {raw_dir}")
    print(f"  Output: {output_dir}")
    
    if not raw_dir.exists():
        print(f"\nERROR: {raw_dir} does not exist!")
        return
    
    # Initialize tokenizer
    print(f"\n{'='*80}")
    print("INITIALIZING TOKENIZER")
    print(f"{'='*80}")
    
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    
    # Initialize chunker
    chunker = TokenRecursiveChunker(
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Process all documents
    all_chunks = []
    
    print(f"\n{'='*80}")
    print("PROCESSING DOCUMENTS")
    print(f"{'='*80}")
    
    doc_dirs = [d for d in raw_dir.iterdir() if d.is_dir()]
    print(f"Found {len(doc_dirs)} document directories")

    if limit is not None:
      doc_dirs = doc_dirs[:limit]
      print(f"  (Limited to first {limit} documents for testing)")
    
    for doc_idx, doc_dir in enumerate(sorted(doc_dirs), 1):
        print(f"\n[{doc_idx}/{len(doc_dirs)}] Processing: {doc_dir.name}")
        
        # Find document file
        html_files = list(doc_dir.glob("*.html"))
        pdf_files = list(doc_dir.glob("*.pdf"))
        
        # Prefer HTML (better structure), fallback to PDF
        if html_files:
            doc_file = html_files[0]
        elif pdf_files:
            doc_file = pdf_files[0]
        else:
            print(f"  ⚠ No HTML/PDF found, skipping")
            continue
        
        # Load metadata
        metadata_file = doc_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata_json = json.load(f)
            metadata = DocumentMetadata.from_json(
                metadata_json,
                default_service=doc_dir.name.split('-')[0].upper()
            )
        else:
            service = doc_dir.name.split('-')[0].upper()
            doc_type = "FAQ" if "faq" in doc_dir.name.lower() else "Guide"
            metadata = DocumentMetadata(service=service, doc_type=doc_type)
        
        # Process document
        try:
            doc_chunks = process_document(doc_file, metadata, chunker)
            all_chunks.extend(doc_chunks)
            print(f"  ✓ Total chunks so far: {len(all_chunks)}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Print statistics
    print(f"\n{'='*80}")
    print("FINAL STATISTICS")
    print(f"{'='*80}")
    print(f"Total chunks: {len(all_chunks)}")
    
    if all_chunks:
        # Token distribution
        token_counts = [c['token_count'] for c in all_chunks]
        print(f"\nToken Distribution:")
        print(f"  Min: {min(token_counts)}")
        print(f"  Max: {max(token_counts)}")
        print(f"  Mean: {sum(token_counts)/len(token_counts):.1f}")
        print(f"  Median: {sorted(token_counts)[len(token_counts)//2]}")
        
        # Token ranges
        print(f"\nToken Ranges:")
        ranges = [(0, 128), (128, 256), (256, 384), (384, 512), (512, 768), (768, 1024)]
        for start, end in ranges:
            count = sum(1 for t in token_counts if start <= t < end)
            pct = count / len(token_counts) * 100
            print(f"  {start:4d}-{end:4d}: {count:6d} ({pct:5.1f}%)")
        
        # Service distribution
        from collections import Counter
        service_counts = Counter(c['metadata']['service'] for c in all_chunks)
        print(f"\nTop Services:")
        for service, count in service_counts.most_common(10):
            pct = count / len(all_chunks) * 100
            print(f"  {service:20s}: {count:6d} ({pct:5.1f}%)")
    
    # Save
    output_file = output_dir / output_filename
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("SAVING CHUNKS")
    print(f"{'='*80}")
    print(f"Output: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            json_line = json.dumps(chunk, ensure_ascii=False)
            f.write(json_line + '\n')
    
    print(f"✓ Saved {len(all_chunks)} chunks")
    print(f"\n{'='*80}")
    print("✓ PIPELINE COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Token-based AWS documentation chunking')
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=512,
        help='Target chunk size in TOKENS (default: 512)'
    )
    parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=0,
        help='Chunk overlap in TOKENS (default: 0)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='chunks_token_based.jsonl',
        help='Output filename (default: chunks_token_based.jsonl)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of documents to process (for testing)'
    )
    
    args = parser.parse_args()
    main(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        output_filename=args.output,
        limit=args.limit
    )