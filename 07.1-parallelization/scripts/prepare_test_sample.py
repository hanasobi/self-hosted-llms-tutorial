#!/usr/bin/env python3
"""
Prepare Stratified Test Sample from Chunks

Creates a representative test sample by stratifying across services.
This ensures test results are representative of the full dataset.

Usage:
    python prepare_test_sample.py --input chunks_all.jsonl --output chunks_test_500.jsonl --sample-size 500
"""

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_chunks(input_file: Path) -> List[Dict[str, Any]]:
    """Load all chunks from JSONL file"""
    chunks = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def stratified_sample(chunks: List[Dict[str, Any]], sample_size: int) -> List[Dict[str, Any]]:
    """Create stratified sample across services"""
    
    # Group by service
    by_service = defaultdict(list)
    for chunk in chunks:
        service = chunk.get('service', 'unknown')
        by_service[service].append(chunk)
    
    logger.info(f"Found {len(by_service)} services:")
    for service, service_chunks in sorted(by_service.items()):
        logger.info(f"  - {service}: {len(service_chunks)} chunks")
    
    # Calculate proportional sample size per service
    total_chunks = len(chunks)
    sampled = []
    
    for service, service_chunks in by_service.items():
        # Proportional allocation
        proportion = len(service_chunks) / total_chunks
        service_sample_size = max(1, int(sample_size * proportion))
        
        # Sample randomly from this service
        if len(service_chunks) <= service_sample_size:
            # Take all if service has fewer chunks than needed
            service_sample = service_chunks
        else:
            service_sample = random.sample(service_chunks, service_sample_size)
        
        sampled.extend(service_sample)
        logger.info(f"Sampled {len(service_sample)} from {service}")
    
    # If we have too many (due to rounding), randomly remove
    if len(sampled) > sample_size:
        sampled = random.sample(sampled, sample_size)
    
    # If we have too few (due to small services), add random extras
    elif len(sampled) < sample_size:
        remaining = sample_size - len(sampled)
        sampled_ids = {c['chunk_id'] for c in sampled}
        available = [c for c in chunks if c['chunk_id'] not in sampled_ids]
        
        if available:
            extras = random.sample(available, min(remaining, len(available)))
            sampled.extend(extras)
            logger.info(f"Added {len(extras)} random extras to reach {sample_size}")
    
    return sampled


def save_chunks(chunks: List[Dict[str, Any]], output_file: Path):
    """Save chunks to JSONL file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Prepare stratified test sample from chunks')
    parser.add_argument('--input', type=Path, required=True,
                       help='Input JSONL file with all chunks')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output JSONL file for test sample')
    parser.add_argument('--sample-size', type=int, default=100,
                       help='Number of chunks in test sample (default: 100)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--stratified', action='store_true',
                       help='Use stratified sampling (recommended)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.random_seed)
    
    # Load chunks
    logger.info(f"Loading chunks from {args.input}")
    chunks = load_chunks(args.input)
    logger.info(f"Loaded {len(chunks)} chunks")
    
    # Sample
    if args.stratified:
        logger.info(f"Creating stratified sample of {args.sample_size} chunks")
        sample = stratified_sample(chunks, args.sample_size)
    else:
        logger.info(f"Creating random sample of {args.sample_size} chunks")
        sample = random.sample(chunks, min(args.sample_size, len(chunks)))
    
    # Save
    logger.info(f"Saving {len(sample)} chunks to {args.output}")
    save_chunks(sample, args.output)
    
    logger.info(f"✓ Done! Test sample ready at {args.output}")


if __name__ == '__main__':
    main()
