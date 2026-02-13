#!/usr/bin/env python3
"""
Parallel QA Pair Generation with vLLM - Post 7.1

This script generates QA pairs in parallel using asyncio and aiohttp,
demonstrating the performance improvements from concurrent processing.

Features:
- Asyncio-based concurrent API calls
- Semaphore-controlled concurrency
- Batched processing
- Automatic retry on errors
- Prometheus metrics collection (optional)
- Progress tracking
- Detailed error logging

Usage:
    # Single concurrency level
    python generate_qa_pairs_parallel.py --concurrency 20 --input chunks.jsonl
    
    # Test multiple concurrency levels
    python generate_qa_pairs_parallel.py --test-mode --input chunks.jsonl
"""

import asyncio
import aiohttp
import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
import argparse
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Keep INFO level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'parallel_generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of a single QA generation request"""
    chunk_id: str
    success: bool
    qa_pairs: Optional[List[Dict]] = None
    error: Optional[str] = None
    latency: Optional[float] = None
    retries: int = 0
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class ConcurrencyTestResult:
    """Results from testing a specific concurrency level"""
    concurrency: int
    total_chunks: int
    successful: int
    failed: int
    success_rate: float
    avg_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    total_duration: float
    throughput: float  # chunks per second
    kv_cache_usage: Optional[float] = None
    gpu_utilization: Optional[float] = None


class PrometheusCollector:
    """Collect metrics from Prometheus during test runs"""
    
    def __init__(self, prometheus_url: str = "http://localhost:9090"):
        self.prometheus_url = prometheus_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def query_metric(self, query: str) -> Optional[float]:
        """Query a single metric from Prometheus"""
        try:
            url = f"{self.prometheus_url}/api/v1/query"
            params = {'query': query}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results = data.get('data', {}).get('result', [])
                    if results:
                        return float(results[0]['value'][1])
        except Exception as e:
            logger.warning(f"Failed to query Prometheus metric: {e}")
        return None
    
    async def get_kv_cache_usage(self) -> Optional[float]:
        """Get current KV-Cache usage percentage"""
        query = 'vllm:kv_cache_usage_perc{job="vllm-service"}'
        return await self.query_metric(query)
    
    async def get_gpu_utilization(self) -> Optional[float]:
        """Get current GPU utilization percentage"""
        query = 'DCGM_FI_DEV_GPU_UTIL{gpu="0"}'
        return await self.query_metric(query)


class ParallelQAGenerator:
    """Generate QA pairs in parallel using asyncio"""
    
    # System prompt for QA generation
    SYSTEM_PROMPT = """You are an expert in AWS documentation. Your task is to create three high-quality question-answer pairs based on a given text passage.

Rules for questions:
- Create three different question types: one factual question, one conceptual question, and one comparison or relationship question
- Questions should be realistic - how actual AWS users would ask
- All answers must be completely answerable from the given context
- Questions should be in English

Rules for answers:
- Extract and provide ALL relevant information from the context
- NEVER add information not explicitly stated in the context
- NEVER use external knowledge or your training data - only use what's in the given context
- Be as detailed as the context allows - short context = short answer, detailed context = detailed answer
- Write in complete, helpful sentences as if answering a colleague
- If comparing items, ONLY compare aspects explicitly mentioned in the context
- If the context doesn't provide enough information for a comparison, create a different question type instead
- Answers should be in English

Generate exactly three question-answer pairs from the given text.

CRITICAL OUTPUT REQUIREMENTS:
- Return ONLY a JSON array
- NO markdown code blocks
- NO explanatory text before or after
- NO extra whitespace or newlines outside the JSON

Required JSON structure:
[
  {"question": "...", "answer": "...", "type": "factual"},
  {"question": "...", "answer": "...", "type": "conceptual"},
  {"question": "...", "answer": "...", "type": "comparison"}
]

Question Type Rules:
- factual: Direct information extraction ("What is X?", "When does Y happen?")
- conceptual: Understanding or process questions ("How does X work?", "Why would you use Y?")
- comparison: Relationships or contrasts ("What's the difference between X and Y?", "How does X compare to Y?")
"""
    
    def __init__(
        self,
        vllm_url: str = "http://localhost:8000",
        concurrency: int = 20,
        batch_size: int = 100,
        timeout: int = 30,
        max_retries: int = 0,  # Changed from 1 to 0 - avoid retry issues
        prometheus_url: Optional[str] = None
    ):
        self.vllm_url = vllm_url.rstrip('/') + '/v1/chat/completions'
        self.concurrency = concurrency
        self.batch_size = batch_size
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.prometheus_url = prometheus_url
        
        self.semaphore = asyncio.Semaphore(concurrency)
        self.session: Optional[aiohttp.ClientSession] = None
        
        logger.info(f"Initialized ParallelQAGenerator:")
        logger.info(f"  - vLLM URL: {self.vllm_url}")
        logger.info(f"  - Concurrency: {concurrency}")
        logger.info(f"  - Batch Size: {batch_size}")
        logger.info(f"  - Timeout: {timeout}s")
        logger.info(f"  - Max Retries: {max_retries}")
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def generate_qa_single(
        self,
        chunk: Dict[str, Any],
        retry_count: int = 0
    ) -> GenerationResult:
        """Generate QA pairs for a single chunk with semaphore control"""
        # Extract chunk_id from metadata if nested, otherwise from top level
        metadata = chunk.get('metadata', {})
        chunk_id = metadata.get('chunk_id') or chunk.get('chunk_id', 'unknown')
        
        logger.info(f"[{chunk_id}] Starting generation...")
        async with self.semaphore:
            start_time = time.time()
            
            try:
                # Prepare request - support both 'content' and 'text' fields
                text_content = chunk.get('content') or chunk.get('text', '')
                if not text_content:
                    raise Exception("Chunk has no 'content' or 'text' field")
                
                full_prompt = f"{self.SYSTEM_PROMPT}\n\nText passage:\n{text_content}"
                
                payload = {
                    "model": "TheBloke/Mistral-7B-Instruct-v0.2-AWQ",  # vLLM model name
                    "messages": [
                        {"role": "user", "content": full_prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1500
                }
                
                # Make API call
                async with self.session.post(self.vllm_url, json=payload) as response:
                    latency = time.time() - start_time
                    
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"API error {response.status}: {error_text}")
                    
                    data = await response.json()
                    content = data['choices'][0]['message']['content']
                    
                    # Parse JSON response
                    qa_data = self._parse_response(content)
                    
                    # Support both formats:
                    # 1. JSON Array: [{...}, {...}, {...}]
                    # 2. Object: {"qa_pairs": [...]}
                    qa_pairs = None
                    if isinstance(qa_data, list):
                        # Format from Post 7: Direct JSON array
                        qa_pairs = qa_data
                    elif isinstance(qa_data, dict) and qa_data.get('qa_pairs'):
                        # Alternative format: Object with qa_pairs
                        qa_pairs = qa_data['qa_pairs']
                    
                    if qa_pairs and len(qa_pairs) > 0:
                        logger.info(f"[{chunk_id}] Success in {latency:.1f}s - {len(qa_pairs)} QA pairs")
                        return GenerationResult(
                            chunk_id=chunk_id,
                            success=True,
                            qa_pairs=qa_pairs,
                            latency=latency,
                            retries=retry_count
                        )
                    else:
                        error_msg = "Invalid response format: no qa_pairs found"
                        if qa_data:
                            error_msg += f" (got: {type(qa_data).__name__})"
                        raise Exception(error_msg)
            
            except Exception as e:
                logger.warning(f"Chunk {chunk_id} failed (attempt {retry_count + 1}): {e}")
                
                # Retry logic
                if retry_count < self.max_retries:
                    logger.info(f"[{chunk_id}] Retrying...")
                    await asyncio.sleep(1)  # Brief delay before retry
                    try:
                        return await self.generate_qa_single(chunk, retry_count + 1)
                    except Exception as retry_error:
                        # Even retry failed - return failed result
                        logger.error(f"Chunk {chunk_id} retry also failed: {retry_error}")
                        return GenerationResult(
                            chunk_id=chunk_id,
                            success=False,
                            error=f"Failed after retry: {str(retry_error)}",
                            latency=time.time() - start_time,
                            retries=retry_count + 1
                        )
                
                # Final failure - no more retries
                logger.info(f"[{chunk_id}] Returning failed result")
                return GenerationResult(
                    chunk_id=chunk_id,
                    success=False,
                    error=str(e),
                    latency=time.time() - start_time,
                    retries=retry_count
                )
    
    def _parse_response(self, content: str) -> Optional[Dict]:
        """Parse JSON response, handling common formatting issues"""
        try:
            # Remove markdown code blocks if present
            content = content.strip()
            if content.startswith('```'):
                # Remove ```json and ``` markers
                lines = content.split('\n')
                content = '\n'.join(lines[1:-1])
            
            # Try to parse
            parsed = json.loads(content)
            return parsed
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            
            # Try to clean common issues
            try:
                # Remove control characters
                import re
                cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', content)
                parsed = json.loads(cleaned)
                logger.info(f"Successfully parsed after removing control characters")
                return parsed
            except:
                pass
            
            return None
        except Exception as e:
            logger.warning(f"Unexpected parse error: {e}")
            return None
    
    async def generate_batch(
        self,
        chunks: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> List[GenerationResult]:
        """Generate QA pairs for a batch of chunks"""
        logger.info(f"Processing batch of {len(chunks)} chunks with concurrency {self.concurrency}")
        
        # Create tasks for all chunks
        tasks = [self.generate_qa_single(chunk) for chunk in chunks]
        
        # Process with progress tracking - ALWAYS catch exceptions
        results = []
        for i, coro in enumerate(asyncio.as_completed(tasks), 1):
            try:
                result = await coro
                results.append(result)
            except Exception as e:
                # This should rarely happen as errors are caught in generate_qa_single
                logger.error(f"Unexpected error in task {i}: {e}", exc_info=True)
                # Create a failed result
                results.append(GenerationResult(
                    chunk_id=f"unknown_task_{i}",
                    success=False,
                    error=f"Unexpected error: {str(e)}"
                ))
            
            if show_progress and (i % 10 == 0 or i == len(tasks)):
                success_count = sum(1 for r in results if r.success)
                failed_count = sum(1 for r in results if not r.success)
                logger.info(f"Progress: {i}/{len(tasks)} - Success: {success_count}, Failed: {failed_count}")
        
        return results
    
    async def generate_all(
        self,
        chunks: List[Dict[str, Any]],
        output_file: Path
    ) -> List[GenerationResult]:
        """Generate QA pairs for all chunks using batched processing"""
        logger.info(f"Starting generation for {len(chunks)} chunks")
        logger.info(f"Using {(len(chunks) + self.batch_size - 1) // self.batch_size} batches of size {self.batch_size}")
        
        all_results = []
        
        # Process in batches
        for batch_num in range(0, len(chunks), self.batch_size):
            batch = chunks[batch_num:batch_num + self.batch_size]
            batch_id = batch_num // self.batch_size + 1
            
            logger.info(f"\n{'='*60}")
            logger.info(f"BATCH {batch_id}: Processing chunks {batch_num + 1}-{batch_num + len(batch)}")
            logger.info(f"{'='*60}")
            
            batch_start = time.time()
            batch_results = await self.generate_batch(batch)
            batch_duration = time.time() - batch_start
            
            all_results.extend(batch_results)
            
            # Batch statistics
            success_count = sum(1 for r in batch_results if r.success)
            failed_count = sum(1 for r in batch_results if not r.success)
            avg_latency = sum(r.latency for r in batch_results if r.latency) / len(batch_results)
            
            logger.info(f"Batch {batch_id} completed in {batch_duration:.1f}s")
            logger.info(f"Success: {success_count}/{len(batch_results)} ({success_count/len(batch_results)*100:.1f}%)")
            logger.info(f"Failed: {failed_count}")
            logger.info(f"Avg latency: {avg_latency:.2f}s")
            
            # Save intermediate results
            self._save_results(batch_results, output_file, append=True)
        
        return all_results
    
    def _save_results(
        self,
        results: List[GenerationResult],
        output_file: Path,
        append: bool = False
    ):
        """Save results to JSONL file with validation"""
        mode = 'a' if append else 'w'
        
        skipped_count = 0
        saved_count = 0
        
        with open(output_file, mode, encoding='utf-8') as f:
            for result in results:
                if result.success and result.qa_pairs:
                    for qa_pair in result.qa_pairs:
                        # Validate required fields
                        if not isinstance(qa_pair, dict):
                            logger.warning(f"Chunk {result.chunk_id}: QA pair is not a dict, skipping")
                            skipped_count += 1
                            continue
                        
                        if 'question' not in qa_pair or 'answer' not in qa_pair:
                            logger.warning(f"Chunk {result.chunk_id}: Missing required field, skipping: {list(qa_pair.keys())}")
                            skipped_count += 1
                            continue
                        
                        output_record = {
                            'chunk_id': result.chunk_id,
                            'question': qa_pair['question'],
                            'answer': qa_pair['answer'],
                            'type': qa_pair.get('type', 'unknown'),
                            'latency': result.latency,
                            'retries': result.retries,
                            'timestamp': result.timestamp
                        }
                        f.write(json.dumps(output_record, ensure_ascii=False) + '\n')
                        saved_count += 1
        
        logger.info(f"Results {'appended to' if append else 'saved to'} {output_file} - Saved: {saved_count}, Skipped: {skipped_count}")
    
    async def collect_metrics(self) -> Dict[str, Optional[float]]:
        """Collect current metrics from Prometheus"""
        if not self.prometheus_url:
            return {}
        
        try:
            async with PrometheusCollector(self.prometheus_url) as collector:
                metrics = {
                    'kv_cache_usage': await collector.get_kv_cache_usage(),
                    'gpu_utilization': await collector.get_gpu_utilization()
                }
                return metrics
        except Exception as e:
            logger.warning(f"Failed to collect Prometheus metrics: {e}")
            return {}


class ConcurrencyTester:
    """Test different concurrency levels to find optimal settings"""
    
    def __init__(
        self,
        vllm_url: str,
        test_levels: List[int] = None,
        batch_size: int = 100,
        prometheus_url: Optional[str] = None
    ):
        self.vllm_url = vllm_url
        self.test_levels = test_levels or [1, 5, 10, 20, 50]
        self.batch_size = batch_size
        self.prometheus_url = prometheus_url
        
        logger.info(f"Initialized ConcurrencyTester:")
        logger.info(f"  - Test levels: {self.test_levels}")
        logger.info(f"  - Batch size: {batch_size}")
    
    async def test_concurrency_level(
        self,
        level: int,
        chunks: List[Dict[str, Any]]
    ) -> ConcurrencyTestResult:
        """Test a specific concurrency level"""
        logger.info(f"\n{'='*80}")
        logger.info(f"TESTING CONCURRENCY LEVEL: {level}")
        logger.info(f"{'='*80}\n")
        
        start_time = time.time()
        
        # Create generator with specific concurrency
        async with ParallelQAGenerator(
            vllm_url=self.vllm_url,
            concurrency=level,
            batch_size=self.batch_size,
            prometheus_url=self.prometheus_url
        ) as generator:
            
            # Wait for system to stabilize
            await asyncio.sleep(2)
            
            # Collect metrics before test
            metrics_before = await generator.collect_metrics()
            
            # Run generation
            results = await generator.generate_batch(chunks, show_progress=True)
            
            # Wait for completion and collect metrics
            await asyncio.sleep(2)
            metrics_after = await generator.collect_metrics()
        
        # Calculate statistics
        duration = time.time() - start_time
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        latencies = sorted([r.latency for r in results if r.latency])
        
        result = ConcurrencyTestResult(
            concurrency=level,
            total_chunks=len(chunks),
            successful=len(successful),
            failed=len(failed),
            success_rate=len(successful) / len(chunks) if chunks else 0,
            avg_latency=sum(latencies) / len(latencies) if latencies else 0,
            p50_latency=latencies[len(latencies) // 2] if latencies else 0,
            p95_latency=latencies[int(len(latencies) * 0.95)] if latencies else 0,
            p99_latency=latencies[int(len(latencies) * 0.99)] if latencies else 0,
            total_duration=duration,
            throughput=len(chunks) / duration if duration > 0 else 0,
            kv_cache_usage=metrics_after.get('kv_cache_usage'),
            gpu_utilization=metrics_after.get('gpu_utilization')
        )
        
        # Log results
        logger.info(f"\n{'-'*80}")
        logger.info(f"CONCURRENCY {level} RESULTS:")
        logger.info(f"{'-'*80}")
        logger.info(f"Duration: {duration:.1f}s")
        logger.info(f"Success Rate: {result.success_rate*100:.1f}% ({result.successful}/{result.total_chunks})")
        logger.info(f"Failed: {result.failed}")
        logger.info(f"Throughput: {result.throughput:.2f} chunks/sec")
        logger.info(f"Latency - Avg: {result.avg_latency:.2f}s, P95: {result.p95_latency:.2f}s, P99: {result.p99_latency:.2f}s")
        
        if result.kv_cache_usage:
            logger.info(f"KV-Cache Usage: {result.kv_cache_usage*100:.2f}%")
        if result.gpu_utilization:
            logger.info(f"GPU Utilization: {result.gpu_utilization:.1f}%")
        
        return result
    
    async def run_all_tests(
        self,
        chunks: List[Dict[str, Any]],
        output_file: Path,
        abort_on_degradation: bool = True
    ) -> List[ConcurrencyTestResult]:
        """Run tests for all concurrency levels"""
        logger.info(f"\n{'#'*80}")
        logger.info(f"STARTING CONCURRENCY TEST SUITE")
        logger.info(f"Test chunks: {len(chunks)}")
        logger.info(f"Levels to test: {self.test_levels}")
        logger.info(f"{'#'*80}\n")
        
        results = []
        
        for level in self.test_levels:
            result = await self.test_concurrency_level(level, chunks)
            results.append(result)
            
            # Check abort criteria
            if abort_on_degradation and len(results) > 1:
                if result.success_rate < 0.95:
                    logger.warning(f"⚠️  Success rate dropped below 95% at concurrency {level}")
                    logger.warning(f"   Stopping tests - Sweet spot likely at concurrency {results[-2].concurrency}")
                    break
                
                if result.p95_latency > 10.0:
                    logger.warning(f"⚠️  P95 latency exceeded 10s at concurrency {level}")
                    logger.warning(f"   Stopping tests - Sweet spot likely at concurrency {results[-2].concurrency}")
                    break
            
            # Small delay between tests
            if level != self.test_levels[-1]:
                logger.info(f"\nWaiting 5s before next test...\n")
                await asyncio.sleep(5)
        
        # Save results
        self._save_test_results(results, output_file)
        
        # Summary
        self._print_summary(results)
        
        return results
    
    def _save_test_results(self, results: List[ConcurrencyTestResult], output_file: Path):
        """Save test results to JSON file"""
        output_data = {
            'test_timestamp': datetime.now().isoformat(),
            'test_levels': self.test_levels,
            'results': [asdict(r) for r in results]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\nTest results saved to {output_file}")
    
    def _print_summary(self, results: List[ConcurrencyTestResult]):
        """Print summary table of all test results"""
        logger.info(f"\n{'='*120}")
        logger.info(f"CONCURRENCY TEST SUMMARY")
        logger.info(f"{'='*120}\n")
        
        # Header
        print(f"{'Concur':<8} {'Success':<10} {'Failed':<8} {'Throughput':<12} {'Latency (P95)':<15} {'KV-Cache':<10} {'GPU Util':<10}")
        print(f"{'-'*8} {'-'*10} {'-'*8} {'-'*12} {'-'*15} {'-'*10} {'-'*10}")
        
        # Data rows
        for r in results:
            kv_str = f"{r.kv_cache_usage*100:.1f}%" if r.kv_cache_usage else "N/A"
            gpu_str = f"{r.gpu_utilization:.0f}%" if r.gpu_utilization else "N/A"
            
            print(f"{r.concurrency:<8} {r.success_rate*100:>6.1f}%   {r.failed:<8} {r.throughput:>6.2f}/s     {r.p95_latency:>8.2f}s        {kv_str:<10} {gpu_str:<10}")
        
        print()
        
        # Find optimal concurrency
        valid_results = [r for r in results if r.success_rate >= 0.95]
        if valid_results:
            optimal = max(valid_results, key=lambda r: r.throughput)
            logger.info(f"🎯 RECOMMENDED CONCURRENCY: {optimal.concurrency}")
            logger.info(f"   - Throughput: {optimal.throughput:.2f} chunks/sec")
            logger.info(f"   - Success Rate: {optimal.success_rate*100:.1f}%")
            logger.info(f"   - P95 Latency: {optimal.p95_latency:.2f}s")
            if optimal.kv_cache_usage:
                logger.info(f"   - KV-Cache Usage: {optimal.kv_cache_usage*100:.1f}%")


def load_chunks(input_file: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load chunks from JSONL file"""
    chunks = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if limit and len(chunks) >= limit:
                break
            chunks.append(json.loads(line))
    
    logger.info(f"Loaded {len(chunks)} chunks from {input_file}")
    return chunks


async def main():
    parser = argparse.ArgumentParser(
        description='Generate QA pairs in parallel using vLLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate with specific concurrency
  python generate_qa_pairs_parallel.py --concurrency 20 --input chunks.jsonl
  
  # Test multiple concurrency levels
  python generate_qa_pairs_parallel.py --test-mode --input chunks.jsonl --test-sample 100
  
  # Full production run
  python generate_qa_pairs_parallel.py --concurrency 15 --input all_chunks.jsonl --output qa_pairs_parallel.jsonl
        """
    )
    
    parser.add_argument('--input', type=Path, required=True,
                       help='Input JSONL file with chunks')
    parser.add_argument('--output', type=Path,
                       help='Output JSONL file for QA pairs (default: qa_pairs_parallel_TIMESTAMP.jsonl)')
    parser.add_argument('--vllm-url', default='http://localhost:8000',
                       help='vLLM API URL (default: http://localhost:8000)')
    parser.add_argument('--concurrency', type=int, default=20,
                       help='Number of concurrent requests (default: 20)')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Number of chunks per batch (default: 100)')
    parser.add_argument('--limit', type=int,
                       help='Limit number of chunks to process')
    parser.add_argument('--prometheus-url', 
                       help='Prometheus URL for metrics collection (e.g., http://localhost:9090)')
    
    # Test mode arguments
    parser.add_argument('--test-mode', action='store_true',
                       help='Run concurrency tests instead of production generation')
    parser.add_argument('--test-levels', type=int, nargs='+', default=[1, 5, 10, 20, 50],
                       help='Concurrency levels to test (default: 1 5 10 20 50)')
    parser.add_argument('--test-sample', type=int, default=100,
                       help='Number of chunks for testing (default: 100)')
    
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.test_mode:
            args.output = Path(f"concurrency_test_results_{timestamp}.json")
        else:
            args.output = Path(f"qa_pairs_parallel_{timestamp}.jsonl")
    
    # Load chunks
    chunks = load_chunks(args.input, limit=args.limit)
    
    if not chunks:
        logger.error("No chunks loaded. Exiting.")
        return
    
    # Run in test mode or production mode
    if args.test_mode:
        # Test mode: evaluate different concurrency levels
        test_chunks = chunks[:args.test_sample]
        
        tester = ConcurrencyTester(
            vllm_url=args.vllm_url,
            test_levels=args.test_levels,
            batch_size=args.batch_size,
            prometheus_url=args.prometheus_url
        )
        
        await tester.run_all_tests(test_chunks, args.output)
    
    else:
        # Production mode: generate with specified concurrency
        async with ParallelQAGenerator(
            vllm_url=args.vllm_url,
            concurrency=args.concurrency,
            batch_size=args.batch_size,
            prometheus_url=args.prometheus_url
        ) as generator:
            
            start_time = time.time()
            results = await generator.generate_all(chunks, args.output)
            total_duration = time.time() - start_time
            
            # Final statistics
            successful = sum(1 for r in results if r.success)
            failed = sum(1 for r in results if not r.success)
            total_qa_pairs = sum(len(r.qa_pairs) for r in results if r.qa_pairs)
            
            logger.info(f"\n{'='*80}")
            logger.info(f"GENERATION COMPLETE")
            logger.info(f"{'='*80}")
            logger.info(f"Total duration: {total_duration/60:.1f} minutes ({total_duration:.0f} seconds)")
            logger.info(f"Chunks processed: {len(chunks)}")
            logger.info(f"Success rate: {successful/len(chunks)*100:.1f}% ({successful}/{len(chunks)})")
            logger.info(f"Failed: {failed}")
            logger.info(f"QA pairs generated: {total_qa_pairs}")
            logger.info(f"Throughput: {len(chunks)/total_duration*60:.1f} chunks/minute")
            logger.info(f"Output saved to: {args.output}")
            
            # List failed chunks for debugging
            if failed > 0:
                failed_chunks = [r.chunk_id for r in results if not r.success]
                logger.warning(f"Failed chunks ({failed}): {', '.join(failed_chunks[:10])}")
                if failed > 10:
                    logger.warning(f"  ... and {failed - 10} more")
            
            logger.info(f"{'='*80}\n")


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        sys.exit(1)