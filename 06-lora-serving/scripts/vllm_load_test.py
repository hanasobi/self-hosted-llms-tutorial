#!/usr/bin/env python3
"""
vLLM Load Test Script
Erzeugt realistische Last mit verschiedenen Mustern für Dashboard-Screenshots
"""

import asyncio
import aiohttp
import time
import json
from datetime import datetime
from typing import List, Dict
import sys

# vLLM Service URL (wird im Cluster verwendet)
VLLM_URL = "http://vllm-service.ml-models:8000/v1/completions"

# Verschiedene Test-Prompts mit unterschiedlicher Komplexität
PROMPTS = {
    "short": [
        "What is Amazon S3?",
        "Explain EC2 instances.",
        "What is DynamoDB?",
        "How does Lambda work?",
        "What is CloudFront?",
    ],
    "medium": [
        "Explain the difference between Amazon S3 and EBS storage.",
        "How does Amazon DynamoDB handle automatic scaling?",
        "What are the main benefits of using AWS Lambda for serverless computing?",
        "Compare Amazon RDS and Aurora database services.",
        "Describe how Amazon CloudWatch monitoring works.",
    ],
    "rag_context": [
        """Context: Amazon S3 is an object storage service that offers industry-leading scalability, data availability, security, and performance. S3 is designed for 99.999999999% (11 9's) of durability.

Question: What is the durability guarantee of Amazon S3?""",
        """Context: Amazon EC2 provides resizable compute capacity in the cloud. You can launch instances with various CPU, memory, storage configurations optimized for different workloads.

Question: What types of resources can you configure when launching EC2 instances?""",
        """Context: DynamoDB is a fully managed NoSQL database service that provides fast and predictable performance with seamless scalability. It supports both document and key-value data models.

Question: What data models does DynamoDB support?""",
    ]
}


class LoadTestStats:
    """Tracks statistics during load test"""
    
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_latency = 0.0
        self.min_latency = float('inf')
        self.max_latency = 0.0
        self.start_time = time.time()
    
    def add_success(self, latency: float):
        self.total_requests += 1
        self.successful_requests += 1
        self.total_latency += latency
        self.min_latency = min(self.min_latency, latency)
        self.max_latency = max(self.max_latency, latency)
    
    def add_failure(self):
        self.total_requests += 1
        self.failed_requests += 1
    
    def get_stats(self) -> Dict:
        elapsed = time.time() - self.start_time
        avg_latency = self.total_latency / self.successful_requests if self.successful_requests > 0 else 0
        rps = self.successful_requests / elapsed if elapsed > 0 else 0
        
        return {
            "elapsed_seconds": round(elapsed, 1),
            "total_requests": self.total_requests,
            "successful": self.successful_requests,
            "failed": self.failed_requests,
            "avg_latency_sec": round(avg_latency, 2),
            "min_latency_sec": round(self.min_latency, 2) if self.min_latency != float('inf') else 0,
            "max_latency_sec": round(self.max_latency, 2),
            "requests_per_sec": round(rps, 2)
        }


async def send_request(session: aiohttp.ClientSession, prompt: str, model: str, 
                       max_tokens: int, stats: LoadTestStats) -> bool:
    """
    Sendet einen einzelnen Request an vLLM
    
    Returns:
        True wenn erfolgreich, False bei Fehler
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    
    start_time = time.time()
    
    try:
        async with session.post(VLLM_URL, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as response:
            if response.status == 200:
                await response.json()  # Response konsumieren
                latency = time.time() - start_time
                stats.add_success(latency)
                return True
            else:
                text = await response.text()
                print(f"❌ HTTP {response.status}: {text[:100]}")
                stats.add_failure()
                return False
                
    except asyncio.TimeoutError:
        print(f"⏱️  Timeout nach 60s")
        stats.add_failure()
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        stats.add_failure()
        return False


async def load_phase(session: aiohttp.ClientSession, stats: LoadTestStats,
                     phase_name: str, duration_sec: int, requests_per_sec: float,
                     concurrency: int, prompt_mix: Dict[str, float]):
    """
    Führt eine Last-Phase aus
    
    Args:
        phase_name: Name der Phase für Logging
        duration_sec: Dauer in Sekunden
        requests_per_sec: Ziel-RPS
        concurrency: Maximale parallele Requests
        prompt_mix: Dict mit Prompt-Typen und deren Anteil (z.B. {"short": 0.5, "medium": 0.3, "rag_context": 0.2})
    """
    print(f"\n{'='*60}")
    print(f"Phase: {phase_name}")
    print(f"Dauer: {duration_sec}s | Target RPS: {requests_per_sec} | Concurrency: {concurrency}")
    print(f"Prompt Mix: {prompt_mix}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    end_time = start_time + duration_sec
    
    # Semaphore für Concurrency-Limit
    semaphore = asyncio.Semaphore(concurrency)
    
    # Counter für Request-Typen
    request_count = {"short": 0, "medium": 0, "rag_context": 0, "base": 0, "lora": 0}
    
    async def rate_limited_request():
        async with semaphore:
            # Wähle Prompt-Typ basierend auf Mix
            import random
            rand = random.random()
            cumulative = 0.0
            prompt_type = "short"
            
            for ptype, weight in prompt_mix.items():
                cumulative += weight
                if rand < cumulative:
                    prompt_type = ptype
                    break
            
            # Wähle zufälligen Prompt aus dem Typ
            prompt = random.choice(PROMPTS[prompt_type])
            
            # 70% LoRA-Adapter, 30% Base Model
            if random.random() < 0.7:
                model = "aws-rag-qa"
                request_count["lora"] += 1
            else:
                model = "TheBloke/Mistral-7B-v0.1-AWQ"
                request_count["base"] += 1
            
            # Max tokens abhängig vom Prompt-Typ
            max_tokens = {
                "short": 50,
                "medium": 100,
                "rag_context": 150
            }[prompt_type]
            
            request_count[prompt_type] += 1
            
            await send_request(session, prompt, model, max_tokens, stats)
    
    # Requests über die Zeit verteilen
    interval = 1.0 / requests_per_sec if requests_per_sec > 0 else 1.0
    
    tasks = []
    last_print = time.time()
    
    while time.time() < end_time:
        # Starte neuen Request
        task = asyncio.create_task(rate_limited_request())
        tasks.append(task)
        
        # Progress alle 5 Sekunden
        if time.time() - last_print > 5.0:
            elapsed = time.time() - start_time
            remaining = end_time - time.time()
            current_stats = stats.get_stats()
            print(f"⏱️  {elapsed:.0f}s elapsed, {remaining:.0f}s remaining | "
                  f"RPS: {current_stats['requests_per_sec']} | "
                  f"Success: {current_stats['successful']} | "
                  f"Failed: {current_stats['failed']}")
            last_print = time.time()
        
        # Warte bis zum nächsten Request
        await asyncio.sleep(interval)
    
    # Warte auf alle laufenden Requests
    print(f"\n⏳ Warte auf {len(tasks)} laufende Requests...")
    await asyncio.gather(*tasks, return_exceptions=True)
    
    print(f"\n✅ Phase abgeschlossen")
    print(f"Request-Verteilung: {request_count}")


async def run_load_test():
    """Hauptfunktion für Load Test mit verschiedenen Phasen"""
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           vLLM Load Test für Dashboard-Screenshots           ║
╚══════════════════════════════════════════════════════════════╝

Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Target: {VLLM_URL}

Test-Phasen:
1. Warmup (2min): Niedrige Last zum Aufwärmen
2. Normal (4min): Moderate konstante Last
3. Spike (3min): Hohe Last-Spitze
4. Recovery (3min): Zurück zu normaler Last
5. Cooldown (2min): Auslaufen lassen

Gesamt: ~14 Minuten
""")
    
    # Initialisiere Stats und HTTP Session
    stats = LoadTestStats()
    
    connector = aiohttp.TCPConnector(limit=100, limit_per_host=100)
    timeout = aiohttp.ClientTimeout(total=None)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        
        # Phase 1: Warmup - Niedrige Last
        await load_phase(
            session, stats,
            phase_name="Warmup",
            duration_sec=120,  # 2 Minuten
            requests_per_sec=2.0,
            concurrency=3,
            prompt_mix={"short": 0.7, "medium": 0.3, "rag_context": 0.0}
        )
        
        # Phase 2: Normal - Moderate konstante Last
        await load_phase(
            session, stats,
            phase_name="Normal Operation",
            duration_sec=240,  # 4 Minuten
            requests_per_sec=5.0,
            concurrency=8,
            prompt_mix={"short": 0.5, "medium": 0.3, "rag_context": 0.2}
        )
        
        # Phase 3: Spike - Hohe Last
        await load_phase(
            session, stats,
            phase_name="Load Spike",
            duration_sec=180,  # 3 Minuten
            requests_per_sec=12.0,
            concurrency=20,
            prompt_mix={"short": 0.3, "medium": 0.4, "rag_context": 0.3}
        )
        
        # Phase 4: Recovery - Zurück zu normal
        await load_phase(
            session, stats,
            phase_name="Recovery",
            duration_sec=180,  # 3 Minuten
            requests_per_sec=5.0,
            concurrency=8,
            prompt_mix={"short": 0.5, "medium": 0.3, "rag_context": 0.2}
        )
        
        # Phase 5: Cooldown - Auslaufen
        await load_phase(
            session, stats,
            phase_name="Cooldown",
            duration_sec=120,  # 2 Minuten
            requests_per_sec=2.0,
            concurrency=3,
            prompt_mix={"short": 0.8, "medium": 0.2, "rag_context": 0.0}
        )
    
    # Finale Statistiken
    final_stats = stats.get_stats()
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    Final Statistics                          ║
╚══════════════════════════════════════════════════════════════╝

Gesamtdauer:        {final_stats['elapsed_seconds']}s
Total Requests:     {final_stats['total_requests']}
Erfolgreich:        {final_stats['successful']} ({100*final_stats['successful']/final_stats['total_requests']:.1f}%)
Fehlgeschlagen:     {final_stats['failed']}

Durchschn. Latenz:  {final_stats['avg_latency_sec']}s
Min Latenz:         {final_stats['min_latency_sec']}s
Max Latenz:         {final_stats['max_latency_sec']}s

Durchschn. RPS:     {final_stats['requests_per_sec']}

Ende: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

✅ Load Test abgeschlossen!
🖼️  Jetzt Grafana-Dashboard Screenshots erstellen!
""")


if __name__ == "__main__":
    try:
        asyncio.run(run_load_test())
    except KeyboardInterrupt:
        print("\n\n⚠️  Load Test abgebrochen (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)