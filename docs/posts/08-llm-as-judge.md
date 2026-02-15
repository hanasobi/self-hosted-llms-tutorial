# Blog Post 8: LLM-as-Judge Self-Hosted – Empirische Evaluation der Grenzen und Möglichkeiten

**Lesezeit:** ~20 Minuten | **Level:** Intermediate  
**Serie:** Self-Hosted LLMs für Datensouveränität | **Code:** [GitHub](https://github.com/hanasobi/self-hosted-llms-tutorial)

> **Hinweis:** Detaillierte Definitionen zu technischen Begriffen findest du im [Glossar](./glossar-self-hosted-llms.md).

In Post 7.2 haben wir Llama-3.1-8B als Dataset-Generator evaluiert: 93.3% A-Quality, kein Halluzinationen, produktionsreif für Batch-Anwendungen. Die Quality-Bewertung erfolgte jedoch mit Claude – unsere Daten verließen die Infrastruktur. Die Data Sovereignty Pipeline hatte eine Lücke.

**Jetzt schließen wir diese Lücke:** Ein self-hosted Llama-3.1-8B Model als LLM-as-Judge evaluiert generierte QA-Pairs. Die entscheidende Frage: Wie gut funktioniert ein 8B Model im Vergleich zu Claude Sonnet 4.5?

---

## TL;DR – Für eilige Leser

**Setup:** Llama-3.1-8B (self-hosted) vs. Claude Sonnet 4.5 (commercial), 171 identische Samples, gleicher Judge-Prompt

**Ergebnisse:**

| Judge | A-Ratings | B-Ratings | C-Ratings | Agreement | Kappa |
|-------|-----------|-----------|-----------|-----------|-------|
| **Claude Sonnet 4.5** | 111 (64.9%) | 50 (29.2%) | 10 (5.8%) | - | - |
| **Llama-3.1-8B** | 167 (97.7%) | 3 (1.8%) | 1 (0.6%) | 65% | 0.30 |

**Kernerkenntnisse:**
- Data Sovereignty machbar: Komplette Pipeline self-hosted
- 8B Models zu optimistisch: Übersehen 90% der B-Ratings, 90% der C-Ratings
- Use-Case abhängig: Pre-Filtering OK, Final Quality Control nicht
- Empirisch validiert: 171 Samples zeigen wahre Performance (für unseren Use Case)
- Framework ready: Llama-70B Test in Post 8.1 möglich

Kein Drop-in Replacement für Claude – aber wir wissen jetzt, wo die Grenzen liegen und für welche Use Cases ein 8B Judge trotzdem nützlich ist.

**Empfehlung:** 8B Judge für Development & Pre-Filtering. Commercial Judge oder größeres Model (70B+) für Final Quality Control.

---

## Inhaltsverzeichnis

- [Intro: Data Sovereignty – Mission Accomplished](#intro-data-sovereignty--mission-accomplished)
- [Warum LLM-as-Judge?](#warum-llm-as-judge)
- [Experiment Setup](#experiment-setup)
- [Der entscheidende Test: Llama vs. Claude](#der-entscheidende-test-llama-vs-claude)
- [Use-Case Analyse: Wofür reicht ein 8B Judge?](#use-case-analyse-wofür-reicht-ein-8b-judge)
- [Self-Evaluation Bias](#self-evaluation-bias)
- [Code & Resources](#code--resources)
- [Fazit](#fazit)

---

## Intro: Data Sovereignty – Mission Accomplished

In den vorherigen Posts haben wir eine komplette self-hosted LLM-Pipeline aufgebaut:
- **Post 1-2:** vLLM Serving auf Kubernetes
- **Post 3-5:** LoRA Fine-Tuning mit MLflow Tracking
- **Post 6:** LoRA Adapter Serving
- **Post 7-7.2:** Self-hosted Dataset Generation mit Quality Evaluation

Aber: In Post 7.2 haben wir **Claude** für Quality Evaluation verwendet. Das bedeutet unsere Daten verlassen trotzdem die eigene Infrastruktur – ein Bruch in der Data Sovereignty Chain.

**Mit Post 8 schließen wir die Lücke:** Ein self-hosted Llama-3.1-8B Model evaluiert die Qualität generierter QA-Pairs. Keine Daten verlassen mehr die eigene Infrastruktur.

**Die entscheidende Frage:** Wie gut funktioniert ein 8B Model als Judge im Vergleich zu Claude Sonnet 4.5? Wir beantworten diese Frage nicht spekulativ, sondern **empirisch** mit 171 evaluierten Samples und klaren Metriken.

**Dieser Post:**
- Implementiert Llama-3.1-8B als LLM-as-Judge
- Vergleicht systematisch mit Claude Sonnet 4.5 (gleiches Setup, gleicher Prompt)
- Quantifiziert Grenzen und Möglichkeiten eines 8B Models
- Gibt pragmatische Empfehlungen für Production Use Cases
- Schafft Framework für Tests mit größeren Models (70B+ in Post 8.1)

**Spoiler:** Ein 8B Model ist kein Drop-in Replacement für Claude - aber die Erkenntnisse sind wertvoll für fundierte Architektur-Entscheidungen.

---

## Warum LLM-as-Judge?

### Das Problem mit manueller Evaluation

Ein manueller Quality Review skaliert nicht:
- Post 7.2: 171 QA-Pairs manuell reviewt (mehrere Stunden/Tage Arbeit)
- Bei 1000+ Samples: Nicht praktikabel
- Subjektive Bewertung, Ermüdung, Inkonsistenz

### LLM-as-Judge als Lösung

**Vorteile:**
- Skaliert auf tausende Samples
- Konsistente Bewertungskriterien
- Automatisierbar in ML-Pipeline
- Kosteneffizient (self-hosted)

**Trade-offs:**
- Nicht perfekt (auch LLMs machen Fehler)
- Model-Größe beeinflusst Qualität
- Braucht guten Prompt und klare Kriterien

### Warum selbst hosten?

**Data Sovereignty:**
- QA-Pairs können sensible Informationen enthalten
- Training-Daten dürfen Infrastruktur nicht verlassen
- Compliance-Anforderungen (DSGVO, Betriebsgeheimnisse)

**Cost:**
- Bei tausenden Evaluierungen wird self-hosted deutlich günstiger.
- Keine API-Kosten pro Request

**Kontrolle:**
- Kein Vendor Lock-in (Model und Prompt selbst kontrolliert)
- Reproduzierbarkeit garantiert (kein Backend-Update ändert Verhalten - gleiches Prompt, gleiches Modell)
- Keine Rate Limits oder unerwartete API-Änderungen

---

## Experiment Setup

### Der Judge-Prompt

Unser Judge bewertet QA-Pairs anhand von drei Texten:
- **Chunk:** Der Quelltext (Ground Truth)
- **Question:** Die generierte Frage
- **Answer:** Die generierte Antwort

**Rating-Kriterien:**

```
RATING A (Perfect):
- Faktisch korrekt und vollständig aus Chunk ableitbar
- Natürliche, hilfreiche Antwort
- Keine Spekulation oder Halluzination

RATING B (Minor Issues):
- Generell korrekt und nutzbar
- Minor Speculation: Logische Implications nicht explizit im Chunk
- Leicht awkward phrasing oder Unvollständigkeit

RATING C (Problematic):
- Faktisch falsch oder widersprüchlich
- HALLUCINATION: Erfundene Fakten (Daten, Zahlen, Namen)
- Nicht aus Chunk ableitbar
```

**Wichtig:** Der Prompt unterscheidet klar zwischen:
- **Speculation (→ B):** "makes deployment easier" wenn Chunk "small file size" sagt
- **Hallucination (→ C):** "Released 2023 by UC Berkeley" ohne jegliche Basis im Chunk

Vollständiger Prompt: [judge_prompt.txt im Repository]

**Anmerkung:** Man könnte und sollte den Scope der Evaluierung noch erweitern, z.B.: Passen die Fragen und der Questions Type (factual, comparison und conceptual) zu einander?

### Warum Llama-3.1-8B?

**Pragmatische Wahl basierend auf Post 7.2:**
- Llama-3.1-8B erreichte **93.3% A-Quality** bei Dataset Generation
- Verfügbar als AWQ-quantisiertes Model (schnelles Inference)
- Passt auf L4 GPU mit 24GB VRAM
- Schon deployed in unserer vLLM-Instanz

**Alternative:** Mistral-7B (nur 90% A-Quality in Post 7.2) → nicht gewählt

### Evaluation-Methodik

**Ansatz:**

Das Ziel ist ein **fairer Vergleich** zwischen self-hosted und commercial Judge:
- Gleicher System Prompt
- Gleiche 171 Samples
- Gleiches Setup (Chunk + Question + Answer)
- Systematische Metrics (Agreement, Cohen's Kappa)

**Warum nur Llama vs Claude?**
- Claude Sonnet 4.5: State-of-the-art commercial baseline
- Llama-3.1-8B: Praktisches self-hosted Model
- Direkter Vergleich zeigt realistisches Performance-Gap

**171 Samples Breakdown:**
- 19 chunks aus Post 7.2
- 3 QA-Pairs pro chunk
- 3 Generator-Models: Mistral-7B, Llama-3.1-8B, GPT-4o-mini
- Jeder Judge bewertet alle 171 mit identischem Prompt

**Metrics:**
- Agreement Rate (% übereinstimmende Ratings)
- Cohen's Kappa (Agreement beyond chance)
- Per-Rating Confusion Matrix
- Disagreement Pattern Analyse

**Hinweis:** Wir hatten ursprünglich 20 Chunk Samples geplant. Die krumme Zahl von 19 ist entstanden, weil Llama und Mistral bei einigen Chunks ein fehlerhaftes JSON zurückgegeben haben. Für diese Chunks hatten wir damit keine QA-Paare.

---

## Der entscheidende Test: Llama vs. Claude

### Setup für fairen Vergleich

**Kritische Erkenntnis während der Arbeit:** Ein Vergleich ist nur fair wenn beide Judges:
- Gleichen System Prompt verwenden
- Gleiche Samples bewerten
- Gleichen Kontext haben (Chunk + Question + Answer)

**Unser Approach:**
- **171 QA-Pairs** aus Post 7.2 (19 chunks × 3 pairs × 3 models)
- **Llama-3.1-8B** (self-hosted via vLLM)
- **Claude Sonnet 4.5** (Anthropic API)
- **Identischer Judge-Prompt:** [judge_prompt.txt](https://github.com/hanasobi/self-hosted-llms-tutorial/blob/main/08-llm-as-judge/judge_prompt.txt)
- **Beide sehen:** Chunk, Question, Answer

Dieser faire A/B-Test zeigt die **wahre Performance-Differenz** zwischen 8B self-hosted und state-of-the-art commercial Model.

### Ergebnisse: Der Reality-Check

**Overall Rating Distribution:**

| Judge | A-Ratings | B-Ratings | C-Ratings | Hallucinations |
|-------|-----------|-----------|-----------|----------------|
| **Claude Sonnet 4.5** | 111 (64.9%) | 50 (29.2%) | 10 (5.8%) | 6 detected |
| **Llama-3.1-8B** | 167 (97.7%) | 3 (1.8%) | 1 (0.6%) | 0 detected |

**Auf den ersten Blick:**
- Claude: 65% A-Quality
- Llama: 98% A-Quality

**Bedeutet das Llama ist besser?** Nein! Llama ist **zu optimistisch**.

### Agreement Analyse

**Overall Statistics:**
- Agreement: **65.0%** (111/171 samples)
- Cohen's Kappa: **0.30** (Fair Agreement)

**Was bedeutet Kappa 0.30?**
- Interpretation: "Fair Agreement" (zwischen 0.2-0.4)
- Agreement beyond chance: Moderat
- **Nicht gut genug für Production Quality Gate**

**Warum nicht höher?**
- Llama übersieht systematisch B- und C-Ratings
- 33% Disagreement (60/171 samples)
- Hauptmuster: Claude findet Probleme, Llama nicht

### Konkrete Beispiele: Wo liegt Llama falsch?

**Beispiel 1: Subtiler faktischer Fehler (Claude B, Llama A)**

```
Chunk: "The free plan will be available for 6 months after account 
        creation. All Free Tier credits must be used within 12 months."

Question: "What benefits does a new AWS customer receive?"

Answer: "Credits are available for 6 months after account creation,
         and any remaining balance automatically applies to AWS bills."

Claude: B ✅
  "The chunk states the FREE PLAN is available for 6 months, 
   but CREDITS must be used within 12 months. The answer 
   incorrectly states credits are available for 6 months."

Llama: A ❌
  "The answer is strictly based on the chunk content, 
   providing accurate information about the Free Tier program."
```

**Llama übersieht den Fehler:** Antwort vermischt "6 months" (free plan) mit "12 months" (credits) - subtil aber falsch!

**Beispiel 2: Irreführende Antwort (Claude C, Llama B)**

```
Chunk: "You can choose between a free plan and a paid plan. 
        If you upgrade to a paid plan, any remaining Free Tier 
        credit balance will automatically apply to your AWS bills."

Question: "What is the difference between free and paid plan?"

Answer: "The free plan is available for 6 months, while the 
         paid plan allows you to use remaining Free Tier credits."

Claude: C ✅
  "The answer is severely incomplete and misleading. It implies 
   only paid plans can use credits, which contradicts the chunk."

Llama: B ❌
  "Generally correct but slightly awkwardly phrased."
```

**Llama bewertet als "minor issue"** was eigentlich eine irreführende Darstellung ist!

**Beispiel 3: Claude zu pedantisch (Llama A, Claude C)**

```
Chunk: "Prices are the same across all regions."

Question: "In which AWS regions are prices identical?"

Answer: "Prices are identical across all AWS regions."

Claude: C ❌
  "Adds 'AWS regions' which is not mentioned in the chunk. 
   This is a hallucination of specific context."

Llama: A ✅
  "Strictly based on chunk content and factually correct."
```

**Hier hat Llama recht!** Die Frage enthält "AWS regions" - die Answer übernimmt diese Terminologie. Das ist kein Hallucination, sondern sinnvolle Clarification im Kontext.

**→ Claude ist gelegentlich zu pedantisch (False Positive), aber Llama übersieht deutlich mehr echte Probleme (False Negatives).**

**Beispiel 4: Nicht beantwortbare Fragen (Claude A, Llama B)**

```
Chunk: "Prices are the same across all regions."

Question: "WHY is there price uniformity across different AWS regions?"

Answer: "The text states prices are the same across all regions,
         but does not provide information on the reason."

Claude: A ✅
  "Excellent answer - explicitly acknowledges that the reason 
   is not provided. Perfect example of staying strictly within 
   the bounds of the given information."

Llama: B ❌
  "Incomplete, does not provide the reason for price uniformity."
```

**Kritischer Fehler:** Llama versteht nicht dass die Frage **unanswerable** ist aus dem Chunk! Die Answer sagt ehrlich "no reason provided" - das ist **perfekt**, nicht "incomplete".

### Disagreement Patterns

**Hauptmuster (60 Disagreements):**

| Pattern | Count | Bedeutung |
|---------|-------|-----------|
| **Claude B → Llama A** | ~45 | Llama übersieht Speculation |
| **Claude C → Llama A/B** | ~8 | Llama übersieht Hallucinations |
| **Claude A → Llama B** | ~2 | Llama zu streng (selten!) |
| **Claude C → Llama A** | ~5 | Claude zu pedantisch |

**Dominantes Problem:** Llama ist **systematisch zu optimistisch**.

### Was bedeuten die Zahlen wirklich?

**Claude's 65% A-Rating:**
- Mix aus echten A-Ratings (~60%)
- Gelegentlich zu pedantisch (~5% False Positives)
- **Konservative, aber akkurate Baseline**

**Llama's 98% A-Rating:**
- Viele echte A-Ratings (~60%)
- Viele übersehene B-Ratings (~30% False Negatives)
- Einige übersehene C-Ratings (~5% False Negatives)
- **Stark zu optimistisch**

**Detection Rates im Detail:**

Von den **60 problematischen Samples** die Claude fand:
- **50 B-Ratings (Speculation):** Llama fand 3 → **6% Detection Rate** (übersieht 94%)
- **10 C-Ratings (Hallucinations):** Llama fand 1 → **10% Detection Rate** (übersieht 90%)

**Das ist signifikant!** Ein 8B Model findet nur einen Bruchteil der Probleme die Claude findet.

**Ground Truth** liegt vielleicht bei ~**70-75% A-Quality** (Claude minus einige False Positives).

### Model-Größe spielt eine Rolle

**Warum ist Llama schwächer?**

**Claude Sonnet 4.5:**
- Größe: Nicht öffentlich, wahrscheinlich **100B+ Parameter**
- Released: Oktober 2024 (neueste Techniken)
- Training: Massive Datenmengen

**Llama-3.1-8B:**
- Größe: **8 Billion Parameter** (12.5× kleiner minimum)
- Quantisiert: AWQ-INT4 (zusätzlich komprimiert)
- Released: Juli 2024

**Das ist nicht überraschend - es ist physikalisch erwartbar:**
- Kleinere Models haben weniger Kapazität für nuanciertes Reasoning
- Subtile Fehler erkennen braucht mehr "Denkleistung"
- "Unanswerable questions" Konzept erfordert Meta-Reasoning

**Die Frage ist nicht "Warum ist Llama schlechter?"**
**Sondern: "Ist ein 8B Model gut genug für praktische Use Cases?"**

---

## Use-Case Analyse: Wofür reicht ein 8B Judge?

### Komplexität der Aufgabe

**LLM-as-Judge ist nicht trivial:**
- 3 Texte in Beziehung setzen (Chunk ↔ Question ↔ Answer)
- 3-stufiges Rating (A/B/C)
- Nuancen erkennen (Hallucination vs. Speculation vs. Perfect)

**Llama-3.1-8B erreicht Kappa 0.30 (Fair Agreement) - das zeigt die Grenzen eines 8B Models bei dieser komplexen Task.**

### Use Case Matrix

| Use Case | Anforderung | Claude Baseline | Llama-3.1-8B | Detection Rate | Empfehlung |
|----------|-------------|-----------------|-------------|----------------|------------|
| **C-Filtering** | Halluzinations aussortieren | 10 C gefunden | ~1 C gefunden | **10%** | ⚠️ Sehr begrenzt |
| **B-Detection** | Speculation erkennen | 50 B gefunden | ~3 B gefunden | **6%** | ❌ Nicht geeignet |
| **A-Acceptance** | Perfekte Samples durchlassen | 111 A | ~167 A | **50 False Positives** | ❌ Zu optimistisch |
| **Pre-Filtering** | Offensichtliche Fehler filtern | Baseline | Funktioniert teilweise | ~20-30% | ⚠️ Mit Vorbehalt |

### Use Case 1: Pre-Filtering großer Datasets

**Empfehlung:** ⚠️ **Mit Vorbehalt geeignet**

Ein 8B Judge kann als erste Filterung bei sehr großen Datasets (>10k samples) dienen, findet aber nur ~10-20% der problematischen Samples. MUSS gefolgt werden von zweiter Review-Stufe (Commercial Judge oder Manual Review). Nicht als einziger Quality Gate verwenden.

**Vorteil:** Kostenlos, schnell, skaliert gut  
**Nachteil:** Übersieht 80-90% der subtilen Fehler

### Use Case 2: Final Quality Control

**Empfehlung:** ❌ **Nicht geeignet**

Zu viele Probleme werden übersehen (90% der B-Ratings, 90% der C-Ratings). Keine verlässliche Quality-Gating möglich. Für Training-Daten unbrauchbar, für Production inakzeptabel.

### Use Case 3: Development & Iteration

**Empfehlung:** ✅ **Gut geeignet**

Für Prompt-Entwicklung und schnelle Iteration während Development ist ein 8B Judge nützlich: Schnelles Feedback zu offensichtlichsten Problemen, kostenlos, gut genug für iterative Verbesserung. Final Validation vor Production sollte mit Claude/GPT-4o erfolgen.

### Wann ein 8B Judge NICHT geeignet ist

**Ein 8B Model ist nicht geeignet für:**
- Mission-Critical QA (Medical, Legal, Finance)
- Small Datasets (<500 samples) → Claude direkt günstiger
- Final Production Quality Gate
- High-Stakes Applications (Safety, Compliance)

**Besser Commercial Judge oder größeres Model verwenden wenn:**
- Kosten vertretbar (<$500)
- Perfect quality kritisch
- Dataset klein genug für manuelle Review
- Regulatory Requirements (auditierbare Evaluation)

**Note:** Diese Limitationen gelten für 8B Models. Größere self-hosted Models (70B+) könnten signifikant bessere Performance zeigen - siehe Post 8.1 Ausblick.

---

## Self-Evaluation Bias

### Testet Llama seine eigenen Outputs nachsichtig?

**Die Frage:** Ist Llama "zu nett" zu Samples die es selbst generiert hat?

**Die kurze Antwort:** Wahrscheinlich nicht signifikant.

**Warum nicht:**

**1. Judge-Prompt ist objektiv:**
- Klare A/B/C Kriterien
- "Is information in chunk?" = factual check
- Nicht subjektiv wie "is this good writing?"

**2. Llama sieht keine Generator-Info:**
- Kein Kontext über welches Model die Answer generiert hat
- Nur: Chunk + Question + Answer
- Keine Möglichkeit "eigene" Samples zu erkennen

**3. Llama ist generell zu optimistisch:**
- 97.7% A-Ratings (unabhängig vom Generator)
- Das ist ein **systematisches** Problem, kein Self-Bias
- Auch bei Claude- oder GPT-generierten Samples zu nachsichtig

### Scope Note

Dieser Post fokussiert auf **answerable questions mit vorhandenem Ground Truth (Chunk)**.

**Nicht systematisch getestet:**
- Adversarial samples (absichtlich misleading)
- Completely nonsense answers
- Edge cases außerhalb normaler QA-Generation

Future Work: Robustness-Testing gegen solche Cases.

---

## Code & Resources

**GitHub Repository:** [self-hosted-llms-tutorial](https://github.com/hanasobi/self-hosted-llms-tutorial)

**Scripts für diesen Post:**
- [`judge_prompt.txt`](https://github.com/hanasobi/self-hosted-llms-tutorial/blob/main/08-llm-as-judge/judge_prompt.txt) - System prompt for judge
- [`llm_as_judge_comparison.py`](https://github.com/hanasobi/self-hosted-llms-tutorial/blob/main/08-llm-as-judge/llm_as_judge_comparison.py) - Multi-judge comparison (Llama + Claude)
- [`compare_judges.py`](https://github.com/hanasobi/self-hosted-llms-tutorial/blob/main/08-llm-as-judge/compare_judges.py) - Agreement analysis & Cohen's Kappa
- [`extract_post8_samples.py`](https://github.com/hanasobi/self-hosted-llms-tutorial/blob/main/08-llm-as-judge/extract_post8_samples.py) - Sample extraction from Post 7.2

### Repository Structure

```
08-llm-as-judge/
├── judge_prompt.txt              # System prompt for judge
├── llm_as_judge_comparison.py   # Multi-judge comparison
├── compare_judges.py             # Agreement analysis
├── extract_post8_samples.py     # Sample extraction from Post 7.2
├── test_samples.jsonl            # 8 curated test samples
└── all_samples_60x3.jsonl        # 171 QA-Pairs for evaluation (60 chunks × 3 models)
```

### Judge Prompt

**Vollständiger Prompt:** [`judge_prompt.txt`](https://github.com/hanasobi/self-hosted-llms-tutorial/blob/main/08-llm-as-judge/judge_prompt.txt)

**Key Sections:**
- Rating Kriterien (A/B/C)
- Hallucination vs Speculation Unterscheidung
- Konkrete Beispiele
- JSON Output Format

### Running the Judge

**Single Judge (Llama):**
```bash
python llm_as_judge.py \
  --samples qa_pairs.jsonl \
  --output ratings.jsonl \
  --vllm-url http://localhost:8000/v1 \
  --model hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4
```

**Multi-Judge Comparison:**
```bash
python llm_as_judge_comparison.py \
  --samples all_samples_60x3.jsonl \ 
  --output comparison.jsonl \
  --llama-url http://localhost:8000/v1 \
  --gpt4o --gpt4o-key $OPENAI_API_KEY \
  --claude --claude-key $ANTHROPIC_API_KEY
```

**Agreement Analysis:**
```bash
python compare_judges.py \
  --llama llama_ratings.jsonl \
  --claude claude_ratings.jsonl
```

### Required Input Format

```json
{
  "chunk_id": "amazon-faq-6",
  "chunk": "With Amazon MQ, you pay only for what you use...",
  "question": "What are the charges for Amazon MQ?",
  "answer": "You are charged for broker instance usage, storage usage, and data transfer fees.",
  "model": "mistral-7b"
}
```

**Kritisch:** `chunk` Feld MUSS gefüllt sein, sonst kann Judge nicht bewerten!

### Output Format

```json
{
  "chunk_id": "amazon-faq-6",
  "model": "mistral-7b",
  "chunk": "...",
  "question": "...",
  "answer": "...",
  "llama_rating": "A",
  "llama_hallucination": false,
  "llama_reasoning": "The answer is strictly based on the chunk content, providing a clear explanation of Amazon MQ charges."
}
```

---

## Fazit

Ein 8B Model als self-hosted Judge erreicht Fair Agreement (Kappa 0.30, 65% Übereinstimmung) mit Claude Sonnet 4.5 - **ehrlich, aber limitiert**.

**Was funktioniert:**
- Development & Iteration: Schnelles Feedback für Prompt-Entwicklung
- Pre-Filtering: Erste Stufe bei großen Datasets (mit Vorbehalt)
- Data Sovereignty: Komplette Pipeline self-hosted
- Cost: Praktisch kostenlos* vs. API-Kosten 

*Wenn GPU vorhanden oder mit scale-to-zero

**Die klaren Grenzen:**
- B-Detection: 6% (übersieht 94% der Speculation)
- C-Detection: 10% (übersieht 90% der Hallucinations)
- Final Quality Gate: Zu viele False Negatives
- Mission-Critical: Nicht geeignet (Medical, Legal, Finance)

**Was wir gelernt haben:**

1. **Model-Größe spielt eine Rolle** - 8B vs 100B+ ist physikalisch erwartbar unterschiedlich
2. **Komplexe Tasks brauchen Kapazität** - 3-Text-Reasoning mit Nuancen überfordert 8B
3. **Use-Case fit ist entscheidend** - Nicht "gut/schlecht" sondern "wofür gut genug?"
4. **Empirisch validieren** - 171 Samples zeigen wahre Performance, nicht 8 Test-Samples

**Der nächste Schritt:**

Die Frage bleibt: Wie viel besser ist Llama-3.1-70B (8.75× größer)? Das Framework steht, Scripts sind ready.

**Post 8.1 (geplant):** Llama-70B Judge Evaluation - kann ein größeres self-hosted Model mit Claude mithalten?

---

**Wichtigste Erkenntnis:**  
> "8B Models haben klare Grenzen bei nuancierten Tasks. Aber wir wissen jetzt, wo diese Grenzen liegen - das ist ein Fortschritt, kein Scheitern."

Data Sovereignty ist machbar - mit bewussten Trade-offs und dem richtigen Model für den Use Case.

---

{% include blog_nav.html current="08-llm-as-judge" %}
