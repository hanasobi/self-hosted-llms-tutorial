# Blog Post 3: Warum Fine-tuning? Wenn RAG und Prompting nicht reichen

**Lesezeit:** ~12 Minuten | **Level:** Einsteiger-Intermediate  
**Serie:** Self-Hosted LLMs für Datensouveränität | **Code:** [GitHub](https://github.com/hanasobi/self-hosted-llms-tutorial.git)

---

## TL;DR – Für eilige Leser

**Die Ausgangslage:** Dein LLM läuft ([Post 2](./02-vllm-kubernetes-basics.md)), aber das Verhalten ist inkonsistent. Prompting allein reicht nicht.

**Die drei Ansätze:**
- **Prompting:** Flexibel, aber variable Ergebnisse
- **RAG:** Fügt Wissen hinzu, aber keine Verhaltensgarantien
- **Fine-tuning:** Brennt konsistentes Verhalten ein

**Wann Fine-tuning?**
- Spezifische Output-Formate (JSON, strukturierte Daten)
- 100% Regeleinhaltung erforderlich
- Latenz-Optimierung durch kürzere Prompts
- Kleinere, spezialisierte Modelle statt große Generalisten

**Unser Ansatz:** Instruction Fine-tuning mit QLoRA – ressourcenschonend und kombinierbar mit RAG.

---

## Inhaltsverzeichnis

- [Das Problem: Dein LLM läuft, aber...](#das-problem-dein-llm-läuft-aber)
- [Die drei Ansätze im Vergleich](#die-drei-ansätze-im-vergleich)
- [Wann lohnt sich Fine-tuning?](#wann-lohnt-sich-fine-tuning)
- [Wann reicht Prompting oder RAG?](#wann-reicht-prompting-oder-rag)
- [Unser Use Case: AWS Documentation Q&A](#unser-use-case-aws-documentation-qa)
- [Der Fine-tuning-Ansatz: LoRA und Instruction Tuning](#der-fine-tuning-ansatz-lora-und-instruction-tuning)
- [Fazit](#fazit)

---

## Das Problem: Dein LLM läuft, aber...

In Post 2 hast du dein erstes selbst gehostetes LLM deployed – Mistral-7B auf Kubernetes mit vLLM. Du kannst es über eine OpenAI-kompatible API ansprechen. Die Infrastruktur steht. Und jetzt?

Die ersten Tests sind vielversprechend. Das Modell antwortet, versteht Fragen, generiert Text. Aber je mehr du es für deinen spezifischen Use Case einsetzt, desto mehr Probleme tauchen auf.

**Inkonsistentes Verhalten trotz sorgfältigem Prompting.** Du hast einen detaillierten System-Prompt geschrieben. Das Modell soll Fragen zu deiner Dokumentation beantworten und dabei nur Informationen aus dem bereitgestellten Kontext verwenden. Meistens funktioniert das. Aber manchmal halluziniert das Modell trotzdem – erfindet Fakten, die nicht im Kontext stehen. Oder es ignoriert deine Formatvorgaben. Der gleiche Prompt, die gleiche Frage, unterschiedliche Ergebnisse.

**RAG hilft, aber löst nicht alles.** Du hast Retrieval-Augmented Generation implementiert. Relevante Dokumente werden dem Prompt hinzugefügt. Das Wissen ist jetzt da. Aber das Modell nutzt es nicht immer korrekt. Es vermischt Kontext-Informationen mit seinem Vorwissen. Es hält sich nicht an deine Anweisung, nur aus dem Kontext zu antworten.

**Spezifische Anforderungen werden nicht zuverlässig erfüllt.** Dein Use Case erfordert ein bestimmtes Antwortformat – etwa JSON mit definierten Feldern. Oder Antworten in einem bestimmten Stil. Oder die strikte Einhaltung von Regeln. Das Modell schafft das in 70% der Fälle. Aber 70% reicht nicht für Produktion.

Das sind keine Bugs. Das ist das erwartete Verhalten eines generischen Modells, das nicht für deinen spezifischen Use Case trainiert wurde.

---

## Die drei Ansätze im Vergleich

Es gibt drei grundlegende Ansätze, um LLM-Verhalten zu steuern. Jeder hat seine Stärken und Grenzen.

**Prompting** ist der einfachste Ansatz. Du nutzt ein vortrainiertes Modell und steuerst sein Verhalten durch sorgfältig formulierte Anweisungen. Das funktioniert überraschend gut für viele Aufgaben – besonders wenn die Anforderungen nicht zu spezifisch sind und das Modell genug Allgemeinwissen mitbringt. Der Nachteil: Die Ergebnisse sind variabel. Das gleiche Prompt kann zu unterschiedlichen Outputs führen. Und je komplexer deine Anforderungen, desto länger und fragiler wird der Prompt.

**Retrieval-Augmented Generation (RAG)** erweitert diesen Ansatz. Statt alles im Prompt zu erklären, fügst du relevante Dokumente als Kontext hinzu. Das Modell kann so auf Wissen zugreifen, das nicht im Training enthalten war – deine interne Dokumentation, aktuelle Informationen, domänenspezifisches Wissen. RAG löst das Wissensproblem, aber nicht das Verhaltensproblem. Das Modell weiß jetzt mehr, aber ob es dieses Wissen korrekt nutzt, ist eine andere Frage.

**Fine-tuning** geht einen Schritt weiter. Du trainierst das Modell auf Beispielen, die zeigen, wie es sich verhalten soll. Das Modell lernt nicht nur Wissen, sondern Patterns – wie es antworten soll, in welchem Format, nach welchen Regeln. Diese Patterns werden "eingebrannt" und sind dadurch konsistenter als Prompt-Instruktionen.

**Vergleichstabelle:**

| Aspekt | Prompting | RAG | Fine-tuning |
|--------|-----------|-----|-------------|
| **Aufwand** | Gering | Mittel | Hoch |
| **Neues Wissen hinzufügen** | Begrenzt (Context Window) | ✅ Stark | ❌ Aufwändig |
| **Konsistentes Verhalten** | ❌ Variabel | ❌ Variabel | ✅ Stark |
| **Spezifische Formate** | Möglich, aber fragil | Möglich, aber fragil | ✅ Robust |
| **Latenz** | Basis | Höher (Retrieval + längerer Prompt) | Niedriger (kürzere Prompts möglich) |
| **Aktualisierung** | Sofort | Schnell (Index update) | Langsam (Retraining) |

Die wichtigste Erkenntnis: Diese Ansätze schließen sich nicht gegenseitig aus. RAG und Fine-tuning lassen sich hervorragend kombinieren – RAG liefert das aktuelle Domänenwissen, Fine-tuning sorgt für konsistentes Antwortverhalten. Genau diese Kombination zeigen wir in dieser Tutorial-Serie.

---

## Wann lohnt sich Fine-tuning?

Fine-tuning ist Aufwand. Du brauchst Trainingsdaten, Compute-Ressourcen, Expertise. Das lohnt sich nicht für jeden Use Case. Aber es gibt klare Indikatoren, wann Fine-tuning der richtige Weg ist.

**Konsistentes Output-Format** ist ein starkes Argument. Wenn dein Modell zuverlässig JSON in einem bestimmten Schema ausgeben soll, oder immer einem bestimmten Antwortmuster folgen soll, ist Fine-tuning oft der robustere Weg. Prompting kann das auch erreichen, aber Fine-tuning macht es konsistenter. In Produktion zählt nicht der Durchschnitt, sondern der schlechteste Fall.

**Strikte Regeleinhaltung** ist ein weiterer Grund. "Antworte nur basierend auf dem bereitgestellten Kontext" – das klingt einfach, ist aber schwer durchzusetzen. Ein fine-tunetes Modell, das auf tausenden Beispielen gelernt hat, wie "nur aus dem Kontext antworten" aussieht, hält sich zuverlässiger daran als ein generisches Modell mit Prompt-Instruktion.

**Domänenspezifische Patterns** profitieren von Fine-tuning. Wenn das Modell bestimmte Fachbegriffe, Abkürzungen oder Zusammenhänge verstehen soll, die es aus dem allgemeinen Training nicht kennt, hilft Fine-tuning. RAG kann Wissen hinzufügen, aber Fine-tuning verändert, wie das Modell dieses Wissen verarbeitet.

**Latenzanforderungen** spielen ebenfalls eine Rolle. Ein fine-tunetes Modell braucht keinen langen System-Prompt, um Verhalten und Format zu erzwingen – das ist bereits trainiert. Selbst in Kombination mit RAG spart das Tokens und damit Latenz. Bei zeitkritischen Anwendungen kann das den Unterschied machen.

**Kostenoptimierung bei hohem Volumen** ist ein wirtschaftlicher Aspekt. Wenn du Millionen von Anfragen pro Monat hast, summieren sich die Kosten für lange Prompts und RAG-Kontexte. Ein fine-tunetes Modell, das mit kürzeren Inputs auskommt, kann langfristig günstiger sein.

**Kleinere Modelle mit Spezialisierung** statt großer Generalisten sind ein oft unterschätzter Vorteil. Ein fine-tunetes 7B-Modell kann bei einer spezifischen Aufgabe mit deutlich größeren Modellen mithalten oder sie sogar übertreffen. In unserem Tutorial erreicht das fine-tuned Mistral-7B eine deutlich höhere Erfolgsrate bei der RAG-QA-Aufgabe als das Base Model. Der Trade-off: Das kleine Modell ist ein Spezialist, kein Generalist. Für viele Enterprise-Use-Cases ist das aber genau richtig. Die praktischen Implikationen sind erheblich: weniger VRAM bedeutet günstigere GPU-Hardware, weniger Parameter bedeuten schnellere Inference und niedrigere Latenz.

> **💡 Praxis-Beispiele aus dem DACH-Markt:**
> - **Compliance-Bot:** Versicherung braucht 100% regelkonforme Antworten zu DSGVO-Anfragen → Fine-tuning für strikte Kontexttreue
> - **Ticket-Routing:** Support-System muss Anfragen als JSON mit `{priority, category, department}` klassifizieren → Fine-tuning für konsistentes Format
> - **Echtzeit-Übersetzung:** Meeting-Tool braucht <500ms Latenz → Fine-tuning erlaubt kürzere Prompts, kleineres Modell

---

## Wann reicht Prompting oder RAG?

Fine-tuning ist nicht immer die richtige Antwort. Es gibt Szenarien, in denen einfachere Ansätze ausreichen.

**In der explorativen Phase** starte mit Prompting. Wenn du noch nicht genau weißt, was du brauchst, ist es schneller zu iterieren. Du lernst, was funktioniert und was nicht, bevor du in Fine-tuning investierst.

**Wenn ein passendes Instruct Model existiert.** Wenn ein bestehendes Open-Source-Modell – etwa Mistral Instruct, Llama Chat oder Qwen – die Aufgabe bereits gut erledigt, lohnt sich Fine-tuning möglicherweise nicht. Hier ist die Frage: Passen die Hosting-Kosten? Ein 70B-Modell braucht deutlich mehr GPU-Ressourcen als ein fine-tunetes 7B-Modell, das die gleiche Aufgabe spezialisiert löst. Aber wenn das größere Modell ohne zusätzlichen Trainingsaufwand funktioniert, kann das der pragmatischere Weg sein.

**Bei geringem Anfragevolumen** überwiegen die Kosten für Fine-tuning möglicherweise nicht den Nutzen. Zeit, Compute, Expertise – das alles hat seinen Preis. Wenn du nur hundert Anfragen pro Tag hast, rechnet sich das selten.

---

## Unser Use Case: AWS Documentation Q&A

Um die Konzepte praktisch zu zeigen, brauchen wir einen konkreten Use Case. Wir haben uns für ein RAG-QA-System für AWS-Dokumentation entschieden.

**Das Szenario:** Ein Benutzer stellt Fragen zu AWS-Services. Das System ruft relevante Dokumentations-Abschnitte ab (RAG) und generiert eine Antwort basierend auf diesem Kontext.

**Warum dieses Beispiel?**

Es ist realistisch. Dokumentations-Assistenten sind einer der häufigsten LLM-Use-Cases in Unternehmen. Die Herausforderungen – Halluzination vermeiden, im Kontext bleiben, konsistentes Format – sind universell.

Es ist reproduzierbar. AWS-Dokumentation ist öffentlich verfügbar. Du kannst das komplette Tutorial nachvollziehen, ohne eigene sensible Daten zu verwenden.

Es zeigt die Kombination von RAG und Fine-tuning. Das Wissen kommt aus dem Retrieval (die Dokumentations-Chunks). Das Verhalten – nur aus dem Kontext antworten, bestimmtes Format einhalten – kommt aus dem Fine-tuning.

**Was wollen wir erreichen?**

Das fine-tunete Modell soll lernen, Fragen ausschließlich basierend auf dem bereitgestellten Kontext zu beantworten. Wenn die Antwort nicht im Kontext steht, soll es das sagen – nicht halluzinieren. Das Format soll konsistent sein. Die Antworten sollen präzise und hilfreich sein.

Das Base Model (Mistral-7B ohne Fine-tuning) erreicht bei dieser Aufgabe eine Erfolgsrate von weniger als 50%. Das fine-tuned Modell erreicht nahezu 100%. Dieser Unterschied zeigt den Wert von Fine-tuning für spezifische Verhaltensanforderungen.

---

## Der Fine-tuning-Ansatz: LoRA und Instruction Tuning

Es gibt verschiedene Wege, ein LLM anzupassen. Für die meisten praktischen Use Cases ist die Kombination aus **LoRA** und **Instruction Fine-tuning** der sinnvollste Ansatz.

**Continued Pre-training vs. Instruction Fine-tuning**

Continued Pre-training trainiert das Modell auf großen Mengen Domänen-Text, um neues Wissen zu verinnerlichen – Fakten, Terminologie, Zusammenhänge. Das braucht viel Daten (Gigabytes bis Terabytes) und erhebliche Compute-Ressourcen. Für die meisten Unternehmens-Use-Cases ist das überdimensioniert.

Instruction Fine-tuning hingegen passt das Verhalten des Modells an: Wie es antwortet, in welchem Format, mit welchem Stil. Du trainierst auf Beispielen von Input-Output-Paaren, die das gewünschte Verhalten demonstrieren. Das ist ressourcenschonender und für die meisten Use Cases der praktikablere Weg – besonders in Kombination mit RAG, das das Domänenwissen liefert.

Diese Tutorial-Serie fokussiert auf Instruction Fine-tuning.

**Full Fine-tuning vs. LoRA**

Bei Full Fine-tuning werden alle Parameter des Modells angepasst. Das erfordert viel GPU-Speicher (das gesamte Modell muss im VRAM liegen, plus Optimizer States) und produziert ein komplett neues Modell, das genauso groß ist wie das Original.

**LoRA (Low-Rank Adaptation)** ist ein effizienter Ansatz. Statt alle Parameter zu ändern, werden kleine "Adapter" trainiert – zusätzliche Gewichte, die auf das Base Model aufgesetzt werden. Das hat mehrere Vorteile: deutlich weniger Speicherbedarf beim Training, die Adapter sind nur wenige Megabytes groß (statt Gigabytes für ein volles Modell), und du kannst mehrere Adapter für verschiedene Tasks auf einem Base Model betreiben.

**QLoRA** geht noch einen Schritt weiter: Das Base Model wird quantisiert (z.B. auf 4-bit), was den Speicherbedarf nochmals reduziert. Training auf Consumer-Hardware wird damit möglich.

Für diese Tutorial-Serie nutzen wir QLoRA – das Base Model wird in 4-bit geladen, die LoRA-Adapter werden in höherer Präzision trainiert. Das ermöglicht Training von Mistral-7B auf einer einzelnen L4-GPU mit 24 GB VRAM.

**Base Model vs. Instruct Model**

Eine häufige Frage: Soll ich ein Base Model oder ein bereits instruction-tuned Modell als Ausgangspunkt nehmen?

Base Models (z.B. `mistralai/Mistral-7B-v0.1`) sind auf Text-Completion trainiert. Sie haben kein eingebautes "Assistenten-Verhalten". Der Vorteil: Du startest mit einer neutralen Basis und trainierst genau das Verhalten, das du willst.

Instruct Models (z.B. `mistralai/Mistral-7B-Instruct-v0.2`) sind bereits auf Instruktionen und Dialog trainiert. Sie verhalten sich "out of the box" wie Assistenten. Der Vorteil: Du baust auf bestehendem Verhalten auf.

Für unseren Use Case starten wir mit einem Base Model, weil wir sehr spezifisches Verhalten trainieren wollen. In anderen Szenarien kann ein Instruct Model als Basis sinnvoller sein.

---

## Fazit

Fine-tuning ist das richtige Werkzeug, wenn du konsistentes Verhalten brauchst, strikte Regeleinhaltung durchsetzen willst, oder spezialisierte Performance aus einem kleineren Modell herausholen musst. Der Schlüssel zum Erfolg liegt nicht im Training selbst, sondern in der Vorbereitung: den richtigen Ansatz wählen und vor allem hochwertige Trainingsdaten erstellen.

**Im nächsten Post** zeigen wir genau das: wie du von Rohdokumenten zu einem hochwertigen Instruction-Dataset kommst – Token-aware Chunking, synthetische QA-Generierung und automatisierte Quality Checks.

{% include blog_nav.html current="03-warum-finetuning" %}