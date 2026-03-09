---
---
# Post 10: Resümee – Was wir über Self-Hosted LLMs gelernt haben

**Lesezeit:** ~8 Minuten | **Level:** Alle  
**Serie:** Self-Hosted LLMs für Datensouveränität | **Code:** [GitHub](https://github.com/hanasobi/self-hosted-llms-tutorial)

Dies ist der letzte Post dieser Serie. Kein neues Feature, kein neues Deployment – stattdessen ein Schritt zurück. Was haben wir in 9 Posts und mehreren Monaten über Self-Hosted LLMs gelernt? Was würden wir heute anders machen? Und was bedeutet das für Unternehmen, die diesen Weg evaluieren?

---

## Was wir gebaut haben

In dieser Serie haben wir eine vollständige LLM-Infrastruktur aufgebaut – Schritt für Schritt, mit echten Problemen und echten Lösungen.

Der Ausgangspunkt war eine einfache Frage: **Kann ein Unternehmen LLMs nutzen, ohne sensible Daten an externe APIs zu senden?**

Die Antwort nach 9 Posts: Ja. Die gesamte Pipeline – vom Deployment über Fine-tuning, Dataset-Generierung und Evaluation bis zur iterativen Verbesserung – läuft ohne einen einzigen externen API-Call. Und sie läuft nicht nur auf Cloud-GPUs, sondern bei Bedarf auch komplett lokal auf Consumer-Hardware.

Aber die interessanteren Erkenntnisse liegen nicht im *Ob*, sondern im *Wie*.

---

## Fünf Dinge, die wir gelernt haben

### 1. Datensouveränität ist kein Alles-oder-Nichts

Zu Beginn der Serie haben wir bewusst GPT-4o-mini für die Dataset-Generierung genutzt – ein pragmatischer Kompromiss, den wir transparent gemacht haben. Das war die richtige Entscheidung. Wer versucht, vom ersten Tag an alles selbst zu hosten, kommt nicht voran.

Der Weg zur Datensouveränität ist ein Prozess: Erst die kritischsten Komponenten unter eigene Kontrolle bringen (Inference, Training), dann schrittweise die restlichen Abhängigkeiten eliminieren. In Post 7 haben wir die Dataset-Generierung self-hosted, in Post 8 die Evaluation. Jeder Schritt war einzeln machbar und hat sofort Wert geliefert.

**Für Entscheider heißt das:** Man muss nicht alles auf einmal umstellen. Ein schrittweiser Ansatz reduziert Risiko und liefert früh Ergebnisse.

### 2. Drei Hebel, nicht einer

Die Qualität eines LLM-Systems hängt an drei Faktoren: den Trainingsdaten, der Modellgröße und dem Prompt Engineering. Die Gewichtung verschiebt sich je nach Aufgabe – aber alle drei sind entscheidend.

Beim Fine-tuning waren die **Daten** der größte Hebel. 200 gezielte negative Samples haben die Hallucination-Rate um 90% gesenkt – nicht ein besseres Modell, nicht mehr Rechenleistung, sondern eine gezielte Veränderung der Datenzusammensetzung. Gleichzeitig haben wir den Großteil unserer Zeit mit Daten verbracht: Dokumente parsen, Chunks erstellen, QA-Paare generieren, Qualität prüfen. Das Fine-tuning selbst war vergleichsweise unkompliziert.

Bei der Evaluation war die **Modellgröße** entscheidend. Llama-3.1-8B als Judge war zu optimistisch und übersah die meisten Qualitätsprobleme. Llama-3.1-70B war deutlich besser, erreichte aber nicht das Niveau eines Commercial-Grade Judges. Für diese Aufgabe konnten bessere Daten allein die Grenzen des Modells nicht kompensieren.

Und überall spielte **Prompt Engineering** eine größere Rolle als erwartet. Bei der Dataset-Generierung entschied der Prompt darüber, ob die negativen Samples thematisch passten oder abdrifteten. Beim LLM-as-Judge bestimmte der Prompt, wie streng oder nachsichtig die Bewertung ausfiel. Kleine Änderungen in der Formulierung hatten oft mehr Einfluss als die Wahl des Modells.

**Für Entscheider heißt das:** Es gibt keinen einzelnen Hebel, der alles löst. Aber die Investition in Datenqualität und Prompt Engineering liefert oft mehr als der Wechsel auf ein größeres Modell – und ist deutlich günstiger.

### 3. Self-Hosted heißt nicht gleich schlechter

Self-Hosted LLMs sind kein 1:1-Ersatz für GPT-4 oder Claude – aber das müssen sie auch nicht sein. Für Fine-tuning, Iteration, Pre-Filtering und Anwendungsfälle, in denen Datensouveränität Priorität hat, sind sie sehr gut einsetzbar. Man muss nur wissen, wo die Grenzen liegen, und das System entsprechend designen: etwa einen self-hosted Judge fürs Development nutzen und für die finale Qualitätskontrolle einen stärkeren Judge einsetzen.

**Für Entscheider heißt das:** Bewerten Sie Self-Hosted LLMs nicht an dem, was sie nicht können, sondern an dem, was sie für Ihren konkreten Anwendungsfall leisten. In vielen Szenarien ist "gut genug und datensouverän" besser als "State-of-the-Art und extern".

### 4. Debugging ist die eigentliche Arbeit

20 Stunden für einen EOS-Token-Bug. OOM-Errors beim ersten 70B-Deployment. 140% Performance-Verlust durch falsche GPU-Offloading-Einstellungen. Diese Geschichten sind kein Beiwerk – sie sind der Kern der Arbeit mit Self-Hosted LLMs.

Jedes Tutorial zeigt den Happy Path. Aber in der Praxis verbringt man den Großteil der Zeit damit, Dinge zum Laufen zu bringen, die laut Dokumentation längst funktionieren sollten. Das ist kein Zeichen von Inkompetenz – es ist die Realität von Infrastrukturarbeit an der Grenze des aktuell Machbaren.

**Für Entscheider heißt das:** Self-Hosting erfordert tiefes technisches Know-how und Geduld für systematisches Debugging. Planen Sie realistisch – nicht nur die GPU-Kosten, sondern vor allem die Engineering-Zeit.

### 5. Die Hardware-Landschaft verändert sich

Wir haben diese Serie mit der Annahme gestartet, dass Self-Hosted LLMs eine Cloud-GPU brauchen. Am Ende lief ein 70B-Modell auf einem Mac Studio – schneller als die Cloud (end-to-end), für null Euro, und ohne auf Instanz-Verfügbarkeit warten zu müssen.

Das ist keine Empfehlung, die Cloud abzuschaffen. Für Production-Scale Serving, verteiltes Training und Burst-Workloads bleibt Cloud-Infrastruktur die bessere Wahl. Aber für Entwicklung, Experimente und kleinere Workloads verschiebt sich die Grenze dessen, was lokal möglich ist, schnell.

**Für Entscheider heißt das:** Evaluieren Sie Self-Hosting nicht nur als Cloud-Alternative, sondern auch als lokale Option. Besonders für regulierte Umgebungen mit strikten Data-Residency-Anforderungen kann das den entscheidenden Unterschied machen.

---

## Was wir heute anders machen würden

**Früher mit negativen Samples starten.** Die Hallucination-Problematik war von Anfang an absehbar. Hätten wir von Post 4 an negative Samples im Dataset gehabt, wären mehrere Debugging-Runden überflüssig gewesen.

**Kleinere Iterationszyklen.** Statt einmal 5.800 Samples zu generieren und dann zu trainieren, würden wir mit 500 Samples starten, schnell evaluieren, und das Dataset iterativ aufbauen. Die Feedback-Schleife wäre kürzer, die Lernkurve steiler.

**Evaluation von Anfang an automatisieren.** Manuelle Evaluation war wertvoll für das Verständnis, skaliert aber nicht. Ein automatisiertes Evaluierungs-Setup ab dem ersten Training hätte uns erlaubt, Änderungen schneller zu bewerten.

---

## Fazit

Self-Hosted LLMs sind machbar, nützlich, und für bestimmte Anwendungsfälle der einzig gangbare Weg. Aber sie sind kein einfacher Ersatz für eine API-Integration. Sie erfordern Engineering-Kompetenz, realistische Erwartungen an die Modellqualität und – vor allem – Investition in die Daten.

Diese Serie hat gezeigt, dass der Weg zur Datensouveränität kein Alles-oder-Nichts-Projekt sein muss. Jeder Schritt – vom ersten Deployment über eigenes Fine-tuning bis zur self-hosted Evaluation – liefert eigenständigen Wert. Man muss nicht alles auf einmal bauen.

Danke an alle, die diese Serie begleitet haben. Der gesamte Code ist auf [GitHub](https://github.com/hanasobi/self-hosted-llms-tutorial) verfügbar – zum Nachbauen, Anpassen und Weiterentwickeln.

{% include blog_nav.html current="10-resumee" %}
