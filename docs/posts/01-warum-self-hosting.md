# Blog Post 1: Warum Self-Hosting? Der Business Case für Datensouveränität

**Lesezeit:** ~10 Minuten | **Level:** Einsteiger  
**Serie:** Self-Hosted LLMs für Datensouveränität | **Code:** [GitHub](https://github.com/hanasobi/self-hosted-llms-tutorial)

---

## TL;DR – Für eilige Leser

**Das Problem:** Unternehmen wollen generative KI nutzen, aber sensible Daten dürfen nicht an externe APIs.

**Die Lösung:** Self-Hosted LLMs bieten volle Kontrolle über Daten und Modellverhalten.

**Diese Tutorial-Serie zeigt:**
- ✅ Von "erstes LLM deployen" bis "vollständige Datensouveränität"
- ✅ Echte Debugging-Stories statt "happy path"
- ✅ Production-Grade Stack: Kubernetes, vLLM, MLflow, Prometheus
- ✅ Schrittweiser Aufbau – nach Post 2 läuft dein erstes selbst gehostetes LLM

**Für wen?** ML Engineers, Data Scientists, Tech Leads und technische Entscheider im DACH-Raum.

---

## Inhaltsverzeichnis

- [Das Problem: KI nutzen, Daten behalten](#das-problem-ki-nutzen-daten-behalten)
- [Die Lösung: Self-Hosted LLMs](#die-lösung-self-hosted-llms)
- [Wann Self-Hosting? Die Entscheidungsmatrix](#wann-self-hosting-die-entscheidungsmatrix)
- [Infrastruktur-Voraussetzungen](#infrastruktur-voraussetzungen)
- [Was diese Tutorial-Serie bietet](#was-diese-tutorial-serie-bietet)
- [Fazit](#fazit)

---

## Das Problem: KI nutzen, Daten behalten

Generative KI ist in aller Munde. ChatGPT, Claude, Gemini – die Möglichkeiten scheinen grenzenlos. Doch wenn es um den Unternehmenseinsatz geht, stellt sich schnell eine entscheidende Frage: **Wohin gehen unsere Daten?**

Die Realität in vielen deutschen Unternehmen sieht so aus: Es gibt einen Use Case, der perfekt für ein LLM wäre – Kundenanfragen automatisiert beantworten, interne Dokumentation durchsuchbar machen, Berichte zusammenfassen. Die Fachabteilung ist begeistert. Dann kommen IT-Security und Datenschutz ins Spiel.

**Typische Hürden:**

Der erste Einwand betrifft den Datenschutz. Kundendaten, Vertragsinformationen, interne Strategiepapiere – all das darf nicht an externe Server gesendet werden. Die DSGVO schreibt vor, dass personenbezogene Daten nur mit entsprechender Rechtsgrundlage verarbeitet werden dürfen. Bei US-amerikanischen Cloud-Diensten kommt zusätzlich die Problematik des transatlantischen Datentransfers hinzu.

Dann gibt es branchenspezifische Regulierung. Finanzdienstleister unterliegen BaFin-Anforderungen zur Auslagerung von IT-Dienstleistungen. Gesundheitsunternehmen müssen besondere Anforderungen an den Schutz von Patientendaten erfüllen. Behörden und öffentliche Einrichtungen haben eigene Vorgaben zur Datenverarbeitung.

Doch es muss nicht immer Regulierung sein. Viele Unternehmen wollen schlicht ihre Betriebsgeheimnisse schützen – proprietäre Prozesse, interne Analysen, Strategiedokumente, die einen Wettbewerbsvorteil darstellen. Diese Informationen an externe APIs zu senden, fühlt sich falsch an, selbst wenn es rechtlich zulässig wäre. Und für Unternehmen aus der Kreativ- und Medienbranche kommt ein weiterer Aspekt hinzu: Sie wollen nicht, dass ihre Inhalte ungefragt oder unentgeltlich zum Training fremder Modelle verwendet werden. Die Nutzungsbedingungen vieler LLM-Anbieter erlauben genau das – oder sind zumindest nicht eindeutig genug, um das Gegenteil zu garantieren.

Schließlich sorgt sich das Management um Vendor Lock-in und Kontrollverlust. Was passiert, wenn OpenAI die Preise erhöht? Was, wenn der Anbieter das Modellverhalten ändert? Wie erklärt man Kunden, dass ihre Daten bei einem US-Unternehmen verarbeitet werden?

Das Ergebnis: Viele vielversprechende KI-Projekte versanden, bevor sie richtig starten.

---

## Die Lösung: Self-Hosted LLMs

Self-Hosted Large Language Models lösen dieses Dilemma. Du betreibst das Modell auf eigener Infrastruktur – in deinem Rechenzentrum, in einer europäischen Cloud-Region, oder auf On-Premise-Hardware. Die Daten verlassen nie deine Kontrolle.

**Was bedeutet das konkret?**

Volle Datensouveränität ist der erste Vorteil. Jede Anfrage, jede Antwort, jedes Training bleibt auf deiner Infrastruktur. Keine Daten fließen zu externen APIs. Du bestimmst, wer Zugriff hat und wie lange Daten gespeichert werden.

Compliance wird einfacher. Wenn die Datenverarbeitung vollständig unter deiner Kontrolle stattfindet, vereinfacht das die Dokumentation für Audits und Zertifizierungen erheblich. Du kannst nachweisen, wo deine Daten sind – weil sie nirgendwo anders hingehen.

Anpassbarkeit ist ein weiterer Vorteil. Durch Fine-tuning kannst du das Modell auf deinen spezifischen Use Case zuschneiden. Das Ergebnis ist oft besser als ein generisches Modell mit noch so ausgefeiltem Prompt, weil das Modell deine Domäne, dein Format, deine Anforderungen "verinnerlicht" hat. Wie Fine-tuning funktioniert und wann es sinnvoll ist, behandeln wir ausführlich in Post 3.

Langfristige Planbarkeit kommt hinzu. Die Kosten für Inference sind vorhersagbar – keine Überraschungen bei der nächsten Rechnung. Das Modellverhalten ändert sich nicht über Nacht, weil der Anbieter ein Update ausrollt.

---

## Wann Self-Hosting? Die Entscheidungsmatrix

Self-Hosting ist nicht für jeden Use Case die richtige Wahl. Die folgende Matrix hilft bei der Orientierung.

| Kriterium | Self-Hosting | Cloud-API |
|---|---|---|
| Sensible Daten im Prompt | ✅ Pflicht | ❌ Risiko |
| Regulatorische Anforderungen (BaFin, DSGVO) | ✅ Pflicht | ❌ Oft nicht möglich |
| Hohes Anfragevolumen (Mio. Tokens/Tag) | ✅ Günstiger | 💰 Teuer |
| Latenz-kritisch | ✅ Volle Kontrolle | ⚠️ Abhängig vom Anbieter |
| Exploration / Prototyping | ⚠️ Aufwändig | ✅ Schneller Start |
| Geringe Nutzung | 💰 Overhead | ✅ Pay-per-Use |
| Keine sensiblen Daten | ⚠️ Unnötiger Aufwand | ✅ Einfacher |
| Kein ML-Ops Know-how | ❌ Hohe Einstiegshürde | ✅ Managed |


**Starke Indikatoren für Self-Hosting:**

Regulatorische Anforderungen machen Self-Hosting oft zur einzigen Option. Wenn Branchenvorschriften explizit verbieten, dass bestimmte Daten an externe Dienste übermittelt werden, bleibt keine Alternative. Das betrifft häufig den Finanzsektor, das Gesundheitswesen und den öffentlichen Dienst.

Sensible Daten im Prompt sind ein weiterer klarer Indikator. Wenn dein Use Case erfordert, dass Kundendaten, Vertragsinformationen oder Geschäftsgeheimnisse an das LLM gesendet werden, ist Self-Hosting der sichere Weg. Bei Cloud-APIs verlierst du die Kontrolle darüber, was mit diesen Daten passiert.

Hohes Anfragevolumen kann Self-Hosting wirtschaftlich attraktiv machen. Ab einer gewissen Schwelle – typischerweise mehrere Millionen Tokens pro Tag – werden die API-Kosten so hoch, dass eigene Infrastruktur günstiger ist. Die genaue Schwelle hängt vom Modell und deinen Anforderungen ab.

Latenz-kritische Anwendungen profitieren davon, dass der Inference-Server im eigenen Netzwerk steht. Keine Internet-Roundtrips, keine geteilte Kapazität mit anderen Kunden, volle Kontrolle über die Hardware.

**Situationen, in denen Cloud-APIs sinnvoller sein können:**

Exploration und Prototyping gehen mit Cloud-APIs schneller. Wenn du noch nicht weißt, ob LLMs für deinen Use Case überhaupt funktionieren, ist es effizienter, erst mit einer API zu experimentieren, bevor du Infrastruktur aufbaust.

Geringe Nutzung macht Self-Hosting unwirtschaftlich. Wenn du nur wenige hundert Anfragen pro Tag hast, lohnt sich der Aufwand für eigene Infrastruktur nicht. Die Fixkosten für GPU-Hardware übersteigen die API-Kosten bei weitem.

Keine sensiblen Daten im Spiel vereinfachen die Entscheidung. Wenn dein Use Case nur mit öffentlich verfügbaren Informationen arbeitet – etwa Zusammenfassungen von Nachrichtenartikeln oder allgemeine Wissensfragen – spricht weniger gegen Cloud-APIs.

Fehlende Expertise oder Infrastruktur sind ebenfalls Faktoren. Self-Hosting erfordert Know-how in ML-Ops, Kubernetes und GPU-Management. Wenn dieses Wissen im Team fehlt und auch nicht aufgebaut werden soll, ist eine Cloud-API die pragmatischere Wahl.

**Die Grauzone:**

Viele Szenarien fallen in die Mitte. Hier hilft ein strukturierter Ansatz: Beginne mit einer Cloud-API für den Proof of Concept. Wenn der Use Case sich bewährt und einer der starken Indikatoren für Self-Hosting zutrifft, plane die Migration. Diese Tutorial-Serie gibt dir das Werkzeug dafür.

---

## Infrastruktur-Voraussetzungen

Self-Hosted LLMs stellen Anforderungen an die Infrastruktur. Hier ein Überblick über das Minimum für den Einstieg:

**Hardware:** Mindestens eine GPU mit 16 GB VRAM (z.B. NVIDIA T4) für kleinere Modelle wie Mistral-7B mit Quantisierung. Für größere Modelle oder höheren Durchsatz entsprechend mehr. Die Details zu GPU-Sizing behandeln wir in Post 2.

**Container-Orchestrierung:** Kubernetes ist der De-facto-Standard für Production-Deployments. Alternativen wie Docker Compose funktionieren für Experimente, skalieren aber nicht.

**Netzwerk:** Interne Erreichbarkeit des Inference-Servers. Je nach Use Case Load Balancer und Ingress.

**Storage:** Persistent Volumes für Model Weights (mehrere GB pro Modell) und optional für Logs und Metriken.

**Monitoring:** Prometheus und Grafana für Metriken. Nicht zwingend für den Einstieg, aber essentiell für Production.

Die genaue Konfiguration hängt von deinem Use Case ab. In den folgenden Posts zeigen wir konkrete Setups mit allen Details.

---

## Was diese Tutorial-Serie bietet

Viele LLM-Tutorials enden dort, wo die echten Herausforderungen beginnen. Diese Serie geht bewusst weiter und zeigt den vollständigen Weg von der ersten Installation bis zur vollständigen Datensouveränität.

**Unser Ansatz:**

Schneller erster Erfolg. Nach Post 2 hast du ein funktionierendes LLM auf deiner eigenen Infrastruktur. Das motiviert und gibt dir eine Basis zum Experimentieren.

Schrittweiser Aufbau. Die Serie folgt einem klaren Lernpfad: Erst verstehen, wie Self-Hosting funktioniert. Dann lernen, wie man das Modell anpasst. Schließlich alle externen Abhängigkeiten eliminieren.

Echte Debugging-Stories. Wir dokumentieren unsere Fehler und Sackgassen. Zum Beispiel haben wir 20 Stunden damit verbracht, ein EOS-Token-Problem zu debuggen. Das zeigen wir – nicht weil wir stolz darauf sind, sondern weil du daraus lernst.

Design Trade-offs. Jede Entscheidung hat Konsequenzen. Context Window → Memory → Hardware → Kosten. Wir erklären, warum wir uns wofür entschieden haben und welche Alternativen wir verworfen haben.

Der Weg zur vollständigen Datensouveränität. Zu Beginn nutzen wir OpenAI für die Dataset-Generierung – das ist pragmatisch und beschleunigt den Einstieg. Aber wir zeigen auch den Weg zur vollständigen Unabhängigkeit: Self-hosted Dataset-Generierung, Training, Inference und Evaluation ohne externe APIs.

Production-Grade Infrastructure. Kubernetes, vLLM, MLflow, Prometheus, Grafana – das sind die Tools, die in Produktion funktionieren und in vielen Unternehmen bereits im Einsatz sind.

---

## Fazit

Self-Hosted LLMs sind kein Allheilmittel. Sie erfordern Investition in Infrastruktur, Expertise und Zeit. Aber für viele Unternehmen sind sie der einzige gangbare Weg, generative KI mit sensiblen Daten zu nutzen.

Diese Tutorial-Serie gibt dir das Werkzeug, um fundierte Entscheidungen zu treffen. Nicht durch Theorie und Benchmarks, sondern durch praktische Erfahrung. Bau einen PoC auf deiner Infrastruktur, mit deinen Daten, für deinen Use Case. Dann weißt du, was funktioniert und was nicht.

**Im nächsten Post** deployen wir dein erstes selbst gehostetes LLM. Mistral-7B auf Kubernetes mit vLLM – Schritt für Schritt, bis du ein funktionierendes System hast, das du über eine OpenAI-kompatible API ansprechen kannst.


{% include blog_nav.html current="01-warum-self-hosting" %}