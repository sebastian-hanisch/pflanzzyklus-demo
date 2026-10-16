# Pflanzzyklus: Wie viel pflanzen, wenn der Verkauf zehn Jahre später ist? – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-pflanzzyklus-demo.streamlit.app/)**

Fall-Demo für die Website „Sebastian Hanisch – Operations Research und Machine Learning“ (`demos.html`). Ein kleiner Weihnachtsbaumhof (rund 2 000 verkaufte Bäume im Jahr) pflanzt Bäume, die erst in 7 bis 12 Jahren verkauft werden. **Akt 1** vergleicht sieben Pflanzregeln auf denselben Marktpfaden mit Pflanzverzug (Spinnweb-Modell), Dürrejahren und gestreutem Wachstum, dazu eine exakte Referenz (lineares Programm, HiGHS). **Akt 2** ist ein kurzer Hebel-Check der Saison (Hofverkauf und Wochenmarkt, 27 Tage): Wo steckt dort noch etwas zu holen?

Entstanden aus der Frage „Weihnachtsbaumplantage mit Verkauf“. Von vier erwogenen Teilfragen (Anbau über Jahre, Größen und Erntezeitpunkt, Saison mit Schlagen/Lagern/Preisen, Selbstschlag) trug in den Vorab-Messreihen nur der Anbau einen Befund; die Saison ist deshalb als kurzer Check mit ehrlichem Negativbefund enthalten, Größen/Erntezeitpunkt und Selbstschlag sind es nicht.

## Warum dieses Problem

Wer heute pflanzt, verkauft in einem Jahrzehnt. Pflanzen alle Betriebe bei hohem Preis mehr, ist der Markt in zehn Jahren überfüllt und der Preis unten; nach Jahren knapper Pflanzung folgt Knappheit (so beschreiben es Berichte zum US-Markt nach 2008). Die Kernfrage: **Lohnt es sich, die Pflanzmenge dem heutigen Preis anzupassen?**

## Modell

Der Hof pflanzt jedes Jahr $P_y$ Bäume; ein Anteil fällt aus (in Dürrejahren mehr), 80 % der Überlebenden sind Verkaufsware, nach 7 bis 12 Jahren (gestreut, je Jahrgang um ±1 Jahr verschoben) hat ein Baum Verkaufsgröße und ist dann `Verkaufsfenster` Jahre lang verkäuflich, danach zu hoch und verworfen. Der Absatz ist fest (Standard 2 000 Bäume im Jahr, ±20 %), der Preis je Baum 45 EUR mal Preisindex des Marktes. Der Marktpreis folgt dem Angebot aus Pflanzungen von vor 7 bis 12 Jahren; die Betriebe im Markt pflanzen nach dem Preis (Marktreaktion = Preisflexibilität mal Pflanzreaktion). Formeln im 📐-Expander der App. Alle Eingangszahlen sind Annahmen oder Verbandsrichtwerte (siehe Grenzen).

## Methodik

Sieben Pflanzregeln auf 300 Marktpfaden à 90 Jahre (die ersten 22 zählen nicht): gleichmäßig pflanzen; dem Preis folgen; antizyklisch; Vorausschau aus den öffentlichen Pflanzzahlen des Marktes; Ausfall nachpflanzen; Lückenfüller nach den erwarteten Ankünften; und als Referenz eine Regel, die den Preis der Ankunftsjahre kennt. Verglichen wird der Jahresüberschuss je Pfad gegen gleichmäßiges Pflanzen (gepaarte Differenz, Standardfehler). Die **exakte Referenz** löst über 40 Pflanzjahre ein lineares Programm (HiGHS) im Erwartungswert mit bekanntem Preispfad. **Akt 2** vergleicht auf denselben Saisons: den Markt als zweiten Absatzweg, die Marktmenge nach Schattenpreis aus einer Rückwärtsrechnung über (Tag, Restbestand), einen Preisplan und die Schranke bei Vorabkenntnis der Nachfrage.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: Standardszenario (35 % Marktanteil in Akt 2, 15 % Dürrejahre, Verkaufsfenster 3 Jahre, Absatz 2 000) und Marktreaktion 1,1, wenn nicht anders genannt; Akt 1 über feste Pfadmengen, Akt 2 über feste Saisons. Jede Zahl steht in `tests/test_claims.py`.

| Frage | Befund |
|---|---|
| Wie viel kostet es, dem Preis zu folgen? (Stärke 1,0) | Bei schwacher Marktreaktion (0,5 bis 1,0, Preisstreuung 0,12 bis 0,20) 0,2 bis 1,6 %. Ab 1,1 (Streuung 0,31) 6,4 %, bei 1,2 (0,59) 27,5 %, bei 1,3 (0,89) 50,5 %, bei 1,4 (1,08) 61,7 % des Jahresüberschusses gegenüber gleichmäßigem Pflanzen |
| Hilft vorsichtiges Folgen? (Stärke 0,5) | Kostet bei 1,1 noch 1,6 %, bei 1,2 11,2 %, bei 1,3 28,0 % |
| Hilft Wissen über die anderen Betriebe (Vorausschau)? | Nein: in keiner Marktreaktion besser als gleichmäßig (0,0 bis 4,8 % des Jahresüberschusses schlechter, bei 1,1 2,7 %) |
| Hilft antizyklisches Pflanzen? | Nein: 1,2 % (bei 0,5) bis 8,6 % (bei 1,4) schlechter |
| Hilft es, den Ausfall nachzupflanzen? | Nein: 3,5 bis 4,1 % schlechter bei 15 % Dürrejahren, 6,8 % bei 35 %; ohne Dürrejahre identisch zu gleichmäßig |
| Gibt es eine Regel, die gleichmäßig schlägt? | Der Lückenfüller, aber nur knapp: 0,7 bis 0,8 % in jeder Marktreaktion, 1,6 % bei 35 % Dürrejahren (Standardfehler unter 0,1 Prozentpunkte) |
| Was bringt der Preis in neun Jahren, wenn man ihn kennt? | Die Referenz-Regel liegt bei 1,1 um 2,4 % unter gleichmäßig und erst bei 1,3 (1,3 %) und 1,4 (4,5 %) darüber |
| Was ist im Erwartungswert überhaupt zu holen? (exakt, 10 Pfade) | Das Optimum bei bekanntem Preispfad liegt 3,8 % (± 0,5) über gleichmäßigem Pflanzen; bei 0,8 sind es 4,5 %, bei 1,3 6,2 %. Gleichmäßig liegt 3,7 % unter dem Optimum, Folgen 8,0 % (bei 1,3: 34,0 %), Lückenfüller 3,6 % |
| Welche Rolle spielt das Verkaufsfenster? | Bei 1 Jahr kostet Folgen 14,0 % und gleichmäßiges Pflanzen verfehlt 216 Verkäufe im Jahr (Standard 31); bei 5 Jahren kostet Folgen nichts (+0,5 %): das Fenster wirkt wie ein Lager |
| Wie gut ist gleichmäßiges Pflanzen im Standardfall? | 31 entgangene Verkäufe (1,6 %) und 38 verworfene Bäume im Jahr; Sollmenge 3 022 gepflanzte Bäume im Jahr |
| Ab wann schwingt der Markt? | Lineare Theorie: Schwelle bei Marktreaktion 1,09 (Nullstellen der charakteristischen Gleichung); darunter klingt eine Störung ab |
| **Akt 2:** Was bringt der Markt? | +10 502 EUR je Saison (+15,4 % des Gewinns ohne Markt, 68 121 EUR). Das folgt aus dem angenommenen Marktanteil von 35 %, ist also kein Befund, sondern eine Annahme |
| **Akt 2:** Was bringen Regeln und Preise? | Marktmenge nach Schattenpreis statt fester Menge +730 EUR (+1,1 %), bester fester Preis statt Richtpreis +330 EUR (+0,5 %), Preisplan nach Tag und Restbestand 0 EUR (der Bestand wird nie knapp, nach der Saison bleiben 787 von 2 000 Bäumen), Vorabkenntnis der Nachfrage +963 EUR (+1,4 %), Wertverlust der Rückläufer 15 EUR |
| **Akt 2:** Wann wird die Saison anspruchsvoll? | Wenn der Hof Rückläufer nicht zurücknimmt (Selbstschlag, 15 % Wertverlust je Tag): Verderb 2 202 EUR je Saison, Preis der Unkenntnis +3 407 EUR (+5,0 %). Bei knappem Bestand (Nachfrage 1,3 mal Bestand, Preisreaktion 2,5): Preisplan +593 EUR (+0,8 %). Ohne Markt: bester fester Preis +2 955 EUR (+3,6 %) |

## Ehrliche Grenzen

- **Angenommener Markt.** Das Spinnweb-Modell ist ein Lehrbuchmodell; wie stark der Zyklus am deutschen Markt ist, ist nicht belegt (Anekdote: US-Markt 2013 bis 2018, Durchschnittspreis von 35 auf 78 USD). Der Regler zeigt, was bei einem so starken Zyklus passiert, nicht, dass er so stark ist. Die Kosten des Folgens wachsen mit der Marktreaktion sehr schnell; bei Preisstreuung 0,31 sind es 6,4 %, ab 0,6 mehr als ein Viertel.
- **Fester Absatz.** Der Hof setzt rund 2 000 Bäume im Jahr ab, unabhängig vom Preis. Bei Händlerabsatz in beliebiger Menge fiele die Rechnung anders aus; deshalb gewinnt „gleichmäßig“ hier so oft.
- **Warum fast jede Regel schlechter ist als gleichmäßig.** Gleichmäßig pflanzen ist im Modell der Maßstab, der den festen Absatz im Erwartungswert genau deckt; das ist keine Naturgesetzlichkeit, sondern Folge der Annahmen. Der Absatz ist fest, ein Baum bringt höchstens den Preis eines Verkaufs: Wer mehr pflanzt, kann nicht mehr verkaufen, zahlt aber Pflege und verwirft zu hohe Bäume, wer weniger pflanzt, verkauft zu wenig. Jede Abweichung von der Sollmenge hat deshalb nur eine Seite mit Risiko, die Gewinnseite ist durch den Absatz gedeckelt. Hinzu kommt, dass der Preis beim Pflanzen wenig über den Preis beim Verkauf sagt: bei Verzug von 7 bis 12 Jahren und einem Zyklus liegen beide oft auf entgegengesetzten Seiten, das Folgen setzt also auf das falsche Vorzeichen, das antizyklische Pflanzen auf ein nur teilweise richtiges. Die exakte Referenz zeigt, dass gleichmäßig nicht optimal ist (3,7 % unter dem Optimum bei bekanntem Preispfad); was fehlt, ist Wissen über den künftigen Preis, das keine der Regeln hat. Mit Händlerabsatz in beliebiger Menge könnte eine Regel mehr verkaufen und die Rechnung änderte sich; das ist nicht gemessen.
- **Ein Preis je Baum.** Größenklassen und der Erntezeitpunkt (wann ein Baum zu groß wird, um ihn zu verkaufen) sind nicht modelliert; im Vorab-Lauf mit größenabhängigem Preis war der Gewinn von der Verkaufsreihenfolge abhängig (13 % Unterschied), ein Artefakt der Preisstruktur, kein Befund.
- **Richtwerte, keine Messung.** Pflanzdichte 7 000 je Hektar, Pflanzenpreis 0,34 EUR, Preis 23 bis 30 EUR je Meter (Verbandsempfehlung 2025), Ausfallspanne und Umtriebszeit stammen aus Verbands- und Presseangaben (Suchzusammenfassungen, nicht im Original geprüft); Dürrehäufigkeit, Verkaufsware 80 %, Verkaufsfenster, Pflegekosten und alle Saisonwerte sind Annahmen.
- **Die Referenz ist keine Obergrenze für den Betrieb.** Das LP rechnet im Erwartungswert (Ausfall gemittelt, Nachfrage fest, Preispfad bekannt). Die Regel „Preis in 9 Jahren bekannt“ ist eine Regel mit Wissen, keine Schranke.
- **Akt 2 ist ein Check, keine Planung.** Marktanteil, Standgebühr, Wertverlust und die Rücknahme der Rückläufer bestimmen die Hebel; der Preisplan lernt nicht aus den ersten Verkaufstagen.
- **Keine Empfehlung für einen Betrieb.** Die Regeln sind Rechenverfahren im Modell.

## Befunde und Korrekturen gegenüber dem Plan

- **Preisstruktur (Entscheidung E3).** Der Plan sah „Preis je Baum fest“ vor; die Vorab-Messreihe rechnete mit höheren Preisen für größere Bäume. Mit festem Preis fielen die Zahlen anders aus: Folgen kostet bei Streuung 0,31 6,4 % (Vorab-Lauf 4,8 %), bei 0,6 27,5 % (23,7 %); „perfekte Voraussicht“ bringt als Regel nicht mehr 2 bis 4 %, sondern liegt bei 1,1 und 1,2 unter gleichmäßig. Erst das LP zeigt mit 3,8 % den Wert des bekannten Preispfads.
- **Fehler im Lückenfüller, vom Orakel gefunden.** Der Test „Lückenfüller im Gleichgewicht pflanzt die Sollmenge“ fiel auf dem ersten Stand durch: die Schleife über künftige Jahrgänge ließ den frühesten (Ankunft nach 6 Jahren) aus, die Regel pflanzte 5,2 % mehr als die Sollmenge und lag 0,5 % des Jahresüberschusses unter gleichmäßig. Nach der Korrektur +0,9 %. Derselbe Versatz steckt im Skript der Vorab-Messreihe (`messreihe_anbau/anbau.py`); dort wurde nur die Fassung vor der Korrektur gemessen, die Aussage „Lückenfüller −5 %“ im Vorab-Ergebnis gilt nicht mehr.
- **Vorausschau.** Der Plan erwartete +1 bis −3 % gegenüber gleichmäßig; gemessen: 0,0 bis 4,8 % schlechter.

## Verifikation

- **Kern:** je Einheit eine Handrechnung auf der Mini-Instanz (`tests/conftest.py`, `tests/test_orchard.py`, `tests/test_rules.py`); unabhängiger Prüfer `check_result` (kein Baum zweimal verkauft, Pflanzmenge in den Grenzen) läuft auch in der App.
- **Orakel:** `tests/test_oracle_exact.py`: das LP gegen Aufzählung auf der Mini-Instanz (375 bei Plan 15/20), gegen eine unabhängige Vorwärtsrechnung (bei Fenster 1 in 25 Zufallsinstanzen gleich), als Obergrenze gegen 25 × 20 Zufallspläne, und die Simulation im Erwartungswert gegen dieselbe Vorwärtsrechnung (drei Regeln, über alle 90 Jahre, Verkäufe und Verwurf gleich). `tests/test_oracle_market.py`: Stabilitätsschwelle des Spinnwebs aus den Nullstellen der charakteristischen Gleichung (1,09), ohne Stöße klingt die Simulation unter der Schwelle ab und schwingt darüber. `tests/test_oracle_season.py`: Rückwärtsrechnung gegen numerische Quadratur (Poisson-Mischung), Hindsight und die Saison von Hand, Bestandserhaltung.
- **Zweig-Tests:** jede Regel weicht von jeder anderen ab; Rücknahme der Rückläufer ändert den Gewinn; der Preisplan weicht nur bei knappem Bestand vom festen Preis ab (die Nullspalte im Hebel-Check ist erklärt, kein toter Zweig).
- **Randfälle und Vorsorge:** alle Regler am Minimum und Maximum einzeln und zusammen, Fenster 1, ein Pfad, Dürre 0 %, Prozess-Test mit zwei Hash-Seeds, Worst-Case am Regler-Maximum, Laufzeit-Importe gegen `requirements.txt`, Presets, Footer, PDF, AppTest (jeder Button, Radio, Permalink).
- **Mutationstests:** `tests/test_mutation_findings.py` (Funde des Mutationslaufs, siehe Abschnitt Tests).

## Tests

159 Tests, rund 3 Minuten (Handrechnungen je Einheit, Orakel, Zweig-Tests, Randfälle, Behauptungen aus diesem README, Prozess-Test, AppTest, PDF).

**Mutationslauf** (`tools/oracle_mutation.py` im Website-Repo, 140 Stellen in `pfz_orchard`, `pfz_rules`, `pfz_exact`, `pfz_market`, `pfz_season`, Seed 1; die schweren Testdateien Behauptungen, Randfälle, App, Prozess-Test und Worst-Case liefen dabei nicht mit): Beim ersten Lauf wurden 61 % der Mutanten gefunden. Die Überlebenden zeigten echte Lücken, die `tests/test_mutation_findings.py` schließt: die Tagesschleife der Saison ließ den letzten Tag aus, ein Baum auf dem Markt zahlte die Standgebühr nicht, der Pool der Rückläufer wurde falsch fortgeschrieben, die Mittelwert-Korrektur der Nachfragefaktoren war ungeprüft, das LP kannte die Nachfragegrenze des letzten Jahres nicht, die Schranke bei Vorabkenntnis ließ den letzten Markttag aus oder verkaufte mehr Bäume, als im Bestand sind, und die Zieljahre des Lückenfüllers (8 bis 10) waren nur im Gleichgewicht, nicht allgemein geprüft. Beim zweiten Lauf mit denselben Stellen: 79 % gefunden; von den 29 Überlebenden ist einer (Pool der Rückläufer bei späterer Mitnahme) inzwischen durch einen weiteren Test geschlossen, die übrigen 28 sind gleichwertig oder liegen unter der Schärfe der Tests (Größe von Arrays, Schleifen über leere Bereiche, doppelte Begrenzung der Marktmenge, Gleichstand der Margen, Normierung der Angebotsstöße im Markt, Quadratur-Näherung der Rückwärtsrechnung).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App (Akt 1, Experiment, Akt 2, Methodenvergleich, Formeln) |
| `pfz_constants.py` | Regler-Grenzen, feste Annahmen, Regelnamen und -texte, Presets |
| `pfz_market.py` | Marktmodell mit Pflanzverzug (Spinnweb) |
| `pfz_orchard.py` | Hof: Jahrgänge, Ausfall, Ankunft, Verkaufsfenster, Kassenfluss, Prüfer |
| `pfz_rules.py` | die sieben Pflanzregeln |
| `pfz_exact.py` | exakte Referenz: LP (HiGHS) und unabhängige Vorwärtsrechnung |
| `pfz_season.py` | Akt 2: Saison, Preisplan, Marktregeln, Schranke bei Vorabkenntnis |
| `pfz_evaluation.py` | Auswertung, Experiment, Verdikte |
| `pfz_presets.py` | Permalink (Begrenzen, Einrasten), Presets, Zufalls-Seed-Button |
| `pfz_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `pfz_pdf_export.py` | PDF-Export (fpdf2) |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Handrechnungen, Orakel, Zweig-Tests, Randfälle, Behauptungen aus diesem README, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Größenklassen und Erntezeitpunkt** (wann welche Klasse geerntet wird): ohne Nachfragemodell je Größe nicht beantwortbar; als Stoppproblem dasselbe Modell wie `value-iteration-demo`.
- **Selbstschlag mit Sortimentserschöpfung:** im Vorab-Lauf kostet eine Fehlmischung des Sortiments 0,6 bis 1,5 %, kein eigener Hook.
- **Händlerabsatz, Lieferung an Haushalte, Warteschlange an der Säge:** Dopplung zu Tourenplanung, `revenue-management-demo` und den Erlang-Demos.
- **Kapitalwert mit Diskontierung, Lernen aus den ersten Verkaufstagen, mehrere Betriebe live im Markt.**

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python-Version in den App-Einstellungen prüfen (lokal Python 3.14, CI Python 3.12); der Code verwendet nur Merkmale, die in Python 3.12 laufen. <!-- lint-ok: dezimalpunkt -->

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, SciPy (HiGHS) und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
