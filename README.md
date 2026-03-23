# CV Parser : Pipeline Hybride 

Extraction structurée de CVs en plusieurs couches ordonnées par fiabilité : Regex → NER → LLM → Post-processing. Supporte PDF (natif et scanné), DOCX, TXT et images. Inclut un évaluateur de performance avec rapport HTML.

---

## Sommaire

- [Architecture](#architecture)
- [Modèle de sortie](#modèle-de-sortie)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage — Parser](#usage--parser)
- [Usage — Évaluateur](#usage--évaluateur)
- [Modèles disponibles](#modèles-disponibles)
- [Formats supportés](#formats-supportés)
- [Structure du projet](#structure-du-projet)

---

## Architecture

Le pipeline traite chaque CV en 5 couches séquentielles. Chaque couche résout ce qu'elle fait le mieux et transmet son résultat à la suivante.

```
Fichier CV (PDF / DOCX / Image / TXT)
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│  COUCHE 0 · TextExtractor                               │
│                                                         │
│  PDF → extraction native (pdfminer)                     │
│      → OCR Tesseract si texte < 50 chars (PDF scanné)   │
│  DOCX → python-docx (paragraphes + tableaux)            │
│  Image (JPG/PNG/WEBP/TIFF) → OCR Tesseract direct       │
└─────────────────────────────────────────────────────────┘
              │  texte brut
              ▼
┌─────────────────────────────────────────────────────────┐
│  COUCHE 1 · RegexExtractor              [déterministe]  │
│                                                         │
│  email · téléphone · LinkedIn · GitHub                  │
│  Résultats prioritaires — priment sur NER et LLM        │
└─────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│  COUCHE 2 · NERExtractor (spaCy)           [structurel] │
│                                                         │
│  nom complet (PER) · localisation (GPE/LOC)             │
│  organisations (ORG) — avec blacklist faux positifs     │
└─────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│  COUCHE 3 · LLMExtractor (Groq)             [sémantique]│
│                                                         │
│  résumé · compétences · expériences · formation         │
│  langues · certifications                               │
│  + contexte enrichi des couches 1 & 2 dans le prompt   │
└─────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│  COUCHE 4 · PostProcessor                               │
│                                                         │
│  Fusion avec règles de priorité par champ               │
│  Validation croisée des valeurs LLM dans le texte source│
│  Normalisation · déduplication · métadonnées debug      │
└─────────────────────────────────────────────────────────┘
              │
              ▼
           CVData (JSON)
```

### Règles de priorité par champ

| Champ | Source primaire | Fallback | Validation croisée |
|---|---|---|---|
| `email` | Regex | LLM | Oui — vérifié dans le texte source |
| `phone` | Regex | LLM | Oui |
| `linkedin` · `github` | Regex | LLM | Oui |
| `full_name` · `location` | NER (spaCy) | LLM | Non |
| `skills` · `summary` | LLM | — | — |
| `experiences` · `education` | LLM | — | — |
| `languages` · `certifications` | LLM | — | — |

> **Validation croisée** : si le LLM propose une valeur pour un champ critique (email, téléphone, etc.), elle est vérifiée contre le texte source avant acceptation. Une valeur introuvable dans le source est invalidée silencieusement.

---

## Modèle de sortie

Chaque CV parsé retourne un objet `CVData` sérialisable en JSON :

```python
@dataclass
class CVData:
    full_name:      str | None       # "Jean Dupont"
    email:          str | None       # "jean@example.com"
    phone:          str | None       # "06 12 34 56 78"
    location:       str | None       # "Paris, France"
    linkedin:       str | None       # "linkedin.com/in/jeandupont"
    github:         str | None       # "github.com/jeandupont"
    summary:        str | None       # Résumé professionnel 2-3 phrases
    skills:         list[str]        # ["python", "docker", "sql"]
    languages:      list[dict]       # [{"language": "Français", "level": "Natif"}]
    experiences:    list[dict]       # [{title, company, location, period, description}]
    education:      list[dict]       # [{degree, institution, location, period}]
    certifications: list[str]        # ["AWS Solutions Architect"]
```

Sortie JSON :
```python
result = parser.parse("cv.pdf")
print(result.to_json())                    # sans métadonnées
print(result.to_json(include_meta=True))   # avec _extraction_meta (debug)
```

---

## Installation

### Dépendances Python

```bash
pip install groq pdfminer.six python-docx spacy pillow pytesseract python-dateutil python-dotenv
```

### Modèle NER spaCy

Choisir selon la langue principale des CVs à traiter :

```bash
# Français (recommandé)
python -m spacy download fr_core_news_sm

# Anglais
python -m spacy download en_core_web_sm

# Les deux (le pipeline charge automatiquement le meilleur disponible)
python -m spacy download fr_core_news_sm en_core_web_sm
```

### Tesseract OCR

Requis uniquement pour les PDFs scannés et les CVs en image.

```bash
# Ubuntu / Debian
apt-get install tesseract-ocr tesseract-ocr-fra tesseract-ocr-eng

# macOS
brew install tesseract tesseract-lang

# Windows
# Télécharger l'installeur : https://github.com/UB-Mannheim/tesseract/wiki
```

### pdf2image *(optionnel, recommandé pour l'OCR PDF)*

Améliore significativement la qualité OCR sur les PDFs scannés.

```bash
pip install pdf2image

# Ubuntu / Debian
apt-get install poppler-utils

# macOS
brew install poppler
```

---

## Configuration

Créer un fichier `.env` à la racine du projet :

```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Obtenir une clé API : [console.groq.com](https://console.groq.com)

---

## Usage — Parser

### API Python

```python
from cv_parser import CVParserFactory

# Choisir un preset selon le besoin
parser = CVParserFactory.fast()           # Volume, CVs standards
parser = CVParserFactory.accurate()      # CVs complexes, profils seniors
parser = CVParserFactory.fastest()       # APIs temps réel (1000 t/s)
parser = CVParserFactory.best()          # Qualité maximale
parser = CVParserFactory.multilingual()  # CVs FR/AR/EN
parser = CVParserFactory.custom("openai/gpt-oss-120b")  # Modèle libre

# Parser un fichier
result = parser.parse("cv.pdf")
result = parser.parse("cv_scanné.jpg")
result = parser.parse("cv.docx")

# Parser du texte brut (benchmark, tests)
result = parser.parse_text("Jean Dupont\njean@email.com\n...")

# Accéder aux champs
print(result.full_name)
print(result.skills)
print(result.to_json())
```

### Traitement en batch

```python
from pathlib import Path
from cv_parser import CVParserFactory
import json

parser = CVParserFactory.fast()
results = []

for cv_path in Path("cvs/").glob("*.pdf"):
    try:
        result = parser.parse(str(cv_path))
        results.append({"file": cv_path.name, "data": json.loads(result.to_json())})
    except Exception as e:
        print(f"Erreur {cv_path.name} : {e}")

with open("resultats.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
```

### CLI

```bash
# Usage de base
python cv_parser.py cv.pdf

# CV scanné (image)
python cv_parser.py cv_scanné.jpg

# DOCX avec modèle spécifique
python cv_parser.py cv.docx openai/gpt-oss-120b

# Formats supportés : pdf, docx, txt, md, jpg, jpeg, png, webp, tiff
```

---

## Usage — Évaluateur

### Format du dataset

Créer un fichier `ground_truth.json` :

```json
[
  {
    "id": "cv_001",
    "text": "Jean Dupont\njean.dupont@example.com\n06 12 34 56 78\n...",
    "expected": {
      "full_name": "Jean Dupont",
      "email": "jean.dupont@example.com",
      "phone": "06 12 34 56 78",
      "location": "Paris",
      "skills": ["python", "sql", "docker"],
      "experiences": [
        {"title": "Développeur Backend", "company": "Acme Corp"}
      ]
    }
  },
  {
    "id": "cv_002",
    "file": "cvs/cv_002.pdf",
    "expected": {
      "full_name": "Marie Martin",
      "email": "marie.martin@example.com"
    }
  }
]
```

> Chaque entrée utilise soit `"text"` (texte inline) soit `"file"` (chemin vers un fichier).  
> Le champ `"expected"` ne doit contenir que les champs à évaluer — les champs absents sont ignorés.

### Lancer le benchmark

```bash
# Benchmark avec options par défaut
python cv_evaluator.py

# Avec options explicites
python cv_evaluator.py \
  --ground-truth ground_truth.json \
  --model        openai/gpt-oss-120b \
  --report       eval_report.html \
  --json-out     eval_results.json
```

### Sorties générées

| Fichier | Format | Contenu |
|---|---|---|
| Console | texte | Progression en temps réel + tableau de métriques agrégées |
| `eval_report.html` | HTML | Dashboard visuel : scores par champ, détail par CV, erreurs scalaires |
| `eval_results.json` | JSON | Données brutes exploitables programmatiquement |

### Métriques

| Type de champ | Métrique | Champs concernés |
|---|---|---|
| Scalaire | Exact Match après normalisation (lowercase + strip) | full_name, email, phone, location, linkedin, github |
| Liste plate | Precision / Recall / F1 sur ensembles | skills, certifications |
| Liste structurée | F1 sur clé principale | experiences (clé : `title`), education (clé : `degree`) |

---

## Modèles disponibles

Tous les modèles production Groq disposent d'un contexte de **131 072 tokens** (limite pratique appliquée : 60 000 chars de texte CV). Mise à jour : mars 2026.

### Production

| Preset | Modèle | Vitesse | Prix input | Recommandé pour |
|---|---|---|---|---|
| `fast()` | `llama-3.1-8b-instant` | 560 t/s | $0.05/1M | Volume, CVs standards (1-2 pages) |
| `accurate()` | `llama-3.3-70b-versatile` | 280 t/s | $0.59/1M | CVs complexes, profils seniors |
| `fastest()` | `openai/gpt-oss-20b` | 1000 t/s | $0.075/1M | APIs temps réel, faible latence |
| `best()` | `openai/gpt-oss-120b` | 500 t/s | $0.15/1M | Qualité maximale, CVs ambigus |

### Preview *(évaluation uniquement — ne pas utiliser en production)*

| Preset | Modèle | Vitesse | Notes |
|---|---|---|---|
| `multilingual()` | `qwen/qwen3-32b` | 400 t/s | CVs FR/AR/EN mixtes |
| `custom(...)` | `meta-llama/llama-4-scout-17b-16e-instruct` | 750 t/s | Très prometteur |
| `custom(...)` | `moonshotai/kimi-k2-instruct-0905` | 200 t/s | Contexte 262k tokens |

### Dépréciés *(ne plus utiliser)*

`mixtral-8x7b-32768` · `llama3-8b-8192` · `llama3-70b-8192` · `gemma2-9b-it`

---

## Formats supportés

| Format | Extension(s) | Méthode | Notes |
|---|---|---|---|
| PDF texte | `.pdf` | pdfminer natif | CVs numériques standard |
| PDF scanné | `.pdf` | OCR Tesseract | Bascule automatique si texte < 50 chars |
| Word | `.docx` | python-docx | Paragraphes + tableaux |
| Texte | `.txt` · `.md` | Lecture directe | — |
| Image | `.jpg` · `.jpeg` · `.png` · `.webp` | OCR Tesseract | CV photographié ou scanné |
| Image HD | `.tiff` · `.tif` · `.bmp` | OCR Tesseract | Documents scannés haute résolution |

> L'OCR fonctionne en français et anglais simultanément (`fra+eng`). Pour d'autres langues, installer le pack Tesseract correspondant et modifier le paramètre `lang` dans `TextExtractor._ocr_pdf()`.

---

## Structure du projet

```
cv_pipeline/
├── cv_parser.py          # Pipeline principal
│   ├── CVData                Modèle de données de sortie
│   ├── TextExtractor         Couche 0 — ingestion multi-format + OCR
│   ├── RegexExtractor        Couche 1 — champs déterministes
│   ├── NERExtractor          Couche 2 — entités nommées (spaCy)
│   ├── LLMExtractor          Couche 3 — extraction sémantique (Groq)
│   ├── PostProcessor         Couche 4 — fusion, validation, normalisation
│   ├── CVParser              Orchestrateur du pipeline
│   └── CVParserFactory       Presets de configuration
│
├── cv_evaluator.py       # Benchmark et évaluation
│   ├── evaluate_single()     Métriques pour un CV
│   ├── aggregate()           Agrégation sur le dataset
│   ├── run_benchmark()       Runner principal
│   ├── print_report()        Rapport console
│   └── generate_html_report() Rapport HTML
│
├── ground_truth.json     # Dataset d'évaluation (à créer)
├── .env                  # GROQ_API_KEY (à créer, ne pas commiter)
└── README.md
```

---

## Dépendances

```
groq>=0.9.0
pdfminer.six>=20221105
python-docx>=1.1.0
spacy>=3.7.0
Pillow>=10.0.0
pytesseract>=0.3.10
python-dateutil>=2.9.0
python-dotenv>=1.0.0
pdf2image>=1.16.0         # optionnel, améliore l'OCR PDF
```
