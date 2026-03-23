"""
CV Parser Evaluator — Pipeline Hybride Stratifié
=================================================
Évalue cv_parser.py sur un dataset annoté (ground_truth.json).

Usage :
    python cv_evaluator.py
    python cv_evaluator.py --ground-truth ground_truth.json --model llama3-70b-8192
    python cv_evaluator.py --report rapport.html --json-out resultats.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from cv_parser import CVParser, CVParserFactory

logger = logging.getLogger("cv_evaluator")


# ---------------------------------------------------------------------------
# Définition des champs et métriques associées
# ---------------------------------------------------------------------------

# Champs scalaires → Exact Match (après normalisation)
SCALAR_FIELDS = ["full_name", "email", "phone", "location", "linkedin", "github"]

# Champs liste plate → Precision / Recall / F1
LIST_FIELDS = ["skills", "certifications"]

# Champs structurés → F1 sur clé principale
STRUCT_FIELDS = {
    "experiences": "title",
    "education":   "degree",
}


def _norm(val: Any) -> str:
    """Normalise une valeur pour comparaison insensible à la casse/espaces."""
    return "" if val is None else str(val).lower().strip()


def _list_f1(predicted: list, expected: list) -> dict:
    """Calcule Precision, Recall, F1 pour deux ensembles."""
    ps = set(_norm(s) for s in (predicted or []))
    es = set(_norm(s) for s in (expected  or []))
    tp   = len(ps & es)
    prec = tp / len(ps) if ps else (1.0 if not es else 0.0)
    rec  = tp / len(es) if es else 1.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def _struct_f1(predicted: list, expected: list, key: str) -> dict:
    """F1 sur la clé principale d'une liste de dicts structurés."""
    pk = set(_norm(d.get(key, "")) for d in (predicted or []))
    ek = set(_norm(d.get(key, "")) for d in (expected  or []))
    tp   = len(pk & ek)
    prec = tp / len(pk) if pk else (1.0 if not ek else 0.0)
    rec  = tp / len(ek) if ek else 1.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def evaluate_single(predicted: dict, expected: dict) -> dict:
    """Évalue un seul CV prédit contre sa vérité terrain."""
    result = {}

    for f in SCALAR_FIELDS:
        result[f] = {
            "match": _norm(predicted.get(f)) == _norm(expected.get(f)),
            "predicted": predicted.get(f),
            "expected":  expected.get(f),
            "type": "scalar",
        }

    for f in LIST_FIELDS:
        result[f] = {
            **_list_f1(predicted.get(f, []), expected.get(f, [])),
            "type": "list",
        }

    for f, key in STRUCT_FIELDS.items():
        result[f] = {
            **_struct_f1(predicted.get(f, []), expected.get(f, []), key),
            "type": "struct",
        }

    return result


def aggregate(per_sample: list) -> dict:
    """Agrège les métriques sur tout le dataset."""
    n = len(per_sample)
    agg = {}

    for f in SCALAR_FIELDS:
        vals = [s[f]["match"] for s in per_sample]
        agg[f] = {"exact_match": sum(vals) / n, "type": "scalar"}

    for f in LIST_FIELDS + list(STRUCT_FIELDS.keys()):
        agg[f] = {
            "f1":        sum(s[f]["f1"]        for s in per_sample) / n,
            "precision": sum(s[f]["precision"] for s in per_sample) / n,
            "recall":    sum(s[f]["recall"]    for s in per_sample) / n,
            "type": "list",
        }

    return agg


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    metrics:       dict
    avg_latency:   float
    failed_parses: int
    per_sample:    list
    timings:       list


def _empty_metrics() -> dict:
    """Métriques à zéro pour un CV en erreur."""
    return {
        **{f: {"match": False, "predicted": None, "expected": None, "type": "scalar"}
           for f in SCALAR_FIELDS},
        **{f: {"f1": 0.0, "precision": 0.0, "recall": 0.0, "type": "list"}
           for f in LIST_FIELDS},
        **{f: {"f1": 0.0, "precision": 0.0, "recall": 0.0, "type": "struct"}
           for f in STRUCT_FIELDS},
    }


def run_benchmark(parser: CVParser, dataset: list) -> BenchmarkResult:
    per_sample = []
    timings    = []
    failed     = 0

    print(f"\n{'─'*60}")
    print(f"  Évaluation — {len(dataset)} CVs")
    print(f"{'─'*60}")

    for item in dataset:
        cv_id = item["id"]
        t0 = time.perf_counter()

        try:
            if "text" in item:
                # Texte inline → parse_text() directement (pas de fichier temporaire)
                result = parser.parse_text(item["text"])
            elif "file" in item:
                result = parser.parse(item["file"])
            else:
                raise ValueError(f"Item '{cv_id}' : clé 'text' ou 'file' requise")

            elapsed   = time.perf_counter() - t0
            predicted = json.loads(result.to_json())
            metrics   = evaluate_single(predicted, item["expected"])
            per_sample.append({"id": cv_id, **metrics})
            timings.append(elapsed)

            scalar_ok = sum(1 for f in SCALAR_FIELDS if metrics[f]["match"])
            list_avg  = (
                sum(metrics[f]["f1"] for f in LIST_FIELDS + list(STRUCT_FIELDS.keys()))
                / (len(LIST_FIELDS) + len(STRUCT_FIELDS))
            )
            status = "✅" if scalar_ok == len(SCALAR_FIELDS) else ("⚠️ " if scalar_ok >= 4 else "❌")
            print(f"  {status} {cv_id:<16}  scalaires {scalar_ok}/{len(SCALAR_FIELDS)}"
                  f"  listes F1={list_avg:.2f}  {elapsed:.2f}s")

        except Exception as e:
            elapsed = time.perf_counter() - t0
            logger.error("Erreur CV '%s' : %s", cv_id, e)
            print(f"  ❌ {cv_id:<16}  ERREUR : {e}")
            failed += 1
            per_sample.append({"id": cv_id, **_empty_metrics()})
            timings.append(elapsed)

    return BenchmarkResult(
        metrics       = aggregate(per_sample),
        avg_latency   = sum(timings) / len(timings) if timings else 0.0,
        failed_parses = failed,
        per_sample    = per_sample,
        timings       = timings,
    )


# ---------------------------------------------------------------------------
# Rapport Console
# ---------------------------------------------------------------------------

def print_report(result: BenchmarkResult, model: str):
    W = 65
    all_fields = SCALAR_FIELDS + LIST_FIELDS + list(STRUCT_FIELDS.keys())

    print(f"\n{'═'*W}")
    print(f"  RÉSULTATS — Pipeline Hybride Stratifié  [{model}]")
    print(f"{'═'*W}")
    print(f"\n  {'Champ':<22} {'Métrique':<24} {'Score':>7}  Barre")
    print(f"  {'─'*58}")

    for f in all_fields:
        m = result.metrics[f]
        if m["type"] == "scalar":
            score = m["exact_match"]
            label = "Exact Match"
        else:
            score = m["f1"]
            label = f"F1  P={m['precision']:.2f} R={m['recall']:.2f}"

        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {f:<22} {label:<24} {score:>6.3f}  {bar}")

    print(f"\n  {'─'*58}")
    print(f"  Latence moy.   : {result.avg_latency:.2f}s")
    print(f"  Erreurs        : {result.failed_parses}")
    print(f"{'═'*W}\n")


# ---------------------------------------------------------------------------
# Rapport HTML
# ---------------------------------------------------------------------------

def generate_html_report(result: BenchmarkResult, model: str, output_path: str):
    all_fields = SCALAR_FIELDS + LIST_FIELDS + list(STRUCT_FIELDS.keys())

    def bar_html(score: float) -> str:
        color = "#22c55e" if score >= 0.8 else ("#f59e0b" if score >= 0.5 else "#ef4444")
        return (
            f'<div style="background:#1e293b;border-radius:4px;height:10px;'
            f'width:160px;display:inline-block;vertical-align:middle">'
            f'<div style="background:{color};height:10px;border-radius:4px;'
            f'width:{score*160:.0f}px"></div></div>'
        )

    field_rows = ""
    for f in all_fields:
        m = result.metrics[f]
        if m["type"] == "scalar":
            score, extra, label = m["exact_match"], "—", "Exact Match"
        else:
            score = m["f1"]
            extra = f"P={m['precision']:.3f} / R={m['recall']:.3f}"
            label = "F1"
        color = "#22c55e" if score >= 0.8 else ("#f59e0b" if score >= 0.5 else "#ef4444")
        field_rows += (
            f"<tr><td><strong>{f}</strong></td>"
            f"<td class='c'>{label}</td>"
            f"<td class='c' style='color:{color};font-weight:700'>{score:.3f}</td>"
            f"<td class='c'>{extra}</td>"
            f"<td class='c'>{bar_html(score)}</td></tr>"
        )

    sample_rows = ""
    for s in result.per_sample:
        scalars  = sum(1 for f in SCALAR_FIELDS if s[f]["match"])
        list_avg = (
            sum(s[f]["f1"] for f in LIST_FIELDS + list(STRUCT_FIELDS.keys()))
            / (len(LIST_FIELDS) + len(STRUCT_FIELDS))
        )
        icon = "✅" if scalars == len(SCALAR_FIELDS) else ("⚠️" if scalars >= 4 else "❌")

        # Détail des erreurs scalaires
        errors = [
            f"{f}: prédit=<em>{s[f]['predicted']}</em> attendu=<em>{s[f]['expected']}</em>"
            for f in SCALAR_FIELDS if not s[f]["match"]
        ]
        error_html = "<br>".join(errors) if errors else "—"

        sample_rows += (
            f"<tr><td>{s['id']}</td>"
            f"<td class='c'>{icon} {scalars}/{len(SCALAR_FIELDS)}</td>"
            f"<td class='c'>{list_avg:.3f} {bar_html(list_avg)}</td>"
            f"<td style='font-size:.8rem;color:#94a3b8'>{error_html}</td></tr>"
        )

    overall_scalar = sum(result.metrics[f]["exact_match"] for f in SCALAR_FIELDS) / len(SCALAR_FIELDS)
    overall_list   = (
        sum(result.metrics[f]["f1"] for f in LIST_FIELDS + list(STRUCT_FIELDS.keys()))
        / (len(LIST_FIELDS) + len(STRUCT_FIELDS))
    )

    html = f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>CV Parser — Évaluation</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0f172a;color:#e2e8f0;padding:2rem}}
h1{{font-size:1.8rem;font-weight:700;color:#f8fafc;margin-bottom:.3rem}}
h2{{font-size:1.05rem;font-weight:600;color:#94a3b8;margin:2rem 0 .8rem;text-transform:uppercase;letter-spacing:.05em}}
.sub{{color:#64748b;font-size:.85rem;margin-bottom:2rem}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:1rem;margin-bottom:2rem}}
.card{{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:1.1rem}}
.card-label{{font-size:.72rem;color:#64748b;text-transform:uppercase;letter-spacing:.05em;margin-bottom:.3rem}}
.card-value{{font-size:1.55rem;font-weight:700}}
.green{{color:#22c55e}}.yellow{{color:#f59e0b}}.red{{color:#ef4444}}.blue{{color:#60a5fa}}
table{{width:100%;border-collapse:collapse;border-radius:12px;overflow:hidden;margin-bottom:1rem}}
th{{background:#0f172a;color:#94a3b8;font-size:.75rem;text-transform:uppercase;letter-spacing:.05em;padding:.75rem 1rem;text-align:left}}
td{{padding:.7rem 1rem;border-bottom:1px solid #0f172a;font-size:.88rem;background:#1e293b}}
tr:hover td{{background:#263248}}
.c{{text-align:center}}
.badge{{display:inline-block;padding:.2rem .7rem;border-radius:999px;font-size:.75rem;font-weight:600;background:#1d4ed8;color:#bfdbfe}}
</style></head><body>
<h1>📊 Rapport d'Évaluation — CV Parser</h1>
<p class="sub">
  Pipeline : <span class="badge">Regex → NER → LLM → Post-processing</span>
  &nbsp;|&nbsp; Modèle LLM : <strong>{model}</strong>
</p>

<div class="cards">
  <div class="card"><div class="card-label">Score Scalaires</div>
    <div class="card-value {'green' if overall_scalar >= 0.8 else 'yellow'}">{overall_scalar:.0%}</div>
    <div style="font-size:.78rem;color:#94a3b8;margin-top:.2rem">Exact Match moyen</div></div>
  <div class="card"><div class="card-label">Score Listes</div>
    <div class="card-value {'green' if overall_list >= 0.7 else 'yellow'}">{overall_list:.0%}</div>
    <div style="font-size:.78rem;color:#94a3b8;margin-top:.2rem">F1 moyen</div></div>
  <div class="card"><div class="card-label">Latence moy.</div>
    <div class="card-value blue">{result.avg_latency:.2f}s</div></div>
  <div class="card"><div class="card-label">Erreurs</div>
    <div class="card-value {'green' if result.failed_parses == 0 else 'red'}">{result.failed_parses}</div>
    <div style="font-size:.78rem;color:#94a3b8;margin-top:.2rem">CVs en erreur</div></div>
</div>

<h2>📋 Métriques par champ</h2>
<table><thead><tr>
  <th>Champ</th><th class="c">Métrique</th><th class="c">Score</th>
  <th class="c">P / R</th><th class="c">Barre</th>
</tr></thead><tbody>{field_rows}</tbody></table>

<h2>🔍 Détail par CV</h2>
<table><thead><tr>
  <th>CV</th><th class="c">Scalaires</th><th class="c">F1 listes</th><th>Erreurs détectées</th>
</tr></thead><tbody>{sample_rows}</tbody></table>

<h2>⚡ Performance</h2>
<table><thead><tr><th>Métrique</th><th class="c">Valeur</th></tr></thead><tbody>
<tr><td>Latence moyenne</td><td class="c">{result.avg_latency:.3f} s</td></tr>
<tr><td>CVs en erreur</td><td class="c">{result.failed_parses}</td></tr>
<tr><td>CVs évalués</td><td class="c">{len(result.per_sample)}</td></tr>
</tbody></table>
</body></html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ Rapport HTML → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Évalue le pipeline CV Parser hybride")
    ap.add_argument("--ground-truth", default="ground_truth.json",
                    help="Fichier JSON de vérité terrain")
    ap.add_argument("--model",        default="llama-3.1-8b-instant",
                    help="Modèle Groq à utiliser")
    ap.add_argument("--report",       default="eval_report.html",
                    help="Chemin du rapport HTML")
    ap.add_argument("--json-out",     default="eval_results.json",
                    help="Chemin des résultats bruts JSON")
    args = ap.parse_args()

    with open(args.ground_truth, encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"📂 Dataset : {len(dataset)} CVs  |  Modèle : {args.model}")

    parser = CVParserFactory.custom(args.model)
    result = run_benchmark(parser, dataset)

    print_report(result, args.model)
    generate_html_report(result, args.model, args.report)

    out = {
        "model":   args.model,
        "metrics": result.metrics,
        "performance": {
            "avg_latency_s": result.avg_latency,
            "failed_parses": result.failed_parses,
            "total_cvs":     len(result.per_sample),
        },
        "per_sample": result.per_sample,
    }
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"📄 Résultats bruts → {args.json_out}")


if __name__ == "__main__":
    main()