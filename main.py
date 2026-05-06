"""
NHTSA API Summary Parser
Fetches recalls and complaints from the NHTSA API,
clusters similar entries, and prints condensed summaries.
"""

import difflib
import json
import re
import urllib.parse
import urllib.request
import urllib.error

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


VPIC_URL = "https://vpic.nhtsa.dot.gov/api/vehicles"
RECALLS_URL = "https://api.nhtsa.gov/recalls/recallsByVehicle"
COMPLAINTS_URL = "https://api.nhtsa.gov/complaints/complaintsByVehicle"


def fetch_json(url: str, params: dict) -> dict:
    """Fetch JSON data from a NHTSA API endpoint."""
    query = urllib.parse.urlencode(params)
    full_url = f"{url}?{query}"

    req = urllib.request.Request(full_url)
    req.add_header("Accept", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 400:
            print(f"  No data available — the vehicle may not have any reports yet, or the make/model/year combo is invalid.")
        else:
            print(f"  HTTP Error {e.code}: {e.reason}")
        return {}
    except urllib.error.URLError as e:
        print(f"  Connection error: {e.reason}")
        return {}


def cluster_summaries(items: list[dict], summary_key: str, component_key: str) -> list[dict]:
    """
    Cluster similar summaries using TF-IDF + Agglomerative Clustering.
    Returns a list of cluster dicts with count, component, and representative summary.
    """
    summaries = []
    components = []

    for item in items:
        text = item.get(summary_key) or ""
        comp = item.get(component_key) or "N/A"
        if text.strip():
            summaries.append(text.strip())
            components.append(comp)

    if not summaries:
        return []

    # Single item — no clustering needed
    if len(summaries) == 1:
        return [{"count": 1, "component": components[0], "summary": summaries[0]}]

    # Vectorize the text
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(summaries)

    # Compute similarity and cluster
    # Use a distance threshold so similar complaints merge together
    n_samples = tfidf_matrix.shape[0]
    if n_samples <= 2:
        n_clusters = 1
    else:
        # Aim for roughly 1 cluster per 10 complaints, min 2, max 15
        n_clusters = max(2, min(15, n_samples // 10))

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(tfidf_matrix.toarray())

    # Build cluster results
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = {"indices": [], "components": []}
        clusters[label]["indices"].append(i)
        clusters[label]["components"].append(components[i])

    results = []
    for label, info in sorted(clusters.items()):
        indices = info["indices"]
        count = len(indices)

        # Pick the most representative summary (closest to cluster center)
        cluster_vectors = tfidf_matrix[indices]
        centroid = np.asarray(cluster_vectors.mean(axis=0))
        similarities = cosine_similarity(cluster_vectors, centroid)
        best_idx = indices[int(np.argmax(similarities))]

        # Most common component in this cluster
        comp_counts = {}
        for c in info["components"]:
            comp_counts[c] = comp_counts.get(c, 0) + 1
        top_component = max(comp_counts, key=comp_counts.get)

        # Truncate summary to ~200 chars for readability
        rep_summary = summaries[best_idx]
        if len(rep_summary) > 250:
            rep_summary = rep_summary[:247] + "..."

        results.append({
            "count": count,
            "component": top_component,
            "summary": rep_summary,
        })

    # Sort by count descending
    results.sort(key=lambda x: x["count"], reverse=True)
    return results


def print_summary(data: dict, label: str, summary_key: str, component_key: str) -> None:
    """Print a single condensed paragraph summarizing all results."""
    results = data.get("results", [])
    count = data.get("Count") or data.get("count", 0)

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")

    if not results:
        print("  No results.\n")
        return

    # Count by component
    comp_counts = {}
    for item in results:
        comp = item.get(component_key) or "OTHER"
        comp_counts[comp] = comp_counts.get(comp, 0) + 1

    # Sort by frequency
    ranked = sorted(comp_counts.items(), key=lambda x: x[1], reverse=True)

    # Build top issues as bullet points
    top_issues_lines = []
    for comp, cnt in ranked[:5]:
        top_issues_lines.append(f"    • {comp} — {cnt} reports")

    # Pick one representative summary from the #1 most common component
    # Truncate at the last complete sentence within ~300 chars
    top_comp = ranked[0][0]
    rep_summary = ""
    for item in results:
        comp = item.get(component_key) or "OTHER"
        text = item.get(summary_key) or ""
        if comp == top_comp and len(text) > 30:
            rep_summary = text.strip()
            if len(rep_summary) > 300:
                # Find the last sentence-ending punctuation within limit
                cut = rep_summary[:300]
                last_period = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
                if last_period > 50:
                    rep_summary = rep_summary[:last_period + 1]
                else:
                    rep_summary = cut.rsplit(" ", 1)[0] + "..."
            break

    print(f"\n  {count} total reports across {len(comp_counts)} categories.\n")
    print("  Top issues:")
    for line in top_issues_lines:
        print(line)
    if rep_summary:
        print(f"\n  Most common issue example:")
        print(f"  \"{rep_summary}\"")
    print()


def parse_vehicle_input(text: str) -> tuple:
    """Parse a freeform string like 'Honda CRV 2014' into (make_query, model_query, year)."""
    # Extract the 4-digit year
    year_match = re.search(r'\b(19|20)\d{2}\b', text)
    if not year_match:
        return None, None, None

    year = year_match.group()
    remaining = text[:year_match.start()] + text[year_match.end():]
    words = remaining.split()

    if len(words) < 2:
        return None, None, None

    # First word = make guess, rest = model guess
    make_query = words[0]
    model_query = " ".join(words[1:])
    return make_query, model_query, year


def resolve_vehicle(make_query: str, model_query: str, year: str) -> tuple:
    """
    Use the vPIC API + fuzzy matching to resolve a make/model query
    into the exact NHTSA make and model names.
    """
    # Step 1: Get all makes and fuzzy match
    makes_data = fetch_json(f"{VPIC_URL}/getallmakes", {"format": "json"})
    all_makes = [r["Make_Name"] for r in makes_data.get("Results", [])]

    make_matches = difflib.get_close_matches(
        make_query.upper(), [m.upper() for m in all_makes], n=1, cutoff=0.5
    )
    if not make_matches:
        print(f"  Could not find a make matching '{make_query}'.")
        return None, None

    # Get the original-cased make name
    matched_make = all_makes[[m.upper() for m in all_makes].index(make_matches[0])]

    # Step 2: Get models for that make + year and fuzzy match
    models_data = fetch_json(
        f"{VPIC_URL}/getmodelsformakeyear/make/{matched_make}/modelyear/{year}",
        {"format": "json"}
    )
    all_models = [r["Model_Name"] for r in models_data.get("Results", [])]

    if not all_models:
        print(f"  No models found for {matched_make} in {year}.")
        return matched_make, None

    model_matches = difflib.get_close_matches(
        model_query.upper(), [m.upper() for m in all_models], n=1, cutoff=0.4
    )
    if not model_matches:
        print(f"  Could not match model '{model_query}' for {matched_make} {year}.")
        print(f"  Available models: {', '.join(sorted(all_models)[:15])}")
        return matched_make, None

    matched_model = all_models[[m.upper() for m in all_models].index(model_matches[0])]
    return matched_make, matched_model


def main():
    query = input("Enter vehicle (e.g. Honda CRV 2014): ").strip()

    if not query:
        print("Please enter a vehicle.")
        return

    make_query, model_query, year = parse_vehicle_input(query)

    # If we couldn't parse all 3 from one line, prompt for missing parts
    if not make_query:
        make_query = query.split()[0] if query.split() else None
    if not make_query:
        print("Please enter at least a vehicle make.")
        return
    if not model_query:
        model_query = input("Enter model (e.g. Camry): ").strip()
    if not model_query:
        print("Model is required.")
        return
    if not year:
        year = input("Enter model year (e.g. 2020): ").strip()
    if not year or not year.isdigit():
        print("A valid year is required.")
        return

    print(f"\nSearching for '{make_query} {model_query} {year}'...")

    make, model = resolve_vehicle(make_query, model_query, year)
    if not make or not model:
        return

    print(f"  Best match: {year} {make} {model}")
    confirm = input("  Is this correct? (Y/n): ").strip().lower()
    if confirm == "n":
        # Show available models so user can retry
        models_data = fetch_json(
            f"{VPIC_URL}/getmodelsformakeyear/make/{make}/modelyear/{year}",
            {"format": "json"}
        )
        all_models = [r["Model_Name"] for r in models_data.get("Results", [])]
        if all_models:
            print(f"\n  Available {make} models for {year}:")
            for m in sorted(all_models):
                print(f"    • {m}")
        print("\n  Please try again with the correct model name.")
        return

    params = {"make": make, "model": model, "modelYear": year}

    # Fetch and display recalls
    print("\nFetching recalls...")
    recalls_data = fetch_json(RECALLS_URL, params)
    print_summary(recalls_data, "RECALLS", "Summary", "Component")

    # Fetch and display complaints
    print("Fetching complaints...")
    complaints_data = fetch_json(COMPLAINTS_URL, params)
    print_summary(complaints_data, "COMPLAINTS", "summary", "components")


if __name__ == "__main__":
    main()
