#!/usr/bin/env python3
"""
ICML / NeurIPS / ICLR 2025 paper keyword report generator.

What this script does:
1. Discovers accepted venue labels from each conference's OpenReview group page.
2. Crawls all accepted papers from OpenReview API2 with retry/backoff.
3. Extracts and normalizes keywords (author keywords + TF-IDF terms).
4. Assigns topic themes (LLM, diffusion, RL, multimodal, etc.).
5. Outputs JSON files and visualization charts.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer


GROUP_URL = "https://openreview.net/group?id={group_id}"
NOTES_API = "https://api2.openreview.net/notes"

CONFERENCE_GROUPS = {
    "ICLR": "ICLR.cc/2025/Conference",
    "NeurIPS": "NeurIPS.cc/2025/Conference",
    "ICML": "ICML.cc/2025/Conference",
}

BROWSER_HEADERS = {
    "accept": "application/json,text/*;q=0.99",
    "accept-language": "en-US,en;q=0.9",
    "cache-control": "no-cache",
    "pragma": "no-cache",
    "referer": "https://openreview.net/",
    "user-agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    ),
}

TRACK_ORDER = {
    "oral": 0,
    "spotlight": 1,
    "spotlightposter": 1,
    "poster": 2,
}

NOISE_TERMS = {
    "learning",
    "model",
    "models",
    "neural network",
    "neural networks",
    "approach",
    "method",
    "methods",
    "framework",
    "analysis",
    "results",
    "paper",
}

KEYWORD_SYNONYMS = [
    (r"\bllms\b", "llm"),
    (r"\blarge language models?\b", "llm"),
    (r"\blanguage models?\b", "llm"),
    (r"\bfoundation models?\b", "foundation model"),
    (r"\bvision language models?\b", "vlm"),
    (r"\bmulti[- ]modal\b", "multimodal"),
    (r"\breinforcement learning\b", "reinforcement learning"),
    (r"\bdeep reinforcement learning\b", "reinforcement learning"),
    (r"\bdiffusion models?\b", "diffusion model"),
    (r"\bgraph neural networks?\b", "gnn"),
    (r"\bretrieval[- ]augmented generation\b", "rag"),
    (r"\bchain[- ]of[- ]thought\b", "chain of thought"),
    (r"\bin[- ]context learning\b", "in context learning"),
]

THEME_PATTERNS = {
    "LLM": [
        r"\bllm\b",
        r"large language model",
        r"language model",
        r"foundation model",
        r"instruction tuning",
    ],
    "Multimodal": [
        r"multimodal",
        r"vision language",
        r"vlm",
        r"text image",
        r"audio visual",
    ],
    "Diffusion": [
        r"diffusion",
        r"denoising",
        r"score[- ]based",
        r"flow matching",
    ],
    "Reinforcement Learning": [
        r"reinforcement learning",
        r"\brl\b",
        r"policy gradient",
        r"offline rl",
        r"reward model",
    ],
    "Alignment and Safety": [
        r"alignment",
        r"safety",
        r"red teaming",
        r"jailbreak",
        r"fairness",
        r"privacy",
        r"toxicity",
    ],
    "Reasoning and Agents": [
        r"reasoning",
        r"agent",
        r"planning",
        r"tool use",
        r"chain of thought",
    ],
    "Vision": [
        r"computer vision",
        r"image",
        r"segmentation",
        r"detection",
        r"vision transformer",
    ],
    "Optimization and Theory": [
        r"optimization",
        r"generalization",
        r"sample complexity",
        r"convergence",
        r"theory",
    ],
    "Robotics and Control": [
        r"robot",
        r"control",
        r"trajectory",
        r"locomotion",
        r"manipulation",
    ],
    "Graphs": [
        r"\bgnn\b",
        r"graph",
        r"graph neural",
        r"knowledge graph",
    ],
}


@dataclass
class CrawlConfig:
    delay_seconds: float = 1.2
    max_retries: int = 7
    page_size: int = 1000
    max_per_venue: Optional[int] = None
    use_playwright: bool = False


def get_field_value(value):
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def parse_keywords(raw_keywords) -> List[str]:
    if raw_keywords is None:
        return []
    if isinstance(raw_keywords, list):
        parts = raw_keywords
    elif isinstance(raw_keywords, str):
        parts = re.split(r"[;,/|]", raw_keywords)
    else:
        parts = [str(raw_keywords)]
    cleaned = []
    for kw in parts:
        kw = str(kw).strip()
        if kw:
            cleaned.append(kw)
    return cleaned


def normalize_keyword(keyword: str) -> str:
    text = keyword.lower().strip()
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9+\- ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    for pattern, replacement in KEYWORD_SYNONYMS:
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"\s+", " ", text).strip()
    if text.endswith("s") and len(text) > 4 and not text.endswith("ss"):
        text = text[:-1]
    return text


def infer_track(venue_name: str) -> str:
    low = venue_name.lower()
    for key in ["oral", "spotlight", "spotlightposter", "poster"]:
        if key in low:
            return key
    return "unknown"


class OpenReviewCrawler:
    def __init__(self, config: CrawlConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update(BROWSER_HEADERS)

    def _sleep_with_jitter(self, base: Optional[float] = None):
        base_delay = self.config.delay_seconds if base is None else base
        time.sleep(base_delay + random.uniform(0.0, 0.35))

    def _request_text(self, url: str, params: Optional[Dict] = None) -> str:
        for attempt in range(1, self.config.max_retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=45)
            except requests.RequestException:
                if attempt == self.config.max_retries:
                    raise
                self._sleep_with_jitter(base=min(10.0, 1.5**attempt))
                continue

            if resp.status_code == 200:
                return resp.text
            if resp.status_code in {429, 500, 502, 503, 504}:
                if attempt == self.config.max_retries:
                    resp.raise_for_status()
                self._sleep_with_jitter(base=min(60.0, 2**attempt))
                continue
            resp.raise_for_status()
        raise RuntimeError("Unreachable retry loop in _request_text")

    def _request_json(self, url: str, params: Optional[Dict] = None) -> Dict:
        text = self._request_text(url, params=params)
        return json.loads(text)

    def fetch_group_html(self, group_id: str) -> str:
        if self.config.use_playwright:
            try:
                # Optional browser simulation path.
                from playwright.sync_api import sync_playwright

                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch(headless=True)
                    context = browser.new_context(
                        user_agent=BROWSER_HEADERS["user-agent"],
                        locale="en-US",
                    )
                    page = context.new_page()
                    page.goto(GROUP_URL.format(group_id=group_id), wait_until="networkidle")
                    html = page.content()
                    context.close()
                    browser.close()
                    return html
            except Exception:
                # Graceful fallback to requests mode when Playwright is unavailable.
                pass
        return self._request_text(GROUP_URL.format(group_id=group_id))

    def discover_accepted_venues(self, group_id: str) -> List[str]:
        html = self.fetch_group_html(group_id)
        escaped = re.findall(r'content\\?\.venue\\?":\\?"([^"\\]+)', html)
        plain = re.findall(r'content\.venue":"([^"]+)', html)
        candidates = sorted(set(escaped + plain))
        accepted = []
        for venue in candidates:
            low = venue.lower()
            if low.startswith("submitted to"):
                continue
            if "reject" in low or "withdraw" in low:
                continue
            accepted.append(venue)
        accepted = sorted(
            accepted,
            key=lambda x: TRACK_ORDER.get(infer_track(x), 99),
        )
        return accepted

    def crawl_venue_notes(self, venue_name: str) -> List[Dict]:
        results: List[Dict] = []
        offset = 0
        limit = self.config.page_size
        while True:
            params = {"content.venue": venue_name, "offset": offset, "limit": limit}
            data = self._request_json(NOTES_API, params=params)
            notes = data.get("notes", [])
            if not notes:
                break
            results.extend(notes)
            offset += len(notes)

            if self.config.max_per_venue is not None and len(results) >= self.config.max_per_venue:
                results = results[: self.config.max_per_venue]
                break
            if len(notes) < limit:
                break
            self._sleep_with_jitter()
        return results


def extract_papers(crawler: OpenReviewCrawler) -> List[Dict]:
    all_papers: List[Dict] = []
    seen_ids = set()

    for conf_name, group_id in CONFERENCE_GROUPS.items():
        venues = crawler.discover_accepted_venues(group_id)
        print(f"[collect] {conf_name} accepted venues: {venues}")
        for venue_name in venues:
            notes = crawler.crawl_venue_notes(venue_name)
            print(f"[collect] {conf_name} | {venue_name}: {len(notes)} papers")
            for note in notes:
                note_id = note.get("id")
                if note_id in seen_ids:
                    continue
                seen_ids.add(note_id)

                content = note.get("content", {})
                title = get_field_value(content.get("title")) or ""
                abstract = get_field_value(content.get("abstract")) or ""
                author_keywords = parse_keywords(get_field_value(content.get("keywords")))
                venue = get_field_value(content.get("venue")) or venue_name
                primary_area = get_field_value(content.get("primary_area")) or ""
                tldr = get_field_value(content.get("TLDR")) or ""
                pdf_path = get_field_value(content.get("pdf")) or ""

                record = {
                    "conference": conf_name,
                    "group_id": group_id,
                    "track": infer_track(venue),
                    "venue": venue,
                    "note_id": note_id,
                    "forum_id": note.get("forum"),
                    "title": title.strip(),
                    "abstract": abstract.strip(),
                    "author_keywords": author_keywords,
                    "primary_area": str(primary_area).strip(),
                    "tldr": str(tldr).strip(),
                    "pdf_url": f"https://openreview.net{pdf_path}" if pdf_path else "",
                    "openreview_url": f"https://openreview.net/forum?id={note.get('forum')}",
                    "cdate": note.get("cdate"),
                    "mdate": note.get("mdate"),
                }
                all_papers.append(record)
    return all_papers


def add_tfidf_keywords(papers: List[Dict], top_k_per_paper: int = 6):
    docs = [(p.get("title", "") + ". " + p.get("abstract", "")).strip() for p in papers]
    non_empty_docs = [doc if doc else "empty document" for doc in docs]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.8,
        max_features=12000,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]{2,}\b",
    )
    matrix = vectorizer.fit_transform(non_empty_docs)
    terms = vectorizer.get_feature_names_out()

    for idx, paper in enumerate(papers):
        row = matrix[idx]
        if row.nnz == 0:
            paper["tfidf_keywords"] = []
            continue
        top_positions = row.data.argsort()[-top_k_per_paper:][::-1]
        tfidf_terms = [terms[row.indices[pos]] for pos in top_positions]
        paper["tfidf_keywords"] = tfidf_terms


def build_combined_keywords(papers: List[Dict]):
    for paper in papers:
        merged = []
        seen = set()
        source_terms = list(paper.get("author_keywords", [])) + list(paper.get("tfidf_keywords", []))
        for term in source_terms:
            normalized = normalize_keyword(term)
            if not normalized:
                continue
            if len(normalized) < 3:
                continue
            if normalized in NOISE_TERMS:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            merged.append(normalized)
        paper["keywords"] = merged


def assign_themes(papers: List[Dict]):
    compiled = {
        theme: [re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns]
        for theme, patterns in THEME_PATTERNS.items()
    }
    for paper in papers:
        search_text = " ".join(
            [
                paper.get("title", ""),
                paper.get("abstract", ""),
                " ".join(paper.get("keywords", [])),
                paper.get("primary_area", ""),
            ]
        ).lower()
        matches = []
        for theme, regex_list in compiled.items():
            if any(regex.search(search_text) for regex in regex_list):
                matches.append(theme)
        if not matches:
            matches = ["Other"]
        paper["themes"] = matches


def top_counter_items(counter: Counter, n: int) -> List[Dict]:
    return [{"term": k, "count": int(v)} for k, v in counter.most_common(n)]


def summarize(papers: List[Dict]) -> Dict:
    conf_keyword_counter: Dict[str, Counter] = {conf: Counter() for conf in CONFERENCE_GROUPS}
    conf_theme_counter: Dict[str, Counter] = {conf: Counter() for conf in CONFERENCE_GROUPS}
    conf_track_counter: Dict[str, Counter] = {conf: Counter() for conf in CONFERENCE_GROUPS}
    global_keyword_counter = Counter()
    global_theme_counter = Counter()

    theme_track_counter = defaultdict(lambda: Counter())
    theme_conf_counter = defaultdict(lambda: Counter())

    for paper in papers:
        conf = paper["conference"]
        track = paper["track"]
        conf_track_counter[conf][track] += 1
        for kw in paper.get("keywords", []):
            conf_keyword_counter[conf][kw] += 1
            global_keyword_counter[kw] += 1
        for theme in paper.get("themes", []):
            conf_theme_counter[conf][theme] += 1
            global_theme_counter[theme] += 1
            theme_track_counter[theme][track] += 1
            theme_conf_counter[theme][conf] += 1

    conf_summary = {}
    for conf in CONFERENCE_GROUPS:
        conf_paper_count = sum(conf_track_counter[conf].values())
        conf_summary[conf] = {
            "paper_count": int(conf_paper_count),
            "track_counts": {k: int(v) for k, v in conf_track_counter[conf].items()},
            "top_keywords": top_counter_items(conf_keyword_counter[conf], 30),
            "top_themes": top_counter_items(conf_theme_counter[conf], 12),
        }

    continuing_hotspots = []
    emerging_trends = []
    total_papers = max(1, len(papers))
    continuing_threshold = max(60, int(total_papers * 0.015))
    emerging_threshold = max(25, int(total_papers * 0.006))

    theme_metrics = []
    for theme, total_count in global_theme_counter.items():
        if theme == "Other":
            continue
        conf_presence = sum(1 for conf in CONFERENCE_GROUPS if theme_conf_counter[theme][conf] > 0)
        oral_spotlight = theme_track_counter[theme]["oral"] + theme_track_counter[theme]["spotlight"] + theme_track_counter[theme]["spotlightposter"]
        oral_spotlight_ratio = oral_spotlight / max(1, total_count)
        max_conf_share = max(theme_conf_counter[theme].values()) / max(1, total_count)

        payload = {
            "theme": theme,
            "paper_mentions": int(total_count),
            "conference_presence": int(conf_presence),
            "oral_spotlight_ratio": round(float(oral_spotlight_ratio), 4),
            "max_conference_share": round(float(max_conf_share), 4),
        }
        theme_metrics.append(payload)

        if conf_presence == 3 and total_count >= continuing_threshold:
            continuing_hotspots.append(payload)
    continuing_hotspots = sorted(
        continuing_hotspots, key=lambda x: x["paper_mentions"], reverse=True
    )
    # Keep the strongest continuing themes; use the rest as potential emerging candidates.
    max_continuing_themes = 8
    continuing_hotspots = continuing_hotspots[:max_continuing_themes]
    continuing_theme_names = {x["theme"] for x in continuing_hotspots}

    # Score non-continuing themes for "emerging" potential.
    # This avoids empty results when all major themes have low oral/spotlight ratio.
    for item in theme_metrics:
        if item["theme"] in continuing_theme_names:
            continue
        if item["paper_mentions"] < emerging_threshold:
            continue
        novelty_bonus = 0.2 if item["conference_presence"] <= 2 else 0.0
        score = (
            0.65 * item["max_conference_share"]
            + 0.35 * item["oral_spotlight_ratio"]
            + novelty_bonus
        )
        emerging_trends.append({**item, "emerging_score": round(float(score), 4)})

    emerging_trends = sorted(emerging_trends, key=lambda x: (x["emerging_score"], x["paper_mentions"]), reverse=True)

    # Add keyword-level trend lens for finer granularity.
    keyword_conf_counter = defaultdict(lambda: Counter())
    keyword_track_counter = defaultdict(lambda: Counter())
    for paper in papers:
        conf = paper["conference"]
        track = paper["track"]
        for kw in paper.get("keywords", []):
            keyword_conf_counter[kw][conf] += 1
            keyword_track_counter[kw][track] += 1

    keyword_trends = []
    keyword_continuing = []
    for kw, total_count in global_keyword_counter.items():
        if total_count < 18:
            continue
        conf_presence = sum(1 for conf in CONFERENCE_GROUPS if keyword_conf_counter[kw][conf] > 0)
        max_conf_share = max(keyword_conf_counter[kw].values()) / max(1, total_count)
        oral_spotlight = (
            keyword_track_counter[kw]["oral"]
            + keyword_track_counter[kw]["spotlight"]
            + keyword_track_counter[kw]["spotlightposter"]
        )
        oral_spotlight_ratio = oral_spotlight / max(1, total_count)
        payload = {
            "keyword": kw,
            "mentions": int(total_count),
            "conference_presence": int(conf_presence),
            "max_conference_share": round(float(max_conf_share), 4),
            "oral_spotlight_ratio": round(float(oral_spotlight_ratio), 4),
        }
        if conf_presence == 3 and total_count >= 85:
            keyword_continuing.append(payload)
        elif max_conf_share >= 0.55 or conf_presence <= 2:
            score = 0.7 * max_conf_share + 0.3 * oral_spotlight_ratio
            keyword_trends.append({**payload, "emerging_score": round(float(score), 4)})

    keyword_trends = sorted(keyword_trends, key=lambda x: (x["emerging_score"], x["mentions"]), reverse=True)
    keyword_continuing = sorted(keyword_continuing, key=lambda x: x["mentions"], reverse=True)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper_count_total": int(len(papers)),
        "conference_summary": conf_summary,
        "global_top_keywords": top_counter_items(global_keyword_counter, 60),
        "global_top_themes": top_counter_items(global_theme_counter, 20),
        "trends": {
            "continuing_hotspots": continuing_hotspots[:12],
            "emerging_trends": emerging_trends[:12],
            "continuing_keywords": keyword_continuing[:20],
            "emerging_keywords": keyword_trends[:20],
            "heuristic_notes": [
                "continuing_hotspots: present in all 3 conferences + high mention count",
                "emerging_trends: theme-level concentration + venue-quality weighted score",
                "keyword trends are term-level signals and may be noisy after normalization",
            ],
        },
    }
    return summary


def plot_top_keywords(summary: Dict, output_path: Path):
    conferences = list(CONFERENCE_GROUPS.keys())
    fig, axes = plt.subplots(1, 3, figsize=(22, 6), constrained_layout=True)
    for idx, conf in enumerate(conferences):
        top = summary["conference_summary"][conf]["top_keywords"][:15]
        terms = [item["term"] for item in top][::-1]
        counts = [item["count"] for item in top][::-1]
        axes[idx].barh(terms, counts, color="#2b6cb0")
        axes[idx].set_title(f"{conf} 2025 Top Keywords")
        axes[idx].set_xlabel("Count")
        axes[idx].tick_params(axis="y", labelsize=9)
    fig.suptitle("Top Keywords by Conference", fontsize=16)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_keyword_heatmap(summary: Dict, output_path: Path):
    top_global = [x["term"] for x in summary["global_top_keywords"][:20]]
    confs = list(CONFERENCE_GROUPS.keys())
    data = []
    for kw in top_global:
        row = {"keyword": kw}
        for conf in confs:
            top_terms = {
                item["term"]: item["count"]
                for item in summary["conference_summary"][conf]["top_keywords"]
            }
            row[conf] = top_terms.get(kw, 0)
        data.append(row)
    frame = pd.DataFrame(data).set_index("keyword")
    plt.figure(figsize=(10, 8))
    sns.heatmap(frame, annot=True, fmt="d", cmap="YlGnBu", linewidths=0.5)
    plt.title("Keyword Frequency Heatmap (Top Global Terms)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_theme_distribution(summary: Dict, output_path: Path):
    confs = list(CONFERENCE_GROUPS.keys())
    theme_set = set()
    for conf in confs:
        theme_set.update(item["term"] for item in summary["conference_summary"][conf]["top_themes"][:10])
    themes = sorted(theme_set)

    records = []
    for theme in themes:
        for conf in confs:
            theme_counts = {
                item["term"]: item["count"]
                for item in summary["conference_summary"][conf]["top_themes"]
            }
            total = max(1, summary["conference_summary"][conf]["paper_count"])
            share = theme_counts.get(theme, 0) / total
            records.append({"theme": theme, "conference": conf, "share": share})

    frame = pd.DataFrame(records)
    plt.figure(figsize=(13, 7))
    sns.barplot(data=frame, x="theme", y="share", hue="conference")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Share of Papers")
    plt.xlabel("Theme")
    plt.title("Theme Distribution Across Conferences")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def generate_markdown_report(summary: Dict, output_path: Path):
    lines = []
    lines.append("# ICML / NeurIPS / ICLR 2025 Keyword Report")
    lines.append("")
    lines.append(f"- Generated at (UTC): `{summary['generated_at_utc']}`")
    lines.append(f"- Total papers: `{summary['paper_count_total']}`")
    lines.append("")

    for conf in CONFERENCE_GROUPS:
        conf_data = summary["conference_summary"][conf]
        lines.append(f"## {conf}")
        lines.append(f"- Paper count: `{conf_data['paper_count']}`")
        lines.append(f"- Track counts: `{conf_data['track_counts']}`")
        lines.append("- Top 15 keywords:")
        for item in conf_data["top_keywords"][:15]:
            lines.append(f"  - {item['term']}: {item['count']}")
        lines.append("")

    lines.append("## Continuing Hotspots")
    for item in summary["trends"]["continuing_hotspots"]:
        lines.append(
            f"- {item['theme']}: mentions={item['paper_mentions']}, "
            f"presence={item['conference_presence']}, "
            f"oral+spotlight ratio={item['oral_spotlight_ratio']}"
        )
    lines.append("")

    lines.append("## Emerging Trends")
    for item in summary["trends"]["emerging_trends"]:
        lines.append(
            f"- {item['theme']}: mentions={item['paper_mentions']}, "
            f"max conference share={item['max_conference_share']}, "
            f"oral+spotlight ratio={item['oral_spotlight_ratio']}"
        )
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, payload: Dict | List):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run(args):
    output_dir = Path(args.output_dir).resolve()
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if args.papers_json:
        papers_json_path = Path(args.papers_json).resolve()
        papers = json.loads(papers_json_path.read_text(encoding="utf-8"))
        print(f"[load] loaded papers from existing JSON: {papers_json_path}")
    else:
        crawler = OpenReviewCrawler(
            CrawlConfig(
                delay_seconds=args.delay_seconds,
                max_retries=args.max_retries,
                page_size=args.page_size,
                max_per_venue=args.max_per_venue,
                use_playwright=args.use_playwright,
            )
        )
        papers = extract_papers(crawler)
        if not papers:
            raise RuntimeError("No papers collected. Please check network access or API limits.")

    add_tfidf_keywords(papers, top_k_per_paper=args.tfidf_top_k)
    build_combined_keywords(papers)
    assign_themes(papers)
    summary = summarize(papers)

    papers_json_path = output_dir / "papers_2025.json"
    summary_json_path = output_dir / "summary_2025.json"
    report_md_path = output_dir / "report_2025.md"
    write_json(papers_json_path, papers)
    write_json(summary_json_path, summary)
    generate_markdown_report(summary, report_md_path)

    plot_top_keywords(summary, figures_dir / "top_keywords_by_conference.png")
    plot_keyword_heatmap(summary, figures_dir / "keyword_heatmap.png")
    plot_theme_distribution(summary, figures_dir / "theme_distribution.png")

    print("[done] output dir:", output_dir)
    print("[done] papers json:", papers_json_path)
    print("[done] summary json:", summary_json_path)
    print("[done] report:", report_md_path)
    print("[done] figures:", figures_dir)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Crawl and analyze ICML/NeurIPS/ICLR 2025 papers from OpenReview."
    )
    parser.add_argument(
        "--output-dir",
        default="conference_2025_keyword_report",
        help="Directory to store JSON/report/charts.",
    )
    parser.add_argument(
        "--papers-json",
        default=None,
        help="Reuse an existing papers JSON to skip crawling.",
    )
    parser.add_argument("--delay-seconds", type=float, default=1.2, help="Base delay between API pages.")
    parser.add_argument("--max-retries", type=int, default=7, help="Max retry attempts per request.")
    parser.add_argument("--page-size", type=int, default=1000, help="OpenReview page size (max 1000).")
    parser.add_argument(
        "--max-per-venue",
        type=int,
        default=None,
        help="Optional cap for debug runs (papers per venue).",
    )
    parser.add_argument("--tfidf-top-k", type=int, default=6, help="TF-IDF keywords to extract per paper.")
    parser.add_argument(
        "--use-playwright",
        action="store_true",
        help="Use Playwright to fetch group pages in a browser-like environment.",
    )
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
