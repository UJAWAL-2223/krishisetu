"""
KRISHI-SETU | Live Data Ingestion Pipeline v3
=============================================
Strategy: HuggingFace slug registry + live myScheme page scraping

Flow:
  1. Load scheme slugs from HuggingFace (myscheme.gov.in registry)
  2. Build live URLs → scrape each scheme page fresh
  3. Filter agriculture schemes
  4. Diff check → only embed new/changed
  5. Update FAISS vector store

Run once:      python ingestion_pipeline.py --max 50
Run scheduled: python ingestion_pipeline.py --schedule
"""
# Suppress noisy logs from playwright
import logging
logging.getLogger("playwright").setLevel(logging.WARNING)
import json
import hashlib
import time
import argparse
import logging
import numpy as np
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log", encoding="utf-8")
    ]
)
log = logging.getLogger("krishi-setu")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    "vector_store_dir": "./vector_store",
    "state_file":       "./vector_store/pipeline_state.json",
    "raw_data_file":    "./vector_store/raw_schemes.json",
    "embedding_model":  "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "refresh_interval_hours": 24,
    "max_schemes": 100,

    "agri_keywords": [
        "farmer", "farming", "agriculture", "agricultural", "kisan", "krishi",
        "crop", "seed", "fertilizer", "irrigation", "tractor", "equipment",
        "soil", "pesticide", "organic", "horticulture", "fisheries", "livestock",
        "dairy", "poultry", "fasal", "bima", "subsidy", "mechanization",
        "agri", "rural", "pm-kisan", "pmfby", "smam", "rkvy",
        "soil health", "drip irrigation", "micro irrigation", "kisaan",
        "beej", "khad", "sinchai", "pashu", "matsya", "fisherman",
        "cultivat", "harvest", "wheat", "rice", "pulses", "oilseed",
    ],

    "headers": {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-IN,en;q=0.9,hi;q=0.8",
        "Connection": "keep-alive",
    }
}


# ─────────────────────────────────────────────
# STEP 1: SLUG LOADER
# Gets scheme slugs from HuggingFace, builds live URLs
# ─────────────────────────────────────────────

# class SlugLoader:
#     """
#     Loads scheme slugs from the HuggingFace dataset
#     (which was sourced from myscheme.gov.in).

#     We use slugs ONLY as a URL directory — the actual
#     content is always scraped live from myscheme.gov.in.
#     This means every run fetches current government data.
#     """

#     def get_scheme_urls(self, max_schemes: int) -> list[str]:
#         log.info("Loading scheme slugs from HuggingFace registry...")

#         try:
#             from datasets import load_dataset
#             ds = load_dataset("shrijayan/gov_myscheme", trust_remote_code=True)
#             rows = list(ds["train"])
#             log.info(f"Registry has {len(rows)} total schemes")

#             urls = []
#             for row in rows:
#                 slug = self._extract_slug(row)
#                 if slug:
#                     urls.append(f"https://www.myscheme.gov.in/schemes/{slug}")

#             # deduplicate while preserving order
#             seen = set()
#             unique_urls = []
#             for u in urls:
#                 if u not in seen:
#                     seen.add(u)
#                     unique_urls.append(u)

#             log.info(f"Built {len(unique_urls)} unique scheme URLs")
#             limited = unique_urls[:max_schemes]
#             log.info(f"Will scrape {len(limited)} schemes this run")
#             return limited

#         except ImportError:
#             log.error("datasets library not installed. Run: pip install datasets")
#             return []
#         except Exception as e:
#             log.error(f"Failed to load slugs from HuggingFace: {e}")
#             return []

#     def _extract_slug(self, row: dict) -> str:
#         """
#         Tries multiple field names to find/build the scheme slug.
#         HuggingFace dataset has inconsistent field names.
#         """
#         # Direct slug field
#         slug = row.get("slug") or row.get("scheme_id") or row.get("id") or ""
#         if slug:
#             return str(slug).strip().lower()

#         # Build from scheme name
#         name = (
#             row.get("Scheme Name") or
#             row.get("scheme_name") or
#             row.get("title") or
#             row.get("name") or ""
#         )
#         if name:
#             import re
#             slug = str(name).lower().strip()
#             slug = re.sub(r'[^a-z0-9\s-]', '', slug)
#             slug = re.sub(r'\s+', '-', slug)
#             slug = slug.strip('-')
#             return slug[:80]  # cap length

#         return ""


class SlugLoader:
    """
    Curated agriculture scheme slugs from myscheme.gov.in.
    URLs are constructed from these slugs and scraped live every run.
    Slugs verified via Google search + myScheme sitemap.
    """

    # Verified agriculture scheme slugs from myscheme.gov.in
    AGRICULTURE_SLUGS = [
        # Central flagship schemes
        "pm-kisan", "pmfby", "kcc", "smam", "rkvy",
        "pkvy", "nmsa", "nfsm", "midh", "per-drop-more-crop",
        "soil-health-card-scheme", "paramparagat-krishi-vikas-yojana",
        "pm-kusum", "agri-infra-fund", "fpo-formation-and-promotion",
        "animal-husbandry-infrastructure-development-fund",
        "pm-matsya-sampada-yojana", "pmmy-kcc",
        "national-beekeeping-honey-mission",
        "formation-promotion-fpos",
        # Kisan schemes
        "mkky", "mksy", "ksyj", "mmksy", "kpyg", "mkky",
        "jkrmy", "namo-shetkari-mahasanman-nidhi-yojana",
        # Irrigation
        "pradhan-mantri-krishi-sinchayee-yojana",
        "pmksy-wdc", "pmksy-aibp", "pmksy-har-khet-ko-pani",
        # Horticulture
        "nfbks", "midh-nho",
        # Livestock & fisheries
        "pmksy-pdmc", "rashtriya-gokul-mission",
        "national-livestock-mission",
        "blue-revolution-integrated-development",
        # State schemes
        "bhavantar-bhugtan-yojana",
        "rythu-bandhu", "ysr-rythu-bharosa",
        "mukhyamantri-kisan-kalyan-yojana",
        "krishak-bandhu", "karshaka-samrudhi",
        "amma-vodi-ap-farmer",
        # Credit & insurance
        "fasal-bima-yojana-up",
        "pradhan-mantri-fasal-bima-yojana",
        "modified-interest-subvention-scheme",
        # Equipment & mechanization
        "farm-mechanization-tnau",
        "smam-agricultural-mechanization",
        # Organic farming
        "national-project-organic-farming",
        "paramparagat-krishi-up",
    ]

    def get_scheme_urls(self, max_schemes: int) -> list[str]:
        urls = [
            f"https://www.myscheme.gov.in/schemes/{slug}"
            for slug in self.AGRICULTURE_SLUGS
        ]
        limited = urls[:max_schemes]
        log.info(f"Loaded {len(limited)} agriculture scheme URLs from curated registry")
        return limited
# ─────────────────────────────────────────────
# STEP 2: SCHEME PAGE SCRAPER
# Visits each scheme URL, extracts structured data
# ─────────────────────────────────────────────

# class SchemeScraper:
#     """
#     Scrapes individual scheme pages from myscheme.gov.in.
#     myScheme is a Next.js app with structured HTML sections.
#     """

#     def __init__(self):
#         import requests
#         self.session = requests.Session()
#         self.session.headers.update(CONFIG["headers"])

#     def scrape(self, url: str) -> dict | None:
#         try:
#             from bs4 import BeautifulSoup

#             resp = self.session.get(url, timeout=15)

#             # 404 = slug doesn't exist on myScheme
#             if resp.status_code == 404:
#                 return None
#             if resp.status_code != 200:
#                 log.warning(f"  HTTP {resp.status_code}: {url}")
#                 return None

#             soup = BeautifulSoup(resp.text, "lxml")
#             slug = url.rstrip("/").split("/")[-1]

#             # ── Scheme name ──────────────────────────────
#             name = self._find_name(soup, slug)

#             # ── Structured sections ──────────────────────
#             sections = self._extract_sections(soup)

#             # ── Ministry ─────────────────────────────────
#             ministry = self._find_ministry(soup)

#             # ── States ───────────────────────────────────
#             states = self._find_states(soup)

#             scheme = {
#                 "scheme_id":           slug,
#                 "scheme_name":         self._clean(name),
#                 "description":         self._clean(sections.get("details", "")),
#                 "eligibility_criteria":self._clean(sections.get("eligibility", "")),
#                 "benefits":            self._clean(sections.get("benefits", "")),
#                 "documents_required":  self._clean(sections.get("documents", "")),
#                 "application_process": self._clean(sections.get("application", "")),
#                 "ministry":            self._clean(ministry),
#                 "applicable_states":   states,
#                 "official_link":       url,
#                 "data_source":         "myscheme.gov.in (live)",
#                 "last_fetched":        datetime.now().isoformat(),
#             }

#             # ── RAG text (what gets embedded) ────────────
#             scheme["rag_text"] = "\n".join([
#                 f"Scheme Name: {scheme['scheme_name']}",
#                 f"Ministry: {scheme['ministry']}",
#                 f"States: {scheme['applicable_states']}",
#                 f"Description: {scheme['description']}",
#                 f"Eligibility: {scheme['eligibility_criteria']}",
#                 f"Benefits: {scheme['benefits']}",
#                 f"Documents: {scheme['documents_required']}",
#             ])

#             # ── Content hash for diff checking ───────────
#             fingerprint = scheme["scheme_name"] + scheme["description"] + scheme["eligibility_criteria"]
#             scheme["content_hash"] = hashlib.md5(fingerprint.encode("utf-8")).hexdigest()

#             return scheme

#         except Exception as e:
#             log.warning(f"  Scrape failed for {url}: {e}")
#             return None

#     def _find_name(self, soup, slug: str) -> str:
#         for tag in ["h1", "h2"]:
#             el = soup.find(tag)
#             if el:
#                 text = el.get_text(strip=True)
#                 if text and len(text) > 3:
#                     return text
#         return slug.replace("-", " ").title()

#     def _extract_sections(self, soup) -> dict:
#         """
#         Maps heading keywords to section content.
#         Works for English and basic Hindi headings.
#         """
#         section_map = {
#             "detail":      "details",
#             "description": "details",
#             "about":       "details",
#             "overview":    "details",
#             "eligib":      "eligibility",
#             "patrta":      "eligibility",
#             "who can":     "eligibility",
#             "benefit":     "benefits",
#             "labh":        "benefits",
#             "what you get":"benefits",
#             "document":    "documents",
#             "dastavej":    "documents",
#             "required doc":"documents",
#             "application": "application",
#             "how to apply":"application",
#             "apply":       "application",
#             "avedan":      "application",
#         }

#         sections = {}
#         for heading in soup.find_all(["h2", "h3", "h4", "h5"]):
#             heading_text = heading.get_text(strip=True).lower()
#             for keyword, section_key in section_map.items():
#                 if keyword in heading_text and section_key not in sections:
#                     parts = []
#                     for sibling in heading.find_next_siblings():
#                         if sibling.name in ["h2", "h3", "h4", "h5"]:
#                             break
#                         text = sibling.get_text(separator=" ", strip=True)
#                         if text:
#                             parts.append(text)
#                     if parts:
#                         sections[section_key] = " ".join(parts)
#                     break

#         # Fallback: grab paragraph text if nothing found
#         if not sections:
#             paras = [
#                 p.get_text(strip=True)
#                 for p in soup.find_all("p")
#                 if len(p.get_text(strip=True)) > 40
#             ]
#             if paras:
#                 sections["details"] = " ".join(paras[:6])

#         return sections

#     def _find_ministry(self, soup) -> str:
#         for text in soup.stripped_strings:
#             t = text.strip()
#             if any(kw in t.lower() for kw in ["ministry", "department", "mantralaya"]):
#                 if 5 < len(t) < 120:
#                     return t
#         return ""

#     def _find_states(self, soup) -> str:
#         state_names = [
#             "Uttar Pradesh", "Punjab", "Haryana", "Rajasthan", "Bihar",
#             "Madhya Pradesh", "Maharashtra", "Gujarat", "Karnataka",
#             "Andhra Pradesh", "Telangana", "West Bengal", "Odisha",
#             "Assam", "Himachal Pradesh", "Uttarakhand", "Jharkhand",
#             "Chhattisgarh", "Tamil Nadu", "Kerala", "Goa", "Sikkim",
#             "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Tripura",
#         ]
#         page_text = soup.get_text(" ", strip=True)
#         found = [s for s in state_names if s in page_text]
#         return ", ".join(found) if found else "All India"

#     def _clean(self, text: str) -> str:
#         import re
#         if not text:
#             return ""
#         text = re.sub(r'<[^>]+>', ' ', str(text))
#         text = re.sub(r'\s+', ' ', text)
#         return text.strip()[:2000]

class SchemeScraper:
    """
    Uses Playwright headless browser to scrape myScheme pages.
    Required because myScheme is JavaScript-rendered (Next.js).
    Playwright runs a real Chromium browser — JS executes, content loads.
    """

    def scrape(self, url: str) -> dict | None:
        try:
            from playwright.sync_api import sync_playwright
            import re

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()

                # Block images/fonts to load faster
                page.route("**/*.{png,jpg,jpeg,gif,svg,woff,woff2}", 
                          lambda route: route.abort())

                page.goto(url, wait_until="networkidle", timeout=20000)

                # Wait for actual content to appear
                try:
                    page.wait_for_selector("h1", timeout=8000)
                except:
                    pass

                html = page.content()
                browser.close()

            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "lxml")
            slug = url.rstrip("/").split("/")[-1]

            name = self._find_name(soup, slug)
            sections = self._extract_sections(soup)
            ministry = self._find_ministry(soup)
            states = self._find_states(soup)

            scheme = {
                "scheme_id":            slug,
                "scheme_name":          self._clean(name),
                "description":          self._clean(sections.get("details", "")),
                "eligibility_criteria": self._clean(sections.get("eligibility", "")),
                "benefits":             self._clean(sections.get("benefits", "")),
                "documents_required":   self._clean(sections.get("documents", "")),
                "application_process":  self._clean(sections.get("application", "")),
                "ministry":             self._clean(ministry),
                "applicable_states":    states,
                "official_link":        url,
                "data_source":          "myscheme.gov.in (live)",
                "last_fetched":         datetime.now().isoformat(),
            }

            scheme["rag_text"] = "\n".join([
                f"Scheme Name: {scheme['scheme_name']}",
                f"Ministry: {scheme['ministry']}",
                f"States: {scheme['applicable_states']}",
                f"Description: {scheme['description']}",
                f"Eligibility: {scheme['eligibility_criteria']}",
                f"Benefits: {scheme['benefits']}",
                f"Documents: {scheme['documents_required']}",
            ])

            fingerprint = scheme["scheme_name"] + scheme["description"] + scheme["eligibility_criteria"]
            scheme["content_hash"] = hashlib.md5(fingerprint.encode("utf-8")).hexdigest()

            # Only return if we got real content
            if scheme["scheme_name"] in ["Quick Links", "", slug.replace("-", " ").title()]:
                if not scheme["description"]:
                    log.warning(f"  No content rendered for {url}")
                    return None

            return scheme

        except Exception as e:
            log.warning(f"  Playwright failed for {url}: {e}")
            return None

    # keep all your existing helper methods below:
    # _find_name, _extract_sections, _find_ministry, _find_states, _clean
    # (no changes needed to those)
# ─────────────────────────────────────────────
# STEP 3: AGRICULTURE FILTER
# ─────────────────────────────────────────────

def is_agriculture_related(scheme: dict) -> bool:
    all_text = " ".join([
        scheme.get("scheme_name", ""),
        scheme.get("description", ""),
        scheme.get("benefits", ""),
        scheme.get("eligibility_criteria", ""),
        scheme.get("ministry", ""),
    ]).lower()
    return any(kw.lower() in all_text for kw in CONFIG["agri_keywords"])


# ─────────────────────────────────────────────
# STEP 4: PIPELINE STATE (DIFF CHECKER)
# ─────────────────────────────────────────────

class PipelineStateManager:

    def __init__(self, state_file: str):
        self.state_file = Path(state_file)
        self.state = self._load()

    def _load(self) -> dict:
        if self.state_file.exists():
            with open(self.state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            count = len(state.get("processed_schemes", {}))
            log.info(f"Loaded pipeline state: {count} schemes tracked")
            return state
        return {"processed_schemes": {}, "last_run": None, "total_runs": 0}

    def save(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)

    def find_diff(self, schemes: list[dict]) -> tuple[list, list, list]:
        processed = self.state.get("processed_schemes", {})
        to_add, to_update, to_skip = [], [], []
        for s in schemes:
            sid = s["scheme_id"]
            new_hash = s["content_hash"]
            if sid not in processed:
                to_add.append(s)
            elif processed[sid]["hash"] != new_hash:
                to_update.append(s)
            else:
                to_skip.append(s)
        log.info(f"Diff → New: {len(to_add)} | Updated: {len(to_update)} | Unchanged: {len(to_skip)}")
        return to_add, to_update, to_skip

    def mark_processed(self, scheme_id: str, content_hash: str):
        self.state["processed_schemes"][scheme_id] = {
            "hash": content_hash,
            "processed_at": datetime.now().isoformat()
        }

    def update_run_metadata(self, added: int, updated: int):
        self.state["last_run"] = datetime.now().isoformat()
        self.state["total_runs"] = self.state.get("total_runs", 0) + 1
        self.state["last_run_stats"] = {"added": added, "updated": updated}


# ─────────────────────────────────────────────
# STEP 5: VECTOR STORE
# ─────────────────────────────────────────────

class VectorStore:

    def __init__(self, store_dir: str, model_name: str):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.index_path    = self.store_dir / "schemes.index"
        self.metadata_path = self.store_dir / "schemes_metadata.json"

        log.info(f"Loading embedding model: {model_name}")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        log.info("Embedding model ready")

        self.metadata = self._load_metadata()
        self.index    = self._load_index()

    def _load_metadata(self) -> list:
        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            log.info(f"Loaded {len(data)} schemes from existing metadata")
            return data
        return []

    def _load_index(self):
        import faiss
        if self.index_path.exists():
            index = faiss.read_index(str(self.index_path))
            log.info(f"Loaded FAISS index: {index.ntotal} vectors")
            return index
        index = faiss.IndexFlatL2(384)
        log.info("Created fresh FAISS index (384 dims)")
        return index

    def add_schemes(self, schemes: list[dict]):
        if not schemes:
            return
        log.info(f"Embedding {len(schemes)} schemes...")
        texts   = [s["rag_text"] for s in schemes]
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=16,
        ).astype(np.float32)
        self.index.add(vectors)
        self.metadata.extend(schemes)
        log.info(f"Vector store total: {self.index.ntotal} schemes")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self.index.ntotal == 0:
            log.warning("Vector store is empty")
            return []
        qvec = self.model.encode(
            [query], normalize_embeddings=True
        )[0].reshape(1, -1).astype(np.float32)
        distances, indices = self.index.search(qvec, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.metadata):
                s = self.metadata[idx].copy()
                s["relevance_score"] = round(float(1 / (1 + dist)), 3)
                results.append(s)
        return results

    def save(self):
        import faiss
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        log.info(f"Vector store saved: {self.index.ntotal} schemes")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

class KrishiSetuPipeline:

    def __init__(self):
        Path(CONFIG["vector_store_dir"]).mkdir(parents=True, exist_ok=True)
        # NOTE: components are instantiated here WITHOUT calling get_scheme_urls
        # max_schemes is passed at run-time only
        self.slug_loader   = SlugLoader()
        self.scraper       = SchemeScraper()
        self.state_manager = PipelineStateManager(CONFIG["state_file"])
        self.vector_store  = VectorStore(CONFIG["vector_store_dir"], CONFIG["embedding_model"])

    def run(self, max_schemes: int = None):
        if max_schemes is None:
            max_schemes = CONFIG["max_schemes"]

        start = time.time()
        log.info("=" * 55)
        log.info("KRISHI-SETU LIVE PIPELINE STARTING")
        log.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log.info(f"Max schemes this run: {max_schemes}")
        log.info("=" * 55)

        # ── 1. GET SCHEME URLs ────────────────────────────
        log.info("\n[1/5] Loading scheme URLs...")
        scheme_urls = self.slug_loader.get_scheme_urls(max_schemes)
        if not scheme_urls:
            log.error("No scheme URLs loaded. Exiting.")
            return

        # ── 2. SCRAPE LIVE PAGES ──────────────────────────
        log.info(f"\n[2/5] Scraping {len(scheme_urls)} live scheme pages...")
        raw_schemes = []
        failed = 0
        for i, url in enumerate(scheme_urls):
            scheme = self.scraper.scrape(url)
            if scheme:
                raw_schemes.append(scheme)
            else:
                failed += 1
            if (i + 1) % 10 == 0:
                log.info(f"  {i+1}/{len(scheme_urls)} | success: {len(raw_schemes)} | failed: {failed}")
            time.sleep(0.4)

        log.info(f"Scraping done — {len(raw_schemes)} successful, {failed} failed/404")

        # Save raw data
        Path(CONFIG["raw_data_file"]).parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG["raw_data_file"], "w", encoding="utf-8") as f:
            json.dump(raw_schemes, f, ensure_ascii=False, indent=2)
        log.info(f"Raw data saved → {CONFIG['raw_data_file']}")

        # ── 3. AGRICULTURE FILTER ─────────────────────────
        log.info("\n[3/5] Filtering for agriculture schemes...")
        agri_schemes = [s for s in raw_schemes if is_agriculture_related(s)]
        log.info(f"Agriculture: {len(agri_schemes)} / {len(raw_schemes)} total scraped")

        if not agri_schemes:
            log.warning("0 agriculture schemes found. Try --max 200 to scrape more.")
            log.info("Tip: raw_schemes.json saved — check what was scraped.")
            return

        # ── 4. DIFF CHECK ─────────────────────────────────
        log.info("\n[4/5] Running diff check...")
        to_add, to_update, to_skip = self.state_manager.find_diff(agri_schemes)

        # ── 5. EMBED + SAVE ───────────────────────────────
        log.info("\n[5/5] Embedding and saving...")
        if to_add:
            self.vector_store.add_schemes(to_add)
            for s in to_add:
                self.state_manager.mark_processed(s["scheme_id"], s["content_hash"])
        if to_update:
            self.vector_store.add_schemes(to_update)
            for s in to_update:
                self.state_manager.mark_processed(s["scheme_id"], s["content_hash"])
        if not to_add and not to_update:
            log.info("All schemes unchanged — vector store is up to date")

        self.vector_store.save()
        self.state_manager.update_run_metadata(len(to_add), len(to_update))
        self.state_manager.save()

        elapsed = time.time() - start
        log.info("\n" + "=" * 55)
        log.info(f"PIPELINE COMPLETE in {elapsed:.1f}s")
        log.info(f"  URLs loaded:       {len(scheme_urls)}")
        log.info(f"  Pages scraped:     {len(raw_schemes)}")
        log.info(f"  Agriculture:       {len(agri_schemes)}")
        log.info(f"  Added to store:    {len(to_add)}")
        log.info(f"  Updated in store:  {len(to_update)}")
        log.info(f"  Skipped (same):    {len(to_skip)}")
        log.info(f"  Total in store:    {self.vector_store.index.ntotal}")
        log.info("=" * 55)

        self._test_search()

    def _test_search(self):
        queries = [
            "tractor subsidy for small farmers",
            "kisan ko beej ke liye paisa",
            "crop insurance wheat farmer",
        ]
        log.info("\n--- Search Tests ---")
        for q in queries:
            results = self.vector_store.search(q, top_k=2)
            log.info(f"Query: '{q}'")
            if results:
                for r in results:
                    log.info(f"  → {r.get('scheme_name')} (score: {r.get('relevance_score')})")
            else:
                log.info("  → No results")


# ─────────────────────────────────────────────
# SCHEDULER
# ─────────────────────────────────────────────

def run_on_schedule(max_schemes: int):
    import schedule
    pipeline = KrishiSetuPipeline()
    log.info(f"Scheduler active — every {CONFIG['refresh_interval_hours']} hours")
    pipeline.run(max_schemes)
    schedule.every(CONFIG["refresh_interval_hours"]).hours.do(
        lambda: pipeline.run(max_schemes)
    )
    while True:
        schedule.run_pending()
        time.sleep(60)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Krishi-Setu Live Data Pipeline")
    parser.add_argument("--schedule", action="store_true", help="Run on 24hr schedule")
    parser.add_argument("--max", type=int, default=100, help="Max schemes per run (default: 100)")
    args = parser.parse_args()

    if args.schedule:
        run_on_schedule(args.max)
    else:
        pipeline = KrishiSetuPipeline()
        pipeline.run(args.max)