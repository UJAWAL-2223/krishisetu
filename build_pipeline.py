"""
KRISHI-SETU | Data Pipeline (PDF approach)
==========================================
Downloads 723 real scheme PDFs from HuggingFace
(sourced from myscheme.gov.in by the dataset author)
Extracts text, filters agriculture, embeds into FAISS.

Run once: python build_pipeline.py
"""

import json
import hashlib
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("krishi-pipeline")

# ── AGRICULTURE KEYWORDS ──────────────────────────────
AGRI_KEYWORDS = [
    "farmer", "farming", "agriculture", "agricultural", "kisan", "krishi",
    "crop", "seed", "fertilizer", "irrigation", "tractor", "equipment",
    "soil", "pesticide", "organic", "horticulture", "fisheries", "livestock",
    "dairy", "poultry", "fasal", "bima", "subsidy", "mechanization",
    "agri", "rural", "pm-kisan", "pmfby", "smam", "rkvy",
    "drip irrigation", "micro irrigation", "kisaan", "beej", "khad",
    "sinchai", "pashu", "matsya", "fisherman", "cultivat",
    "harvest", "wheat", "rice", "pulses", "oilseed", "horticulture",
    "sericulture", "apiculture", "aquaculture", "plantation",
]

def is_agriculture(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in AGRI_KEYWORDS)


# ── STEP 1: DOWNLOAD PDFs ─────────────────────────────
def download_pdfs(local_dir: str = "./scheme_pdfs") -> str:
    from huggingface_hub import snapshot_download
    log.info("Downloading scheme PDFs from HuggingFace...")
    log.info("This is a one-time download (~70MB). Will be cached after.")
    path = snapshot_download(
        repo_id="shrijayan/gov_myscheme",
        repo_type="dataset",
        local_dir=local_dir,
    )
    log.info(f"Downloaded to: {path}")
    return path


# ── STEP 2: EXTRACT TEXT FROM PDFs ───────────────────
def extract_from_pdf(pdf_path: Path) -> dict | None:
    """
    Uses PyMuPDF to extract text from a scheme PDF.
    Each PDF is one scheme from myscheme.gov.in
    """
    import fitz  # PyMuPDF

    try:
        doc = fitz.open(str(pdf_path))
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()

        if len(full_text.strip()) < 100:
            return None  # skip empty/corrupt PDFs

        # Slug from filename
        # e.g. "pm-kisan-scheme.pdf" → "pm-kisan-scheme"
        slug = pdf_path.stem.lower().replace(" ", "-")

        # Try to extract scheme name from first few lines
        lines = [l.strip() for l in full_text.split("\n") if l.strip()]
        name = lines[0] if lines else slug.replace("-", " ").title()

        # Cap name length — first line sometimes has garbage
        if len(name) > 120:
            name = slug.replace("-", " ").title()

        # Build structured scheme object
        scheme = {
            "scheme_id":            slug,
            "scheme_name":          name,
            "description":          full_text[:1500].strip(),
            "eligibility_criteria": extract_section(full_text, ["eligib", "who can", "patrta"]),
            "benefits":             extract_section(full_text, ["benefit", "labh", "what you get"]),
            "documents_required":   extract_section(full_text, ["document", "required", "dastavej"]),
            "application_process":  extract_section(full_text, ["apply", "application", "avedan"]),
            "ministry":             extract_section(full_text, ["ministry", "department", "mantralaya"], single_line=True),
            "applicable_states":    "All India",
            "official_link":        f"https://www.myscheme.gov.in/schemes/{slug}",
            "data_source":          "myscheme.gov.in (official PDF)",
            "last_fetched":         datetime.now().isoformat(),
            "full_text":            full_text[:5000],
        }

        # RAG text — what gets embedded
        scheme["rag_text"] = f"""
Scheme Name: {scheme['scheme_name']}
Ministry: {scheme['ministry']}
Description: {scheme['description'][:800]}
Eligibility: {scheme['eligibility_criteria'][:500]}
Benefits: {scheme['benefits'][:500]}
Documents: {scheme['documents_required'][:300]}
        """.strip()

        # Content hash for diff checking
        fingerprint = scheme["scheme_name"] + scheme["description"][:500]
        scheme["content_hash"] = hashlib.md5(
            fingerprint.encode("utf-8")
        ).hexdigest()

        return scheme

    except Exception as e:
        log.warning(f"Failed to extract {pdf_path.name}: {e}")
        return None


def extract_section(text: str, keywords: list, single_line: bool = False) -> str:
    """
    Finds a section in the PDF text by searching for keyword headings.
    Returns the text that follows that heading.
    """
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if any(kw in line.lower() for kw in keywords):
            if single_line:
                return line.strip()[:200]
            # Grab next 10 lines after the heading
            section_lines = lines[i+1 : i+12]
            section = " ".join(l.strip() for l in section_lines if l.strip())
            if section:
                return section[:800]
    return ""


# ── STEP 3: BUILD VECTOR STORE ────────────────────────
def build_vector_store(schemes: list[dict], store_dir: str = "./vector_store"):
    """
    Embeds all schemes and saves FAISS index + metadata.
    """
    import faiss
    from sentence_transformers import SentenceTransformer

    store_path = Path(store_dir)
    store_path.mkdir(parents=True, exist_ok=True)

    log.info("Loading embedding model...")
    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    log.info("Model ready")

    log.info(f"Embedding {len(schemes)} schemes...")
    texts = [s["rag_text"] for s in schemes]
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=16,
    ).astype(np.float32)

    # Build FAISS index
    index = faiss.IndexFlatL2(384)
    index.add(vectors)
    log.info(f"FAISS index built: {index.ntotal} vectors")

    # Save index
    faiss.write_index(index, str(store_path / "schemes.index"))

    # Save metadata (scheme details for lookup after search)
    with open(store_path / "schemes_metadata.json", "w", encoding="utf-8") as f:
        json.dump(schemes, f, ensure_ascii=False, indent=2)

    log.info(f"Vector store saved to {store_dir}/")
    log.info(f"Files: schemes.index + schemes_metadata.json")

    # Quick test search
    log.info("\n--- Test Searches ---")
    test_queries = [
        "tractor subsidy for small farmers",
        "crop insurance for wheat",
        "kisan ko beej ke liye paisa",
        "irrigation support drip",
    ]
    for query in test_queries:
        qvec = model.encode(
            [query], normalize_embeddings=True
        )[0].reshape(1, -1).astype(np.float32)
        distances, indices = index.search(qvec, 3)
        log.info(f"\nQuery: '{query}'")
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(schemes):
                score = round(1 / (1 + float(dist)), 3)
                log.info(f"  → {schemes[idx]['scheme_name']} (score: {score})")


# ── MAIN ──────────────────────────────────────────────
def main():
    log.info("=" * 55)
    log.info("KRISHI-SETU | Building Knowledge Base")
    log.info("=" * 55)

    # 1. Download PDFs
    pdf_dir = download_pdfs("./scheme_pdfs")

    # 2. Find all PDFs
    pdf_files = list(Path(pdf_dir).rglob("*.pdf"))
    log.info(f"Found {len(pdf_files)} PDF files")

    # 3. Extract text from each PDF
    log.info("Extracting text from PDFs...")
    all_schemes = []
    for i, pdf_path in enumerate(pdf_files):
        scheme = extract_from_pdf(pdf_path)
        if scheme:
            all_schemes.append(scheme)
        if (i + 1) % 50 == 0:
            log.info(f"  Processed {i+1}/{len(pdf_files)}")

    log.info(f"Successfully extracted: {len(all_schemes)} schemes")

    # 4. Filter for agriculture
    agri_schemes = [s for s in all_schemes if is_agriculture(s["full_text"])]
    log.info(f"Agriculture schemes: {len(agri_schemes)} / {len(all_schemes)}")

    # Save filtered schemes for reference
    with open("./vector_store/agriculture_schemes.json", "w", encoding="utf-8") as f:
        json.dump(agri_schemes, f, ensure_ascii=False, indent=2)

    # 5. Build vector store
    build_vector_store(agri_schemes)

    log.info("\n" + "=" * 55)
    log.info("PIPELINE COMPLETE")
    log.info(f"Agriculture schemes in knowledge base: {len(agri_schemes)}")
    log.info("Next step: run the FastAPI backend")
    log.info("=" * 55)


if __name__ == "__main__":
    main()