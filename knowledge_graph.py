"""
KRISHI-SETU | Knowledge Graph
==============================
Builds a NetworkX graph from your 1350 scheme documents.

Nodes:
  - Farmer (with attributes: state, crop, land_size, income)
  - Scheme (with attributes: name, ministry, benefits)
  - Crop (wheat, rice, cotton etc.)
  - Region (states)
  - Document (aadhaar, land record etc.)

Relationships:
  - Farmer --GROWS--> Crop
  - Farmer --LOCATED_IN--> Region
  - Scheme --APPLICABLE_IN--> Region
  - Scheme --COVERS--> Crop
  - Scheme --REQUIRES--> Document
  - Farmer --ELIGIBLE_FOR--> Scheme (computed)

Run: python knowledge_graph.py (to build and save)
Import: from knowledge_graph import KrishiSetuGraph
"""

import json
import pickle
import re
import logging
from pathlib import Path

import networkx as nx

log = logging.getLogger("krishi-graph")

# ── Document keywords — what papers schemes typically need ──
DOCUMENT_PATTERNS = {
    "Aadhaar Card":       ["aadhaar", "aadhar", "uid", "identity"],
    "Land Record":        ["land record", "khasra", "khatauni", "patta", "jamabandi"],
    "Bank Passbook":      ["bank passbook", "bank account", "bank details"],
    "Caste Certificate":  ["caste certificate", "sc ", "st ", "obc"],
    "Income Certificate": ["income certificate", "income proof", "bpl"],
    "Farmer ID":          ["farmer id", "kisan id", "farmer registration"],
    "Crop Insurance":     ["crop insurance", "fasal bima"],
    "Photo":              ["passport photo", "photograph"],
    "Mobile Number":      ["mobile number", "phone number"],
}

# ── Crop keywords ──────────────────────────────────────────
CROP_PATTERNS = {
    "Wheat":       ["wheat", "gehun", "gehu"],
    "Rice":        ["rice", "paddy", "dhan"],
    "Cotton":      ["cotton", "kapas"],
    "Sugarcane":   ["sugarcane", "ganna"],
    "Maize":       ["maize", "corn", "makka"],
    "Pulses":      ["pulses", "dal", "lentil", "gram", "chana"],
    "Oilseeds":    ["oilseed", "mustard", "sarson", "sunflower", "soybean"],
    "Vegetables":  ["vegetable", "sabzi", "horticulture"],
    "Fruits":      ["fruit", "mango", "banana", "apple", "orchard"],
    "Spices":      ["spice", "turmeric", "chilli", "pepper"],
    "All Crops":   ["all crop", "any crop", "kharif", "rabi"],
}

# ── State names ────────────────────────────────────────────
ALL_STATES = [
    "Uttar Pradesh", "Punjab", "Haryana", "Rajasthan", "Bihar",
    "Madhya Pradesh", "Maharashtra", "Gujarat", "Karnataka",
    "Andhra Pradesh", "Telangana", "West Bengal", "Odisha",
    "Assam", "Himachal Pradesh", "Uttarakhand", "Jharkhand",
    "Chhattisgarh", "Tamil Nadu", "Kerala", "Goa", "Sikkim",
    "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Tripura",
    "Arunachal Pradesh", "All India",
]


class KrishiSetuGraph:
    """
    Knowledge Graph for Krishi-Setu.

    WHY A KNOWLEDGE GRAPH?
    ----------------------
    FAISS finds semantically similar schemes — great for discovery.
    But it can't reason: "Is this farmer with 3 acres eligible for
    a scheme that requires land < 2 acres?" That's deterministic
    logic, not semantic similarity.

    The Knowledge Graph handles eligibility reasoning explicitly.
    Nodes = entities (farmers, schemes, crops, regions, documents)
    Edges = relationships (eligible_for, covers, requires, located_in)

    Together: FAISS discovers candidates → Graph verifies eligibility.
    This is the architecture your synopsis describes.
    """

    def __init__(self):
        self.graph = nx.DiGraph()  # Directed graph — relationships have direction
        self.scheme_index = {}     # scheme_id → node data for fast lookup

    def build(self, schemes: list[dict]) -> None:
        """
        Builds the full knowledge graph from scheme documents.
        Called once — graph is then saved and reloaded.
        """
        log.info(f"Building knowledge graph from {len(schemes)} schemes...")

        # ── Add region nodes ──────────────────────────────
        for state in ALL_STATES:
            self.graph.add_node(
                f"region:{state}",
                node_type="region",
                name=state
            )

        # ── Add crop nodes ────────────────────────────────
        for crop in CROP_PATTERNS:
            self.graph.add_node(
                f"crop:{crop}",
                node_type="crop",
                name=crop
            )

        # ── Add document nodes ────────────────────────────
        for doc in DOCUMENT_PATTERNS:
            self.graph.add_node(
                f"doc:{doc}",
                node_type="document",
                name=doc
            )

        # ── Add scheme nodes + relationships ──────────────
        for scheme in schemes:
            sid = scheme.get("scheme_id", "")
            if not sid:
                continue

            node_id = f"scheme:{sid}"
            full_text = (
                scheme.get("full_text", "") +
                scheme.get("rag_text", "") +
                scheme.get("eligibility_criteria", "")
            ).lower()

            # Parse eligibility rules from text
            land_limit  = self._extract_land_limit(full_text)
            income_limit = self._extract_income_limit(full_text)
            small_farmer = self._is_small_farmer_scheme(full_text)

            # Add scheme node
            self.graph.add_node(
                node_id,
                node_type="scheme",
                scheme_id=sid,
                scheme_name=scheme.get("scheme_name", sid),
                ministry=scheme.get("ministry", ""),
                benefits=scheme.get("benefits", "")[:300],
                eligibility_text=scheme.get("eligibility_criteria", "")[:500],
                land_limit_acres=land_limit,
                income_limit=income_limit,
                small_farmer_only=small_farmer,
                official_link=scheme.get("official_link", ""),
            )

            # Index for fast lookup
            self.scheme_index[sid] = node_id

            # ── Scheme → Region edges ─────────────────────
            applicable_states = scheme.get("applicable_states", "All India")
            if "all india" in applicable_states.lower() or not applicable_states:
                # Applicable in all states
                for state in ALL_STATES:
                    self.graph.add_edge(
                        node_id, f"region:{state}",
                        relation="APPLICABLE_IN"
                    )
            else:
                for state in ALL_STATES:
                    if state.lower() in applicable_states.lower():
                        self.graph.add_edge(
                            node_id, f"region:{state}",
                            relation="APPLICABLE_IN"
                        )

            # ── Scheme → Crop edges ───────────────────────
            covers_any = False
            for crop, keywords in CROP_PATTERNS.items():
                if any(kw in full_text for kw in keywords):
                    self.graph.add_edge(
                        node_id, f"crop:{crop}",
                        relation="COVERS"
                    )
                    covers_any = True

            # If no specific crop found, assume all crops
            if not covers_any:
                self.graph.add_edge(
                    node_id, "crop:All Crops",
                    relation="COVERS"
                )

            # ── Scheme → Document edges ───────────────────
            doc_text = scheme.get("documents_required", "").lower()
            for doc, keywords in DOCUMENT_PATTERNS.items():
                if any(kw in doc_text or kw in full_text[:2000] for kw in keywords):
                    self.graph.add_edge(
                        node_id, f"doc:{doc}",
                        relation="REQUIRES"
                    )

        log.info(
            f"Graph built — "
            f"{self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )

    def check_eligibility(
        self,
        scheme_ids: list[str],
        farmer: dict
    ) -> list[dict]:
        """
        THE CORE REASONING FUNCTION.

        Takes a list of candidate scheme IDs (from FAISS search)
        and a farmer profile, then reasons over the graph to determine:
        - Is this farmer likely eligible?
        - What documents will they need?
        - Why are they eligible or not?

        This is deterministic reasoning — not LLM guessing.
        The graph encodes the rules; we traverse them explicitly.
        """
        results = []

        farmer_state = farmer.get("state", "")
        farmer_crop  = farmer.get("crop", "")
        farmer_land  = float(farmer.get("land_size", 0))
        farmer_income = farmer.get("income_band", "")

        for sid in scheme_ids:
            node_id = self.scheme_index.get(sid)
            if not node_id or node_id not in self.graph:
                continue

            node_data = self.graph.nodes[node_id]
            reasons_eligible = []
            reasons_not_eligible = []

            # ── Check 1: State applicability ─────────────
            state_node = f"region:{farmer_state}"
            state_applicable = self.graph.has_edge(node_id, state_node)
            all_india = self.graph.has_edge(node_id, "region:All India")

            if state_applicable or all_india:
                reasons_eligible.append(f"Available in {farmer_state}")
            else:
                reasons_not_eligible.append(f"Not available in {farmer_state}")

            # ── Check 2: Crop coverage ────────────────────
            crop_matched = False
            if farmer_crop:
                for crop_name, keywords in CROP_PATTERNS.items():
                    if any(kw in farmer_crop.lower() for kw in keywords):
                        crop_node = f"crop:{crop_name}"
                        if self.graph.has_edge(node_id, crop_node) or \
                           self.graph.has_edge(node_id, "crop:All Crops"):
                            reasons_eligible.append(f"Covers {farmer_crop}")
                            crop_matched = True
                            break

                if not crop_matched:
                    if self.graph.has_edge(node_id, "crop:All Crops"):
                        reasons_eligible.append("Covers all crops")
                        crop_matched = True

            # ── Check 3: Land size limit ──────────────────
            land_limit = node_data.get("land_limit_acres")
            if land_limit and farmer_land > 0:
                if farmer_land <= land_limit:
                    reasons_eligible.append(
                        f"Land size {farmer_land} acres within limit ({land_limit} acres)"
                    )
                else:
                    reasons_not_eligible.append(
                        f"Land size {farmer_land} acres exceeds limit ({land_limit} acres)"
                    )

            # ── Check 4: Small farmer preference ─────────
            if node_data.get("small_farmer_only") and farmer_land > 5:
                reasons_not_eligible.append("Scheme preferred for small/marginal farmers (< 5 acres)")

            # ── Get required documents from graph ─────────
            required_docs = [
                self.graph.nodes[neighbor]["name"]
                for neighbor in self.graph.successors(node_id)
                if self.graph.nodes[neighbor].get("node_type") == "document"
            ]

            # ── Compute eligibility score ─────────────────
            total_checks = len(reasons_eligible) + len(reasons_not_eligible)
            if total_checks == 0:
                eligibility_score = 0.5
            else:
                eligibility_score = len(reasons_eligible) / total_checks

            likely_eligible = eligibility_score >= 0.5 and len(reasons_not_eligible) == 0

            results.append({
                "scheme_id":        sid,
                "scheme_name":      node_data.get("scheme_name", sid),
                "ministry":         node_data.get("ministry", ""),
                "benefits_preview": node_data.get("benefits", ""),
                "eligibility_text": node_data.get("eligibility_text", ""),
                "official_link":    node_data.get("official_link", ""),
                "likely_eligible":  likely_eligible,
                "eligibility_score":round(eligibility_score, 2),
                "reasons_eligible":     reasons_eligible,
                "reasons_not_eligible": reasons_not_eligible,
                "required_documents":   required_docs if required_docs else [
                    "Aadhaar Card", "Land Record", "Bank Passbook"
                ],
            })

        # Sort by eligibility score — most eligible first
        results.sort(key=lambda x: x["eligibility_score"], reverse=True)
        return results

    def get_graph_stats(self) -> dict:
        """Returns stats about the graph — useful for presentation."""
        node_types = {}
        for _, data in self.graph.nodes(data=True):
            t = data.get("node_type", "unknown")
            node_types[t] = node_types.get(t, 0) + 1

        return {
            "total_nodes":  self.graph.number_of_nodes(),
            "total_edges":  self.graph.number_of_edges(),
            "node_types":   node_types,
            "scheme_count": node_types.get("scheme", 0),
            "region_count": node_types.get("region", 0),
            "crop_count":   node_types.get("crop", 0),
            "doc_count":    node_types.get("document", 0),
        }

    # ── Private helpers ────────────────────────────────────

    def _extract_land_limit(self, text: str) -> float | None:
        """
        Extracts land size limits from scheme text.
        e.g. "land holding up to 2 hectares" → 4.94 acres
             "less than 5 acres" → 5.0
        """
        # Look for acre patterns
        acre_match = re.search(
            r'(\d+\.?\d*)\s*(?:acre|acres)', text
        )
        if acre_match:
            return float(acre_match.group(1))

        # Look for hectare patterns and convert
        hectare_match = re.search(
            r'(\d+\.?\d*)\s*(?:hectare|hectares|ha)', text
        )
        if hectare_match:
            return round(float(hectare_match.group(1)) * 2.47, 2)

        return None

    def _extract_income_limit(self, text: str) -> str | None:
        """Extracts income limits from scheme text."""
        income_match = re.search(
            r'(?:income|annual income)[^\d]*(\d[\d,]*)\s*(?:rupees|rs|₹)?',
            text
        )
        if income_match:
            return income_match.group(1)
        return None

    def _is_small_farmer_scheme(self, text: str) -> bool:
        """Checks if scheme is specifically for small/marginal farmers."""
        return any(kw in text for kw in [
            "small farmer", "marginal farmer",
            "small and marginal", "small & marginal",
            "laghu kisan", "seema kisan"
        ])

    def save(self, path: str = "./vector_store/knowledge_graph.pkl"):
        """Saves graph to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "graph": self.graph,
                "scheme_index": self.scheme_index
            }, f)
        log.info(f"Knowledge graph saved to {path}")

    @classmethod
    def load(cls, path: str = "./vector_store/knowledge_graph.pkl") -> "KrishiSetuGraph":
        """Loads graph from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        kg = cls()
        kg.graph = data["graph"]
        kg.scheme_index = data["scheme_index"]
        log.info(
            f"Knowledge graph loaded — "
            f"{kg.graph.number_of_nodes()} nodes, "
            f"{kg.graph.number_of_edges()} edges"
        )
        return kg


# ── Build and save when run directly ──────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    # Load schemes from pipeline output
    schemes_path = "./vector_store/agriculture_schemes.json"
    if not Path(schemes_path).exists():
        print(f"ERROR: {schemes_path} not found.")
        print("Run build_pipeline.py first.")
        exit(1)

    with open(schemes_path, encoding="utf-8") as f:
        schemes = json.load(f)

    log.info(f"Loaded {len(schemes)} schemes")

    # Build graph
    kg = KrishiSetuGraph()
    kg.build(schemes)

    # Print stats
    stats = kg.get_graph_stats()
    print("\n=== Knowledge Graph Stats ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Test eligibility check
    print("\n=== Test Eligibility Check ===")
    test_farmer = {
        "state": "Punjab",
        "crop": "wheat",
        "land_size": 2.0,
        "income_band": "below 1 lakh"
    }
    # Use first 3 scheme IDs as test
    test_ids = [s["scheme_id"] for s in schemes[:3]]
    results = kg.check_eligibility(test_ids, test_farmer)
    for r in results:
        print(f"\nScheme: {r['scheme_name']}")
        print(f"  Eligible: {r['likely_eligible']} (score: {r['eligibility_score']})")
        print(f"  Reasons for: {r['reasons_eligible']}")
        print(f"  Reasons against: {r['reasons_not_eligible']}")
        print(f"  Docs needed: {r['required_documents'][:3]}")

    # Save
    kg.save()
    print("\n✅ Knowledge graph built and saved!")
    print("Next: python main.py")