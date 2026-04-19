# save this as check_raw.py and run it
import json

with open("./vector_store/raw_schemes.json") as f:
    schemes = json.load(f)

# Check first 3 schemes
for s in schemes[:3]:
    print("=" * 50)
    print("Name:", s.get("scheme_name"))
    print("Description:", s.get("description")[:200] if s.get("description") else "EMPTY")
    print("Eligibility:", s.get("eligibility_criteria")[:200] if s.get("eligibility_criteria") else "EMPTY")
    print("RAG text:", s.get("rag_text")[:300] if s.get("rag_text") else "EMPTY")