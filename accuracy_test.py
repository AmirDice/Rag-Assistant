"""Quick RAG accuracy test — checks if the expected doc appears in top results."""
import json
import subprocess
import sys

API = "http://localhost:8000"

# Each test: (query, primary_expected_doc_substring, acceptable_alternatives)
# primary = the MOST specific doc that should answer this
# alternatives = general-purpose docs that also validly answer it
# Generic test set for the bundled Acme Cloud demo corpus (see demo_docs/).
TESTS = [
    (
        "How do I sign in with single sign-on?",
        "getting_started",
        ["admin_security"],
    ),
    (
        "How do I enable two-factor authentication for my team?",
        "admin_security",
        [],
    ),
    (
        "How do I upgrade my plan?",
        "billing",
        [],
    ),
    (
        "Why did my file upload fail?",
        "troubleshooting",
        [],
    ),
    (
        "How do I create an API key?",
        "admin_security",
        ["troubleshooting"],
    ),
    (
        "How do I share a file with someone outside my team?",
        "getting_started",
        [],
    ),
    (
        "What plans are available and what do they include?",
        "billing",
        [],
    ),
    (
        "How do I recover a deleted file?",
        "troubleshooting",
        [],
    ),
]

def run_query(question, top_k=5):
    payload = json.dumps({
        "question": question,
        "tenant_id": "demo",
        "top_k": top_k,
    })
    result = subprocess.run(
        ["curl", "-s", "-X", "POST", f"{API}/query",
         "-H", "Content-Type: application/json",
         "-d", payload],
        capture_output=True, text=True, timeout=30
    )
    try:
        return json.loads(result.stdout)
    except:
        return None

def check_match(source_doc, expected, alternatives):
    """Returns: 'primary', 'alternative', or 'miss'."""
    if not source_doc:
        return "miss"
    for pattern in [expected]:
        if pattern.lower() in source_doc.lower():
            return "primary"
    for alt in alternatives:
        if alt.lower() in source_doc.lower():
            return "alternative"
    return "miss"

print("=" * 90)
print("RAG ACCURACY TEST")
print("=" * 90)

top1_primary = 0
top3_primary = 0
top1_relevant = 0  # primary OR alternative
top3_relevant = 0
total = len(TESTS)

for i, (query, expected, alts) in enumerate(TESTS, 1):
    print(f"\n{'─'*90}")
    print(f"Q{i}: {query}")
    print(f"    Expected: {expected}")
    
    resp = run_query(query)
    if not resp or "answer_chunks" not in resp:
        print(f"    ERROR: No response from API")
        continue
    
    chunks = resp["answer_chunks"]
    if not chunks:
        print(f"    NO RESULTS")
        continue
    
    # Show top 3 results
    for j, chunk in enumerate(chunks[:3]):
        doc = chunk.get("source_doc", "?")
        score = chunk.get("score", 0)
        match = check_match(doc, expected, alts)
        marker = " ✓ PRIMARY" if match == "primary" else (" ~ ALT" if match == "alternative" else "")
        print(f"    #{j+1} [{score:.3f}] {doc}{marker}")
    
    # Score top-1
    top1_match = check_match(chunks[0].get("source_doc", ""), expected, alts)
    if top1_match == "primary":
        top1_primary += 1
        top1_relevant += 1
    elif top1_match == "alternative":
        top1_relevant += 1
    
    # Score top-3
    found_primary = False
    found_relevant = False
    for chunk in chunks[:3]:
        m = check_match(chunk.get("source_doc", ""), expected, alts)
        if m == "primary":
            found_primary = True
            found_relevant = True
        elif m == "alternative":
            found_relevant = True
    if found_primary:
        top3_primary += 1
    if found_relevant:
        top3_relevant += 1

print(f"\n{'='*90}")
print(f"RESULTS ({total} queries)")
print(f"{'='*90}")
print(f"  Top-1 Primary Match:   {top1_primary}/{total} ({top1_primary/total*100:.0f}%)")
print(f"  Top-1 Relevant Match:  {top1_relevant}/{total} ({top1_relevant/total*100:.0f}%)")
print(f"  Top-3 Primary Match:   {top3_primary}/{total} ({top3_primary/total*100:.0f}%)")
print(f"  Top-3 Relevant Match:  {top3_relevant}/{total} ({top3_relevant/total*100:.0f}%)")
print(f"\nKey:")
print(f"  Primary  = the MOST specific document for the query")
print(f"  Relevant = primary OR an acceptable alternative doc")
