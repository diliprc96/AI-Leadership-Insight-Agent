"""Quick smoke test for PDF parsing."""
import sys
sys.path.insert(0, ".")
from leadership_agent.ingestion.pdf_parser import parse_document
from pathlib import Path

chunks = parse_document(Path("data/raw/test.pdf"))
print(f"PDF chunks produced: {len(chunks)}")
if chunks:
    m = chunks[0]["metadata"]
    print(f"Metadata: company={m['company']}, year={m['year']}, section={m['section']}, file={m['source_file']}")
    print(f"Text preview: {chunks[0]['text'][:200]}")
else:
    print("WARNING: No chunks produced!")
