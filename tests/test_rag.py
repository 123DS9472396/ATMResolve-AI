from src.rag.citations import format_citation

def test_citation_shows_id_title_and_source():
    value = format_citation({"citation_id": "POL-1", "title": "ATM policy", "source_url": "https://example.org/policy"})
    assert "POL-1" in value
    assert "ATM policy" in value
    assert "https://example.org/policy" in value
