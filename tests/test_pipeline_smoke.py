"""离线冒烟：从合成 PDF 到 FAISS 检索，整条链路不依赖任何外部模型。"""

from pathlib import Path

from scripts.make_sample_pdf import build_pdf


def test_pdf_generation_works(tmp_path: Path):
    pdf = tmp_path / "doc.pdf"
    build_pdf(pdf)
    assert pdf.exists()
    assert pdf.stat().st_size > 5000  # 至少几 KB


def test_pipeline_load_split_index_retrieve(tmp_path: Path):
    pdf = tmp_path / "doc.pdf"
    build_pdf(pdf)

    from ragdoc.loaders import SplitConfig, load_and_split
    docs = load_and_split(pdf, SplitConfig(chunk_size=300, chunk_overlap=60))
    assert len(docs) >= 5
    for d in docs:
        assert "source" in d.metadata
        assert "section" in d.metadata or d.metadata.get("section") == ""

    from ragdoc.embeddings import HashingEmbeddings
    from ragdoc.vectorstore import build_index, retrieve_with_score

    vs = build_index(docs, HashingEmbeddings(dim=128))
    hits = retrieve_with_score(vs, "深圳住宿每晚多少", k=3)
    assert len(hits) == 3
    top_text = "\n".join(d.page_content for d, _ in hits)
    assert "600" in top_text  # top-3 至少应包含住宿标准 600
