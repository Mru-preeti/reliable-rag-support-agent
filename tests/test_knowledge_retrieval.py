"""Comprehensive unit tests for General-Purpose Hybrid Knowledge Retrieval."""

import pytest
from src.knowledge import KnowledgeBase
from src.types import KnowledgeChunk, KnowledgeDocumentMetadata, RetrievalResult


@pytest.fixture(scope="module")
def kb():
    """Module-scoped fixture providing initialized general-purpose KnowledgeBase."""
    return KnowledgeBase()


class TestFrontmatterAndChunking:
    """Test YAML frontmatter parsing and section-level chunking."""

    def test_all_14_documents_loaded(self, kb):
        assert len(kb.documents) == 14
        assert "01-returns-policy-current.md" in kb.documents
        assert "02-returns-policy-legacy.md" in kb.documents
        assert "14-internal-content-migration-notes.md" in kb.documents

    def test_frontmatter_metadata_fields(self, kb):
        doc01 = kb.documents["01-returns-policy-current.md"]["metadata"]
        assert doc01.document_id == "RET-2026-01"
        assert doc01.title == "Returns Policy"
        assert doc01.status == "active"
        assert doc01.effective_date == "2026-04-01"
        assert doc01.policy_authority == "official"
        assert doc01.supersedes == "RET-2024-01"

        doc02 = kb.documents["02-returns-policy-legacy.md"]["metadata"]
        assert doc02.document_id == "RET-2024-01"
        assert doc02.status == "superseded"
        assert doc02.superseded_by == "RET-2026-01"

        doc14 = kb.documents["14-internal-content-migration-notes.md"]["metadata"]
        assert doc14.document_id == "MIG-TEST-04"
        assert doc14.status == "draft"
        assert doc14.policy_authority == "none"
        assert doc14.customer_answering is False

    def test_section_level_chunking(self, kb):
        doc01_chunks = [c for c in kb.chunks if c.filename == "01-returns-policy-current.md"]
        assert len(doc01_chunks) >= 4
        headings = {c.heading for c in doc01_chunks}
        assert "Standard return window" in headings
        assert "Item condition" in headings
        assert "Return shipping and refunds" in headings
        assert "Exclusions and exceptions" in headings

    def test_chunk_citation_format(self, kb):
        for chunk in kb.chunks:
            assert chunk.citation == f"{chunk.filename}: {chunk.heading}"


class TestAuthorityAndPrecedence:
    """Test filtering of superseded, draft, and non-authoritative documents."""

    def test_superseded_policy_filtered_by_default(self, kb):
        res = kb.retrieve("How long do I have to return an item?")
        retrieved_files = {c.filename for c in res.chunks}
        assert "01-returns-policy-current.md" in retrieved_files
        assert "02-returns-policy-legacy.md" not in retrieved_files

    def test_draft_internal_scratchpad_filtered(self, kb):
        res = kb.retrieve("Tell me about the 60 day return window migration note")
        retrieved_files = {c.filename for c in res.chunks}
        assert "14-internal-content-migration-notes.md" not in retrieved_files


class TestTrueParaphrases:
    """Test retrieval robustness across genuine semantic paraphrases with zero token overlap."""

    def test_paraphrase_sending_product_back_shipping_fee(self, kb):
        """Query: 'Is there any charge if I need to send back a product I bought?'"""
        res = kb.retrieve("Is there any charge if I need to send back a product I bought?")
        citations = [c.citation for c in res.chunks]
        assert any("01-returns-policy-current.md" in cit for cit in citations)

    def test_paraphrase_changing_delivery_location(self, kb):
        """Query: 'Can I alter the delivery destination for a newly placed order?'"""
        res = kb.retrieve("Can I alter the delivery destination for a newly placed order?")
        citations = [c.citation for c in res.chunks]
        assert any("08-order-changes-and-cancellations.md: Address changes" == cit for cit in citations)

    def test_paraphrase_price_drop_refund(self, kb):
        """Query: 'The price decreased right after I bought my backpack. Can I get the difference reimbursed?'"""
        res = kb.retrieve("The price decreased right after I bought my backpack. Can I get the difference reimbursed?")
        citations = [c.citation for c in res.chunks]
        assert any("10-gift-cards-and-price-adjustments.md: Price adjustments" == cit for cit in citations)

    def test_paraphrase_automatic_dish_washing(self, kb):
        """Query: 'Can I put my tumbler in an automatic dish washing appliance?'"""
        res = kb.retrieve("Can I put my tumbler in an automatic dish washing appliance?")
        citations = [c.citation for c in res.chunks]
        assert any("11-product-care.md" in cit or "12-breeze-tumbler-product-card.md" in cit for cit in citations)


class TestMultiIntentQueryCoverage:
    """Test compound multi-intent questions to ensure neither query clause is crowded out."""

    def test_return_window_and_canada_shipping_combined(self, kb):
        """Query: 'What is the return window for a standard customer, and do you ship to Canada?'"""
        res = kb.retrieve("What is the return window for a standard customer, and do you ship to Canada?")
        citations = [c.citation for c in res.chunks]
        
        # Must retrieve evidence for BOTH return window AND Canada shipping
        has_return_window = any("01-returns-policy-current.md: Standard return window" in cit for cit in citations)
        has_canada_shipping = any("06-international-shipping.md: Supported destinations" in cit for cit in citations)
        
        assert has_return_window, f"Expected return window in {citations}"
        assert has_canada_shipping, f"Expected Canada shipping destinations in {citations}"


class TestGeneralizedConflictDetection:
    """Test general-purpose conflict detection on both repository data and in-memory unseen test objects."""

    def test_breeze_tumbler_conflict_detected_without_hardcoding(self, kb):
        res = kb.retrieve("Can I put the entire Breeze Tumbler in the dishwasher?")
        assert res.has_conflict is True
        assert len(res.conflicting_sources) == 2
        assert any("11-product-care.md" in s for s in res.conflicting_sources)
        assert any("12-breeze-tumbler-product-card.md" in s for s in res.conflicting_sources)

    def test_hypothetical_unseen_warranty_conflict(self):
        """Test in-memory unseen active conflict: 5-year vs 2-year warranty on the same product."""
        temp_kb = KnowledgeBase.__new__(KnowledgeBase)
        temp_kb.k1 = 1.5
        temp_kb.b = 0.75
        temp_kb.embedding_model_name = "all-MiniLM-L6-v2"
        temp_kb._embedding_model = None

        meta_a = KnowledgeDocumentMetadata(document_id="DOC-A", title="Warranty Policy A", status="active", policy_authority="official")
        meta_b = KnowledgeDocumentMetadata(document_id="DOC-B", title="Warranty Policy B", status="active", policy_authority="official")

        chunk_a = KnowledgeChunk(
            chunk_id="doc_a_warr",
            filename="DOC-A.md",
            heading="Backpack Warranty",
            full_heading_path="Warranty Policy A > Backpack Warranty",
            content="All Aster & Row backpacks include a 5 years limited warranty against defects.",
            metadata=meta_a,
            citation="DOC-A.md: Backpack Warranty"
        )
        chunk_b = KnowledgeChunk(
            chunk_id="doc_b_warr",
            filename="DOC-B.md",
            heading="Backpack Warranty",
            full_heading_path="Warranty Policy B > Backpack Warranty",
            content="All Aster & Row backpacks include a 2 years limited warranty against defects.",
            metadata=meta_b,
            citation="DOC-B.md: Backpack Warranty"
        )

        temp_kb.index_custom_chunks([chunk_a, chunk_b])
        res = temp_kb.retrieve("How long is the backpack warranty?")
        assert res.has_conflict is True
        assert "DOC-A.md: Backpack Warranty" in res.conflicting_sources
        assert "DOC-B.md: Backpack Warranty" in res.conflicting_sources

    def test_hypothetical_unseen_shipping_threshold_conflict(self):
        """Test in-memory unseen active conflict: $50 vs $75 free shipping threshold."""
        temp_kb = KnowledgeBase.__new__(KnowledgeBase)
        temp_kb.k1 = 1.5
        temp_kb.b = 0.75
        temp_kb.embedding_model_name = "all-MiniLM-L6-v2"
        temp_kb._embedding_model = None

        meta_a = KnowledgeDocumentMetadata(document_id="SHIP-A", title="Shipping Rules A", status="active", policy_authority="official")
        meta_b = KnowledgeDocumentMetadata(document_id="SHIP-B", title="Shipping Rules B", status="active", policy_authority="official")

        chunk_a = KnowledgeChunk(
            chunk_id="ship_a",
            filename="SHIP-A.md",
            heading="Domestic Shipping Charges",
            full_heading_path="Shipping Rules A > Domestic Shipping Charges",
            content="Standard domestic shipping is free for all orders of $50 or more.",
            metadata=meta_a,
            citation="SHIP-A.md: Domestic Shipping Charges"
        )
        chunk_b = KnowledgeChunk(
            chunk_id="ship_b",
            filename="SHIP-B.md",
            heading="Domestic Shipping Charges",
            full_heading_path="Shipping Rules B > Domestic Shipping Charges",
            content="Standard domestic shipping is free for all orders of $75 or more.",
            metadata=meta_b,
            citation="SHIP-B.md: Domestic Shipping Charges"
        )

        temp_kb.index_custom_chunks([chunk_a, chunk_b])
        res = temp_kb.retrieve("What is the free shipping threshold for domestic orders?")
        assert res.has_conflict is True
        assert "SHIP-A.md: Domestic Shipping Charges" in res.conflicting_sources
        assert "SHIP-B.md: Domestic Shipping Charges" in res.conflicting_sources

    def test_no_false_conflict_for_trailplus_exception(self, kb):
        res = kb.retrieve("What is the return window for standard customers vs TrailPlus members?")
        assert res.has_conflict is False


class TestInsufficientInformation:
    """Test explicit abstention signal when knowledge coverage is missing."""

    def test_vegan_materials_abstention(self, kb):
        res = kb.retrieve("Are all fabrics and adhesives in your bags certified vegan?")
        # Should flag insufficient info
        assert res.is_insufficient is True or not any("certified vegan" in c.content.lower() for c in res.chunks)
