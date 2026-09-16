from datetime import datetime
from typing import List, Optional
from uuid import UUID

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


class MarketCompanyLink(Base):
    __tablename__ = "market_company_link"

    market_id: Mapped[UUID] = mapped_column(
        ForeignKey("markets.id", ondelete="CASCADE"),
        primary_key=True
    )
    company_id: Mapped[UUID] = mapped_column(
        ForeignKey("companies.id", ondelete="CASCADE"),
        primary_key=True
    )


class MarketMetricLink(Base):
    __tablename__ = "market_metric_link"

    market_id: Mapped[UUID] = mapped_column(
        ForeignKey("markets.id", ondelete="CASCADE"),
        primary_key=True
    )
    metric_id: Mapped[UUID] = mapped_column(
        ForeignKey("esg_metric_definitions.id", ondelete="CASCADE"),
        primary_key=True
    )


class Market(Base):
    __tablename__ = "markets"

    id: Mapped[UUID] = mapped_column(
        primary_key=True,
        server_default=text("gen_random_uuid()")
    )
    name: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    sector_code: Mapped[str | None] = mapped_column(String(50), nullable=True)
    sasb_sector: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(50), default="pending", index=True, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("now()"))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("now()"), onupdate=text("now()"))

    companies: Mapped[list["Company"]] = relationship(
        secondary="market_company_link",
        back_populates="markets"
    )
    metrics: Mapped[list["ESGMetricDefinition"]] = relationship(
        secondary="market_metric_link",
        back_populates="markets"
    )


class Company(Base):
    __tablename__ = "companies"

    id: Mapped[UUID] = mapped_column(
        primary_key=True,
        server_default=text("gen_random_uuid()")
    )
    name: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    country: Mapped[str | None] = mapped_column(String(70), nullable=True)
    ticker: Mapped[str | None] = mapped_column(String(20), nullable=True)
    has_public_esg: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    esg_scoring: Mapped[str | None] = mapped_column(String(20), default="pending", nullable=True, index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("now()"))

    markets: Mapped[list[Market]] = relationship(
        secondary="market_company_link",
        back_populates="companies"
    )
    metric_values: Mapped[list["CompanyMetricValue"]] = relationship(
        back_populates="company",
        cascade="all, delete-orphan"
    )
    esg_signals: Mapped[list["CompanyESGSignals"]] = relationship(
        back_populates="company",
        cascade="all, delete-orphan"
    )
    evidence_claims: Mapped[list["CompanyEvidenceClaim"]] = relationship(
        back_populates="company",
        cascade="all, delete-orphan"
    )


class ESGMetricDefinition(Base):
    __tablename__ = "esg_metric_definitions"

    id: Mapped[UUID] = mapped_column(
        primary_key=True,
        server_default=text("gen_random_uuid()")
    )
    key: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    unit: Mapped[str | None] = mapped_column(String(50), nullable=True)
    category: Mapped[str] = mapped_column(String(1), nullable=False)       # E, S, or G
    sasb_sector: Mapped[str | None] = mapped_column(String(50), nullable=True)  # null = universal
    source_framework: Mapped[str | None] = mapped_column(String(50), nullable=True)
    is_universal: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    markets: Mapped[list[Market]] = relationship(
        secondary="market_metric_link",
        back_populates="metrics"
    )
    company_values: Mapped[list["CompanyMetricValue"]] = relationship(
        back_populates="metric",
        cascade="all, delete-orphan"
    )


class CompanyMetricValue(Base):
    __tablename__ = "company_metric_values"

    id: Mapped[UUID] = mapped_column(
        primary_key=True,
        server_default=text("gen_random_uuid()")
    )
    company_id: Mapped[UUID] = mapped_column(
        ForeignKey("companies.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    metric_id: Mapped[UUID] = mapped_column(
        ForeignKey("esg_metric_definitions.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    value: Mapped[str | None] = mapped_column(Text, nullable=True)
    numeric_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    reporting_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source: Mapped[str | None] = mapped_column(String(100), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)   # 0.0 – 1.0
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)       # explainability text (ESG pillar scores only)
    # Ensemble scorer uncertainty output (agentic_ensemble_v1 source only;
    # NULL for every other source -- see db_migrations/005_ensemble_cutover.sql
    # and agentic_estimation/layer_3/confidence_gate.py).
    low_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    high_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence_label: Mapped[str | None] = mapped_column(String(10), nullable=True)  # 'high'|'medium'|'low'
    verdict: Mapped[str | None] = mapped_column(String(30), nullable=True)   # 'skipped'|'passed'|'passed_after_retry'|'refuted'
    needs_review: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    __table_args__ = (
        UniqueConstraint("company_id", "metric_id", "reporting_year", "source",
                         name="uq_company_metric_year_source"),
    )

    company: Mapped[Company] = relationship(back_populates="metric_values")
    metric: Mapped[ESGMetricDefinition] = relationship(back_populates="company_values")


class CompanyESGSignals(Base):
    __tablename__ = "company_esg_signals"

    id: Mapped[UUID] = mapped_column(
        primary_key=True,
        server_default=text("gen_random_uuid()")
    )
    company_id: Mapped[UUID] = mapped_column(
        ForeignKey("companies.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    source: Mapped[str] = mapped_column(String(60), nullable=False)
    signal_text: Mapped[str] = mapped_column(Text, nullable=False)
    gathered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), nullable=False
    )

    __table_args__ = (
        UniqueConstraint("company_id", "source", name="uq_company_signal_source"),
    )

    company: Mapped["Company"] = relationship(back_populates="esg_signals")


class CompanyEvidenceClaim(Base):
    """
    Layer 2 Extractor output: a single typed, source-attributed claim about a
    company's E/S/G posture. Replaces freehand LLM scores — see
    db_migrations/002_evidence_claims.sql for the "no source, no claim" rule
    enforced at the DB level.
    """
    __tablename__ = "company_evidence_claims"

    id: Mapped[UUID] = mapped_column(
        primary_key=True,
        server_default=text("gen_random_uuid()")
    )
    company_id: Mapped[UUID] = mapped_column(
        ForeignKey("companies.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    pillar: Mapped[str] = mapped_column(String(1), nullable=False)          # E, S, or G
    factor: Mapped[str] = mapped_column(String(100), nullable=False, index=True)

    polarity: Mapped[int] = mapped_column(Integer, nullable=False)           # -1, 0, +1
    strength: Mapped[float] = mapped_column(Float, nullable=False)           # 0-1
    confidence: Mapped[float] = mapped_column(Float, nullable=False)         # 0-1
    value: Mapped[float | None] = mapped_column(Float, nullable=True)
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)

    source_signal_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("company_esg_signals.id", ondelete="SET NULL"), nullable=True
    )
    source_note: Mapped[str | None] = mapped_column(Text, nullable=True)

    produced_by: Mapped[str] = mapped_column(String(60), nullable=False)
    method: Mapped[str] = mapped_column(String(30), nullable=False, default="extracted")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), nullable=False
    )

    company: Mapped["Company"] = relationship(back_populates="evidence_claims")


class CountryESGBaseline(Base):
    __tablename__ = "country_esg_baseline"

    id: Mapped[UUID] = mapped_column(
        primary_key=True,
        server_default=text("gen_random_uuid()")
    )
    country: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    year: Mapped[int] = mapped_column(Integer, nullable=False)
    e_score: Mapped[float] = mapped_column(Float, nullable=False)
    s_score: Mapped[float] = mapped_column(Float, nullable=False)
    g_score: Mapped[float] = mapped_column(Float, nullable=False)
    indicator_count: Mapped[int] = mapped_column(Integer, nullable=False)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), nullable=False
    )

class BCorpLookup(Base):
    __tablename__ = "bcorp_lookup"

    company_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    company_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    country: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    state: Mapped[str | None] = mapped_column(String(100), nullable=True)
    city: Mapped[str | None] = mapped_column(String(100), nullable=True)
    industry: Mapped[str | None] = mapped_column(String(255), nullable=True)
    industry_category: Mapped[str | None] = mapped_column(String(255), nullable=True)
    sector: Mapped[str | None] = mapped_column(String(100), nullable=True)
    sasb_sector: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)
    size: Mapped[str | None] = mapped_column(String(50), nullable=True)
    website: Mapped[str | None] = mapped_column(String(500), nullable=True)
    ownership: Mapped[str | None] = mapped_column(String(100), nullable=True)
    current_status: Mapped[str | None] = mapped_column(String(50), nullable=True)
    assessment_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    overall_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    impact_area_environment: Mapped[float | None] = mapped_column(Float, nullable=True)
    impact_area_governance: Mapped[float | None] = mapped_column(Float, nullable=True)
    impact_area_workers: Mapped[float | None] = mapped_column(Float, nullable=True)
    impact_area_community: Mapped[float | None] = mapped_column(Float, nullable=True)
    impact_area_customers: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("now()"))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("now()"))


class ClimateTraceOwner(Base):
    """
    Climate TRACE v7 owner lookup (name -> id), harvested periodically by
    climate_trace_harvester.py via GET /v7/owners?name=<a-z> enumeration.
    Not joined to `companies` by FK -- matched at query time by fuzzy name
    comparison (same pattern as company_metadata.py's _name_overlap()),
    since Climate TRACE owner names don't always match a company's legal name.
    """
    __tablename__ = "climate_trace_owners"

    owner_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("now()"))


class ClimateTraceOwnerEmissions(Base):
    """Per-facility real emissions for a known Climate TRACE owner (GET /v7/sources?ownerIds=)."""
    __tablename__ = "climate_trace_owner_emissions"

    id: Mapped[UUID] = mapped_column(primary_key=True, server_default=text("gen_random_uuid()"))
    owner_id: Mapped[str] = mapped_column(
        ForeignKey("climate_trace_owners.owner_id", ondelete="CASCADE"), nullable=False, index=True
    )
    source_id: Mapped[int] = mapped_column(nullable=False)
    source_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    country_iso3: Mapped[str | None] = mapped_column(String(3), nullable=True)
    sector: Mapped[str | None] = mapped_column(String(60), nullable=True)
    subsector: Mapped[str | None] = mapped_column(String(60), nullable=True)
    # TEXT, not String(60): some subsectors (e.g. pulp/paper) return long
    # descriptive asset_type strings, not short codes -- found live during
    # the full owner-emissions harvest and crashed a VARCHAR(60) column.
    asset_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    gas: Mapped[str] = mapped_column(String(20), nullable=False, default="co2e_100yr")
    emissions_quantity: Mapped[float | None] = mapped_column(Float, nullable=True)
    activity: Mapped[float | None] = mapped_column(Float, nullable=True)
    activity_units: Mapped[str | None] = mapped_column(Text, nullable=True)
    capacity: Mapped[float | None] = mapped_column(Float, nullable=True)
    capacity_units: Mapped[str | None] = mapped_column(Text, nullable=True)
    year: Mapped[int] = mapped_column(nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("now()"))

    __table_args__ = (
        UniqueConstraint("owner_id", "source_id", "gas", "year", name="uq_ct_owner_emissions"),
    )


class ClimateTraceCountryEmissions(Base):
    """
    Country totals AND country+sector breakdown (GET /v7/sources/emissions
    ?gadmId=&year=). sector IS NULL -> that country's grand total; non-null
    -> one row per sector. subsector follows the same NULL-means-parent-
    total pattern one level deeper.

    No separate global/worldwide table -- verified live that summing a
    sector across every country here equals Climate TRACE's own no-gadmId
    global total (within float rounding), so worldwide sector totals are a
    GROUP BY query over this table (see the climate_trace_global_sector_
    emissions SQL view created in the migration), not separately harvested.
    """
    __tablename__ = "climate_trace_country_emissions"

    id: Mapped[UUID] = mapped_column(primary_key=True, server_default=text("gen_random_uuid()"))
    country_iso3: Mapped[str] = mapped_column(String(3), nullable=False)
    sector: Mapped[str | None] = mapped_column(String(60), nullable=True)
    subsector: Mapped[str | None] = mapped_column(String(60), nullable=True)
    gas: Mapped[str] = mapped_column(String(20), nullable=False, default="co2e_100yr")
    emissions_quantity: Mapped[float] = mapped_column(Float, nullable=False)
    percentage_of_total: Mapped[float | None] = mapped_column(Float, nullable=True)
    year: Mapped[int] = mapped_column(nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("now()"))


