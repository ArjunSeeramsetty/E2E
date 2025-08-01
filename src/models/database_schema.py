"""
Database schema definitions for PostgreSQL
"""

from sqlalchemy import create_engine, Column, Integer, String, Numeric, Date, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime

Base = declarative_base()

class Reports(Base):
    """Master table for all ingested reports"""
    __tablename__ = 'reports'
    
    report_id = Column(Integer, primary_key=True, autoincrement=True)
    report_date = Column(Date, nullable=False)
    source_entity = Column(String(10), nullable=False)
    source_url = Column(Text)
    ingestion_timestamp = Column(DateTime, default=datetime.now)
    
    # Relationships
    regional_summaries = relationship("RegionalSummaries", back_populates="report")
    state_summaries = relationship("StateSummaries", back_populates="report")

class RegionalSummaries(Base):
    """Regional-level power supply data"""
    __tablename__ = 'regional_summaries'
    
    summary_id = Column(Integer, primary_key=True, autoincrement=True)
    report_id = Column(Integer, ForeignKey('reports.report_id'), nullable=False)
    region_code = Column(String(3), nullable=False)
    peak_demand_met_mw = Column(Numeric(10, 2))
    peak_shortage_mw = Column(Numeric(10, 2))
    energy_met_mu = Column(Numeric(10, 2))
    energy_shortage_mu = Column(Numeric(10, 2))
    max_demand_met_day_mw = Column(Numeric(10, 2))
    time_of_max_demand = Column(String(10))
    
    # Relationships
    report = relationship("Reports", back_populates="regional_summaries")
    state_summaries = relationship("StateSummaries", back_populates="regional_summary")
    generation_by_source = relationship("GenerationBySource", back_populates="regional_summary")

class StateSummaries(Base):
    """State-level power supply data"""
    __tablename__ = 'state_summaries'
    
    state_summary_id = Column(Integer, primary_key=True, autoincrement=True)
    summary_id = Column(Integer, ForeignKey('regional_summaries.summary_id'), nullable=False)
    state_name = Column(String(50), nullable=False)
    max_demand_met_mw = Column(Numeric(10, 2))
    shortage_at_max_demand_mw = Column(Numeric(10, 2))
    drawal_schedule_mu = Column(Numeric(10, 2))
    over_under_drawal_mu = Column(Numeric(10, 2))
    
    # Relationships
    regional_summary = relationship("RegionalSummaries", back_populates="state_summaries")

class GenerationBySource(Base):
    """Generation breakdown by fuel type"""
    __tablename__ = 'generation_by_source'
    
    gen_id = Column(Integer, primary_key=True, autoincrement=True)
    summary_id = Column(Integer, ForeignKey('regional_summaries.summary_id'), nullable=False)
    source_type = Column(String(50), nullable=False)
    gross_generation_mu = Column(Numeric(10, 2))
    percentage_share = Column(Numeric(5, 2))
    
    # Relationships
    regional_summary = relationship("RegionalSummaries", back_populates="generation_by_source")

# Region-specific tables
class SRStationGeneration(Base):
    """Southern Region station-level generation"""
    __tablename__ = 'sr_station_generation'
    
    station_gen_id = Column(Integer, primary_key=True, autoincrement=True)
    summary_id = Column(Integer, ForeignKey('regional_summaries.summary_id'), nullable=False)
    state_name = Column(String(50), nullable=False)
    station_name = Column(String(100), nullable=False)
    installed_capacity_mw = Column(Numeric(10, 2))
    net_gen_mu = Column(Numeric(10, 2))
    avg_mw = Column(Numeric(10, 2))

class SRReservoirLevels(Base):
    """Southern Region reservoir data"""
    __tablename__ = 'sr_reservoir_levels'
    
    reservoir_id = Column(Integer, primary_key=True, autoincrement=True)
    summary_id = Column(Integer, ForeignKey('regional_summaries.summary_id'), nullable=False)
    reservoir_name = Column(String(100), nullable=False)
    current_level_mts = Column(Numeric(10, 2))
    current_energy_mu = Column(Numeric(10, 2))
    last_year_energy_mu = Column(Numeric(10, 2))

class NRReliabilityIndices(Base):
    """Northern Region reliability metrics"""
    __tablename__ = 'nr_reliability_indices'
    
    reliability_id = Column(Integer, primary_key=True, autoincrement=True)
    summary_id = Column(Integer, ForeignKey('regional_summaries.summary_id'), nullable=False)
    corridor_name = Column(String(100), nullable=False)
    ttc_violation_percent = Column(Numeric(5, 2))
    atc_violation_percent = Column(Numeric(5, 2)) 