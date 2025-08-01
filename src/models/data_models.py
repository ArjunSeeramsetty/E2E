"""
Data models for the Power Supply Data Warehouse
Defines Pydantic models for all data structures
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import date, datetime
from decimal import Decimal
import re

class Report(BaseModel):
    """Core report metadata"""
    report_date: date
    source_entity: str = Field(..., pattern="^(NLDC|SRLDC|NRLDC|WRLDC|ERLDC|NERLDC)$")
    source_url: Optional[str] = None
    ingestion_timestamp: datetime = Field(default_factory=datetime.now)
    
    @field_validator('source_entity')
    @classmethod
    def validate_source_entity(cls, v):
        valid_entities = ['NLDC', 'SRLDC', 'NRLDC', 'WRLDC', 'ERLDC', 'NERLDC']
        if v not in valid_entities:
            raise ValueError(f'Invalid source entity. Must be one of {valid_entities}')
        return v

class RegionalSummary(BaseModel):
    """Regional-level power supply data"""
    region_code: str = Field(..., pattern="^(NR|WR|SR|ER|NER)$")
    peak_demand_met_mw: Optional[Decimal] = None
    peak_shortage_mw: Optional[Decimal] = None
    energy_met_mu: Optional[Decimal] = None
    energy_shortage_mu: Optional[Decimal] = None
    max_demand_met_day_mw: Optional[Decimal] = None
    time_of_max_demand: Optional[str] = None
    
    @field_validator('region_code')
    @classmethod
    def validate_region_code(cls, v):
        valid_regions = ['NR', 'WR', 'SR', 'ER', 'NER']
        if v not in valid_regions:
            raise ValueError(f'Invalid region code. Must be one of {valid_regions}')
        return v

class StateSummary(BaseModel):
    """State-level power supply data"""
    state_name: str
    max_demand_met_mw: Optional[Decimal] = None
    shortage_at_max_demand_mw: Optional[Decimal] = None
    drawal_schedule_mu: Optional[Decimal] = None
    over_under_drawal_mu: Optional[Decimal] = None
    
    @field_validator('state_name')
    @classmethod
    def clean_state_name(cls, v):
        """Clean and standardize state names"""
        # Remove trailing periods and extra spaces
        cleaned = re.sub(r'\.$', '', v.strip())
        # Standardize common variations
        state_mappings = {
            'Harvana.': 'Haryana',
            'Harvana': 'Haryana',
            'Uttar Pradesh.': 'Uttar Pradesh',
            'Madhya Pradesh.': 'Madhya Pradesh',
        }
        return state_mappings.get(cleaned, cleaned)

class GenerationBySource(BaseModel):
    """Generation breakdown by fuel type"""
    source_type: str
    gross_generation_mu: Optional[Decimal] = None
    percentage_share: Optional[Decimal] = None
    
    @field_validator('source_type')
    @classmethod
    def validate_source_type(cls, v):
        valid_sources = [
            'Coal', 'Lignite', 'Hydro', 'Nuclear', 'Gas/Naptha/Diesel',
            'Solar', 'Wind', 'RES', 'Total'
        ]
        if v not in valid_sources:
            # Allow custom sources but log them
            print(f"Warning: Unknown source type: {v}")
        return v

class FrequencyProfile(BaseModel):
    """Grid frequency stability data"""
    frequency_band: str
    percentage_time: Optional[Decimal] = None
    fvi: Optional[Decimal] = None  # Frequency Variation Index

class InterRegionalExchange(BaseModel):
    """Power exchange between regions"""
    from_region: str
    to_region: str
    scheduled_mu: Optional[Decimal] = None
    actual_mu: Optional[Decimal] = None
    deviation_mu: Optional[Decimal] = None

class TransnationalExchange(BaseModel):
    """Power exchange with neighboring countries"""
    country: str
    scheduled_mu: Optional[Decimal] = None
    actual_mu: Optional[Decimal] = None
    deviation_mu: Optional[Decimal] = None

class GenerationOutage(BaseModel):
    """Generation capacity unavailable"""
    region_code: str
    sector: str  # 'Central' or 'State'
    outage_mw: Optional[Decimal] = None

# Region-specific models
class SRStationGeneration(BaseModel):
    """Southern Region station-level generation"""
    state_name: str
    station_name: str
    installed_capacity_mw: Optional[Decimal] = None
    net_gen_mu: Optional[Decimal] = None
    avg_mw: Optional[Decimal] = None

class SRReservoirLevel(BaseModel):
    """Southern Region reservoir data"""
    reservoir_name: str
    current_level_mts: Optional[Decimal] = None
    current_energy_mu: Optional[Decimal] = None
    last_year_energy_mu: Optional[Decimal] = None

class NRReliabilityIndex(BaseModel):
    """Northern Region reliability metrics"""
    corridor_name: str
    ttc_violation_percent: Optional[Decimal] = None
    atc_violation_percent: Optional[Decimal] = None

class ParsedReport(BaseModel):
    """Complete parsed report structure"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    report: Report
    regional_summaries: List[RegionalSummary] = []
    state_summaries: List[StateSummary] = []
    generation_by_source: List[GenerationBySource] = []
    frequency_profile: Optional[FrequencyProfile] = None
    inter_regional_exchanges: List[InterRegionalExchange] = []
    transnational_exchanges: List[TransnationalExchange] = []
    generation_outages: List[GenerationOutage] = []
    
    # Region-specific data
    sr_station_generation: Optional[List[SRStationGeneration]] = None
    sr_reservoir_levels: Optional[List[SRReservoirLevel]] = None
    nr_reliability_indices: Optional[List[NRReliabilityIndex]] = None 