#!/usr/bin/env python3
"""
Processor Factory and Detection System

This module provides a unified table detection and processor selection system.
"""

import re
from typing import List, Tuple, Dict, Any, Optional
from abc import ABC, abstractmethod
import pandas as pd
from loguru import logger

# Hindi word patterns for garbage table detection
HINDI_WORDS = [
    'आई०ई०जी०', 'ध र', 'प्र वध न', 'अन  र', 'दिन क', 'अखिल भ रत',
    'त ीर् प्रण ली', 'की िैदनक', 'दग्रड', 'दनष्प िन', 'ररपोर् र',
    'वेब', 'इर्', 'उप्लब्ध', 'है', 'की', 'के', 'क', 'में', 'से',
    'पर', 'दिन', 'तारीख', 'तिथि', 'दिनांक', 'समय', 'घंटा', 'मिनट'
]

def _contains_hindi_words(table_text: str) -> bool:
    """
    Check if table contains Hindi words indicating it's a garbage table.
    
    Args:
        table_text: Table content as text
    
    Returns:
        True if Hindi words are found, False otherwise
    """
    table_text_lower = table_text.lower()
    for hindi_word in HINDI_WORDS:
        if hindi_word.lower() in table_text_lower:
            return True
    return False

# Import all logical processors
from .logical_processors.regional_summary_logical_processor import RegionalSummaryLogicalProcessor
from .logical_processors.state_summary_processor import StateSummaryProcessor
from .logical_processors.generation_by_source_logical_processor import GenerationBySourceLogicalProcessor
from .logical_processors.inter_region_transmission_processor import InterRegionTransmissionProcessor
from .logical_processors.solar_non_solar_peak_logical_processor import SolarNonSolarPeakLogicalProcessor
from .logical_processors.ddf_processor import DDFProcessor
from .logical_processors.scada_timeseries_logical_processor import ScadaTimeseriesLogicalProcessor
from .logical_processors.transnational_exchange_logical_processor import TransnationalExchangeLogicalProcessor
from .logical_processors.transnational_transmission_logical_processor import TransnationalTransmissionLogicalProcessor
from .logical_processors.regional_import_export_summary_logical_processor import RegionalImportExportSummaryLogicalProcessor
from .logical_processors.generation_outages_logical_processor import GenerationOutagesLogicalProcessor
from .logical_processors.share_logical_processor import ShareLogicalProcessor
from .logical_processors.transnational_summary_logical_processor import TransnationalSummaryLogicalProcessor
from .logical_processors.frequency_profile_logical_processor import FrequencyProfileLogicalProcessor

# Import LLM processors as fallback
from .llm_processors.regional_import_export_summary_processor import RegionalImportExportSummaryProcessor
from .llm_processors.transnational_transmission_processor import TransnationalTransmissionProcessor
from .llm_processors.generation_by_source_processor import GenerationBySourceProcessor
from .llm_processors.generation_outages_processor import GenerationOutagesProcessor
from .llm_processors.share_processor import ShareProcessor
from .llm_processors.scada_timeseries_processor import ScadaTimeseriesProcessor

# Unified table type detection patterns
TABLE_TYPE_PATTERNS = {
    "regional_summary": {
        "keywords": ["regional summary", "region", "nr", "wr", "sr", "er", "ner", "all india", "energy met (mu)", "energy met (mw)", "peak met (mw)", "energy met", "peak met"],
        "required_columns": ["region", "energy_met_mu", "peak_met_mw", "report_date"],
        "logical_processor": RegionalSummaryLogicalProcessor,
        "llm_processor": None
    },
    "state_summary": {
        "keywords": ["state summary", "state wise", "state-wise", "state data", "state summary data", "state wise summary", "state-wise summary", "state summary table", "state wise data", "states", "max.demand", "energy met", "drawal schedule", "energy shortage"],
        "required_columns": ["state_name", "region_name", "peak_demand_mw", "energy_met_mu", "report_date"],
        "logical_processor": StateSummaryProcessor,
        "llm_processor": StateSummaryProcessor
    },
    "generation_by_source": {
        "keywords": ["sourcewise generation", "gross generation", "fuel type", "coal", "lignite", "hydro", "nuclear", "gas naptha diesel", "res wind solar biomass", "sourcewise", "generation by source", "source wise", "% share", "source wise generation", "generation by source table", "sourcewise generation table"],
        "required_columns": ["region_name", "source_name", "energy_generation_mu", "report_date"],
        "logical_processor": GenerationBySourceLogicalProcessor,
        "llm_processor": GenerationBySourceProcessor
    },
    "inter_regional_transmission": {
        "keywords": ["inter regional transmission", "inter-regional transmission", "inter region", "transmission", "region", "line name", "max (mw)", "min (mw)", "avg (mw)", "energy exchange (mu)", "voltage level", "line details", "no. of circuit", "import (mu)", "export (mu)", "net (mu)", "import/export"],
        "required_columns": ["region", "line_name", "power_max_mw", "power_min_mw", "power_avg_mw", "energy_exchange_mu", "report_date"],
        "logical_processor": InterRegionTransmissionProcessor,
        "llm_processor": None
    },
    "solar_non_solar_peak": {
        "keywords": ["solar non solar peak", "solar", "non solar", "peak", "solar peak", "non solar peak", "solar hr", "non-solar hr", "max demand met", "shortage"],
        "required_columns": ["region", "solar_peak_mw", "non_solar_peak_mw", "report_date"],
        "logical_processor": SolarNonSolarPeakLogicalProcessor,
        "llm_processor": None
    },
    "ddf": {
        "keywords": ["ddf", "demand drawal factor", "demand drawal", "drawal factor", "based on regional max demands", "based on state max demands", "regional max demands", "state max demands"],
        "required_columns": ["calculation_basis", "regional_value", "report_date"],
        "logical_processor": DDFProcessor,
        "llm_processor": None
    },
    "scada_timeseries": {
        "keywords": ["15 min (instantaneous)", "all india grid frequency", "generation & demand met", "scada data", "time", "frequency (hz)", "demand met (mw)", "nuclear (mw)", "wind (mw)", "solar (mw)", "hydro** (mw)", "gas (mw)", "thermal (mw)", "others* (mw)", "net demand met (mw)", "total generation (mw)", "net transnational exchange (mw)", "scada", "timeseries", "instantaneous"],
        "required_columns": ["timestamp", "frequency_hz", "power_demand_mw", "report_date"],
        "logical_processor": ScadaTimeseriesLogicalProcessor,
        "llm_processor": ScadaTimeseriesProcessor
    },
    "transnational_exchange": {
        "keywords": ["country", "ppa", "bilateral", "dam iex", "dam pxil", "dam hpx", "rtm iex", "rtm pxil", "rtm hpx", "total", "export", "import", "net", "exchange", "transnational exchange"],
        "required_columns": ["country", "ppa", "bilateral", "dam_iex", "dam_pxil", "dam_hpx", "rtm_iex", "rtm_pxil", "rtm_hpx", "total", "exchange_type", "report_date"],
        "logical_processor": TransnationalExchangeLogicalProcessor,
        "llm_processor": None
    },
    "transnational_transmission": {
        "keywords": ["international exchanges", "state", "region", "line name", "max (mw)", "min (mw)", "avg (mw)", "energy exchange (mu)", "transnational transmission", "international transmission", "line details", "voltage level", "no. of circuit", "bhutan", "nepal", "bangladesh", "400kv", "220kv", "132kv", "hvdc", "line flow", "transmission line", "transnational transmission line", "international transmission line", "line flow data"],
        "required_columns": ["state", "region", "line_name", "power_max_mw", "power_min_mw", "power_avg_mw", "energy_exchange_mu", "report_date"],
        "logical_processor": TransnationalTransmissionLogicalProcessor,
        "llm_processor": TransnationalTransmissionProcessor
    },
    "regional_import_export_summary": {
        "keywords": ["schedule(mu)", "actual(mu)", "o/d/u/d(mu)", "regional import export summary", "schedule", "actual", "o/d/u/d", "regional", "import", "export", "overdrawal", "underdrawal"],
        "required_columns": ["region", "energy_schedule_mu", "energy_actual_mu", "energy_overdrawal_mu", "report_date"],
        "logical_processor": RegionalImportExportSummaryLogicalProcessor,
        "llm_processor": RegionalImportExportSummaryProcessor
    },
    "generation_outages": {
        "keywords": ["outage", "generation outage", "sector", "central sector", "state sector", "mw", "central sector", "state sector", "total", "% share", "capacity", "outage capacity"],
        "required_columns": ["sector_name", "region_name", "power_outage_capacity_mw", "report_date"],
        "logical_processor": GenerationOutagesLogicalProcessor,
        "llm_processor": GenerationOutagesProcessor
    },
    "share": {
        "keywords": ["share percentage", "share data", "generation share", "share of res", "share of non-fossil", "share of hydro", "share of nuclear", "% share", "percentage share", "res share", "non-fossil share"],
        "required_columns": ["measure", "region_name", "share_percentage", "report_date"],
        "logical_processor": ShareLogicalProcessor,
        "llm_processor": ShareProcessor
    },
    "frequency_profile": {
        "keywords": ["frequency profile", "fvi", "frequency variation index", "frequency distribution", "49.7", "49.8", "49.9", "50.05", "frequency bands", "frequency ranges", "percentage time", "frequency variation", "frequency profile data", "frequency variation index (fvi)", "frequency profile table"],
        "required_columns": ["region", "frequency_variation_index", "percentage_time_less_49_7", "percentage_time_49_7_to_49_8", "percentage_time_49_8_to_49_9", "percentage_time_less_49_9", "percentage_time_49_9_to_50_05", "percentage_time_greater_50_05", "report_date"],
        "logical_processor": FrequencyProfileLogicalProcessor,
        "llm_processor": None
    },
    "transnational_summary": {
        "keywords": ['bhutan', 'nepal', 'bangladesh', 'godda', 'actual', 'day peak', 'mu', 'mw', 'transnational summary', 'international summary', 'country summary', 'actual (mu)', 'day peak (mw)', 'transnational exchanges summary'],
        "required_columns": ["country", "energy_actual_mu", "power_day_peak_mw", "report_date"],
        "logical_processor": TransnationalSummaryLogicalProcessor,
        "llm_processor": None
    }
}

# Global variable to track used table types
_USED_TABLE_TYPES = set()

# Table structure patterns for more accurate detection
TABLE_STRUCTURE_PATTERNS = {
    "regional_summary": {
        "keywords": ["regional summary", "region", "nr", "wr", "sr", "er", "ner", "all india", "energy met (mu)", "energy met (mw)", "peak met (mw)", "energy met", "peak met", "evening peak hrs", "from rldcs"],
        "required_columns": ["region_code", "peak_demand_met_evng_mw", "energy_met_total_mu", "report_date"],
        "structure": {"min_rows": 6, "max_rows": 12, "min_cols": 6, "max_cols": 8, "expected_cols": ["NR", "WR", "SR", "ER", "NER", "TOTAL"]},
        "logical_processor": RegionalSummaryLogicalProcessor,
        "llm_processor": None
    },
    "state_summary": {
        "keywords": ["state summary", "state wise", "state-wise", "state data", "state summary data", "state wise summary", "state-wise summary", "state summary table", "state wise data"],
        "required_columns": ["state_name", "region_name", "peak_demand_mw", "energy_met_mu", "report_date"],
        "structure": {"min_rows": 30, "max_rows": 50, "min_cols": 8, "max_cols": 10, "expected_cols": ["Region", "States"]},
        "logical_processor": StateSummaryProcessor,
        "llm_processor": StateSummaryProcessor
    },
    "generation_by_source": {
        "keywords": ["sourcewise generation", "gross generation", "fuel type", "coal", "lignite", "hydro", "nuclear", "gas naptha diesel", "res wind solar biomass", "sourcewise", "generation by source", "source wise", "% share", "source wise generation", "generation by source table", "sourcewise generation table"],
        "required_columns": ["region_name", "source_name", "energy_generation_mu", "report_date"],
        "structure": {"min_rows": 5, "max_rows": 10, "min_cols": 7, "max_cols": 9, "expected_cols": ["All India"]},
        "logical_processor": GenerationBySourceLogicalProcessor,
        "llm_processor": GenerationBySourceProcessor
    },
    "inter_regional_transmission": {
        "keywords": ["inter regional transmission", "inter-regional transmission", "inter region", "transmission", "region", "line name", "max (mw)", "min (mw)", "avg (mw)", "energy exchange (mu)"],
        "required_columns": ["region", "line_name", "power_max_mw", "power_min_mw", "power_avg_mw", "energy_exchange_mu", "report_date"],
        "structure": {"min_rows": 70, "max_rows": 85, "min_cols": 8, "max_cols": 10, "expected_cols": ["Sl No", "Voltage Level", "Line Details"]},
        "logical_processor": InterRegionTransmissionProcessor,
        "llm_processor": None
    },
    "solar_non_solar_peak": {
        "keywords": ["solar non solar peak", "solar", "non solar", "peak", "solar peak", "non solar peak"],
        "required_columns": ["region", "solar_peak_mw", "non_solar_peak_mw", "report_date"],
        "structure": {"min_rows": 1, "max_rows": 3, "min_cols": 3, "max_cols": 5, "expected_cols": ["Max Demand Met(MW)", "Time"]},
        "logical_processor": SolarNonSolarPeakLogicalProcessor,
        "llm_processor": None
    },
    "ddf": {
        "keywords": ["ddf", "demand drawal factor", "demand drawal", "drawal factor"],
        "required_columns": ["calculation_basis", "regional_value", "report_date"],
        "structure": {"min_rows": 1, "max_rows": 3, "min_cols": 1, "max_cols": 3, "expected_cols": ["Based on Regional Max Demands", "Based on State Max Demands"]},
        "logical_processor": DDFProcessor,
        "llm_processor": None
    },
    "scada_timeseries": {
        "keywords": ["15 min (instantaneous)", "all india grid frequency", "generation & demand met", "scada data", "time", "frequency (hz)", "demand met (mw)", "nuclear (mw)", "wind (mw)", "solar (mw)", "hydro** (mw)", "gas (mw)", "thermal (mw)", "others* (mw)", "net demand met (mw)", "total generation (mw)", "net transnational exchange (mw)", "scada", "timeseries", "instantaneous"],
        "required_columns": ["timestamp", "frequency_hz", "power_demand_mw", "report_date"],
        "structure": {"min_rows": 90, "max_rows": 100, "min_cols": 12, "max_cols": 15, "expected_cols": ["TIME", "FREQUENCY", "DEMAND MET", "NUCLEAR", "WIND", "SOLAR", "HYDRO", "GAS", "THERMAL", "OTHERS", "NET DEMAND MET", "TOTAL GENERATION", "NET TRANSNATIONAL EXCHANGE"]},
        "logical_processor": ScadaTimeseriesLogicalProcessor,
        "llm_processor": ScadaTimeseriesProcessor
    },
    "transnational_exchange": {
        "keywords": ["country", "ppa", "bilateral", "dam iex", "dam pxil", "dam hpx", "rtm iex", "rtm pxil", "rtm hpx", "total", "export", "import", "net", "exchange", "transnational exchange", "gna", "t-gna", "collective", "idam", "rtm"],
        "required_columns": ["country", "ppa", "bilateral", "dam_iex", "dam_pxil", "dam_hpx", "rtm_iex", "rtm_pxil", "rtm_hpx", "total", "exchange_type", "report_date"],
        "structure": {"min_rows": 6, "max_rows": 10, "min_cols": 9, "max_cols": 12, "expected_cols": ["Country", "GNA", "T-GNA", "TOTAL", "BILATERAL", "COLLECTIVE", "IDAM", "RTM"], "max_occurrences": 3},
        "logical_processor": TransnationalExchangeLogicalProcessor,
        "llm_processor": None
    },
    "transnational_transmission": {
        "keywords": ["international exchanges", "state", "region", "line name", "max (mw)", "min (mw)", "avg (mw)", "energy exchange (mu)", "transnational transmission", "international transmission", "line details", "voltage level", "no. of circuit", "bhutan", "nepal", "bangladesh", "400kv", "220kv", "132kv", "hvdc", "line flow", "transmission line", "transnational transmission line", "international transmission line", "line flow data"],
        "required_columns": ["state", "region", "line_name", "power_max_mw", "power_min_mw", "power_avg_mw", "energy_exchange_mu", "report_date"],
        "structure": {"min_rows": 10, "max_rows": 20, "min_cols": 6, "max_cols": 8, "expected_cols": ["State", "Region", "Line Name"]},
        "logical_processor": TransnationalTransmissionLogicalProcessor,
        "llm_processor": TransnationalTransmissionProcessor
    },
    "regional_import_export_summary": {
        "keywords": ["schedule(mu)", "actual(mu)", "o/d/u/d(mu)", "regional import export summary", "schedule", "actual", "o/d/u/d", "regional", "import", "export", "overdrawal", "underdrawal"],
        "required_columns": ["region", "energy_schedule_mu", "energy_actual_mu", "energy_overdrawal_mu", "report_date"],
        "structure": {"min_rows": 2, "max_rows": 5, "min_cols": 6, "max_cols": 8, "expected_cols": ["Schedule(MU)", "Actual(MU)", "O/D/U/D(MU)"]},
        "logical_processor": RegionalImportExportSummaryLogicalProcessor,
        "llm_processor": RegionalImportExportSummaryProcessor
    },
    "generation_outages": {
        "keywords": ["outage", "generation outage", "sector", "central sector", "state sector", "mw", "central sector", "state sector", "total", "% share", "capacity", "outage capacity"],
        "required_columns": ["sector_name", "region_name", "power_outage_capacity_mw", "report_date"],
        "structure": {"min_rows": 2, "max_rows": 5, "min_cols": 7, "max_cols": 9, "expected_cols": ["Central Sector", "State Sector", "Total", "% Share"]},
        "logical_processor": GenerationOutagesLogicalProcessor,
        "llm_processor": GenerationOutagesProcessor
    },
    "share": {
        "keywords": ["share percentage", "share data", "generation share", "share of res", "share of non-fossil", "share of hydro", "share of nuclear", "% share", "percentage share", "res share", "non-fossil share"],
        "required_columns": ["measure", "region_name", "share_percentage", "report_date"],
        "structure": {"min_rows": 1, "max_rows": 3, "min_cols": 6, "max_cols": 8, "expected_cols": ["Share of RES", "Share of Non-fossil"]},
        "logical_processor": ShareLogicalProcessor,
        "llm_processor": ShareProcessor
    },
    "frequency_profile": {
        "keywords": ["frequency profile", "fvi", "frequency variation index", "frequency distribution", "49.7", "49.8", "49.9", "50.05", "frequency bands", "frequency ranges", "percentage time", "frequency variation", "frequency profile data", "frequency variation index (fvi)", "frequency profile table"],
        "required_columns": ["region", "frequency_variation_index", "percentage_time_less_49_7", "percentage_time_49_7_to_49_8", "percentage_time_49_8_to_49_9", "percentage_time_less_49_9", "percentage_time_49_9_to_50_05", "percentage_time_greater_50_05", "report_date"],
        "structure": {"min_rows": 1, "max_rows": 3, "min_cols": 7, "max_cols": 9, "expected_cols": ["Region", "FVI", "< 49.7", "49.7 - 49.8"]},
        "logical_processor": FrequencyProfileLogicalProcessor,
        "llm_processor": None
    },
    "transnational_summary": {
        "keywords": ['bhutan', 'nepal', 'bangladesh', 'godda', 'actual', 'day peak', 'mu', 'mw', 'transnational summary', 'international summary', 'country summary', 'actual (mu)', 'day peak (mw)', 'transnational exchanges summary'],
        "required_columns": ["country", "energy_actual_mu", "power_day_peak_mw", "report_date"],
        "structure": {"min_rows": 1, "max_rows": 3, "min_cols": 4, "max_cols": 6, "expected_cols": ["Bhutan", "Nepal", "Bangladesh", "Godda"]},
        "logical_processor": TransnationalSummaryLogicalProcessor,
        "llm_processor": None
    }
}

def _calculate_structure_score(table_df: pd.DataFrame, structure_pattern: dict) -> float:
    """
    Calculate how well a table matches the expected structure.
    
    Args:
        table_df: DataFrame to analyze
        structure_pattern: Structure pattern from TABLE_STRUCTURE_PATTERNS
    
    Returns:
        Structure match score (0.0 to 1.0)
    """
    rows, cols = table_df.shape
    structure = structure_pattern["structure"]
    
    # Check row count
    row_score = 0.0
    if structure["min_rows"] <= rows <= structure["max_rows"]:
        row_score = 1.0
    else:
        # Calculate distance from expected range
        if rows < structure["min_rows"]:
            row_score = max(0.0, 1.0 - (structure["min_rows"] - rows) / structure["min_rows"])
        else:
            row_score = max(0.0, 1.0 - (rows - structure["max_rows"]) / structure["max_rows"])
    
    # Check column count
    col_score = 0.0
    if structure["min_cols"] <= cols <= structure["max_cols"]:
        col_score = 1.0
    else:
        # Calculate distance from expected range
        if cols < structure["min_cols"]:
            col_score = max(0.0, 1.0 - (structure["min_cols"] - cols) / structure["min_cols"])
        else:
            col_score = max(0.0, 1.0 - (cols - structure["max_cols"]) / structure["max_cols"])
    
    # Check expected columns
    expected_cols = structure.get("expected_cols", [])
    col_match_score = 0.0
    if expected_cols:
        table_cols = [str(col).strip().lower() for col in table_df.columns]
        matches = sum(1 for exp_col in expected_cols if any(exp_col.lower() in col for col in table_cols))
        col_match_score = matches / len(expected_cols) if expected_cols else 0.0
    
    # Combine scores (weight: 40% rows, 30% cols, 30% column names)
    total_score = (0.4 * row_score) + (0.3 * col_score) + (0.3 * col_match_score)
    
    return total_score

def _calculate_keyword_score(table_text: str, keywords: list) -> float:
    """
    Calculate keyword match score.
    
    Args:
        table_text: Table content as text
        keywords: List of keywords to match
    
    Returns:
        Keyword match score (0.0 to 1.0)
    """
    table_text_lower = table_text.lower()
    matches = sum(1 for keyword in keywords if keyword.lower() in table_text_lower)
    return matches / len(keywords) if keywords else 0.0

def detect_table_type(table_df: pd.DataFrame, exclude_used: bool = True) -> Tuple[str, float]:
    """
    Unified table type detection system using both keywords and structure.
    
    Args:
        table_df: DataFrame to analyze
        exclude_used: Whether to exclude already used table types
    
    Returns:
        Tuple of (detected_table_type, confidence_score)
    """
    # Convert table content to text for keyword matching
    table_text = table_df.to_string()
    
    # Check for Hindi words first - if found, classify as unknown (garbage table)
    if _contains_hindi_words(table_text):
        logger.info("Hindi words detected - classifying as unknown (garbage table)")
        return "unknown", 1.0
    
    best_type = "unknown"
    best_score = 0
    
    for table_type, pattern_info in TABLE_STRUCTURE_PATTERNS.items():
        # Skip if this table type has already been used and we're excluding used types
        if exclude_used and table_type in _USED_TABLE_TYPES:
            # Special case: transnational_exchange can be used up to 3 times
            if table_type == "transnational_exchange":
                used_count = sum(1 for used_type in _USED_TABLE_TYPES if used_type == "transnational_exchange")
                if used_count >= 3:
                    continue
            else:
                continue
        
        # Calculate keyword score
        keyword_score = _calculate_keyword_score(table_text, pattern_info["keywords"])
        
        # Calculate structure score
        structure_score = _calculate_structure_score(table_df, pattern_info)
        
        # Combined score (60% keywords, 40% structure)
        combined_score = (0.6 * keyword_score) + (0.4 * structure_score)
        
        if combined_score > best_score:
            best_score = combined_score
            best_type = table_type
    
    return best_type, best_score

def classify_all_tables(tables: List[pd.DataFrame]) -> List[Tuple[int, str, float]]:
    """
    Classify all tables at once, ensuring each table type is only identified once
    (except transnational_exchange which can be identified up to 3 times).
    
    Args:
        tables: List of table DataFrames
    
    Returns:
        List of (table_index, detected_type, confidence) tuples
    """
    global _USED_TABLE_TYPES
    _USED_TABLE_TYPES.clear()
    
    classifications = []
    transnational_exchange_count = 0
    
    for i, table_df in enumerate(tables):
        # First, check if this table contains Hindi words (garbage table)
        table_text = table_df.to_string()
        if _contains_hindi_words(table_text):
            classifications.append((i, "unknown", 1.0))
            logger.info(f"Table {i+1}: Hindi words detected - classified as unknown (garbage table)")
            continue
        
        detected_type, confidence = detect_table_type(table_df, exclude_used=True)
        
        # Special handling for transnational_exchange - limit to 3
        if detected_type == "transnational_exchange":
            transnational_exchange_count += 1
            if transnational_exchange_count > 3:
                logger.info(f"Table {i+1}: transnational_exchange limit reached (3), trying alternative detection")
                # Try detection without transnational_exchange in exclusion
                temp_used = _USED_TABLE_TYPES.copy()
                _USED_TABLE_TYPES.discard("transnational_exchange")
                detected_type, confidence = detect_table_type(table_df, exclude_used=True)
                _USED_TABLE_TYPES = temp_used
        
        classifications.append((i, detected_type, confidence))
        
        # Mark as used (except transnational_exchange which can be used multiple times)
        if detected_type != "unknown":
            if detected_type == "transnational_exchange":
                # Don't add to used types, just track count
                pass
            else:
                _USED_TABLE_TYPES.add(detected_type)
    
    return classifications

def mark_table_type_as_used(table_type: str) -> None:
    """
    Mark a table type as used to prevent re-detection.
    
    Args:
        table_type: The table type to mark as used
    """
    global _USED_TABLE_TYPES
    _USED_TABLE_TYPES.add(table_type)
    logger.info(f"Marked table type '{table_type}' as used. Used types: {_USED_TABLE_TYPES}")

def reset_used_table_types() -> None:
    """
    Reset the list of used table types.
    """
    global _USED_TABLE_TYPES
    _USED_TABLE_TYPES.clear()
    logger.info("Reset used table types list")

def get_processor(table_df: pd.DataFrame, prefer_logical: bool = True, detected_type: str = None) -> Optional[Any]:
    """
    Get the appropriate processor for a table.
    
    Args:
        table_df: DataFrame to process
        prefer_logical: Whether to prefer logical processors over LLM processors
        detected_type: Pre-detected table type (if provided, skips detection)
    
    Returns:
        Processor instance or None if no suitable processor found
    """
    # If detected_type is provided, use it directly
    if detected_type and detected_type in TABLE_STRUCTURE_PATTERNS:
        pattern_info = TABLE_STRUCTURE_PATTERNS[detected_type]
        
        # Try logical processor first if preferred
        if prefer_logical and pattern_info.get("logical_processor"):
            try:
                processor = pattern_info["logical_processor"]()
                logger.info(f"Selected logical processor: {processor.__class__.__name__} for type: {detected_type}")
                return processor
            except Exception as e:
                logger.warning(f"Failed to instantiate logical processor for {detected_type}: {e}")
        
        # Try LLM processor as fallback
        if pattern_info.get("llm_processor"):
            try:
                processor = pattern_info["llm_processor"]()
                logger.info(f"Selected LLM processor: {processor.__class__.__name__} for type: {detected_type}")
                return processor
            except Exception as e:
                logger.warning(f"Failed to instantiate LLM processor for {detected_type}: {e}")
        
        # If no processor found for the detected type, try logical processor anyway
        if pattern_info.get("logical_processor"):
            try:
                processor = pattern_info["logical_processor"]()
                logger.info(f"Selected logical processor (fallback): {processor.__class__.__name__} for type: {detected_type}")
                return processor
            except Exception as e:
                logger.warning(f"Failed to instantiate logical processor (fallback) for {detected_type}: {e}")
    
    # If no detected_type provided or no processor found, fall back to detection
    detected_type, confidence = detect_table_type(table_df)
    
    if detected_type == "unknown":
        return None
    
    pattern_info = TABLE_STRUCTURE_PATTERNS.get(detected_type)
    if not pattern_info:
        return None
    
    # Try logical processor first if preferred
    if prefer_logical and pattern_info.get("logical_processor"):
        try:
            processor = pattern_info["logical_processor"]()
            logger.info(f"Selected logical processor: {processor.__class__.__name__} for type: {detected_type}")
            return processor
        except Exception as e:
            logger.warning(f"Failed to instantiate logical processor for {detected_type}: {e}")
    
    # Try LLM processor as fallback
    if pattern_info.get("llm_processor"):
        try:
            processor = pattern_info["llm_processor"]()
            logger.info(f"Selected LLM processor: {processor.__class__.__name__} for type: {detected_type}")
            return processor
        except Exception as e:
            logger.warning(f"Failed to instantiate LLM processor for {detected_type}: {e}")
    
    return None

def get_processor_info(table_df: pd.DataFrame, detected_type: str = None) -> Dict[str, Any]:
    """
    Get detailed information about processor selection for a table.
    
    Args:
        table_df: The table DataFrame
        detected_type: Pre-detected table type (if None, will detect)
    
    Returns:
        Dictionary with detection and processor information
    """
    if detected_type is None:
        detected_type, confidence = detect_table_type(table_df)
    else:
        # If type is provided, we assume high confidence
        confidence = 1.0
        
    pattern_info = TABLE_STRUCTURE_PATTERNS.get(detected_type, {})
    
    return {
        "detected_type": detected_type,
        "confidence": confidence,
        "available_logical": pattern_info.get("logical_processor") is not None,
        "available_llm": pattern_info.get("llm_processor") is not None,
        "keywords": pattern_info.get("keywords", []),
        "required_columns": pattern_info.get("required_columns", [])
    } 