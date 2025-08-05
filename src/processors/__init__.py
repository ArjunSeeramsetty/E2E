#!/usr/bin/env python3
"""
Processors Module

This module contains specialized processors for different types of power supply data tables.
Each processor handles the specific structural and data format requirements of its target table type.
"""

import pandas as pd
from typing import Optional

# Import base processor
from .base_processor import BaseProcessor

# Import LLM-based processors
from .llm_processors import (
    RegionalSummaryProcessor,
    GenerationBySourceProcessor,
    GenerationOutagesProcessor,
    TransnationalTransmissionProcessor,
    TransnationalExchangeProcessor,
    SolarNonSolarPeakProcessor,
    TransnationalSummaryProcessor,
    ShareProcessor,
    RegionalImportExportSummaryProcessor,
    ScadaTimeseriesProcessor,
    FrequencyProfileProcessor,
)

# Import logical/rule-based processors
from .logical_processors import (
    StateSummaryProcessor,
    DDFProcessor,
    InterRegionTransmissionProcessor,
    RegionalSummaryLogicalProcessor,
    GenerationBySourceLogicalProcessor,
    FrequencyProfileLogicalProcessor,
    SolarNonSolarPeakLogicalProcessor,
    TransnationalSummaryLogicalProcessor,
    # Legacy functions for backward compatibility
    process_state_table_robustly,
    validate_state_table_data,
    get_state_table_summary,
    detect_row_structure_robust,
    extract_numeric_value_robust,
)

# Map table types to their corresponding processor classes
PROCESSOR_MAP = {
    "regional_daily_summary": RegionalSummaryLogicalProcessor,  # Use logical processor
    "state_daily_summary": StateSummaryProcessor,
    "generation_by_source": GenerationBySourceLogicalProcessor,  # Use logical processor
    "generation_outages": GenerationOutagesProcessor,
    "inter_regional_transmission": InterRegionTransmissionProcessor,
    "transnational_transmission": TransnationalTransmissionProcessor,
    "transnational_exchange": TransnationalExchangeProcessor,
    "transnational_summary": TransnationalSummaryLogicalProcessor,  # Use logical processor
    "regional_import_export_summary": RegionalImportExportSummaryProcessor,
    "share": ShareProcessor,
    "solar_non_solar_peak": SolarNonSolarPeakLogicalProcessor,  # Use logical processor
    "frequency_profile": FrequencyProfileLogicalProcessor,  # Use logical processor
    "scada_timeseries": ScadaTimeseriesProcessor,
    "ddf": DDFProcessor,
}

def get_processor(table_df: pd.DataFrame) -> Optional[BaseProcessor]:
    """
    Factory function to detect table type and return an instance
    of the appropriate processor.
    """
    # Use the detection logic from the BaseProcessor
    table_type = BaseProcessor.detect_table_type(table_df)
    
    processor_class = PROCESSOR_MAP.get(table_type)
    
    if processor_class:
        return processor_class()
    
    return None

__all__ = [
    # Legacy functions for backward compatibility
    'process_state_table_robustly',
    'validate_state_table_data', 
    'get_state_table_summary',
    'detect_row_structure_robust',
    'extract_numeric_value_robust',
    # New processor architecture
    'BaseProcessor',
    'RegionalSummaryProcessor', 
    'StateSummaryProcessor',
    'GenerationBySourceProcessor',
    'GenerationOutagesProcessor',
    'InterRegionTransmissionProcessor',
    'TransnationalTransmissionProcessor',
    'TransnationalExchangeProcessor',
    'TransnationalSummaryProcessor',
    'RegionalImportExportSummaryProcessor',
    'ShareProcessor',
    'SolarNonSolarPeakProcessor',
    'FrequencyProfileProcessor',
    'ScadaTimeseriesProcessor',
    'DDFProcessor',
    # Logical processors
    'RegionalSummaryLogicalProcessor',
    'GenerationBySourceLogicalProcessor',
    'FrequencyProfileLogicalProcessor',
    'SolarNonSolarPeakLogicalProcessor',
    'TransnationalSummaryLogicalProcessor',
    'get_processor',
    'PROCESSOR_MAP'
] 