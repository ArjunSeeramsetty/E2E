#!/usr/bin/env python3
"""
Processors Module

This module contains specialized processors for different types of power supply data tables.
Each processor handles the specific structural and data format requirements of its target table type.
"""

import pandas as pd
from typing import Optional

# Import base processor and specific processors
from .base_processor import BaseProcessor
from .regional_summary_processor import RegionalSummaryProcessor
from .state_summary_processor import StateSummaryProcessor
from .generation_by_source_processor import GenerationBySourceProcessor
from .generation_outages_processor import GenerationOutagesProcessor
from .inter_region_transmission_processor import InterRegionTransmissionProcessor
from .transnational_transmission_processor import TransnationalTransmissionProcessor
from .transnational_exchange_processor import TransnationalExchangeProcessor
from .transnational_summary_processor import TransnationalSummaryProcessor
from .regional_import_export_summary_processor import RegionalImportExportSummaryProcessor
from .share_processor import ShareProcessor
from .solar_non_solar_peak_processor import SolarNonSolarPeakProcessor
from .frequency_profile_processor import FrequencyProfileProcessor
from .scada_timeseries_processor import ScadaTimeseriesProcessor
from .ddf_processor import DDFProcessor

# Import legacy functions for backward compatibility
from .raw_state_table_processing import (
    process_state_table_robustly,
    validate_state_table_data,
    get_state_table_summary,
    detect_row_structure_robust,
    extract_numeric_value_robust
)

# Map table types to their corresponding processor classes
PROCESSOR_MAP = {
    "regional_daily_summary": RegionalSummaryProcessor,
    "state_daily_summary": StateSummaryProcessor,
    "generation_by_source": GenerationBySourceProcessor,
    "generation_outages": GenerationOutagesProcessor,
    "inter_regional_transmission": InterRegionTransmissionProcessor,
    "transnational_transmission": TransnationalTransmissionProcessor,
    "transnational_exchange": TransnationalExchangeProcessor,
    "transnational_summary": TransnationalSummaryProcessor,
    "regional_import_export_summary": RegionalImportExportSummaryProcessor,
    "share": ShareProcessor,
    "solar_non_solar_peak": SolarNonSolarPeakProcessor,
    "frequency_profile": FrequencyProfileProcessor,
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
    'get_processor',
    'PROCESSOR_MAP'
] 