#!/usr/bin/env python3
"""
Logical Processors Module

This module contains processors that use rule-based or logical processing for data
extraction. These processors rely on deterministic algorithms and pattern matching
rather than LLM-based processing.
"""

# Import all logical/rule-based processors
from .state_summary_processor import StateSummaryProcessor
from .ddf_processor import DDFProcessor
from .inter_region_transmission_processor import InterRegionTransmissionProcessor
from .regional_summary_logical_processor import RegionalSummaryLogicalProcessor
from .generation_by_source_logical_processor import GenerationBySourceLogicalProcessor
from .frequency_profile_logical_processor import FrequencyProfileLogicalProcessor
from .solar_non_solar_peak_logical_processor import SolarNonSolarPeakLogicalProcessor
from .transnational_summary_logical_processor import TransnationalSummaryLogicalProcessor
from .raw_state_table_processing import (
    process_state_table_robustly,
    validate_state_table_data,
    get_state_table_summary,
    detect_row_structure_robust,
    extract_numeric_value_robust
)

__all__ = [
    'StateSummaryProcessor',
    'DDFProcessor',
    'InterRegionTransmissionProcessor',
    'RegionalSummaryLogicalProcessor',
    'GenerationBySourceLogicalProcessor',
    'FrequencyProfileLogicalProcessor',
    'SolarNonSolarPeakLogicalProcessor',
    'TransnationalSummaryLogicalProcessor',
    # Legacy functions for backward compatibility
    'process_state_table_robustly',
    'validate_state_table_data',
    'get_state_table_summary',
    'detect_row_structure_robust',
    'extract_numeric_value_robust',
] 