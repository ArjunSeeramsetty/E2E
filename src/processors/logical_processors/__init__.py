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
    # Legacy functions for backward compatibility
    'process_state_table_robustly',
    'validate_state_table_data',
    'get_state_table_summary',
    'detect_row_structure_robust',
    'extract_numeric_value_robust',
] 