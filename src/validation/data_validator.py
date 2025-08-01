"""
Data validation for parsed reports
"""

from typing import List, Dict, Any
from src.models.data_models import ParsedReport
from loguru import logger
from dataclasses import dataclass

@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]

class DataValidator:
    """Validator for parsed report data"""
    
    def validate(self, parsed_report: ParsedReport) -> ValidationResult:
        """Validate parsed report data"""
        errors = []
        warnings = []
        
        # Basic validation
        if not parsed_report.report.report_date:
            errors.append("Report date is required")
        
        if not parsed_report.report.source_entity:
            errors.append("Source entity is required")
        
        # Validate regional summaries
        for regional_summary in parsed_report.regional_summaries:
            if not regional_summary.region_code:
                errors.append("Region code is required for regional summary")
            
            # Check for negative values
            if regional_summary.peak_demand_met_mw and regional_summary.peak_demand_met_mw < 0:
                warnings.append(f"Negative peak demand for region {regional_summary.region_code}")
        
        # Validate state summaries
        for state_summary in parsed_report.state_summaries:
            if not state_summary.state_name:
                errors.append("State name is required for state summary")
        
        # Check data consistency
        if parsed_report.regional_summaries and parsed_report.state_summaries:
            # Basic consistency check - should have some data
            if len(parsed_report.regional_summaries) == 0:
                warnings.append("No regional summaries found")
            
            if len(parsed_report.state_summaries) == 0:
                warnings.append("No state summaries found")
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings
        ) 