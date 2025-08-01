"""
GPU-accelerated NLDC parser implementation
"""

from typing import Dict, Any, List
from src.parsers.gpu_base_parser import GPUBaseParser
from src.models.data_models import RegionalSummary, StateSummary, GenerationBySource
from loguru import logger
from decimal import Decimal

class GPUNLDCParser(GPUBaseParser):
    """GPU-accelerated NLDC parser implementation"""
    
    def _get_table_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Define NLDC table mappings based on real PDF structures"""
        return {
            "regional_summary": {
                "table_name": "Power Supply Position at All India and Regional level",
                "expected_columns": ["NR", "WR", "SR", "ER", "NER", "TOTAL"],
                "type": "regional_summary"
            },
            "state_summary": {
                "table_name": "Power Supply Position in States",
                "expected_columns": ["States", "Max.Demand Met", "Shortage", "Drawal Schedule", "OD(+)/UD(-)"],
                "type": "state_summary"
            },
            "generation_by_source": {
                "table_name": "Sourcewise generation (Gross) (MU)",
                "expected_columns": ["Source", "All India"],
                "type": "generation_by_source"
            },
            "frequency_profile": {
                "table_name": "Frequency Profile",
                "expected_columns": ["Frequency Band", "Percentage Time", "FVI"],
                "type": "frequency_profile"
            },
            "inter_regional_exchange": {
                "table_name": "Inter-Regional Exchange",
                "expected_columns": ["From", "To", "Scheduled", "Actual", "Deviation"],
                "type": "inter_regional_exchange"
            },
            "transnational_exchange": {
                "table_name": "Transnational Exchange",
                "expected_columns": ["Country", "Scheduled", "Actual", "Deviation"],
                "type": "transnational_exchange"
            },
            "time_series_data": {
                "table_name": "Time Series Data",
                "expected_columns": ["Time", "Frequency", "Demand", "Generation"],
                "type": "time_series_data"
            }
        }
    
    def parse_regional_summary(self, table_data: List[List[str]]) -> List[RegionalSummary]:
        """Parse regional summary table with enhanced handling for 'All India' row"""
        try:
            summaries = []
            
            if len(table_data) < 2:
                logger.warning("Regional summary table has insufficient data")
                return summaries
            
            # Get headers and data rows
            headers = table_data[0]
            data_rows = table_data[1:]
            
            logger.info(f"Parsing regional summary with headers: {headers}")
            logger.info(f"Found {len(data_rows)} data rows")
            
            # Skip header row and process data rows
            for row_idx, row in enumerate(data_rows):
                if len(row) < 2:  # Need at least region code and one value
                    logger.warning(f"Skipping row {row_idx + 1}: insufficient columns")
                    continue
                
                try:
                    # Extract region code from first column
                    region_code = row[0].strip()
                    
                    # Handle "TOTAL" or "All India" row
                    if region_code.upper() in ["TOTAL", "ALL INDIA", "ALLINDIA"]:
                        region_code = "All India"
                    
                    # Extract numeric values (skip first column which is region code)
                    numeric_values = []
                    for i in range(1, len(row)):
                        value = self._clean_numeric_value(row[i])
                        numeric_values.append(value)
                    
                    # Create RegionalSummary object
                    # For NLDC, we typically have: Peak Demand, Peak Met, Energy Requirement, Energy Met
                    if len(numeric_values) >= 4:
                        summary = RegionalSummary(
                            region=region_code,
                            demand=numeric_values[0],  # Peak Demand
                            generation=numeric_values[1],  # Peak Met
                            deficit=numeric_values[2],  # Energy Requirement
                            frequency=numeric_values[3] if len(numeric_values) > 3 else None  # Energy Met or other metric
                        )
                        summaries.append(summary)
                        logger.info(f"Parsed regional summary for {region_code}: {summary}")
                    else:
                        logger.warning(f"Insufficient numeric values for {region_code}: {numeric_values}")
                        
                except Exception as e:
                    logger.warning(f"Error parsing regional summary row {row_idx + 1} {row}: {str(e)}")
                    continue
            
            logger.info(f"Successfully parsed {len(summaries)} regional summaries")
            return summaries
            
        except Exception as e:
            logger.error(f"Error parsing regional summary: {str(e)}")
            return []
    
    def parse_state_summary(self, table_data: List[List[str]]) -> List[StateSummary]:
        """Parse state summary table with enhanced column handling"""
        try:
            summaries = []
            
            if len(table_data) < 2:
                logger.warning("State summary table has insufficient data")
                return summaries
            
            # Get headers and data rows
            headers = table_data[0]
            data_rows = table_data[1:]
            
            logger.info(f"Parsing state summary with headers: {headers}")
            logger.info(f"Found {len(data_rows)} data rows")
            
            # Skip header row and process data rows
            for row_idx, row in enumerate(data_rows):
                if len(row) < 2:  # Need at least state name and one value
                    logger.warning(f"Skipping row {row_idx + 1}: insufficient columns")
                    continue
                
                try:
                    # Extract state name from first column
                    state_name = row[0].strip()
                    
                    # Skip empty or header-like rows
                    if not state_name or state_name.upper() in ["STATES", "STATE", "TOTAL", "ALL INDIA"]:
                        continue
                    
                    # Extract numeric values (skip first column which is state name)
                    numeric_values = []
                    for i in range(1, len(row)):
                        value = self._clean_numeric_value(row[i])
                        numeric_values.append(value)
                    
                    # Create StateSummary object
                    # For NLDC state summary, we typically have: Max Demand Met, Shortage, Drawal Schedule, OD/UD
                    if len(numeric_values) >= 3:
                        summary = StateSummary(
                            state=state_name,
                            demand=numeric_values[0],  # Max Demand Met
                            generation=numeric_values[1],  # Shortage (or other metric)
                            deficit=numeric_values[2],  # Drawal Schedule
                            frequency=numeric_values[3] if len(numeric_values) > 3 else None  # OD/UD
                        )
                        summaries.append(summary)
                        logger.info(f"Parsed state summary for {state_name}: {summary}")
                    else:
                        logger.warning(f"Insufficient numeric values for {state_name}: {numeric_values}")
                        
                except Exception as e:
                    logger.warning(f"Error parsing state summary row {row_idx + 1} {row}: {str(e)}")
                    continue
            
            logger.info(f"Successfully parsed {len(summaries)} state summaries")
            return summaries
            
        except Exception as e:
            logger.error(f"Error parsing state summary: {str(e)}")
            return []
    
    def parse_generation_by_source(self, table_data: List[List[str]]) -> List[GenerationBySource]:
        """Parse generation by source table with enhanced handling"""
        try:
            generations = []
            
            if len(table_data) < 2:
                logger.warning("Generation by source table has insufficient data")
                return generations
            
            # Get headers and data rows
            headers = table_data[0]
            data_rows = table_data[1:]
            
            logger.info(f"Parsing generation by source with headers: {headers}")
            logger.info(f"Found {len(data_rows)} data rows")
            
            # Skip header row and process data rows
            for row_idx, row in enumerate(data_rows):
                if len(row) < 2:  # Need at least source name and generation value
                    logger.warning(f"Skipping row {row_idx + 1}: insufficient columns")
                    continue
                
                try:
                    # Extract source name from first column
                    source_name = row[0].strip()
                    
                    # Skip empty or header-like rows
                    if not source_name or source_name.upper() in ["SOURCE", "TOTAL", "ALL INDIA"]:
                        continue
                    
                    # Extract generation value (second column)
                    generation_value = self._clean_numeric_value(row[1])
                    
                    # Extract percentage if available (third column)
                    percentage_value = None
                    if len(row) > 2:
                        percentage_value = self._clean_numeric_value(row[2])
                    
                    # Create GenerationBySource object
                    generation = GenerationBySource(
                        source=source_name,
                        generation=generation_value,
                        percentage=percentage_value
                    )
                    generations.append(generation)
                    logger.info(f"Parsed generation by source for {source_name}: {generation}")
                        
                except Exception as e:
                    logger.warning(f"Error parsing generation by source row {row_idx + 1} {row}: {str(e)}")
                    continue
            
            logger.info(f"Successfully parsed {len(generations)} generation by source entries")
            return generations
            
        except Exception as e:
            logger.error(f"Error parsing generation by source: {str(e)}")
            return [] 