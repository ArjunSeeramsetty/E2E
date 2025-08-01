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
        """Define NLDC table mappings"""
        return {
            "regional_summary": {
                "headers": ["Region", "Demand", "Generation", "Deficit", "Frequency"],
                "type": "regional_summary"
            },
            "state_summary": {
                "headers": ["State", "Demand", "Generation", "Deficit", "Frequency"],
                "type": "state_summary"
            },
            "generation_by_source": {
                "headers": ["Source", "Generation", "Percentage"],
                "type": "generation_by_source"
            },
            "frequency_profile": {
                "headers": ["Time", "Frequency", "Demand", "Generation"],
                "type": "frequency_profile"
            },
            "time_series_data": {
                "headers": ["Time", "Frequency", "Demand", "Generation"],
                "type": "time_series_data"
            }
        }
    
    def parse_regional_summary(self, table_data: List[List[str]]) -> List[RegionalSummary]:
        """Parse regional summary table"""
        try:
            summaries = []
            
            # Skip header row
            for row in table_data[1:]:
                if len(row) >= 5:
                    try:
                        summary = RegionalSummary(
                            region=row[0].strip(),
                            demand=self._clean_numeric_value(row[1]),
                            generation=self._clean_numeric_value(row[2]),
                            deficit=self._clean_numeric_value(row[3]),
                            frequency=self._clean_numeric_value(row[4])
                        )
                        summaries.append(summary)
                    except Exception as e:
                        logger.warning(f"Error parsing regional summary row {row}: {str(e)}")
                        continue
            
            return summaries
            
        except Exception as e:
            logger.error(f"Error parsing regional summary: {str(e)}")
            return []
    
    def parse_state_summary(self, table_data: List[List[str]]) -> List[StateSummary]:
        """Parse state summary table"""
        try:
            summaries = []
            
            # Skip header row
            for row in table_data[1:]:
                if len(row) >= 5:
                    try:
                        summary = StateSummary(
                            state=row[0].strip(),
                            demand=self._clean_numeric_value(row[1]),
                            generation=self._clean_numeric_value(row[2]),
                            deficit=self._clean_numeric_value(row[3]),
                            frequency=self._clean_numeric_value(row[4])
                        )
                        summaries.append(summary)
                    except Exception as e:
                        logger.warning(f"Error parsing state summary row {row}: {str(e)}")
                        continue
            
            return summaries
            
        except Exception as e:
            logger.error(f"Error parsing state summary: {str(e)}")
            return []
    
    def parse_generation_by_source(self, table_data: List[List[str]]) -> List[GenerationBySource]:
        """Parse generation by source table"""
        try:
            generations = []
            
            # Skip header row
            for row in table_data[1:]:
                if len(row) >= 3:
                    try:
                        generation = GenerationBySource(
                            source=row[0].strip(),
                            generation=self._clean_numeric_value(row[1]),
                            percentage=self._clean_numeric_value(row[2])
                        )
                        generations.append(generation)
                    except Exception as e:
                        logger.warning(f"Error parsing generation by source row {row}: {str(e)}")
                        continue
            
            return generations
            
        except Exception as e:
            logger.error(f"Error parsing generation by source: {str(e)}")
            return [] 