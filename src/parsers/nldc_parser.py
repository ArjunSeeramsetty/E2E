"""
NLDC (National Load Despatch Centre) specific parser
"""

from typing import Dict, Any, List
from src.parsers.base_parser import BaseParser
from src.models.data_models import RegionalSummary, StateSummary, GenerationBySource
from loguru import logger
from decimal import Decimal

class NLDCParser(BaseParser):
    """Parser for NLDC daily PSP reports"""
    
    def _get_table_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Define NLDC-specific table mappings"""
        return {
            "regional_summary": {
                "table_name": "Power Supply Position at All India and Regional level",
                "expected_columns": ["Region", "Demand Met during Evening Peak hrs(MW)", "Peak Shortage (MW)", "Energy Met (MU)", "Energy Shortage (MU)"]
            },
            "state_summary": {
                "table_name": "Power Supply Position in States",
                "expected_columns": ["States", "Max.Demand Met during the day (MW)", "Shortage at Max.Demand (MW)", "Drawal Schedule (MU)", "OD(+)/UD(-)(MU)"]
            },
            "generation_by_source": {
                "table_name": "Sourcewise generation (Gross) (MU)",
                "expected_columns": ["Source", "NR", "WR", "SR", "ER", "NER", "All India"]
            }
        }
    
    def parse_regional_summary(self, table_data: List[List[str]]) -> List[RegionalSummary]:
        """Parse NLDC Table A: Power Supply Position at All India and Regional level"""
        regional_summaries = []
        
        try:
            # Skip header row
            for row in table_data[1:]:
                if len(row) < 5:  # Minimum required columns
                    continue
                
                region_code = row[0].strip()
                if region_code in ["NR", "WR", "SR", "ER", "NER", "All India"]:
                    regional_summary = RegionalSummary(
                        region_code=region_code,
                        peak_demand_met_mw=self._clean_numeric_value(row[1]),
                        peak_shortage_mw=self._clean_numeric_value(row[2]),
                        energy_met_mu=self._clean_numeric_value(row[3]),
                        energy_shortage_mu=self._clean_numeric_value(row[4])
                    )
                    regional_summaries.append(regional_summary)
            
            logger.info(f"Parsed {len(regional_summaries)} regional summaries")
            return regional_summaries
            
        except Exception as e:
            logger.error(f"Error parsing regional summary: {str(e)}")
            return []
    
    def parse_state_summary(self, table_data: List[List[str]]) -> List[StateSummary]:
        """Parse NLDC Table C: Power Supply Position in States"""
        state_summaries = []
        
        try:
            # Skip header row
            for row in table_data[1:]:
                if len(row) < 5:  # Minimum required columns
                    continue
                
                state_name = row[0].strip()
                if state_name and state_name not in ["States", "Total"]:
                    state_summary = StateSummary(
                        state_name=state_name,
                        max_demand_met_mw=self._clean_numeric_value(row[1]),
                        shortage_at_max_demand_mw=self._clean_numeric_value(row[2]),
                        drawal_schedule_mu=self._clean_numeric_value(row[3]),
                        over_under_drawal_mu=self._clean_numeric_value(row[4])
                    )
                    state_summaries.append(state_summary)
            
            logger.info(f"Parsed {len(state_summaries)} state summaries")
            return state_summaries
            
        except Exception as e:
            logger.error(f"Error parsing state summary: {str(e)}")
            return []
    
    def parse_generation_by_source(self, table_data: List[List[str]]) -> List[GenerationBySource]:
        """Parse NLDC Table G: Sourcewise generation (Gross) (MU)"""
        generation_sources = []
        
        try:
            # Skip header row
            for row in table_data[1:]:
                if len(row) < 2:  # Minimum required columns
                    continue
                
                source_type = row[0].strip()
                if source_type and source_type not in ["Source", "Total"]:
                    # Parse All India column (usually the last column)
                    all_india_value = self._clean_numeric_value(row[-1]) if len(row) > 6 else None
                    
                    generation_source = GenerationBySource(
                        source_type=source_type,
                        gross_generation_mu=all_india_value
                    )
                    generation_sources.append(generation_source)
            
            logger.info(f"Parsed {len(generation_sources)} generation sources")
            return generation_sources
            
        except Exception as e:
            logger.error(f"Error parsing generation by source: {str(e)}")
            return [] 