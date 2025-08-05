#!/usr/bin/env python3
"""
Generation By Source Processor

This processor handles the 'Generation By Source' table type.
It extracts source-wise generation data with regions and energy sources.
"""

from ..base_processor import BaseProcessor
from typing import List, Dict
from loguru import logger

class GenerationBySourceProcessor(BaseProcessor):
    """Processor for the 'Generation By Source' table."""
    TABLE_TYPE = "generation_by_source"
    KEYWORDS = ['sourcewise generation', 'gross generation', 'fuel type', 'coal', 'lignite', 'hydro', 'nuclear', 'gas naptha diesel', 'res wind solar biomass', 'sourcewise']
    REQUIRED_COLUMNS = [
        "region_name", "source_name", "energy_generation_mu", "report_date"
    ]
    PROMPT_TEMPLATE = """
You are an expert data extraction agent. Your task is to analyze the following raw table text of generation by source data and convert it into a structured JSON array. Adhere strictly to the specified format.

**Raw Table Text:**
```
{input_table}
```

**Instructions:**
1. Parse the text to identify data for each region (NR, WR, SR, ER, NER, All India) and each source type.
2. For each region-source combination, create a JSON object with the following keys.
3. Use the provided `report_date` for all records.
4. If a value is missing or cannot be determined, use `null`.
5. Ensure all numerical values are cleaned and converted to numbers.
6. **IMPORTANT: Extract all source-wise generation data for each region.**

**Target JSON Schema:**
- `region_name`: (String) Region name (NR, WR, SR, ER, NER, All India).
- `source_name`: (String) Source name (Coal, Lignite, Hydro, Nuclear, Gas Naptha & Diesel, RES, Total).
- `energy_generation_mu`: (Number) Energy generation in MU.
- `report_date`: (String) "{report_date}"

**EXACT COLUMN MAPPING (based on actual data structure):**
The table has sources as rows and regions as columns:
- Column 1: Source names (Coal, Lignite, Hydro, Nuclear, Gas Naptha & Diesel, RES, Total)
- Subsequent columns: Regions (NR, WR, SR, ER, NER, All India)
- Values: Energy generation in MU for each source-region combination

**Output ONLY the final JSON array. Do not include any explanations or surrounding text.**
"""

    def validate(self, data: List[Dict]) -> bool:
        """Stricter validation for the generation by source table."""
        if not super().validate(data):
            return False
        
        # Check for expected sources
        expected_sources = {"Coal", "Lignite", "Hydro", "Nuclear", "Gas, Naptha & Diesel", "RES (Wind, Solar, Biomass & Others)", "Total"}
        found_sources = {record.get("source_name") for record in data}

        if not found_sources.intersection(expected_sources):
            logger.warning(f"Validation warning for {self.TABLE_TYPE}: No expected sources found.")

        # Check for plausible numeric values
        for record in data:
            if record.get("energy_generation_mu") is not None and not isinstance(record["energy_generation_mu"], (int, float)):
                logger.error(f"Validation failed for {self.TABLE_TYPE}: 'energy_generation_mu' is not a number for source {record.get('source_name')} in region {record.get('region_name')}.")
                return False

        logger.info(f"Strict validation passed for {self.TABLE_TYPE}.")
        return True 