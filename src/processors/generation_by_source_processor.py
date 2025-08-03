#!/usr/bin/env python3
"""
Generation By Source Processor

This processor handles the 'Generation By Source' table type.
It extracts source-wise generation data and structures it for database storage.
"""

from .base_processor import BaseProcessor
from typing import List, Dict
from loguru import logger

class GenerationBySourceProcessor(BaseProcessor):
    """Processor for the 'Generation By Source' table."""
    TABLE_TYPE = "generation_by_source"
    KEYWORDS = ['source', 'sourcewise', 'gross generation', 'fuel type', 'coal', 'hydro', 'nuclear', 'thermal']
    REQUIRED_COLUMNS = [
        "region_name", "source_name", "energy_generation_mu", "report_date"
    ]
    PROMPT_TEMPLATE = """
You are an expert data extraction agent. Your task is to analyze the following raw table text of source-wise generation data and convert it into a structured JSON array. Adhere strictly to the specified format.

**Raw Table Text:**
```
{input_table}
```

**Instructions:**
1. Parse the text to identify data for each generation source (Coal, Lignite, Hydro, Nuclear, Gas Naptha & Diesel, RES (Wind, Solar, Biomass & Others), etc.).
2. For each source AND each region, create a separate JSON object with the following keys.
3. Extract data for ALL regions: NR, WR, SR, ER, NER, and All India.
4. Use the provided `report_date` for all records.
5. If a value is missing or cannot be determined, use `null`.
6. Ensure all numerical values are cleaned and converted to numbers.
7. Create separate records for each region-source combination.

**Target JSON Schema:**
- `region_name`: (String) Region name (e.g., "NR", "WR", "SR", "ER", "NER", "All India").
- `source_name`: (String) Name of the generation source (e.g., "Coal", "Lignite", "Hydro", "Nuclear").
- `energy_generation_mu`: (Number) Energy Generation (MU).
- `report_date`: (String) "{report_date}"

**Important:** Create separate JSON objects for each region-source combination. For example, if Coal has values for NR, WR, SR, ER, NER, and All India, create 6 separate records, one for each region.

**Output ONLY the final JSON array. Do not include any explanations or surrounding text.**
"""

    def validate(self, data: List[Dict]) -> bool:
        """Stricter validation for the generation by source table."""
        if not super().validate(data):
            return False
        
        expected_sources = {"Coal", "Lignite", "Hydro", "Nuclear", "Gas, Naptha & Diesel", "RES (Wind, Solar, Biomass & Others)", "Total"}
        found_sources = {record.get("source_name") for record in data}

        if not expected_sources.issubset(found_sources):
            logger.warning(f"Validation warning for {self.TABLE_TYPE}: Not all expected sources found. Missing: {expected_sources - found_sources}")

        # Check for plausible numeric values
        for record in data:
            if record.get("energy_generation_mu") is not None and not isinstance(record["energy_generation_mu"], (int, float)):
                logger.error(f"Validation failed for {self.TABLE_TYPE}: 'energy_generation_mu' is not a number for source {record.get('source_name')}.")
                return False

        logger.info(f"Strict validation passed for {self.TABLE_TYPE}.")
        return True 