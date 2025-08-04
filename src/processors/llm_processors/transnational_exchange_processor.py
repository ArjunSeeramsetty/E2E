#!/usr/bin/env python3
"""
Transnational Exchange Processor

This processor handles the 'Transnational Exchange' table type.
It extracts country-wise transnational exchange data and structures it for database storage.
"""

from ..base_processor import BaseProcessor
from typing import List, Dict
from loguru import logger

class TransnationalExchangeProcessor(BaseProcessor):
    """Processor for the 'Transnational Exchange' table."""
    TABLE_TYPE = "transnational_exchange"
    KEYWORDS = ['country', 'bilateral', 'gna', 'isga', 'ppa', 'idam', 'rtm', 'iex', 'pxil', 'hpx', 'bhutan', 'nepal', 'bangladesh', 'myanmar']
    REQUIRED_COLUMNS = [
        "country", "energy_gna_ppa_mu", "energy_bilateral_mu", "energy_idam_iex_mu", "energy_idam_pxil_mu", "energy_idam_hpx_mu", "energy_rtm_iex_mu", "energy_rtm_pxil_mu", "energy_rtm_hpx_mu", "energy_total_mu", "report_date"
    ]
    PROMPT_TEMPLATE = """
You are an expert data extraction agent. Your task is to analyze the following raw table text of transnational exchange data and convert it into a structured JSON array. Adhere strictly to the specified format.

**Raw Table Text:**
```
{input_table}
```

**Instructions:**
1. Parse the text to identify data for each country (Bhutan, Nepal, Bangladesh, Myanmar).
2. For each country, create a JSON object with the following keys.
3. Use the provided `report_date` for all records.
4. If a value is missing or cannot be determined, use `null`.
5. Ensure all numerical values are cleaned and converted to numbers.
6. **IMPORTANT: Extract ALL energy exchange data including the detailed breakdown.**

**Target JSON Schema:**
- `country`: (String) Country name (Bhutan, Nepal, Bangladesh, Myanmar).
- `energy_gna_ppa_mu`: (Number) Energy through General Network Access/Power Purchase Agreement (MU).
- `energy_bilateral_mu`: (Number) Energy through Bilateral agreements (MU).
- `energy_idam_iex_mu`: (Number) Energy through Day Ahead Market in Indian Energy Exchange (MU).
- `energy_idam_pxil_mu`: (Number) Energy through Day Ahead Market in PXIL (MU).
- `energy_idam_hpx_mu`: (Number) Energy through Day Ahead Market in HPX (MU).
- `energy_rtm_iex_mu`: (Number) Energy through Real Time Market in Indian Energy Exchange (MU).
- `energy_rtm_pxil_mu`: (Number) Energy through Real Time Market in PXIL (MU).
- `energy_rtm_hpx_mu`: (Number) Energy through Real Time Market in HPX (MU).
- `energy_total_mu`: (Number) Total energy exchange (MU).
- `report_date`: (String) "{report_date}"

**EXACT COLUMN MAPPING (based on actual data structure):**
For any country row: "Country,Col2,Col3,Col4,Col5,Col6,Col7,Col8,Col9,Col10"
- Column 1: Country name → country
- Column 2: GNA (ISGS/PPA) value → energy_gna_ppa_mu
- Column 3: T-GNA (Bilateral) value → energy_bilateral_mu
- Column 4: TOTAL (IDAM IEX) value → energy_idam_iex_mu
- Column 5: IDAM PXIL value → energy_idam_pxil_mu
- Column 6: IDAM HPX value → energy_idam_hpx_mu
- Column 7: RTM IEX value → energy_rtm_iex_mu
- Column 8: RTM PXIL value → energy_rtm_pxil_mu
- Column 9: RTM HPX value → energy_rtm_hpx_mu
- Column 10: Total value → energy_total_mu

**IMPORTANT: The table has hierarchical headers that need to be understood:**
- Row 1: Main headers (Country, GNA (ISGS/PPA), T-GNA, TOTAL, Unnamed: 0, Unnamed: 1, Unnamed: 2, Unnamed: 3, Unnamed: 4, Unnamed: 5)
- Row 2: Sub-headers (BILATERAL TOTAL, COLLECTIVE, etc.)
- Row 3: Sub-headers (IDAM, RTM, etc.)
- Row 4: Sub-headers (IEX, PXIL, HPX, IEX, PXIL, HPX, etc.)

**Output ONLY the final JSON array. Do not include any explanations or surrounding text.**
"""

    def validate(self, data: List[Dict]) -> bool:
        """Stricter validation for the transnational exchange table."""
        if not super().validate(data):
            return False
        
        # Check for expected countries
        expected_countries = {"Bhutan", "Nepal", "Bangladesh", "Myanmar"}
        found_countries = {record.get("country") for record in data}

        if not found_countries.intersection(expected_countries):
            logger.warning(f"Validation warning for {self.TABLE_TYPE}: No expected countries found.")

        # Check for plausible numeric values
        for record in data:
            if record.get("energy_total_mu") is not None and not isinstance(record["energy_total_mu"], (int, float)):
                logger.error(f"Validation failed for {self.TABLE_TYPE}: 'energy_total_mu' is not a number for country {record.get('country')}.")
                return False

        logger.info(f"Strict validation passed for {self.TABLE_TYPE}.")
        return True 