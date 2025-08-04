#!/usr/bin/env python3
"""
SCADA Time Series Processor

This processor handles the 'SCADA Time Series' table type.
It extracts SCADA time series data and structures it for database storage.
"""

from ..base_processor import BaseProcessor
from typing import List, Dict
from loguru import logger

class ScadaTimeseriesProcessor(BaseProcessor):
    """Processor for the 'SCADA Time Series' table."""
    TABLE_TYPE = "scada_timeseries"
    KEYWORDS = ['scada', 'time', 'timestamp', 'instantaneous', '15-minute', 'all india scada', 'frequency', 'demand']
    REQUIRED_COLUMNS = [
        "timestamp", "frequency_hz", "power_demand_mw", "report_date"
    ]
    PROMPT_TEMPLATE = """
You are an expert data extraction agent. Your task is to analyze the following raw table text of SCADA time series data and convert it into a structured JSON array. Adhere strictly to the specified format.

**Raw Table Text:**
```
{input_table}
```

**Instructions:**
1. Parse the text to identify data for each time interval (typically 15-minute intervals).
2. For each time interval, create a JSON object with the following keys.
3. Use the provided `report_date` for all records.
4. If a value is missing or cannot be determined, use `null`.
5. Ensure all numerical values are cleaned and converted to numbers.

**Target JSON Schema:**
- `timestamp`: (String) Time stamp in HH:MM format (e.g., "00:00", "00:15", "00:30").
- `frequency_hz`: (Number) System frequency (Hz).
- `power_demand_mw`: (Number) All India demand (MW).
- `power_nuclear_mw`: (Number) Nuclear power used in the 15 minute interval (MW).
- `power_wind_mw`: (Number) Wind power used in the 15 minute interval (MW).
- `power_solar_mw`: (Number) Solar power used in the 15 minute interval (MW).
- `power_hydro_mw`: (Number) Hydro power used in the 15 minute interval (MW).
- `power_gas_mw`: (Number) Gas power used in the 15 minute interval (MW).
- `power_thermal_mw`: (Number) Thermal power used in the 15 minute interval (MW).
- `power_other_mw`: (Number) Other power used in the 15 minute interval (MW).
- `power_net_demand_met_mw`: (Number) Demand met excluding the variable power of wind and solar (MW) .
- `power_total_generation_mw`: (Number) Total generating power used (Sum of all sources used) in the 15 minute interval (MW).
- `power_net_exchange_mw`: (Number) Net Transnational exchange (MW).
- `report_date`: (String) "{report_date}"

**Output ONLY the final JSON array. Do not include any explanations or surrounding text.**
"""

    def validate(self, data: List[Dict]) -> bool:
        """Stricter validation for the SCADA time series table."""
        if not super().validate(data):
            return False
        
        # Check for expected time intervals (96 intervals for 24 hours)
        if len(data) < 90:  # Allow some flexibility
            logger.warning(f"Validation warning for {self.TABLE_TYPE}: Found fewer than 90 time intervals ({len(data)}).")

        # Check for expected frequency range (49.5-50.5 Hz)
        for record in data:
            freq = record.get("frequency_hz")
            if freq is not None and isinstance(freq, (int, float)):
                if not (49.0 <= freq <= 51.0):
                    logger.warning(f"Validation warning for {self.TABLE_TYPE}: Frequency {freq} Hz is outside expected range (49.0-51.0 Hz).")

        # Check for plausible numeric values
        for record in data:
            if record.get("power_demand_mw") is not None and not isinstance(record["power_demand_mw"], (int, float)):
                logger.error(f"Validation failed for {self.TABLE_TYPE}: 'power_demand_mw' is not a number for timestamp {record.get('timestamp')}.")
                return False

        logger.info(f"Strict validation passed for {self.TABLE_TYPE}.")
        return True 