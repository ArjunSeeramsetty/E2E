#!/usr/bin/env python3
"""
Base Processor for Power Supply Data Tables

This module contains the base class for all table processors.
Each processor handles the specific structural and data format requirements of its target table type.
"""

import requests
import json
import time
import os
from typing import List, Dict, Any, Optional
import pandas as pd
from loguru import logger

class BaseProcessor:
    """
    Base class for all table processors. Contains shared logic for
    LLM interaction, type detection, and validation.
    """
    # To be defined by subclasses
    TABLE_TYPE = "unknown"
    KEYWORDS = []
    PROMPT_TEMPLATE = ""
    REQUIRED_COLUMNS = []

    # Shared LLM Configuration
    LLM_ENDPOINT = "https://api.openai.com/v1/chat/completions"
    LLM_MODEL = "gpt-3.5-turbo"
    LLM_TIMEOUT = 120
    LLM_MAX_RETRIES = 2

    @staticmethod
    def detect_table_type(table_df: pd.DataFrame) -> str:
        """
        Detects table type based on keywords in its content.
        This method is now static and can be called from the factory.
        """
        if table_df.empty:
            return 'unknown'
        
        content_sample = " ".join(map(str, table_df.columns)) + " " + " ".join(map(str, table_df.head(2).values.flatten()))
        content_sample = content_sample.lower()

        # This part needs to be aware of all processor types
        from . import PROCESSOR_MAP
        scores = {}
        for type_name, processor_class in PROCESSOR_MAP.items():
            score = sum(1 for pattern in processor_class.KEYWORDS if pattern in content_sample)
            scores[type_name] = score

        if not scores:
            return 'unknown'
            
        best_type = max(scores, key=scores.get)
        return best_type if scores[best_type] > 0 else 'unknown'

    def call_llm_with_retry(self, prompt: str) -> Optional[str]:
        """Calls the OpenAI API with a retry mechanism."""
        for attempt in range(self.LLM_MAX_RETRIES):
            try:
                headers = {
                    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": self.LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are a data extraction expert. Return only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 4000
                }
                
                response = requests.post(self.LLM_ENDPOINT, json=payload, headers=headers, timeout=self.LLM_TIMEOUT)
                response.raise_for_status()
                
                result = response.json()
                return result["choices"][0]["message"]["content"]
                
            except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
                logger.warning(f"LLM call attempt {attempt + 1} for {self.TABLE_TYPE} failed: {e}")
                if attempt < self.LLM_MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
        return None

    def validate(self, data: List[Dict]) -> bool:
        """
        Validates the structured data from the LLM.
        Subclasses should override this for more specific checks.
        """
        if not isinstance(data, list) or not data:
            logger.error(f"Validation failed for {self.TABLE_TYPE}: Data is not a non-empty list.")
            return False

        # Check if all required columns are present in the first record
        first_record = data[0]
        if not all(key in first_record for key in self.REQUIRED_COLUMNS):
            missing_keys = [key for key in self.REQUIRED_COLUMNS if key not in first_record]
            logger.error(f"Validation failed for {self.TABLE_TYPE}: Missing required keys {missing_keys}.")
            return False
        
        logger.info(f"Basic validation passed for {self.TABLE_TYPE}.")
        return True

    def process(self, table_df: pd.DataFrame, report_date: str) -> Optional[List[Dict]]:
        """Main processing logic for a table."""
        logger.info(f"Processing with {self.__class__.__name__}...")
        input_table_str = table_df.to_string(index=False, header=True)
        prompt = self.PROMPT_TEMPLATE.format(input_table=input_table_str, report_date=report_date)

        llm_response = self.call_llm_with_retry(prompt)

        if not llm_response:
            logger.error(f"LLM failed to process table for {self.TABLE_TYPE}.")
            return None

        try:
            structured_data = json.loads(llm_response)
            
            # Handle case where LLM returns a dict with "data" key instead of direct array
            if isinstance(structured_data, dict) and "data" in structured_data:
                logger.info(f"LLM returned dict with 'data' key, extracting array for {self.TABLE_TYPE}.")
                structured_data = structured_data["data"]
            
            if self.validate(structured_data):
                logger.info(f"Successfully processed and validated data for {self.TABLE_TYPE}.")
                return structured_data
            else:
                logger.error(f"Validation failed for LLM response for {self.TABLE_TYPE}.")
                return None
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            try:
                import re
                # Look for JSON wrapped in markdown code blocks
                json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', llm_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    structured_data = json.loads(json_str)
                    
                    # Handle case where LLM returns a dict with "data" key instead of direct array
                    if isinstance(structured_data, dict) and "data" in structured_data:
                        logger.info(f"LLM returned dict with 'data' key, extracting array for {self.TABLE_TYPE}.")
                        structured_data = structured_data["data"]
                    
                    if self.validate(structured_data):
                        logger.info(f"Successfully processed and validated data for {self.TABLE_TYPE}.")
                        return structured_data
                    else:
                        logger.error(f"Validation failed for LLM response for {self.TABLE_TYPE}.")
                        return None
                else:
                    logger.error(f"Failed to decode JSON from LLM response for {self.TABLE_TYPE}.")
                    return None
            except (json.JSONDecodeError, AttributeError):
                logger.error(f"Failed to decode JSON from LLM response for {self.TABLE_TYPE}.")
                return None 