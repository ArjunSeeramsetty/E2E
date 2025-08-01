"""
Factory for creating appropriate parsers based on source entity
"""

from typing import Dict, Type
from src.parsers.base_parser import BaseParser
from src.parsers.nldc_parser import NLDCParser
from loguru import logger

class ParserFactory:
    """Factory for creating report parsers"""
    
    def __init__(self):
        self._parsers: Dict[str, Type[BaseParser]] = {
            'NLDC': NLDCParser,
            # Add other parsers as they're implemented
            # 'SRLDC': SRLDCParser,
            # 'NRLDC': NRLDCParser,
        }
    
    def get_parser(self, source_entity: str) -> BaseParser:
        """Get appropriate parser for source entity"""
        if source_entity not in self._parsers:
            raise ValueError(f"No parser available for source entity: {source_entity}")
        
        parser_class = self._parsers[source_entity]
        return parser_class()
    
    def register_parser(self, source_entity: str, parser_class: Type[BaseParser]):
        """Register a new parser"""
        self._parsers[source_entity] = parser_class
        logger.info(f"Registered parser for {source_entity}") 