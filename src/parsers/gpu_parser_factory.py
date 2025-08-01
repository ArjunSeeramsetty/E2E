"""
GPU-accelerated parser factory
"""

from typing import Dict, Type
from src.parsers.gpu_base_parser import GPUBaseParser
from src.parsers.gpu_nldc_parser import GPUNLDCParser
from loguru import logger

class GPUParserFactory:
    """Factory for creating GPU-accelerated parsers"""
    
    def __init__(self):
        self.parsers: Dict[str, Type[GPUBaseParser]] = {
            "NLDC": GPUNLDCParser,
            # Add other regional parsers here as they are implemented
            # "SRLDC": GPUSRLDCParser,
            # "NRLDC": GPUNRLDCParser,
            # "WRLDC": GPUWRLDCParser,
            # "ERLDC": GPUERLDCParser,
            # "NERLDC": GPUNERLDCParser,
        }
    
    def get_parser(self, source_entity: str) -> GPUBaseParser:
        """Get a parser instance for the given source entity"""
        try:
            if source_entity in self.parsers:
                parser_class = self.parsers[source_entity]
                return parser_class()
            else:
                logger.warning(f"No GPU parser found for {source_entity}, falling back to NLDC")
                return GPUNLDCParser()
        except Exception as e:
            logger.error(f"Error creating GPU parser for {source_entity}: {str(e)}")
            # Fallback to NLDC parser
            return GPUNLDCParser()
    
    def get_available_parsers(self) -> list:
        """Get list of available parser types"""
        return list(self.parsers.keys()) 