"""
Database connection and operations
"""

import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from src.models.database_schema import Base
from src.models.data_models import ParsedReport
from loguru import logger
from typing import Optional

class DatabaseConnection:
    """Database connection manager"""
    
    def __init__(self, connection_string: Optional[str] = None):
        if connection_string is None:
            # Default to SQLite for development
            connection_string = "sqlite:///power_supply_data.db"
        
        self.engine = create_engine(connection_string)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create tables
        self._create_tables()
    
    def _create_tables(self):
        """Create all tables if they don't exist"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except SQLAlchemyError as e:
            logger.error(f"Error creating tables: {str(e)}")
            raise
    
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    def store_report(self, parsed_report: ParsedReport) -> bool:
        """Store parsed report in database"""
        try:
            with self.get_session() as session:
                # Convert Pydantic models to SQLAlchemy models
                from src.models.database_schema import Reports, RegionalSummaries, StateSummaries, GenerationBySource
                
                # Create report record
                report = Reports(
                    report_date=parsed_report.report.report_date,
                    source_entity=parsed_report.report.source_entity,
                    source_url=parsed_report.report.source_url,
                    ingestion_timestamp=parsed_report.report.ingestion_timestamp
                )
                session.add(report)
                session.flush()  # Get the report_id
                
                # Store regional summaries
                for regional_summary in parsed_report.regional_summaries:
                    db_regional = RegionalSummaries(
                        report_id=report.report_id,
                        region_code=regional_summary.region_code,
                        peak_demand_met_mw=regional_summary.peak_demand_met_mw,
                        peak_shortage_mw=regional_summary.peak_shortage_mw,
                        energy_met_mu=regional_summary.energy_met_mu,
                        energy_shortage_mu=regional_summary.energy_shortage_mu,
                        max_demand_met_day_mw=regional_summary.max_demand_met_day_mw,
                        time_of_max_demand=regional_summary.time_of_max_demand
                    )
                    session.add(db_regional)
                    session.flush()  # Get the summary_id
                    
                    # Store state summaries for this region
                    for state_summary in parsed_report.state_summaries:
                        if state_summary.state_name:  # Basic validation
                            db_state = StateSummaries(
                                summary_id=db_regional.summary_id,
                                state_name=state_summary.state_name,
                                max_demand_met_mw=state_summary.max_demand_met_mw,
                                shortage_at_max_demand_mw=state_summary.shortage_at_max_demand_mw,
                                drawal_schedule_mu=state_summary.drawal_schedule_mu,
                                over_under_drawal_mu=state_summary.over_under_drawal_mu
                            )
                            session.add(db_state)
                    
                    # Store generation by source for this region
                    for gen_source in parsed_report.generation_by_source:
                        db_gen = GenerationBySource(
                            summary_id=db_regional.summary_id,
                            source_type=gen_source.source_type,
                            gross_generation_mu=gen_source.gross_generation_mu,
                            percentage_share=gen_source.percentage_share
                        )
                        session.add(db_gen)
                
                session.commit()
                logger.info(f"Successfully stored report for {parsed_report.report.report_date}")
                return True
                
        except SQLAlchemyError as e:
            logger.error(f"Database error storing report: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error storing report: {str(e)}")
            return False
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.get_session() as session:
                result = session.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error(f"Database connection test failed: {str(e)}")
            return False 