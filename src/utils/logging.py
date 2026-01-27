"""
Centralized logging configuration.

Provides consistent logging across all components:
- Scripts
- API
- Pipeline
- Background jobs

Usage:
    from src.utils.logging import setup_logging, get_logger
    
    # At application startup
    setup_logging()
    
    # In any module
    logger = get_logger(__name__)
    logger.info("Starting process")
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.config import settings


# =============================================================================
# Log Formatters
# =============================================================================

class ColoredFormatter(logging.Formatter):
    """Colored console formatter for development."""
    
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname:8}{self.RESET}"
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON formatter for production/structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        import json
        
        log_record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        
        if hasattr(record, "extra"):
            log_record.update(record.extra)
        
        return json.dumps(log_record)


# =============================================================================
# Setup Functions
# =============================================================================

_logging_configured = False


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[Path] = None,
    json_format: bool = False,
    force: bool = False,
) -> logging.Logger:
    """
    Configure application-wide logging.
    
    Call once at application startup (main script, API startup, etc.)
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR). Defaults to config.
        log_file: Optional file path. Defaults to config.
        json_format: Use JSON format (recommended for production).
        force: Force reconfiguration even if already configured.
        
    Returns:
        Root application logger
    """
    global _logging_configured
    
    if _logging_configured and not force:
        return logging.getLogger("betting_model")
    
    level = level or settings.log_level
    log_file = log_file or settings.log_file
    
    # Root logger for our application
    root_logger = logging.getLogger("betting_model")
    root_logger.setLevel(getattr(logging, level.upper()))
    root_logger.handlers = []  # Clear existing handlers
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    if json_format:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(ColoredFormatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
    
    root_logger.addHandler(console_handler)
    
    # File handler (if configured)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        root_logger.addHandler(file_handler)
    
    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    _logging_configured = True
    root_logger.debug("Logging configured")
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger under the application namespace.
    
    Args:
        name: Logger name (usually __name__ or module name)
        
    Returns:
        Configured logger instance
        
    Usage:
        logger = get_logger(__name__)
        logger.info("Processing started")
    """
    # Remove common prefixes for cleaner names
    if name.startswith("src."):
        name = name[4:]
    
    return logging.getLogger(f"betting_model.{name}")


# =============================================================================
# Context Managers
# =============================================================================

class LogContext:
    """
    Context manager for adding context to log messages.
    
    Usage:
        with LogContext(match_id=123, league="PL"):
            logger.info("Processing match")  # Includes context
    """
    
    def __init__(self, **context):
        self.context = context
        self._old_factory = None
    
    def __enter__(self):
        self._old_factory = logging.getLogRecordFactory()
        
        def factory(*args, **kwargs):
            record = self._old_factory(*args, **kwargs)
            record.extra = self.context
            return record
        
        logging.setLogRecordFactory(factory)
        return self
    
    def __exit__(self, *args):
        logging.setLogRecordFactory(self._old_factory)
