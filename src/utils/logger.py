import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Setup logging at pipeline start"""
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        level=getattr(logging, level.upper())
    )

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)