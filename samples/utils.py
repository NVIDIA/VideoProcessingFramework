import sys
import logging

SERVICE_LOGGING_FORMAT = (
    "[{filename:s}][{funcName:s}:{lineno:d}]" + "[{levelname:s}] {message:s}"
)
SERVICE_LOGGING_STREAM = sys.stdout


def get_logger(logger_name, log_level="info"):

    SERVICE_LOGGING_LEVEL = getattr(logging, log_level.upper(), None)

    logger = logging.getLogger(logger_name)
    logger.setLevel(SERVICE_LOGGING_LEVEL)
    ch = logging.StreamHandler(SERVICE_LOGGING_STREAM)
    formatter = logging.Formatter(SERVICE_LOGGING_FORMAT, style="{")
    ch.setFormatter(formatter)
    ch.setLevel(SERVICE_LOGGING_LEVEL)
    logger.addHandler(ch)
    logger.propagate = False

    return logger


logger = get_logger(__file__)
