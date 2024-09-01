import os
import logging
import time
from colorlog import ColoredFormatter

_logger_level_dict = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARN,
    "error": logging.ERROR,
    "fatal": logging.FATAL,
}

def setup_logger(args):
    if hasattr(args, 'logger_level'):
        logger_level = _logger_level_dict[args.logger_level]
    else:
        logger_level = logging.INFO

    format_head = f"%(log_color)s[%(asctime)s][%(levelname)s] %(message)s%(reset)s"

    logger = logging.getLogger()
    logger.setLevel(logger_level)

    console = logging.StreamHandler()
    console.setLevel(logger_level)
    
    formatter = ColoredFormatter(format_head, datefmt="%m/%d-%I:%M:%S")
    console.setFormatter(formatter)
    logger.addHandler(console)







