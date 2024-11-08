import logging

def set_up_logger():
    logger = logging.getLogger("AIR")
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def set_log_name(name:str):
    file_handler = logging.FileHandler(f"{name}.log")
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

logger = set_up_logger()

