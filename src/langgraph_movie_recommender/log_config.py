import logging

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "huggingface_hub",
    "sentence_transformers",
)


def configure_logging(level_name="INFO"):
    """Configure root logging from the LOG_LEVEL setting."""
    level = logging.getLevelNamesMapping().get(str(level_name).strip().upper(), logging.INFO)
    logging.basicConfig(level=level, format=LOG_FORMAT, force=True)

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(max(level, logging.WARNING))
