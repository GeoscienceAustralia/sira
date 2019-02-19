import logging
import logging.config


def configure_logger(log_path, loglevel='DEBUG'):
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'console_formatter': {
                'format': '%(levelname)-8s '+
                          '[%(name)s:%(lineno)s] %(message)s'},
            'file_formatter': {
                'format': '%(asctime)s,%(msecs)s - %(levelname)-8s '+
                          '[%(name)s:%(lineno)s] %(message)s'}
        },
        'handlers': {
            'console': {
                'level': loglevel,
                'class': 'logging.StreamHandler',
                'formatter': 'console_formatter',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'level': 'INFO',
                'class': 'logging.FileHandler',
                'formatter': 'file_formatter',
                'filename': log_path,
                'mode': 'w'
            }
        },
        'loggers': {
            '': {
                'level': 'DEBUG',
                'handlers': ['console', 'file']
            }
        },
    })
