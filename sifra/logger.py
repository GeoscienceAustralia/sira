import time
import coloredlogs, logging
import os

coloredlogs.DEFAULT_LOG_FORMAT = \
    '%(asctime)s [%(threadName)s] ' \
    '%(levelname)-8.8s %(message)s'

coloredlogs.COLOREDLOGS_LEVEL_STYLES = \
    'spam=22;debug=28;verbose=34;' \
    'notice=220;warning=202;success=118,bold;' \
    'error=124;critical=background=red'

# # Original coloredlog format defaults:
# coloredlogs.DEFAULT_LOG_FORMAT = \
#     '%(asctime)s %(name)s[%(process)d] ' \
#     '%(levelname)s %(message)s'


class Logger():
    def __init__(self):

        # define logger object to reference in modules and display logs
        logging.captureWarnings(True)
        self.logger = logging.getLogger('py.warnings')
        self.logger.setLevel(logging.DEBUG)

        # path to save logs
        # ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # if not os.path.exists(os.path.join(ROOT_DIR,'logs')):
            # os.makedirs(os.path.join(ROOT_DIR,'logs'))

        # self.log_path = os.path.join(ROOT_DIR,'logs')

        # name of the file
        # self.timestamp = time.strftime('%Y%m%d_%H%M%S')
        # self.logfile = os.path.join(self.log_path, 'sifralog_'+str(self.timestamp)+'.log')

        # handler to display LOGS to CONSOLE
        # ---------------------------------------------------------------------
        coloredlogs.install(level='DEBUG')
        coloredlogs.install(milliseconds=True)

        # handler to save LOGS to FILE
        # ---------------------------------------------------------------------
        self.logFormatter = logging.Formatter("%(asctime)s [%(threadName)s] %(levelname)-8.8s  %(message)s")

    def set_log_file_path(self, file_path):

        # if os.path.exists(file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.fileHandler = logging.FileHandler(file_path, mode='w+')
        self.fileHandler.setFormatter(self.logFormatter)
        self.logger.addHandler(self.fileHandler)

    # change log level, it has to be a logging.level object eg logging.INFO
    def set_log_level(self, input_verbose_level):

        level = logging.DEBUG

        if input_verbose_level is not None:
            if input_verbose_level.upper() == "DEBUG":
                level = logging.DEBUG

            elif input_verbose_level.upper() == "INFO":
                level = logging.INFO

            elif input_verbose_level.upper() == "WARNING":
                level = logging.WARNING

            elif input_verbose_level.upper() == "ERROR":
                level = logging.ERROR

            elif input_verbose_level.upper() == "CRITICAL":
                level = logging.CRITICAL

        self.logger.setLevel(level)

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def critical(self, msg):
        self.logger.critical(msg)


rootLogger = Logger()
