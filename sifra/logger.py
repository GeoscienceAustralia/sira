import datetime
import logging
import os

class Logger():
    def __init__(self):

        # formate to display log messages
        self.logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")

        # define logger object to reference in modules and display logs
        self.logger = logging.getLogger("rootLogger")

        # path to save logs
        if os.path.exists(os.path.join(os.getcwd(),'logs')):
            self.logPath = 'logs'
        else:
            self.logPath = 'sifra/logs'
        # name of the file
        self.time_start = self.get_round_off_time()

        # handler to save logs to file
        self.fileHandler = logging.FileHandler("{0}/{1}.log".format(self.logPath, self.time_start), mode='a')
        self.fileHandler.setFormatter(self.logFormatter)
        self.logger.addHandler(self.fileHandler)

        # handler to display logs to console
        self.consoleHandler = logging.StreamHandler()
        self.consoleHandler.setFormatter(self.logFormatter)
        self.logger.addHandler(self.consoleHandler)

        # default option
        self.logger.setLevel(logging.INFO)

    def get_round_off_time(self):
        round_mins = 5
        now = datetime.datetime.now()
        mins = now.minute - (now.minute % round_mins)
        time_start = str(datetime.datetime(now.year, now.month, now.day, now.hour, mins) + datetime.timedelta(
            minutes=round_mins)).replace(' ', '_').replace(':', '-')
        return time_start

    # change log level, it has to be a logging.level object eg logging.INFO
    def set_log_level(self,level):
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







