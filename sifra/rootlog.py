# rootLogger = logging.getLogger("rootLogger")
# logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
# consoleHandler = logging.StreamHandler()
# consoleHandler.setFormatter(logFormatter)
# rootLogger.addHandler(consoleHandler)
# rootLogger.setLevel(logging.INFO)
#
# logging.info('its something')



import logging
import datetime
import os

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger("rootLogger")



round_mins = 5
now = datetime.datetime.now()
mins = now.minute - (now.minute % round_mins)

logPath = os.path.abspath('./sifra/logs')
if not os.path.exists(logPath):
    os.makedirs(logPath)

time_start = str(datetime.datetime(now.year, now.month, now.day, now.hour, mins) + datetime.timedelta(minutes=round_mins)).replace(' ', '_').replace(':', '-')
fileHandler = logging.FileHandler("{0}/{1}.log".format(logPath, time_start), mode='a')
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)
#
# args={verbose}
# args.verbose = "INFO"
# if args.verbose is not None:
#     if args.verbose.upper() == "DEBUG":
#         rootLogger.setLevel(logging.DEBUG)
#
#     elif args.verbose.upper() == "INFO":
#         rootLogger.setLevel(logging.INFO)
#
#     elif args.verbose.upper() == "WARNING":
#         rootLogger.setLevel(logging.WARNING)
#
#     elif args.verbose.upper() == "ERROR":
#         rootLogger.setLevel(logging.ERROR)
#
#     elif args.verbose.upper() == "CRITICAL":
#         rootLogger.setLevel(logging.CRITICAL)
# else:
#     # default option
rootLogger.setLevel(logging.INFO)
