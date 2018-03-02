# rootLogger = logging.getLogger("rootLogger")
# logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
# consoleHandler = logging.StreamHandler()
# consoleHandler.setFormatter(logFormatter)
# rootLogger.addHandler(consoleHandler)
# rootLogger.setLevel(logging.INFO)
#
# logging.info('its something')



import logging
import time

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger("rootLogger")

logPath='sifra/logs'
time_start = time.strftime("%Y%m%d-%H%M")
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
