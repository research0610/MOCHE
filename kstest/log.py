import logging.config
import json
import os

config = json.load(open("logging.json"))
try:
    log_folder = config["handlers"]["info_file_handler"]["filename"].split("/")[0]
    if os.path.isdir(log_folder) == False:
        os.mkdir(log_folder)
        print("Creating log folder {}/{}".format(os.getcwd(), log_folder))
except:
    pass

logging.config.dictConfig(config)


def getLogger(name):
    logger = logging.getLogger(name)
    return logger
