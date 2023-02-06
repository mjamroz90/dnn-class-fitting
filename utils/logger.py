import logging


def log(clazz):
    def logger(name):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        if logger.hasHandlers():
            logger.handlers.clear()

        #create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # create formatter
        formatter = logging.Formatter('[%(filename)s:%(lineno)d] - %(name)s - %(levelname)s - %(message)s')

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(ch)
        return logger

    clazz.logger = logger(clazz.__name__)
    return clazz
