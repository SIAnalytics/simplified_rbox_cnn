import os
import time
import logging
import logging.handlers


class LoggingManager(object):
    instance = None

    def __new__(cls):
        if not isinstance(cls.instance, logging.Logger):
            if not os.path.exists('log'):
                os.makedirs('log')

            logger = logging.getLogger('tensorflow')
            logger.handlers = []
            logger.setLevel(logging.CRITICAL)
            cls.instance = logging.getLogger('logger')

            fomatter = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')
            struct_time = time.gmtime()
            date = [str(struct_time[i]) for i in range(7)]
            date_str = '-'.join(date)
            file_handle = logging.FileHandler('log/' + date_str + '.log')
            file_handle.setFormatter(fomatter)
            stream_handle = logging.StreamHandler()
            stream_handle.setFormatter(fomatter)

            cls.instance.handlers = []
            cls.instance.addHandler(file_handle)
            cls.instance.addHandler(stream_handle)
            cls.instance.setLevel(logging.DEBUG)

        return cls.instance


if LoggingManager.instance is None:
    LoggingManager()
