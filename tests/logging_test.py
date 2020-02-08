import logging
import logging.handlers
import ipdb
logger = logging.getLogger('')
logger.handlers = []


class MemoryHandler(logging.handlers.MemoryHandler):
    def flush(self):
        self.acquire()
        try:
            for record in self.buffer:
                self.target.handle(record)
            self.buffer = []
            print('')
        finally:
            self.release()


sh = logging.StreamHandler()
sh.terminator = ''
formatter = logging.Formatter(" %(message)s |")
sh.setFormatter(formatter)
mh = MemoryHandler(100, target=sh)
logger.addHandler(mh)
logger.setLevel(logging.INFO)



logging.info("test123")
logging.info("test123")
logger.info("test123")
logger.debug("test123")
#ipdb.set_trace()
mh.flush()
logger.info("test123")
logger.info("test123")
