import time
import logging


class print_time:
    def __init__(self, obj=None, enter_msg=None, exit_msg=None):
        self.enter_msg = enter_msg if obj is None else "Starting to {}".format(obj)
        self.exit_msg = exit_msg if obj is None else "Successfully {}!".format(obj)

    def __enter__(self):
        logging.info(self.enter_msg)
        self.time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        tot = time.time() - self.time
        if tot >= 60:
            res = '{} minutes and {} seconds'.format(int(int(tot) / 60), int(tot) % 60)
        else:
            res = '{} seconds'.format(int(tot))
        logging.info(self.exit_msg + ': ' + res)
