import logging


class RunMock(object):
    def __init__(self) -> None:
        super().__init__()
        self._logger = logging.getLogger("run_logger")

    def log(self, name, value, description=""):
        self._logger.info(f"{name}, {value}")

    def log_list(self, name, value, description=""):
        self._logger.info(f"{name}, {value}")

    def log_now(self, name, description, **kwargs):
        self._logger.info(f"{name}")

    def log_table(self, name, value, description=""):
        self._logger.info(f"{name}, {value}")

    def log_image(self, name, path=None, plot=None):
        self._logger.info(f"{name}")

    def tag(self, key, value=None):
        self._logger.info(f"{key}, {value}")

    def upload_file(self, name, path_or_stream):
        self._logger.info(f"{name}")
