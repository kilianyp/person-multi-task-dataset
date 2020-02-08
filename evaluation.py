from abc import abstractmethod


class Evaluation(object):
    def __init__(self, name):
        # relevant endpoints to write out
        self.writer = None
        self.output_path = None
        self.name = name

    @abstractmethod
    def get_writer(self, output_path):
        self.output_path = output_path
        pass

    @abstractmethod
    def score(self):
        pass

    def before_infere(self, data):
        return data

    def before_saving(self, endpoints, data):
        return {**endpoints, **data}
