import pickle
class PickleWriter(object):
    def __init__(self, output_file):
        self.output_file = output_file
    
    def write(self, **data_to_write):
        pickle.dump(data_to_write, self.f)

    def __enter__(self):
        self.f = open(self.output_file, 'wb')
        return self
