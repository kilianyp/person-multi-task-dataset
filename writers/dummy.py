class DummyWriter(object):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def write(self, **kwargs):
        pass



