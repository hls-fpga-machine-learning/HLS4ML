
class Writer(object):
    def __init__(self):
        pass

    def write_hls(self, model):
        raise NotImplementedError


writer_map = {}


def register_writer(name, writer_cls):
    if name in writer_map:
        raise Exception('Writer {} already registered'.format(name))

    writer_map[name] = writer_cls


def get_writer(name):
    return writer_map[name]()
