
class Backend(object):
    def __init__(self, name):
        self.name = name
        self.config_templates = {}
        self.function_templates = {}
        self.include_lists = {}

    def get_config_template(self, kind):
        return self.config_templates.get(kind)

    def get_function_template(self, kind):
        return self.function_templates.get(kind)

    def get_include_list(self, kind):
        return self.include_lists.get(kind, [])

    def register_templates(self, name, function_template, config_template, include_list=[]):
        self.function_templates[name] = function_template
        self.config_templates[name] = config_template
        self.include_lists[name] = include_list

    def register_source(self, file_name, source, destination_dir='nnet_utils'):
        raise NotImplementedError


backend_map = {}


def register_backend(name, backend_cls):
    if name in backend_map:
        raise Exception('Backend {} already registered'.format(name))

    backend_map[name] = backend_cls()


def get_backend(name):
    return backend_map[name]
