
class OptimizerPass(object):
    def __init__(self):
        pass

    def match(self, node):
        raise NotImplementedError

    def transform(self, model, node):
        raise NotImplementedError


optimizer_map = {}


def register_pass(name, opt_cls):
    if name in optimizer_map:
        raise Exception('Optimization pass {} already registered'.format(name))

    if type(name) in [list, tuple]:
        for n in name:
            optimizer_map[n] = opt_cls
    else:
        optimizer_map[name] = opt_cls


def get_optimizer(name):
    return optimizer_map[name]()


def get_available_passes():
    return list(optimizer_map.keys())


def optimize_model(model, passes=None):
    if passes is None:
        passes = optimizer_map.keys()
    optimizers = [get_optimizer(opt_pass) for opt_pass in passes]
    optimization_done = False
    while not optimization_done:
        for opt in optimizers:
            for node in model.graph.values():
                if opt.match(node):
                    res = opt.transform(model, node)
                    if res:
                        break
            else:
                continue
            break
        else:
            optimization_done = True
