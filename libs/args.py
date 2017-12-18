import yaml


class Args(object):

    def __init__(self, yaml_path):
        self.dargs = None
        self.load(yaml_path)

    def __setattr__(self, key, value):
        object.__setattr__(self, key, value)
        if value is not None and not isinstance(value, dict):
            self.dargs[key] = value

    def load(self, path):
        with open(path, 'r') as f:
            d = yaml.load(f)
        self.dargs = d
        for k, v in d.items():
            setattr(self, k, v)

    def dump(self):
        r = ''
        for k, v in self.dargs.items():
            r += '{} : {}\n'.format(k, v)
        return r
