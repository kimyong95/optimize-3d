import inspect

class collect_calls:
    def __init__(self, target_method, arg_names=None):
        self.obj, self.func_name = target_method.__self__, target_method.__name__
        self.original_func = getattr(self.obj, self.func_name)
        self.param_names = list(inspect.signature(self.original_func).parameters)
        self.names_to_collect = set(arg_names or [])
        self.data = []

    def __enter__(self):
        def wrapper(*args, **kwargs):
            res = self.original_func(*args, **kwargs)
            all_args = {**dict(zip(self.param_names, args)), **kwargs}
            self.data.append({'args': {k: v for k, v in all_args.items() if k in self.names_to_collect}, 'return': res})
            return res
        setattr(self.obj, self.func_name, wrapper)
        return self.data

    def __exit__(self, *args):
        setattr(self.obj, self.func_name, self.original_func)