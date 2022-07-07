from smart.errors import assertion


# enables / disables auto detection, administrated by DummyWriter
class Config:
    VERBOSE = True
    _AUTO_DETECTION = True


# allows to temporarily change config with python's with statement
class ConfigOff:
    def __init__(self, config):
        self.config = config
        self.items = {}

    def __call__(self, **kwargs):
        for key, val in kwargs.items():
            self.items[key] = get_config(key)
            set_config(key, val)
        return self

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        for key, val in self.items.items():
            set_config(key, val)
        return self


config = Config()
temp_config = ConfigOff(config)

def set_config(key, val):
    match key:
        case "VERBOSE":
            assertion(isinstance(val, bool), "VERBOSE must be a boolean value")
            config.VERBOSE = val
        case "_AUTO_DETECTION": # prefered not to change it 
            assertion(isinstance(val, bool), "AUTO_DETECTION must be a boolean value")
            config._AUTO_DETECTION = val

def get_config(key):
    match key:
        case "VERBOSE":
            return config.VERBOSE
        case "_AUTO_DETECTION": 
            return config._AUTO_DETECTION
        case _:
            assertion(False, f"Config: key {key} not found")