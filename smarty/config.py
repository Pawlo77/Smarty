from smarty.errors import assertion


# enables / disables auto detection, administrated by DummyWriter
class Config:
    """Smarty config

    :var bool VERBOSE: model's verbosity
    :var bool _AUTO_DETECTION: **enables/disables** *smarty.datasets.datasets.DataSet* **auto dtype detection**, preferable not to change by set param but by *smarty.datasets.datasets.DataSet.dummy_writer*
    """
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
"""
Accuall *smarty.config.Config()* object that holds liblary configuration (note: after program's execution any changes are gone). 

.. note::
    It is not reccommended to change the config directly.
"""

temp_config = ConfigOff(config)
"""Allows to temporarily change the config with auto-restore to previous option, ex:

.. code-block:: python
    :linenos:
    
    from smarty.config import temp_config

    with temp_config(VERBOSE=False):
        model.predict(test_ds)
"""


def set_config(key, val):
    """Set smarty.config.config property

    :param str key: smarty.config.Config() class var
    :param val: new value for smarty.config.config.key

    :raises: AssertionError if provided key's val is in wrong dtype or if provided key is not *smarty.config.Config* var
    """
    match key:
        case "VERBOSE":
            assertion(isinstance(val, bool), "VERBOSE must be a boolean value")
            config.VERBOSE = val
        case "_AUTO_DETECTION": # prefered not to change it 
            assertion(isinstance(val, bool), "AUTO_DETECTION must be a boolean value")
            config._AUTO_DETECTION = val
        case _:
            assertion(False, f"Config: key {key} not found")

def get_config(key):
    """Get smarty.config.config property

    :param str key: smarty.config.Config() class var
    :raises: AssertionError if provided key is not *smarty.config.Config* var
    """

    match key:
        case "VERBOSE":
            return config.VERBOSE
        case "_AUTO_DETECTION": 
            return config._AUTO_DETECTION
        case _:
            assertion(False, f"Config: key {key} not found")