from importlib import resources

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

__version__ = "0.6.0"

_cfg = tomllib.loads(resources.read_text("strappy", "config.toml"))