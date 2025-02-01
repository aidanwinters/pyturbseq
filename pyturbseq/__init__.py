from importlib.metadata import version

try:
    __version__ = version("pyturbseq")
except ImportError:
    # Package is not installed
    __version__ = "unknown"