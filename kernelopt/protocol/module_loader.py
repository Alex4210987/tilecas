"""Load a module from a path, for candidates and exported programs alike.

Both exporters need this and neither owns it; it lived in the CUDA exporter
only because that one was written first.
"""
import importlib.util


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
