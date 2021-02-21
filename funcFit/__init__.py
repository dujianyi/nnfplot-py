from os.path import dirname, basename, isfile, join
import glob
import importlib

modules = glob.glob(join(dirname(__file__), "*.py"))
modules_to_import = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]

# print(__name__)
# Copy those names into the current name space
g = globals()
for module in modules_to_import:
	mod = importlib.import_module(__name__+'.'+module)
	g[module] = getattr(mod, module)