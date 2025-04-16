import os
import importlib.util


def import_all_providers(package_path):
    # Iterate over all Python files in the package folder.
    for filename in os.listdir(package_path):
        if filename.endswith(".py") and filename != "__init__.py":
            module_name = filename[:-3]
            module_path = os.path.join(package_path, filename)
            spec = importlib.util.spec_from_file_location(
                module_name, module_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)


# Use the function with the chat provider directory (adjust path accordingly)
import_all_providers(os.path.join(os.path.dirname(__file__), "chat_providers"))

# Now ChatFactory should have all providers registered.
