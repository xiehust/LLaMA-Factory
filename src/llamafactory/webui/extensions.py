import os
import sys
import traceback
import re
from collections import namedtuple
import gradio as gr
import scripts
import importlib.util

extensions = []
extensions_dir = os.path.join(os.path.dirname(__file__), "extensions")


class Extension:
    def __init__(self, name, path, enabled=True):
        self.name = name
        self.path = path
        self.enabled = enabled
        self.status = ''

    def load_modules(self):
        modules = {}
        
        # 遍历目录
        for filename in os.listdir(self.path):
            if filename.endswith('.py') and not filename.startswith('__'):
                # 获取完整的文件路径
                file_path = os.path.join(self.path, filename)
                
                module_name = filename[:-3]
                
                try:
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    modules[module_name] = module
                    print(f"Successfully imported {module_name}")
                except Exception as e:
                    print(f"Failed to import {module_name}: {str(e)}")
        return modules


    def list_files(self, subdir, extension):

        dirpath = os.path.join(self.path, subdir)
        if not os.path.isdir(dirpath):
            return []

        res = []
        for filename in sorted(os.listdir(dirpath)):
            res.append(scripts.ScriptFile(self.path, filename, os.path.join(dirpath, filename)))

        res = [x for x in res if os.path.splitext(x.path)[1].lower() == extension and os.path.isfile(x.path)]

        return res


def list_extensions():
    # list sub dir of extensions_dir
    extension_sub_dirname = os.listdir(extensions_dir)
    print(extension_sub_dirname)
    paths = []
    for dirname in extension_sub_dirname:
        path = os.path.join(extensions_dir,dirname)
        if not os.path.isdir(path):
            continue

        paths.append((dirname, path))

    extensions = []
    for dirname, path in paths:
        extension = Extension(name=dirname, path=path)
        extensions.append(extension)
    return extensions

def load_extensions():
    global extensions
    extensions = list_extensions()
    modules={}
    for extension in extensions:
        if modules.get(extension.name):
            print(f"Extension {extension.name} already loaded")
            continue
        modules[extension.name]=extension.load_modules()
    return modules
    
if __name__ == "__main__":    
    # 导入所有.py文件
    imported_modules = load_extensions()
    
    # 使用导入的模块
    for module_name, module in imported_modules.items():
        # 获取模块中的所有属性
        for attr_name in dir(module):
            if not attr_name.startswith('__'):
                # 获取属性
                attr = getattr(module, attr_name)
                print(f"Module: {module_name}, Attribute: {attr_name}, Type: {type(attr)}")
                
                # 如果是函数，可以调用它
                if callable(attr):
                    try:
                        result = attr()
                        print(f"Called {attr_name}() from {module_name}, result: {result}")
                    except Exception as e:
                        print(f"Error calling {attr_name}: {str(e)}")

