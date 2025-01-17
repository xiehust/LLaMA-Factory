import os
import sys
import traceback
import re
from collections import namedtuple
import extensions

import gradio as gr

scripts_data = []
ScriptFile = namedtuple("ScriptFile", ["basedir", "filename", "path"])
ScriptClassData = namedtuple("ScriptClassData", ["script_class", "path", "basedir", "module"])
SCRIP_PATH = os.path.dirname(__file__)

class Script:
    filename = None
    args_from = None
    args_to = None
    alwayson = False

    """A gr.Group component that has all script's UI inside it"""
    group = None

    infotext_fields = None
    """if set in ui(), this is a list of pairs of gradio component + text; the text will be used when
    parsing infotext to set the value for the component; see ui.py's txt2img_paste_fields for an example
    """

    def title(self):
        """this function should return the title of the script. This is what will be displayed in the dropdown menu."""

        raise NotImplementedError()

    def ui(self, is_img2img):
        """this function should create gradio UI elements. See https://gradio.app/docs/#components
        The return value should be an array of all components that are used in processing.
        Values of those returned componenbts will be passed to run() and process() functions.
        """

        pass

def list_scripts(scriptdirname, extension):
    scripts_list = []

    basedir = os.path.join(SCRIP_PATH, scriptdirname)
    if os.path.exists(basedir):
        for filename in sorted(os.listdir(basedir)):
            scripts_list.append(ScriptFile(SCRIP_PATH, filename, os.path.join(basedir, filename)))

    for ext in extensions.active():
        scripts_list += ext.list_files(scriptdirname, extension)

    scripts_list = [x for x in scripts_list if os.path.splitext(x.path)[1].lower() == extension and os.path.isfile(x.path)]

    return scripts_list