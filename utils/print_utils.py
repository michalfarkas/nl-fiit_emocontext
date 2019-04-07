import contextlib
import sys
import os
import _io

class MuteFile(object):
    def write(self, x): pass

class IndentFile(object):
    def __init__(self, indent, old_std):
        self.indent = indent
        self.old_std = old_std
    def write(self, x):
        x_strip = x.rstrip()
        if len(x) > 0 and x[0] == "\n":
            self.old_std.write(os.linesep)
        elif len(x) > 0:
            if x[0] == '\r':
                # weird cases, for example used by progressbars :(
                text_with_indent = "\r" + self.indent + x[1:]
            else:
                text_with_indent = self.indent + x_strip
            self.old_std.write(text_with_indent)
    def flush(self):
        self.old_std.flush()

@contextlib.contextmanager
def nostd(mode='out'):
    """
    Used to mute output or errors. Any output in the with block will be muted.

    Usage:
    with nostd('):
        whatever()

    :param mode: stream to mute
    :return: None
    """
    if mode == 'out':
        save_stdout = sys.stdout
        sys.stdout = MuteFile()
        yield
        sys.stdout = save_stdout
    elif mode == 'err':
        save_stderr = sys.stderr
        sys.stderr = MuteFile()
        yield
        sys.stderr = save_stderr
    else:
        raise NotImplementedError

@contextlib.contextmanager
def add_indent(level, mode='out', char='.', char_per_level=2):
    """
    Used to add indent to the output or errors. Anything from the with block will be indented.

    Usage:
    with add_indent(2):
        whatever()

    :param level: how much to indent
    :param mode: stream to indent
    :param char: char to use as indent
    :param char_per_level: how many chars per level
    :return:
    """
    if mode == 'out':
        indent = char*level*char_per_level
        save_stdout = sys.stdout
        sys.stdout = IndentFile(indent, save_stdout)
        yield
        sys.stdout = save_stdout
    elif mode == 'err':
        indent = char * level * char_per_level
        save_stderr = sys.stderr
        sys.stderr = IndentFile(indent, save_stderr)
        yield
        sys.stderr = save_stderr
