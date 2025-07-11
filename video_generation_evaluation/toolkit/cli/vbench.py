import argparse
import importlib
import subprocess

vbench_cmd = ['evaluate', 'static_filter']

def main():
    parser = argparse.ArgumentParser(prog="toolkit", formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(title='toolkit subcommands')

    for cmd in vbench_cmd:
        module = importlib.import_module(f'toolkit.cli.{cmd}')
        module.register_subparsers(subparsers)
    parser.set_defaults(func=help)
    args = parser.parse_args()
    args.func(args)

def help(args):
    subprocess.run(['toolkit', '-h'], check=True)
