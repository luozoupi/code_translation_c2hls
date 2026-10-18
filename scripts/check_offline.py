"""Run tests with real network and FPGA executable invocations blocked."""
import os
import re
import socket
import subprocess
import sys

os.environ.setdefault('C2HLS_PART', 'xcu280-fsvh2892-2L-e')
os.environ.setdefault('C2HLS_CLOCK_NS', '3.33')
os.environ.setdefault('C2HLS_VITIS_VERSION', '2023.2')
os.environ['C2HLS_RUN_VITIS_INTEGRATION'] = '0'

def deny_network(*args, **kwargs):
    raise RuntimeError('Offline validation blocks real network connections')

socket.socket.connect = deny_network
socket.socket.connect_ex = deny_network
original_popen = subprocess.Popen

class OfflinePopen(original_popen):
    def __init__(self, args, *positional, **kwargs):
        command = args if isinstance(args, str) else ' '.join(map(str, args))
        if '--version' not in command and re.search(r'(^|[\s/;&])(?:vitis-run|vitis_hls|v\+\+|vivado)(?:\s|$)', command):
            raise RuntimeError('Offline validation blocks real FPGA tool execution')
        super().__init__(args, *positional, **kwargs)

subprocess.Popen = OfflinePopen
sys.path.insert(0, os.getcwd())
import pytest
raise SystemExit(pytest.main(['-p', 'no:cacheprovider', *sys.argv[1:]]))
