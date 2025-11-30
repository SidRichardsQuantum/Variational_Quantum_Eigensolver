import subprocess
import sys
import shutil

def run_cmd(cmd):
    return subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def test_vqe_cli_works():
    if shutil.which("vqe") is None:
        # Entry point not installed, skip inside CI before install step
        return
    proc = run_cmd("vqe --molecule H2 --steps 1")
    assert proc.returncode == 0

def test_qpe_cli_works():
    if shutil.which("qpe") is None:
        return
    proc = run_cmd("qpe --molecule H2 --ancillas 1")
    assert proc.returncode == 0
