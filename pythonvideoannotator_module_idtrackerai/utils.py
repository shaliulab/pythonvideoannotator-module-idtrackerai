import subprocess
import os
from confapp import conf

try:
    import sys

    sys.path.append(os.getcwd())
    import local_settings # type: ignore

    conf += local_settings
except Exception as e:
    print(e)
    pass

def show_message(message):
    print(message)
    subprocess.Popen(["notify-send"] + [f"\"{message}\""])


def notify_propagation(n_past, n_future):

    message = f"{n_past} <- UPDATED -> {n_future}"
    show_message(message)
