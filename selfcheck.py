"""Offline self-check — proves the classifier loads and runs with NO network access.

It hard-disables socket creation, then loads the local model and classifies a sample. If any code
path tried to reach the network it would raise; a clean pass is positive proof that no external /
API calls are required.

Run:  python selfcheck.py
Exit 0 = offline-ready · non-zero = a network dependency leaked in.
"""
import os
import socket
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _block_network():
    """Block host resolution and outbound connections so any network attempt raises immediately.
    (Patches DNS + connection entry points rather than the socket class, which some libs subclass.)"""
    def _no(*_a, **_k):
        raise RuntimeError("network access attempted during offline self-check")

    socket.getaddrinfo = _no
    socket.create_connection = _no
    socket.socket.connect = _no
    socket.socket.connect_ex = _no


def main() -> int:
    _block_network()
    import pandas as pd
    from girp import load_local_model, classify_columns

    model, dev = load_local_model()                      # local weights only
    df = pd.DataFrame({"text": ["Call Sarah Lee on 02 9000 0000 about the refund.",
                                "The quarterly report is ready."]})
    out = classify_columns(model, df, ["text"], progress=False)
    levels = out["girp_level"].tolist()
    assert levels[0] in ("Private", "Confidential"), levels
    assert levels[1] == "Public", levels
    print(f"OFFLINE SELF-CHECK PASSED on {dev}: model loaded + classified with networking disabled. "
          f"levels={levels}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"OFFLINE SELF-CHECK FAILED: {type(e).__name__}: {e}")
        sys.exit(1)
