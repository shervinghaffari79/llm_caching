"""
Minimal Clore.ai launcher for the GPU runs (REST API, 1 request/s limit).

    python scripts/clore.py balance
    python scripts/clore.py create SERVER_ID --ssh-key ~/.ssh/key.pub [--image IMG]
    python scripts/clore.py orders            # active orders with SSH host/port
    python scripts/clore.py cancel ORDER_ID

The API key is read from CLORE_API_KEY in .env and never printed.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

API = "https://api.clore.ai/v1/"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = "cloreai/jupyter:ubuntu24.04-v2"


def _key() -> str:
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("CLORE_API_KEY="):
            return line.split("=", 1)[1].strip().strip("'\"")
    raise SystemExit("CLORE_API_KEY missing from .env")


def call(endpoint: str, body: dict | None = None) -> dict:
    for attempt in range(5):
        req = urllib.request.Request(
            API + endpoint, headers={"auth": _key(), "Content-Type": "application/json"},
            data=json.dumps(body).encode() if body is not None else None,
            method="POST" if body is not None else "GET")
        with urllib.request.urlopen(req, timeout=30) as r:
            out = json.load(r)
        if out.get("code") != 5:                  # 5 = rate limited
            return out
        time.sleep(1.5 * (attempt + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("balance")
    c = sub.add_parser("create")
    c.add_argument("server", type=int)
    c.add_argument("--ssh-key", required=True)
    c.add_argument("--image", default=DEFAULT_IMAGE)
    sub.add_parser("orders")
    x = sub.add_parser("cancel")
    x.add_argument("order", type=int)
    a = ap.parse_args()

    if a.cmd == "balance":
        for w in call("wallets").get("wallets", []):
            print(f"{w['name']:18s} {w['balance']}")
    elif a.cmd == "create":
        body = {"currency": "USD-Blockchain", "image": a.image,
                "renting_server": a.server, "type": "on-demand",
                "ports": {"22": "tcp"},
                "ssh_key": Path(a.ssh_key).expanduser().read_text().strip()}
        print(json.dumps(call("create_order", body)))
    elif a.cmd == "orders":
        for o in call("my_orders").get("orders", []):
            print(json.dumps({k: o.get(k) for k in
                              ("id", "si", "image", "price", "pub_cluster", "tcp_ports",
                               "creation_fee", "spend", "expired")}))
    elif a.cmd == "cancel":
        print(json.dumps(call("cancel_order", {"id": a.order})))


if __name__ == "__main__":
    main()
