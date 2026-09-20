"""Local-only HTTP API and static UI; use private Codespaces port forwarding."""

import argparse
import json
import os
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .adapter import catalogue, json_bytes
from .jobs import Jobs

STATIC = Path(__file__).parent / "static"


def make_server(host="127.0.0.1", port=8000):
    jobs = Jobs()
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, body, content_type="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'self'; script-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def allowed_host(self):
            hostname = urlsplit("//" + self.headers.get("Host", "")).hostname
            allowed = {"localhost", "127.0.0.1", "::1"}
            codespace = os.environ.get("CODESPACE_NAME")
            domain = os.environ.get(
                "GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN", "app.github.dev"
            )
            if codespace:
                allowed.add(f"{codespace}-{self.server.server_port}.{domain}")
            return hostname in allowed

        def do_GET(self):
            if not self.allowed_host():
                self.send(403, json_bytes({"error": "Unrecognized Host"}))
                return
            path = urlsplit(self.path).path
            if path == "/api/catalogue":
                self.send(200, json_bytes({**catalogue(), "submission_token": token}))
            elif path == "/api/runs":
                # Avoid transferring statevectors/parameter histories on every poll.
                rows = jobs.history()
                for row in rows:
                    if "result" in row:
                        row["result"] = {
                            k: v
                            for k, v in row["result"].items()
                            if k
                            not in {
                                "final_state_real",
                                "final_state_imag",
                                "params_history",
                                "final_params",
                                "environment",
                            }
                        }
                self.send(200, json_bytes(rows))
            elif path.startswith("/api/runs/"):
                run_id = path.removeprefix("/api/runs/")
                row = next((r for r in jobs.history() if r["id"] == run_id), None)
                self.send(
                    200 if row else 404, json_bytes(row or {"error": "Run not found"})
                )
            elif path in {
                "/",
                "/app.js",
                "/model.mjs",
                "/style.css",
                "/ui.mjs",
                "/charts.mjs",
                "/comparison.mjs",
            }:
                name = "index.html" if path == "/" else path[1:]
                mime = {
                    "html": "text/html",
                    "js": "text/javascript",
                    "mjs": "text/javascript",
                    "css": "text/css",
                }[name.rsplit(".", 1)[1]]
                self.send(200, (STATIC / name).read_bytes(), mime + "; charset=utf-8")
            else:
                self.send(404, json_bytes({"error": "Not found"}))

        def do_POST(self):
            if not self.allowed_host() or not secrets.compare_digest(
                self.headers.get("X-Studio-Token", ""), token
            ):
                self.send(
                    403, json_bytes({"error": "Reload the studio before submitting"})
                )
                return
            if self.path.startswith("/api/runs/") and self.path.endswith("/cancel"):
                run_id = self.path[len("/api/runs/") : -len("/cancel")]
                try:
                    self.send(200, json_bytes(jobs.cancel(run_id)))
                except KeyError:
                    self.send(404, json_bytes({"error": "Run not found"}))
                return
            if self.path != "/api/runs":
                self.send(404, json_bytes({"error": "Not found"}))
                return
            try:
                if (
                    self.headers.get("Content-Type", "").split(";")[0]
                    != "application/json"
                ):
                    raise ValueError("Content-Type must be application/json")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16384:
                    raise ValueError("Expected a JSON body of at most 16 KiB")
                raw = json.loads(self.rfile.read(length))
                self.send(202, json_bytes(jobs.submit(raw)))
            except (ValueError, UnicodeError) as exc:
                self.send(400, json_bytes({"error": str(exc)}))

    try:
        server = ThreadingHTTPServer((host, port), Handler)
    except BaseException:
        jobs.close()
        raise
    server.jobs = jobs
    return server


def main():
    parser = argparse.ArgumentParser(description="Optional VQE Experiment Studio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()
    server = make_server(args.host, args.port)
    print(f"VQE Experiment Studio: http://{args.host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Cancelling queued and running experiments…", flush=True)
    finally:
        server.server_close()
        server.jobs.close()
