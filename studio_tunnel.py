"""Reconnectable, localhost-only SSH tunnel for a remotely hosted movie studio.

Optional desktop dependency: pip install 'paramiko>=3.5,<5'
Uses the existing SSH key and known_hosts; never accepts an unknown host key.
"""
from __future__ import annotations

import argparse
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import socket
import socketserver
import threading

import paramiko

LOG = logging.getLogger("studio-tunnel")


class Connection:
    def __init__(self, args: argparse.Namespace):
        self.args = args

    def open(self, port: int, origin: tuple[str, int]) -> paramiko.Channel:
        # Isolate browser sockets so a stalled long-lived video/SSE connection
        # cannot block every asset on a shared SSH transport.
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        known = Path.home() / ".ssh" / "known_hosts"
        if known.is_file():
            client.load_host_keys(str(known))
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
        try:
            client.connect(self.args.host, port=self.args.port, username=self.args.user,
                           key_filename=str(self.args.key), allow_agent=True, look_for_keys=False,
                           timeout=15, auth_timeout=30, banner_timeout=45, compress=False)
            transport = client.get_transport()
            assert transport is not None
            transport.set_keepalive(15)
            channel = transport.open_channel("direct-tcpip", ("127.0.0.1", port), origin, timeout=15)
            channel.studio_client = client
            return channel
        except Exception:
            client.close()
            raise

class ForwardServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    connection: Connection
    remote_port: int


class Forward(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        channel = None
        try:
            channel = self.server.connection.open(self.server.remote_port, self.client_address)
            LOG.info("Forward channel opened to port %s", self.server.remote_port)
            # Blocking pumps avoid Windows select() interactions with Paramiko's
            # synthetic channel file descriptors, including large asset stalls.
            def send_request() -> None:
                try:
                    while block := self.request.recv(65536):
                        channel.sendall(block)
                except (OSError, EOFError, paramiko.SSHException):
                    pass
                finally:
                    try:
                        channel.shutdown_write()
                    except OSError:
                        pass
            threading.Thread(target=send_request, daemon=True).start()
            channel.settimeout(60)
            while block := channel.recv(65536):
                self.request.sendall(block)
        except Exception as exc:
            LOG.warning("Forward unavailable (%s); next request reconnects", type(exc).__name__)
            if channel is None:
                try:
                    self.request.sendall(b"HTTP/1.1 503 Service Unavailable\r\nConnection: close\r\n"
                                         b"Content-Type: text/plain\r\n\r\n"
                                         b"Reconnecting to the movie studio. Reload this page shortly.")
                except OSError:
                    pass
        finally:
            if channel is not None:
                channel.close()
                channel.studio_client.close()
            try:
                self.request.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--user", default="root")
    parser.add_argument("--key", type=Path, default=Path.home() / ".ssh" / "id_ed25519")
    parser.add_argument("--http-port", type=int, default=7860)
    parser.add_argument("--oauth-port", type=int, default=7862)
    args = parser.parse_args()
    handler = RotatingFileHandler(Path(__file__).with_name("studio-tunnel.log"),
                                  maxBytes=512000, backupCount=2, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOG.addHandler(handler)
    LOG.setLevel(logging.INFO)
    connection = Connection(args)
    servers = []
    try:
        for port in (args.http_port, args.oauth_port):
            server = ForwardServer(("127.0.0.1", port), Forward)
            server.connection, server.remote_port = connection, port
            servers.append(server)
            threading.Thread(target=server.serve_forever, daemon=True).start()
        LOG.info("Local listeners ready: http://localhost:%s", args.http_port)
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    main()
