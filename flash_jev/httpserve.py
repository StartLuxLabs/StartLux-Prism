"""A minimal HTTP/1.1 server for a JSON API: one thread per connection, keep-alive, the request line and the few
headers that matter parsed by hand.  http.server parses every request's headers through email.parser and reads through
buffered file objects - a few hundred microseconds per request in this interpreter, which is a measurable share of a
5 ms decision.  Requests on one connection are handled in order; bodies need a Content-Length (no chunked uploads).

    serve(host, port, handle)    handle(method, path, body: bytes) -> (status, body: bytes)
"""
from __future__ import annotations

import socket
import threading
import uuid

REASONS = {200: "OK", 400: "Bad Request", 404: "Not Found", 411: "Length Required", 413: "Payload Too Large",
           422: "Unprocessable Entity", 431: "Request Header Fields Too Large", 500: "Internal Server Error",
           501: "Not Implemented"}
MAX_HEADER = 64 << 10
MAX_BODY = 64 << 20


def _response(status, body):
    return (f"HTTP/1.1 {status} {REASONS.get(status, '')}\r\nContent-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\nx-typesafe-request-id: {uuid.uuid4().hex}\r\n\r\n").encode() + body


def _connection(conn, handle):
    buf = bytearray()
    with conn:
        while True:
            end = buf.find(b"\r\n\r\n")
            while end < 0:
                if len(buf) > MAX_HEADER:
                    conn.sendall(_response(431, b'{"error": "headers too large"}'))
                    return
                chunk = conn.recv(65536)
                if not chunk:
                    return
                buf += chunk
                end = buf.find(b"\r\n\r\n")
            lines = bytes(buf[:end]).split(b"\r\n")
            del buf[:end + 4]
            try:
                method, path, version = lines[0].decode("latin-1").split(" ", 2)
            except ValueError:
                conn.sendall(_response(400, b'{"error": "malformed request line"}'))
                return
            length, close = 0, version != "HTTP/1.1"
            for line in lines[1:]:
                k, _, v = line.partition(b":")
                k = k.strip().lower()
                if k == b"content-length":
                    length = int(v)
                elif k == b"connection":
                    v = v.strip().lower()
                    close = v == b"close" or (close and v != b"keep-alive")
                elif k == b"transfer-encoding":
                    conn.sendall(_response(501, b'{"error": "chunked bodies are not supported"}'))
                    return
            if length > MAX_BODY:
                conn.sendall(_response(413, b'{"error": "body too large"}'))
                return
            while len(buf) < length:
                chunk = conn.recv(max(65536, length - len(buf)))
                if not chunk:
                    return
                buf += chunk
            body = bytes(buf[:length])
            del buf[:length]
            status, data = handle(method, path, body)
            conn.sendall(_response(status, data))      # one write: status line, headers and body in one segment
            if close:
                return


def serve(host, port, handle, backlog=1024):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(backlog)
    while True:
        conn, _ = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)   # responses must not wait for delayed ACKs
        threading.Thread(target=_connection, args=(conn, handle), daemon=True).start()
