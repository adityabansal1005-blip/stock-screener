"""Import the dashboard and exercise its HTML route without startup jobs."""
from unittest.mock import patch
import socket

_connect = socket.socket.connect

def local_only(sock, address):
    # Windows asyncio uses a loopback socket pair even for in-process ASGI.
    if isinstance(address, tuple) and address[0] in ('127.0.0.1', '::1'):
        return _connect(sock, address)
    raise RuntimeError('External network disabled during smoke test')

if __name__ == '__main__':
    with patch('socket.socket.connect', new=local_only):
        from fastapi.testclient import TestClient
        import main
        client=TestClient(main.app)
        response=client.get('/')
        assert response.status_code==200
        assert 'NSE Stock Signal Dashboard' in response.text
        client.close()
    print('PASS: dashboard imports and HTML route returns 200; startup scan/alerts not invoked.')
