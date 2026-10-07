import argparse
import socket
import threading
import webbrowser

import uvicorn


def main():
    parser = argparse.ArgumentParser(description='本地学习工作台')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-open', action='store_true', help='不自动打开浏览器')
    args = parser.parse_args()
    port = args.port
    while port < args.port + 20:
        with socket.socket() as probe:
            try:
                probe.bind(('127.0.0.1', port))
                break
            except OSError:
                port += 1
    else:
        raise SystemExit('没有可用端口，请用 --port 指定。')
    url = f'http://127.0.0.1:{port}'
    print(f'学习工作台：{url}  （Ctrl+C 停止）', flush=True)
    if not args.no_open:
        threading.Timer(1.3, webbrowser.open, args=(url,)).start()
    uvicorn.run('app.server:app', host='127.0.0.1', port=port, log_level='warning')


if __name__ == '__main__':
    main()
