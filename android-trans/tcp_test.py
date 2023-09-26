import socket
import threading
import time


def receiveVideo(sock, addr):
    print("Accept new connection from %s:%s ..." % addr)
    sock.send(b"Welcome")
    while True:
        data = sock.recv(1024)
        time.sleep(1)
        if not data or data.decode('utf-8') == 'exit':
            break
        sock.send(('Hello, %s!' % data.decode('utf-8')).encode('utf-8'))
    sock.close()
    print('Connection from %s:%s closed.' % addr)
    

if __name__ == "__main__":
    address = ("0.0.0.0", 6010)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(address)
    s.listen(5)
    print(f"[*] Listening: {address}")
    while True:
        sock, addr = s.accept()
        t = threading.Thread(target=receiveVideo, args=(sock, addr))
        t.start()
        t.join()