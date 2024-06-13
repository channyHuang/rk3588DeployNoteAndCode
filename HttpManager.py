import argparse
import cv2
import numpy as np
import time
import json
import requests
import http.server
import socketserver

from ImageProcess import imageProcess

class HttpManager(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        url = self.path
        enc = 'UTF-8'
        self.send_response(200)
        self.send_header('Content-type', 'text/html;charset=%s' % enc)
        self.end_headers()
        respond_json = {'responds': 0}
        self.wfile.write(json.dumps(respond_json))

    def do_POST(self):
        enc = 'UTF-8'
        url = str(self.path)
        content_length = int(self.headers['content-length'])
        lendata = self.rfile.read(2)
        len = int.from_bytes(lendata, byteorder='little')
        jdata = self.rfile.read(len)
        jsonData = json.loads(jdata)
		
        imagebody = self.rfile.read(content_length - 2 - len)
        imagenp = np.frombuffer(imagebody, np.uint8)
        img = cv2.imdecode(imagenp, cv2.IMREAD_COLOR)
        cv2.imshow('img', img)
        cv2.waitKey(1)

        if url == '/folder/fun':
            self.send_response(200)
            self.send_header('Content-type', 'text/html;charset=%s' % enc)
            self.end_headers()
            respond_json = {'responds': 0}
            self.wfile.write(json.dumps(respond_json))

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='--server is_server')
	parser.add_argument('--server', action='store_true', default=False, help = 'video or image')
	parser.add_argument('--client', action='store_true', default=False, help = 'video or image')
	args = parser.parse_args()

	if args.server:
		server_address = ('', 8080)
		httpd = socketserver.TCPServer(server_address, HttpManager)
		httpd.serve_forever()
	elif args.client:
		url = 'http://127.0.0.1:8080/folder/fun'
		img = cv2.imread('bus.jpg')
		while True:
			data = imageProcess.image2bytes(img)
			localtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
			sendData = {'time':localtime}
			jSendData = json.dumps(sendData).encode()
			head = int(len(jSendData)).to_bytes(length = 2, byteorder='little', signed=False)
			content = head + jSendData + data

			res = requests.post(url = url, data = content)
			print('send data', time.time())
			time.sleep(5)
	else:
		print('param error')