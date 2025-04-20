import socket
import cv2
import pickle
import struct

cap = cv2.VideoCapture(0)
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('172.25.0.1', 9999))
server_socket.listen(1)
conn, addr = server_socket.accept()

while True:
    ret, frame = cap.read()
    # print(frame)
    # cv2.imshow("Received in WSL", frame)
    # if cv2.waitKey(1) == ord('q'):
    #     break
    if not ret:
        break
    data = pickle.dumps(frame)
    message = struct.pack("Q", len(data)) + data
    conn.sendall(message)