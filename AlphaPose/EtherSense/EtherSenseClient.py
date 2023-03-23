#!/usr/bin/python
import sys
import signal
import asyncio
import numpy as np
import pickle
import socket
import struct
import cv2
import argparse
import zmq
import zmq.asyncio
import os
from multiprocessing import Process, Queue, Pool, Barrier
from datetime import datetime, date

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(description="Ethersense client.")
parser.add_argument("--gui", action="store_true")

args = parser.parse_args()
args.gui = True

mc_ip_address = "224.0.0.1"
save_root_addr = "/media/nistring/8d91e3e1-365a-4ecc-af00-b66fcdfe3e21"
port = 1024
chunk_size = 4096
FPS = 15
width = 640
height = 480
max_distance = 3.0  # m
min_distance = 0.3  # m


def RGB2D(depth_map):
    b = depth_map[:, :, 2].astype(np.float16)
    g = depth_map[:, :, 1].astype(np.float16)
    r = depth_map[:, :, 0].astype(np.float16)
    d = np.zeros_like(b, dtype=np.float16)

    rr = np.logical_and(r >= g, r >= b)
    rg = np.logical_and(rr, g >= b)
    d[rg] = g[rg]

    gg = np.logical_and(g >= r, g >= b)
    d[gg] = (509 + b - r)[gg]

    bb = np.logical_and(b >= r, b >= g)
    d[bb] = (1019 + r - g)[bb]

    rb = np.logical_and(rr, b >= g)
    d[rb] = (1529 - b)[rb]

    d = min_distance + (max_distance - min_distance) * d / 1529.0

    return d


def D2RGB(d):

    # Compute RGB channels from depth map
    r = np.zeros_like(d)
    g = np.zeros_like(d)
    b = np.zeros_like(d)

    # Normalize depth
    d_normal = 1529.0 * (d - min_distance) / (max_distance - min_distance)
    d_normal = np.rint(d_normal)

    # D2R
    r[np.logical_or(d_normal < 255, 1275 <= d_normal)] = 255

    condition = np.logical_and(255 <= d_normal, d_normal < 510)
    r[condition] = (509 - d_normal)[condition]

    condition = np.logical_and(1020 <= d_normal, d_normal < 1275)
    r[condition] = (d_normal - 1020)[condition]

    # D2G
    condition = d_normal < 255
    g[condition] = d_normal[condition]

    g[np.logical_and(255 <= d_normal, d_normal < 765)] = 255

    condition = np.logical_and(765 <= d_normal, d_normal < 1020)
    g[condition] = (1019 - d_normal)[condition]

    # D2B
    condition = np.logical_and(510 <= d_normal, d_normal < 765)
    b[condition] = (d_normal - 510)[condition]

    b[np.logical_and(765 <= d_normal, d_normal < 1275)] = 255

    condition = 1275 <= d_normal
    b[condition] = (1529 - d_normal)[condition]

    # Set RGB channels to RGB image
    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)

    return rgb

def release_video(rgb_out, depth_out):
    rgb_out.release()
    depth_out.release()
        

def video_writer(addr, queue):

    p = None

    while True:
        hour = None
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        path = os.path.join(save_root_addr, addr)
        rgb_path = os.path.join(path, 'rgb')
        depth_path = os.path.join(path, 'depth')
        if not os.path.exists(path):
            os.makedirs(path)

        if not os.path.exists(rgb_path):
            os.makedirs(rgb_path)
        if not os.path.exists(depth_path):
            os.makedirs(depth_path)

        today = datetime.now()
        rgb_out = cv2.VideoWriter(os.path.join(rgb_path, today.strftime("%d-%m-%Y-%H-%M-%S") + "_rgb.mp4"), fourcc, FPS, (width, height))
        depth_out = cv2.VideoWriter(os.path.join(depth_path, today.strftime("%d-%m-%Y-%H-%M-%S") + "_depth.mp4"), fourcc, FPS, (width, height))

        # Save video every hours
        while True:
            received_data = queue.get()
            now = datetime.fromtimestamp(received_data["timestamp"])
            rgb_out.write(received_data["color_array"])
            depth_out.write(received_data["depth_array"])
            if hour:
                if hour != now.hour:
                    break
            else:
                hour = now.hour

        if p:
            p.join()
        p = Process(target=release_video, args=(rgb_out, depth_out))
        p.start()


async def receive_from_zmq(zmq_socket, queue, save_queue):
    received_data = {}
    while True:
        try:
            topic, data = await zmq_socket.recv_multipart()
            if topic == b"TS":
                if "timestamp" not in received_data:
                    received_data["timestamp"] = struct.unpack("<d", data[0:8])[0]
                    continue
            if topic == b"RGB":
                if "RGB" not in received_data:
                    received_data["color_array"] = pickle.loads(data)
                    continue
            if topic == b"DEPTH":
                if "DEPTH" not in received_data:
                    received_data["depth_array"] = pickle.loads(data)
                    continue

            queue.put_nowait(received_data)
            save_queue.put_nowait(received_data)
            received_data = {}

        except asyncio.CancelledError:
            print("cancelled")
            raise


async def send_ping(transport, address):
    while True:
        try:
            transport.sendto(b"ping", address)
            # print(f'Sent ping to {str(address)}')
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            raise


async def display_data(address, queue):
    while True:
        received_data = await queue.get()

        array = np.concatenate((received_data['color_array'], received_data['depth_array']), axis=1)
        
        if args.gui:
            cv2.imshow(address, array)
            key = cv2.waitKey(1)
            # if key == 27:
            #     break
        queue.task_done()


class DiscoveryClientProtocol:
    def __init__(self, loop):
        self.loop = loop
        self.transport = None
        self.ctx = None

        self.display_task = None

    def connection_made(self, transport):
        self.transport = transport
        sock = self.transport.get_extra_info("socket")
        sock.settimeout(0)
        ttl = struct.pack("b", 1)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)

        self.ping_task = asyncio.ensure_future(send_ping(self.transport, (mc_ip_address, port)))

    def datagram_received(self, data, addr):
        # print("Received {!r} from {}".format(data, addr))

        if self.ctx is None:
            self.ctx = zmq.asyncio.Context()
            zmq_socket = self.ctx.socket(zmq.SUB)
            address = f"{addr[0]}:{addr[1]}"
            zmq_socket.connect(f"tcp://{address}")

            zmq_socket.subscribe(b"TS")
            zmq_socket.subscribe(b"RGB")
            zmq_socket.subscribe(b"DEPTH")
            
            queue = asyncio.Queue()
            save_queue = Queue()
            self.receive_task = asyncio.ensure_future(receive_from_zmq(zmq_socket, queue, save_queue))
            if args.gui:
                self.display_task = asyncio.ensure_future(display_data(address, queue))
            p = Process(target=video_writer, args=(address, save_queue))
            p.start()

    def error_received(self, exc):
        print("Error received:", exc)

    def connection_lost(self, exc):
        if self.ping_task:
            self.ping_task.cancel()
        if self.receive_task:
            self.receive_task.cancel()
        if self.display_task:
            self.display_task.cancel()


def main():
    loop = asyncio.get_event_loop()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    connect = loop.create_datagram_endpoint(lambda: DiscoveryClientProtocol(loop), sock=sock)
    transport, protocol = loop.run_until_complete(connect)

    def signal_handler():
        loop.stop()

    loop.add_signal_handler(signal.SIGINT, signal_handler)

    try:
        loop.run_forever()
    finally:
        transport.close()
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


if __name__ == "__main__":
    main()
