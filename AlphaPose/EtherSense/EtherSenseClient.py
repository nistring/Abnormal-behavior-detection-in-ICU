#!/usr/bin/python
import signal
import asyncio
import numpy as np
import socket
import struct
import cv2
import zmq
import zmq.asyncio
import os
from multiprocessing import Process, Queue, Pool, Barrier
from datetime import datetime, date
import time
import pickle
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse

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

stop_flag = {}


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


def decode(data):
    ts = struct.unpack("<d", data["TS"][0:8])[0]
    rgb = cv2.imdecode(np.asarray(bytearray(data["RGB"])), cv2.IMREAD_COLOR)
    depth = cv2.imdecode(np.asarray(bytearray(data["DEPTH"])), cv2.IMREAD_COLOR)
    return (ts, rgb, depth)


def video_writer(addr, queue):

    p = None
    while True:
        path = os.path.join(save_root_addr, addr)
        rgb_path = os.path.join(path, "rgb")
        depth_path = os.path.join(path, "depth")
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(rgb_path):
            os.makedirs(rgb_path)
        if not os.path.exists(depth_path):
            os.makedirs(depth_path)

        hour = None
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        today = datetime.now()
        rgb_out = cv2.VideoWriter(os.path.join(rgb_path, today.strftime("%d-%m-%Y-%H-%M-%S") + "_rgb.mp4"), fourcc, FPS, (width, height))
        depth_out = cv2.VideoWriter(
            os.path.join(depth_path, today.strftime("%d-%m-%Y-%H-%M-%S") + "_depth.mp4"), fourcc, FPS, (width, height)
        )

        # Save video every hours
        while not stop_flag[addr]:
            (ts, rgb, depth) = queue.get()

            now = datetime.fromtimestamp(ts)
            rgb_out.write(rgb)
            depth_out.write(depth)
            if hour:
                if hour != now.hour:
                    break
            else:
                hour = now.hour

        if p:
            p.join()
        p = Process(target=release_video, args=(rgb_out, depth_out))
        p.start()

        while stop_flag[addr]:
            time.sleep(1000)


async def display_data(address, queue):

    fps = 0
    idx = 0
    last = 0
    while True:
        (ts, rgb, depth) = await queue.get()

        array = np.concatenate((rgb, depth), axis=1)

        if idx % FPS == 0:
            cur = ts
            if last:
                fps = 1 / (cur - last) * FPS
            last = cur

        cv2.putText(array, f"FPS : {fps:4.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow(address, array)
        key = cv2.waitKey(1)
        idx += 1
        queue.task_done()


async def send_data(dict_queues):
    queues = []
    for v in dict_queues.values():
        queues.extend(v)

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://127.0.0.1:5555")
    data = {}

    while True:
        for address, q in queues:
            data[address] = await q.get()["DEPTH"]
            q.task_done()
        socket.recv()
        socket.send(pickle.dumps(data))


async def receive_from_zmq(zmq_socket, address, queue, async_queue):
    received_data = {}
    while True:
        while stop_flag[address]:
            await asyncio.sleep(1)
        try:
            for i in range(3):
                topic, data = await zmq_socket.recv_multipart()
                topic = topic.decode()
                received_data[topic] = data

            decoded = decode(received_data)
            queue.put(decoded)
            await async_queue.put(decoded)
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


def stop(*args):
    on, address = args
    stop_flag[address] = not on


class DiscoveryClientProtocol():
    def __init__(self, loop):
        self.loop = loop
        self.transport = None
        self.ctx = None
        self.addr_dict = {}
        self.receive_task = {}
        self.display_task = {}
        self.processes = {}
        self.queues = {}
        self.async_queues = {}

    def connection_made(self, transport):
        self.transport = transport
        sock = self.transport.get_extra_info("socket")
        sock.settimeout(0)
        ttl = struct.pack("b", 1)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)

        self.ping_task = asyncio.ensure_future(send_ping(self.transport, (mc_ip_address, port)))

    def datagram_received(self, data, addr):
        # print("Received {!r} from {}".format(data, addr))

        [addr, port] = addr
        if addr not in self.addr_dict or self.addr_dict[addr] != data:
            self.addr_dict[addr] = data
            self.reset(addr)
            for i in range(int(data.decode())):
                address = f"{addr}:{int(port)+i}"
                self.ctx = zmq.asyncio.Context()
                zmq_socket = self.ctx.socket(zmq.SUB)
                zmq_socket.connect(f"tcp://{address}")
                zmq_socket.subscribe(b"TS")
                zmq_socket.subscribe(b"RGB")
                zmq_socket.subscribe(b"DEPTH")
                queue = Queue(FPS // 2)
                async_queue = asyncio.Queue(FPS // 2)
                self.queues[addr].append(queue)
                self.async_queues[addr].append((address, async_queue))

                stop_flag[address] = False
                self.receive_task[addr].append(asyncio.ensure_future(receive_from_zmq(zmq_socket, address, queue, async_queue)))
                if args.gui:
                    cv2.namedWindow(address)
                    cv2.createButton(address, stop, address, cv2.QT_CHECKBOX, 1)
                    self.display_task[addr].append(
                        asyncio.ensure_future(
                            display_data(address, async_queue),
                        )
                    )

                p = Process(target=video_writer, args=(address, queue))
                p.start()
                self.processes[addr].append(p)

            if not args.gui:
                self.send_task = asyncio.ensure_future(send_data(self.async_queues))

    def error_received(self, exc):
        print("Error received:", exc)

    def connection_lost(self, exc):
        if self.ping_task:
            self.ping_task.cancel()
        for tasks in self.receive_task.values():
            for t in tasks:
                t.cancel()
        for tasks in self.display_task.values():
            for t in tasks:
                t.cancel()
        for processes in self.processes.values():
            for p in processes:
                p.terminate()
                p.close()
        if self.send_task:
            self.send_task.cancel()

    def reset(self, addr):
        try:
            cv2.destroyAllWindows()
            for task in self.receive_task[addr]:
                task.cancel()
            for task in self.display_task[addr]:
                task.cancel()
            for processes in self.processes[addr]:
                processes.terminate()
                processes.close()
            del self.queues[addr]
            del self.async_queues[addr]
            self.send_task.cancel()
        except:
            pass
        finally:
            self.receive_task[addr] = []
            self.display_task[addr] = []
            self.processes[addr] = []
            self.queues[addr] = []
            self.async_queues[addr] = []


def main():
    args.gui = False
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
    args.gui = True
    main()
