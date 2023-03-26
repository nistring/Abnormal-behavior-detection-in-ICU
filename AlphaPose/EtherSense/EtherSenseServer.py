#!/usr/bin/python
import pyrealsense2 as rs
import sys
import asyncio
import numpy as np
from multiprocessing import Process, Queue
import socket
import zmq
import zmq.asyncio
import signal
import struct
import time
import cv2
import time
import os

mc_ip_address = "224.0.0.1"
port = 1024
chunk_size = 4096
depth_scale = 0.0010000000474974513
max_distance = 3.0  # m
min_distance = 0.3  # m
FPS = 15
width = 640
height = 480


def create_pipeline():
    ctx = rs.context()
    devices = ctx.query_devices()
    pipelines = []

    for device_id in range(devices.size()):
        device = devices[device_id]
        detected_camera = device.get_info(rs.camera_info.serial_number)
        camera_name = device.get_info(rs.camera_info.name)
        del device

        print(f"Detected {camera_name}, Serial: {detected_camera}")

        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(detected_camera)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, FPS)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, FPS)
        pipe.start(cfg)

        pipelines.append(pipe)

    return pipelines


def get_camera_data(pipeline, image_filters, align):
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    color = aligned_frames.get_color_frame()
    depth = aligned_frames.get_depth_frame()

    if depth and color:
        for filter in image_filters:
            depth = filter.process(depth)

        color_mat = np.asanyarray(color.as_frame().get_data())
        depth_mat = np.asanyarray(depth.as_frame().get_data())
        ts = time.time()

        return ts, color_mat, depth_mat
    else:
        return None, None, None

def encoder(in_queues, out_queues, loop):
    while True:
        for in_q, out_q in zip(in_queues, out_queues):
            (ts, color, depth) = in_q.get()

            ts = struct.pack('<d', ts)
            color = cv2.imencode('.jpg', color)[1]
            color = np.array(color).tobytes()
            depth = cv2.imencode('.jpg', depth)[1]
            depth = np.array(depth).tobytes()
            out_q.put((ts, color, depth))


async def stream_data(pipeline, image_filters, align, zmq_socket, in_queue, out_queue):
    while True:
        in_queue.put(get_camera_data(pipeline, image_filters, align))
        ts, color_data, depth_data = out_queue.get()

        await zmq_socket.send_multipart([b"TS", ts])
        await zmq_socket.send_multipart([b"RGB", color_data])
        await zmq_socket.send_multipart([b"DEPTH", depth_data])

        await asyncio.sleep(0)


class MulticastServerProtocol:
    def __init__(self, loop):

        print("Launching Realsense Camera Server")
        try:
            self.pipelines = create_pipeline()
        except:
            print("Unexpected error: ", sys.exc_info()[1])
            sys.exit(1)

        self.num_cameras = len(self.pipelines)

        # Post processing filters
        self.spatial_filter = rs.spatial_filter()
        self.temporal_filter = rs.temporal_filter()
        self.depth_to_disparity = rs.disparity_transform(True)
        self.disparity_to_depth = rs.disparity_transform(False)
        self.color_filter = rs.colorizer()

        # Filter options
        self.spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.6)
        self.spatial_filter.set_option(rs.option.filter_smooth_delta, 8)
        self.temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
        self.color_filter.set_option(rs.option.max_distance, max_distance)
        self.color_filter.set_option(rs.option.min_distance, min_distance)
        self.color_filter.set_option(rs.option.histogram_equalization_enabled, 0)
        self.color_filter.set_option(rs.option.color_scheme, 9)

        self.image_filters = [
            #self.spatial_filter,
            # self.temporal_filter,
            self.color_filter,
        ]

        self.frame_data = ""

        align_to = rs.stream.color
        self.align = rs.align(align_to)

        self.stream_task = []
        in_queues = []
        out_queues = []
        ctx = zmq.asyncio.Context()
        for i, pipe in enumerate(self.pipelines):
            zmq_socket = ctx.socket(zmq.PUB)
            zmq_socket.bind("tcp://*:%d" % (port+i))
            in_queue = Queue(FPS//2)
            out_queue = Queue(FPS//2)
            in_queues.append(in_queue)
            out_queues.append(out_queue)
            self.stream_task.append(asyncio.ensure_future(stream_data(pipe, self.image_filters, self.align, zmq_socket, in_queue, out_queue)))

        self.p = Process(target=encoder, args=(in_queues, out_queues, loop))
        self.p.start()

    def connection_made(self, transport):
        self.transport = transport

    def connection_lost(self, exc):
        self.p.terminate()
        self.p.close()
        for task in self.stream_task:
            task.cancel()
        for pipe in self.pipelines:
            pipe.stop()

    def datagram_received(self, data, addr):
        self.transport.sendto(str(len(self.pipelines)).encode(), addr)


            

def main(argv):
    loop = asyncio.get_event_loop()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    server_address = ("", port)
    sock.bind(server_address)

    num_cameras = rs.context().query_devices().size()

    connect = loop.create_datagram_endpoint(lambda: MulticastServerProtocol(loop), sock=sock)

    transport, protocol = loop.run_until_complete(connect)
    # async def shutdown_handler():
    #     print(1)
    #     loop.stop()

    # loop.add_signal_handler(signal.SIGINT, lambda:asyncio.create_task(shutdown_handler()))
    # async def shutdown():
    #     while True:
    #         n = rs.context().query_devices().size()
    #         if num_cameras != n:
    #             return
    #         await asyncio.sleep(0)

    # loop.run_until_complete(shutdown())
    loop.run_forever()
    # transport.close()
    # loop.run_until_complete(loop.shutdown_asyncgens())
    # loop.stop()
    loop.close()

    # try:
    #     loop.run_forever()
        
    # finally:
    #     transport.close()
    #     loop.run_until_complete(loop.shutdown_asyncgens())
    #     loop.close()


if __name__ == "__main__":
    main(sys.argv[1:])
