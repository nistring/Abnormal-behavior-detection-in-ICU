#!/usr/bin/python
import pyrealsense2 as rs
import sys
import argparse
import asyncio
import numpy as np
import pickle
import socket
import zmq
import zmq.asyncio
import os
import signal


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(description="Ethersense client.")

args = parser.parse_args()

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

    return pipe


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
        ts = frames.get_timestamp()

        return color_mat, depth_mat, ts
    else:
        return None, None


async def stream_data(pipeline, image_filters, align, zmq_socket):
    while True:
        color, depth, ts = get_camera_data(pipeline, image_filters, align)

        color_data = pickle.dumps(color)
        depth_data = pickle.dumps(depth)

        await zmq_socket.send_multipart([b"TS", ts])
        await zmq_socket.send_multipart([b"RGB", color_data])
        await zmq_socket.send_multipart([b"DEPTH", depth_data])

        await asyncio.sleep(0)


class MulticastServerProtocol:
    def __init__(self, loop):

        print("Launching Realsense Camera Server")
        try:
            self.pipeline = create_pipeline()
        except:
            print("Unexpected error: ", sys.exc_info()[1])
            sys.exit(1)

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
            self.spatial_filter,
            self.temporal_filter,
            self.depth_to_disparity,
            self.disparity_to_depth,
            self.color_filter,
        ]

        self.frame_data = ""

        align_to = rs.stream.color
        self.align = rs.align(align_to)

        ctx = zmq.asyncio.Context()
        zmq_socket = ctx.socket(zmq.PUB)
        zmq_socket.bind("tcp://*:%d" % port)
        self.stream_task = asyncio.ensure_future(stream_data(self.pipeline, self.image_filters, self.align, zmq_socket, self.plugins))

    def connection_made(self, transport):
        self.transport = transport

    def connection_lost(self, exc):
        self.stream_task.cancel()

    def datagram_received(self, data, addr):
        self.transport.sendto(b"pong", addr)


def main(argv):
    loop = asyncio.get_event_loop()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    server_address = ("", port)
    sock.bind(server_address)

    connect = loop.create_datagram_endpoint(lambda: MulticastServerProtocol(loop), sock=sock)

    transport, protocol = loop.run_until_complete(connect)

    def shutdown_handler():
        loop.stop()

    loop.add_signal_handler(signal.SIGINT, shutdown_handler)

    try:
        loop.run_forever()
    finally:
        transport.close()
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


if __name__ == "__main__":
    main(sys.argv[1:])
