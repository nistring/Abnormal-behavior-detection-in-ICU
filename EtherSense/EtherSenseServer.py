#!/usr/bin/python
import pyrealsense2 as rs
import sys
import asyncore
import numpy as np
import pickle
import socket
import struct
import cv2
import os
import datetime

root = os.path.abspath(os.path.dirname(__file__))

mc_ip_address = '224.0.0.1'
port = 1024
chunk_size = 4096
depth_scale = 0.0010000000474974513
max_distance = 3.0 # m
min_distance = 0.3 # m
FPS = 15
width = 640
height = 480
		
class EtherSenseServer(asyncore.dispatcher):
    def __init__(self, address):
        asyncore.dispatcher.__init__(self)
        print("Launching Realsense Camera Server")
        try:
            self._open_pipeline()
        except:
            print("Unexpected error: ", sys.exc_info()[1])
            sys.exit(1)
        self.create_socket(socket.AF_INET, socket.SOCK_STREAM)
        client_address = (address[0], port)
        print('sending acknowledgement to', client_address)

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

        self.frame_data = ''
        self.connect(client_address)
        self.packet_id = 0        

    def handle_connect(self):
        print("connection received")

    def writable(self):
        return True

    def update_frame(self):
        depth, depth_map = self._get_depth()

        if depth is not None:
        # convert the depth image to a string for broadcast
            data = pickle.dumps(depth)
        # capture the lenght of the data portion of the message	
            length = struct.pack('<I', len(data))
        # for the message for transmission
            self.frame_data = b''.join([length, data])

        if depth_map is not None:
            self.depth_out.write(depth_map)

        # Save video every hours
        today = datetime.datetime.now()
        if self.hour != today.hour:
            self.depth_out.release()
            self._open_video_writer()


    def handle_write(self):
        # first time the handle_write is called
        if not hasattr(self, 'frame_data'):
            self.update_frame()
        # the frame has been sent in it entirety so get the latest frame
        if len(self.frame_data) == 0:
            self.update_frame()
        else:
	    # send the remainder of the frame_data until there is no data remaining for transmition
            remaining_size = self.send(self.frame_data)
            self.frame_data = self.frame_data[remaining_size:]
	

    def handle_close(self):
        self.pipeline.stop()
        self.close()

    def _open_pipeline(self):
        self.cfg = rs.config()
        self.cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, FPS)
        self.cfg.enable_stream(rs.stream.depth, width, height, rs.format.bgr8, FPS)
        self.pipeline = rs.pipeline()
        self.pipeline_profile = self.pipeline.start(self.cfg)

        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def _get_depth(self):
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        # take owner ship of the frame for further processing
        frames.keep()
        depth = aligned_frames.get_depth_frame()
        # color = aligned_frames.get_color_frame()

        if depth:
            # Post-processing
            depth = self.depth_to_disparity.process(depth)
            depth = self.spatial_filter.process(depth)
            depth = self.temporal_filter.process(depth)
            depth = self.disparity_to_depth.process(depth)
            # original_depth = np.clip(np.asanyarray(depth.get_data()) * depth_scale, min_distance, max_distance)
            depth_map = self.color_filter.process(depth)
            
            depth = np.asanyarray(depth.get_data())
            depth_map = np.asanyarray(depth_map.get_data())

            return depth, depth_map
        else:
            return None, None

    def _open_video_writer(self):
        self.today = datetime.datetime.now()
        self.hour = self.today.hour
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.depth_out = cv2.VideoWriter(os.path.join(root, self.today.strftime("%d-%m-%Y-%H-%M-%S")+'_depth.mp4'), fourcc, FPS, (width, height))

            

class MulticastServer(asyncore.dispatcher):
    def __init__(self):
        asyncore.dispatcher.__init__(self)
        # Listen only to a multicast group.
        server_address = ('', port)
        self.create_socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.bind(server_address)
        # Join a multicast group on a local interface.
        mreq = struct.pack("4sl", socket.inet_aton(mc_ip_address), socket.INADDR_ANY)
        self.socket.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

    def handle_read(self):
        data, addr = self.socket.recvfrom(42)
        print('Recived Multicast message %s bytes from %s' % (data, addr))
	# Once the server recives the multicast signal, open the frame server
        server = EtherSenseServer(addr)
        print(sys.stderr, data)

    def writable(self): 
        return False # don't want write notifies

    def handle_close(self):
        self.close()


def main():
    # initalise the multicast receiver 
    server = MulticastServer()
    # hand over excicution flow to asyncore
    asyncore.loop()
   
if __name__ == '__main__':
    main()