import pyrealsense2 as rs
import numpy as np
import cv2 as cv
import os
import timeit
import sys
import os
import shutil
import datetime

root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(root)
depth_scale = 0.0010000000474974513
max_distance = 3.0 # m
min_distance = 0.3 # m
FPS = 15
width = 640
height = 480

# Post processing filters
spatial_filter = rs.spatial_filter()
temporal_filter = rs.temporal_filter()
depth_to_disparity = rs.disparity_transform(True)
disparity_to_depth = rs.disparity_transform(False)
color_filter = rs.colorizer()

# Filter options
spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.6)
spatial_filter.set_option(rs.option.filter_smooth_delta, 8)
temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
color_filter.set_option(rs.option.max_distance, max_distance)
color_filter.set_option(rs.option.min_distance, min_distance)
color_filter.set_option(rs.option.histogram_equalization_enabled, 0)
color_filter.set_option(rs.option.color_scheme, 9)

# training data
#tossturn900-1300
angle_list = ['15', '30', '45']
action_list = ['normal', 'tube', 'escape', 'tossturn', 'seizure']
state = 0

def create_folder():

    name = input('Name of the participant : ')
    directory = os.path.join(root, name)
    try:
        os.makedirs(directory)
    except:
        pass
    
    print(angle_list)
    while True:
        angle = input('Choose one of the angles : ')
        if angle in angle_list:
            break
    directory = os.path.join(directory, angle)
    try:
        os.makedirs(directory)
    except:
        pass

    for action in action_list:
        action_dir = os.path.join(directory, action)
        try:
            if os.path.exists(action_dir):
                while True:
                    overwrite = input(f'You already have files in folder\nContinue? [y/n]')
                    if overwrite in ('y', 'n'):
                        break
                if overwrite == 'y':
                    shutil.rmtree(directory)
                else:
                    sys.exit()
            for n in ('depth_data', 'depth_img', 'color_img', 'res', 'labels'):
                os.makedirs(os.path.join(action_dir, n))
        except OSError:
            print('Error: Creating directory. ' + directory)
    
    return name, action, directory
            

def training_video():

    name, action, angle_dir = create_folder()
    directory = os.path.join(angle_dir, action_list[state])

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, FPS)

    profile = pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)

    try:
        save = False
        idx = 0
        cv.namedWindow('depth_frame', cv.WINDOW_NORMAL)
        cv.resizeWindow('depth_frame', width=width, height=height)
        cv.namedWindow('color_frame', cv.WINDOW_NORMAL)
        cv.resizeWindow('color_frame', width=width, height=height)
        
        while True:

            start_t = timeit.default_timer()

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not depth_frame or not color_frame:
                continue
            
            # Post-processing
            depth_frame = depth_to_disparity.process(depth_frame)
            depth_frame = spatial_filter.process(depth_frame)
            depth_frame = temporal_filter.process(depth_frame)
            depth_frame = disparity_to_depth.process(depth_frame)
            
            depth_data = np.asanyarray(depth_frame.get_data()) * depth_scale
            depth_data = np.clip(depth_data, min_distance, max_distance).astype(np.float32)
            depth_map = ((depth_data-min_distance) * (255 / (max_distance-min_distance)))
            depth_map = np.clip(depth_map, 0, 255).astype(np.uint8)
            depth_map = np.stack([depth_map, depth_map, depth_map], axis=-1)
            color_map = np.asanyarray(color_frame.get_data())

            if save is True:
                np.save(os.path.join(directory, 'depth_data', str(idx) + '.npy'), depth_data)
                cv.imwrite(os.path.join(directory, 'depth_img', str(idx) + '.jpg'), depth_map)
                cv.imwrite(os.path.join(directory, 'color_img', str(idx) + '.jpg'), color_map)
                cv.circle(color_map, (17, 69), 6, (0,0,255), -1)
                cv.putText(color_map, 'REC', (30, 75), cv.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(0,0,255), thickness=2)
                idx += 1

            k = cv.waitKey(1)
            # Press q to quit
            if k == ord('q'):
                cv.destroyAllWindows()
                break
            # Press r to save the frames
            if k == ord('r'):
                if save is False:
                    save = True
                else:
                    save = False
            if k in list(map(ord, ('1','2','3','4','5'))):
                state = k - ord('1')
                directory = os.path.join(angle_dir, action_list[state])
            
            cv.putText(color_map, f'Name : {name}', (480, 25), cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,128,255), thickness=2)
            cv.putText(color_map, f'Action : {action_list[state]}', (480, 50), cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,128,255), thickness=2)

            terminate_t = timeit.default_timer()
            cv.putText(color_map, 'FPS : ' + str(int(1./(terminate_t - start_t))), (15, 25), cv.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(0,255,128), thickness=2)
            cv.putText(color_map, f'Index : {idx}', (15, 50), cv.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(0,255,128), thickness=2)
            cv.imshow('depth_frame', depth_map)
            cv.imshow('color_frame', color_map)
            

    finally:
        pipeline.stop()

def test_video():
    today = datetime.datetime.now()
    hour = today.hour

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, FPS)

    profile = pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)


    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    depth_out = cv.VideoWriter(os.path.join(root, today.strftime("%d-%m-%Y-%H-%M-%S")+'_depth.mp4'), fourcc, FPS, (width, height))
    color_out = cv.VideoWriter(os.path.join(root, today.strftime("%d-%m-%Y-%H-%M-%S")+'_color.mp4'), fourcc, FPS, (width, height))

    try:
        cv.namedWindow('depth_frame', cv.WINDOW_NORMAL)
        cv.resizeWindow('depth_frame', width=width, height=height)
        cv.namedWindow('color_frame', cv.WINDOW_NORMAL)
        cv.resizeWindow('color_frame', width=width, height=height)
        
        while True:

            start_t = timeit.default_timer()

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not depth_frame or not color_frame:
                continue
            
            # Post-processing
            depth_frame = depth_to_disparity.process(depth_frame)
            depth_frame = spatial_filter.process(depth_frame)
            depth_frame = temporal_filter.process(depth_frame)
            depth_frame = disparity_to_depth.process(depth_frame)
            # original_depth = np.clip(np.asanyarray(depth_frame.get_data()) * depth_scale, min_distance, max_distance)
            depth_frame = color_filter.process(depth_frame)
            
            depth_map = np.asanyarray(depth_frame.get_data())
            color_map = np.asanyarray(color_frame.get_data())
            # retrieved_depth = RGB2D(depth_map)
            # print(np.mean(np.abs(original_depth - retrieved_depth)))

            k = cv.waitKey(1)
            # Press q to quit
            if k == ord('q'):
                cv.destroyAllWindows()
                break

            depth_out.write(depth_map)
            color_out.write(color_map)
            today = datetime.datetime.now()
            if hour != today.hour:
                print('save')
                depth_out.release()
                color_out.release()
                hour = today.hour
                depth_out = cv.VideoWriter(os.path.join(root, today.strftime("%d-%m-%Y-%H-%M-%S")+'_depth.mp4'), fourcc, FPS, (width, height))
                color_out = cv.VideoWriter(os.path.join(root, today.strftime("%d-%m-%Y-%H-%M-%S")+'_color.mp4'), fourcc, FPS, (width, height))
            
            terminate_t = timeit.default_timer()
            cv.putText(depth_map, 'FPS : ' + str(int(1./(terminate_t - start_t))), (15, 25), cv.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(0,255,128), thickness=2)
            cv.imshow('depth_frame', depth_map)
            cv.imshow('color_frame', color_map)
            
    finally:
        pipeline.stop()

def RGB2D(depth_map):
    b = depth_map[:,:,2].astype(np.float16)
    g = depth_map[:,:,1].astype(np.float16)
    r = depth_map[:,:,0].astype(np.float16)
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

def D2RGB(depth_map):
    # Convert depth map to float type
    d = depth_map.astype(np.float)

    # Initialize empty RGB image
    rgb = np.zeros((d.shape[0], d.shape[1], 3), dtype=np.uint8)

    # Compute RGB channels from depth map
    r = np.zeros_like(d)
    g = np.zeros_like(d)
    b = np.zeros_like(d)

    d_normal = 1529.0 * (d - min_distance) / (max_distance - min_distance)
    d_normal = np.rint(d_normal)

    r[np.logical_or(d_normal < 255, 1275 <= d_normal)] = 255

    condition = np.logical_and(255 <= d_normal, d_normal < 510)
    r[condition] = (509 - d_normal)[condition]

    condition = np.logical_and(1020 <= d_normal, d_normal < 1275)
    r[condition] = (d_normal - 1020)[condition]


    condition = d_normal < 255
    g[condition] = d_normal[condition]

    g[np.logical_and(255 <= d_normal, d_normal < 765)] = 255

    condition = np.logical_and(765 <= d_normal, d_normal < 1020)
    g[condition] = (1019 - d_normal)[condition]


    condition = np.logical_and(510 <= d_normal, d_normal < 765)
    b[condition] = (d_normal - 510)[condition]

    b[np.logical_and(765 <= d_normal, d_normal < 1275)] = 255

    condition = 1275 <= d_normal
    b[condition] = (1529 - d_normal)[condition]

    # Set RGB channels to RGB image
    rgb[:,:,0] = r
    rgb[:,:,1] = g
    rgb[:,:,2] = b

    return rgb


# unsigned short RGBtoD(unsigned char r, unsigned char g, unsigned char b)
# {
#     // conversion from RGB color to quantized depth value
#     if (b + g + r < 255)
#     {
#         return 0;
#     }
#     else if (r >= g && r >= b)
#     {
#         if (g >= b)
#         {   
#             return g - b;
#         }
#         else
#         {
#             return (g - b) + 1529;
#         }
#     }
#     else if (g >= r && g >= b)
#     {
#         return b - r + 510;
#     }
#     else if (b >= g && b >= r)
# {
#         return r - g + 1020;
#     }
# }

if __name__ == '__main__':
    test_video()