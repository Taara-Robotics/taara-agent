import time
import struct
import pyzed.sl as sl
import cv2
import gzip
import numpy as np
import multiprocessing as mp
import argparse
from math import floor
from queue import Queue, Full, Empty
from scipy.spatial.transform import Rotation
from multiprocessing.shared_memory import SharedMemory

from .agent_message import AgentMessage
from .recording_writer import RecordingWriter
from .introduction_message import IntroductionMessage
from .frame_message import FrameMessage
from .fps_counter import FpsCounter

# Parse args
parser = argparse.ArgumentParser(
    description="write Taara recording to .trr file"
)

parser.add_argument(
    "--desc",
    default="",
    metavar="recording_description",
    type=str,
    help="recordings description in snake_case that will be added to the file metadata",
)

parser.add_argument(
    "--name",
    default=argparse.SUPPRESS,
    metavar="output_name",
    type=str,
    help="recordings output name. If not specified then will be generated automatically",
)

args = parser.parse_args()

if not "name" in args:
    args.name = str(round(time.time()))

if not "desc" in args:
    args.desc = ""

# Create a ZED camera object
zed = sl.Camera()

# Set configuration parameters
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.SVGA # SVGA/HD1200/HD1080
init_params.camera_fps = 15
init_params.depth_mode = sl.DEPTH_MODE.NEURAL
init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
init_params.coordinate_units = sl.UNIT.MILLIMETER

# Open the camera
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    print("Failed to open camera")
    exit(1)

# Enable positional tracking with default parameters
tracking_parameters = sl.PositionalTrackingParameters()
err = zed.enable_positional_tracking(tracking_parameters)
if err != sl.ERROR_CODE.SUCCESS:
    print("Failed to enable positional tracking")
    exit(1)

# Create RuntimeParameters after opening the camera
runtime_parameters = sl.RuntimeParameters()

# Get camera extrinsics
camera_position = (0, 0, 0)
camera_euler_angles_deg = (0, 0, 0)

# Get camera intrinsics
camera_configuration = zed.get_camera_information().camera_configuration
calibration_params = camera_configuration.calibration_parameters

width, height = camera_configuration.resolution.width, camera_configuration.resolution.height #, camera_configuration.fps
fx, fy, cx, cy = calibration_params.left_cam.fx, calibration_params.left_cam.fy, calibration_params.left_cam.cx, calibration_params.left_cam.cy
k1, k2, p1, p2, k3 = calibration_params.left_cam.disto[:5]

# Depth range
depth_range_min, depth_range_max = (0.3, 10)

# Generate TRR
start_time = time.time()
recording_name = f'zed_{args.name}'

# Write frame messages
i = 0
fps_counter = FpsCounter()

zed_pose = sl.Pose()
zed_translation = sl.Translation()
zed_orientation = sl.Orientation()
sensors_data = sl.SensorsData()
color_mat = sl.Mat()
depth_mat = sl.Mat()

# payload process
def run_payload_process():
    while True:
        try:
            # Get data
            data = data_queue.get()
            frame_time, translation, eulers, color_data, depth_data = data
            
            # construct frame message payload
            payload = struct.pack("<I", frame_time)
            payload += struct.pack("<fff", *translation)
            payload += struct.pack("<fff", *eulers)
            
            # convert color to jpeg
            color_jpeg = cv2.imencode(".jpg", color_data, [int(cv2.IMWRITE_JPEG_QUALITY), 90])[1].tobytes()
            payload += struct.pack("<I", len(color_jpeg))
            payload += color_jpeg

            # compress depth using gzip
            depth_uint16 = np.array(depth_data, dtype=np.uint16)
            depth_gzip = gzip.compress(depth_uint16, 9)
            payload += struct.pack("<I", len(depth_gzip))
            payload += depth_gzip

            # Push payload to queue
            payload_queue.put(payload)
        except KeyboardInterrupt:
            break

    print("Stopped payload process")

def run_recording_process():
    connection_id = 1
    recorder = RecordingWriter(f"recordings/{recording_name}.trr", False, args.desc)

    # Write introduction message
    payload = struct.pack("<Q", floor(time.time()*1000))
    payload += struct.pack("<fff", *camera_position)
    payload += struct.pack("<fff", *camera_euler_angles_deg)
    payload += struct.pack("<HH", width, height) # color resolution
    payload += struct.pack("<HH", width, height) # depth resolution
    payload += struct.pack("<ff", depth_range_min, depth_range_max)
    payload += struct.pack("<ffff", fx, fy, cx, cy)
    payload += struct.pack("<fffff", k1, k2, p1, p2, k3)

    introduction_message = IntroductionMessage(payload)
    recorder.write(connection_id, introduction_message)

    # Write frame messages
    while True:
        try:
            payload = payload_queue.get()
            frame_message = FrameMessage(payload)
            recorder.write(connection_id, frame_message)
            print("WROTE", len(payload))
        except KeyboardInterrupt:
            break
    
    recorder.close()
    print("Stopped recording process")

data_queue = mp.Queue(1)
payload_queue = mp.Queue(1)

payload_process = mp.Process(target=run_payload_process)
payload_process.start()

recording_process = mp.Process(target=run_recording_process)
recording_process.start()

while True:
    try:
        fps_counter.start_frame()

        # A new image is available if grab() returns SUCCESS
        if zed.grab(runtime_parameters) != sl.ERROR_CODE.SUCCESS:
            print("Failed to grab frame")
            break
        
        # Update frame time
        frame_time = floor((time.time() - start_time)*1000)

        # Get IMU pose
        zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE)
        imu_pose = sensors_data.get_imu_data().get_pose()
        
        # Get the pose of the left eye of the camera with reference to the world frame
        # zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD)

        # Convert translation from mm to meters
        translation = imu_pose.get_translation().get() / 1000
        # translation = zed_pose.get_translation(zed_translation).get() / 1000
        
        # Display the orientation quaternion
        # quaternion = zed_pose.get_orientation(zed_orientation).get()
        quaternion = imu_pose.get_orientation().get()
        eulers = Rotation.from_quat(quaternion).as_euler("xyz", degrees=True)
        print(eulers, translation)
        
        # Retrieve left color image
        zed.retrieve_image(color_mat, sl.VIEW.LEFT)
        
        # Retrieve depth map - depth is aligned on the left image
        zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)

        # Get datas
        color_data = color_mat.get_data()
        depth_data = depth_mat.get_data()

        # Push data to queue
        data_queue.put((frame_time, translation, eulers, color_data, depth_data))

        i += 1
        fps_counter.end_frame()
        
        # print fps status
        print(f"[{i}] {fps_counter}")
    except KeyboardInterrupt:
        break

print("Stopped camera process")
zed.close()

# Close
payload_process.join()
recording_process.join()
