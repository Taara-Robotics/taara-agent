import time
import struct
import cv2
import gzip
from matplotlib import pyplot as plt
import numpy as np
import multiprocessing as mp
import argparse
from math import floor
from scipy.spatial.transform import Rotation
from queue import Empty

from .recording_writer import RecordingWriter
from .introduction_message import IntroductionMessage
from .frame_message import FrameMessage
from .fps_counter import FpsCounter
from .visual_odometry import VisualOdometry
from .rgbd_odometry import RgbdOdometry

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

parser.add_argument(
    "--camera",
    default="orbbec",
    metavar="camera_type",
    type=str,
    help="camera type to record from",
)

args = parser.parse_args()

if not "name" in args:
    args.name = str(round(time.time()))

if not "desc" in args:
    args.desc = ""

# Initialize camera
if args.camera == "orbbec":
    from .orbbec_camera import OrbbecCamera
    camera = OrbbecCamera()

camera.start()

# payload process
def run_payload_process():
    global camera, running

    while running:
        try:
            # Get data
            data = data_queue.get_nowait()
            frame_time, translation, eulers, color_data, depth_data = data
            
            # construct frame message payload
            payload = struct.pack("<I", frame_time)
            payload += struct.pack("<fff", *translation)
            payload += struct.pack("<fff", *eulers)

            # convert color to jpeg
            if camera.color_format == "JPG":
                color_jpeg = color_data.tobytes()
            else:
                color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
            
            color_jpeg = cv2.imencode(".jpg", color_data, [int(cv2.IMWRITE_JPEG_QUALITY), 90])[1].tobytes()

            payload += struct.pack("<I", len(color_jpeg))
            payload += color_jpeg

            # compress depth using gzip
            depth_gzip = gzip.compress(depth_data.tobytes(), 0)
            payload += struct.pack("<I", len(depth_gzip))
            payload += depth_gzip

            # Push payload to queue
            payload_queue.put(payload)
        except Empty:
            time.sleep(0.01)

    print("Stopped payload process")

def run_recording_process():
    global camera, running

    connection_id = 1
    recorder = RecordingWriter(f"recordings/{recording_name}.trr", False, args.desc)

    camera_matrix, distortion_coefficients = camera.get_intrinsics()
    color_shape = camera.get_color_shape()
    depth_shape = camera.get_depth_shape()
    depth_range = camera.get_depth_range()

    # Write introduction message
    payload = struct.pack("<Q", floor(time.time()*1000))
    payload += struct.pack("<fff", *camera_position)
    payload += struct.pack("<fff", *camera_euler_angles_deg)
    payload += struct.pack("<HH", color_shape[1], color_shape[0])
    payload += struct.pack("<HH", depth_shape[1], depth_shape[0])
    payload += struct.pack("<ff", *depth_range)
    payload += struct.pack("<ffff", camera_matrix[0,0], camera_matrix[1,1], camera_matrix[0,2], camera_matrix[1,2])
    payload += struct.pack("<fffff", *distortion_coefficients)

    introduction_message = IntroductionMessage(payload)
    recorder.write(connection_id, introduction_message)

    # Write frame messages
    while running:
        try:
            payload = payload_queue.get_nowait()
            frame_message = FrameMessage(payload)
            recorder.write(connection_id, frame_message)
            print("WROTE", len(payload))
        except Empty:
            time.sleep(0.01)
    
    recorder.close()
    print("Stopped recording process")

# Camera extrinsics
camera_position = (0, 0, 0)
camera_euler_angles_deg = (0, 0, 0)

# Generate TRR
start_time = time.time()
recording_name = f'orbbec_{args.name}'

# Write frame messages
i = 0
fps_counter = FpsCounter()

running = True

data_queue = mp.Queue(10)
payload_queue = mp.Queue(10)

payload_process = mp.Process(target=run_payload_process)
payload_process.start()

recording_process = mp.Process(target=run_recording_process)
recording_process.start()

camera_matrix, distortion_coefficients = camera.get_intrinsics()
# vo = VisualOdometry(camera_matrix, distortion_coefficients, camera.get_depth_scale(), camera.get_depth_range()[0], camera.get_depth_range()[1])
vo = RgbdOdometry(256, 192, camera_matrix / camera.get_color_shape()[1] * 256, camera.get_depth_scale(), camera.get_depth_range()[0], camera.get_depth_range()[1])
prev_pose = None
trajectory = []

while running:
    try:
        fps_counter.start_frame()

        # Wait for frames
        s, color_data, depth_data = camera.wait_for_frames()

        if not s:
            continue
        
        # Update frame time
        frame_time = floor((time.time() - start_time)*1000)

        if camera.color_format == "JPG":
            color = cv2.imdecode(color_data, cv2.IMREAD_COLOR)
        else:
            color = color_data

        cv2.imshow("color", color)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_data, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow("depth", depth_colormap)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # VO pose
        # pose = vo.process_frame(color, depth_data)
        pose = vo.process_frame(cv2.resize(color, (256, 128)), cv2.resize(depth_data, (256, 128)))

        # if prev_pose is not None:
        #     odom = np.linalg.inv(prev_pose) @ pose
            
        #     if np.linalg.norm(odom[:3, 3]) < 0.05 and np.linalg.norm(np.arccos((np.trace(odom[:3, :3]) - 1) / 2)) < 0.05:
        #         continue

        # trajectory.append(pose[:3, 3])
        # prev_pose = pose

        # Push data to queue
        translation = pose[:3, 3]
        eulers = Rotation.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True)
        data_queue.put((frame_time, translation, eulers, color, depth_data))

        # Update fps counter
        i += 1
        fps_counter.end_frame()
        
        # print fps status
        print(f"[{i}] {fps_counter}")
    except KeyboardInterrupt:
        break

# # Plot trajectory
# trajectory = np.array(trajectory)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# plt.show()


# Stop camera
running = False
camera.stop()

# Close
payload_process.join()
recording_process.join()

# Close windows
cv2.destroyAllWindows()
