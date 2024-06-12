import cv2
import numpy as np
import imufusion

from pyorbbecsdk import Pipeline, Config, FrameSet, OBSensorType, OBAlignMode, OBFormat, Frame
from threading import Lock

class OrbbecCamera:
    color_format = "JPG"
    depth_format = "Y16"

    def __init__(self):
        self._stopping = False

        self._pipeline = Pipeline()
        device = self._pipeline.get_device()
        device_info = device.get_device_info()
        device_pid = device_info.get_pid()
        self._config = Config()
        
        profile_list = self._pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        # self._color_profile = profile_list.get_default_video_stream_profile()
        # self._color_profile = profile_list.get_video_stream_profile(1920, 1080, OBFormat.MJPG, 30)
        self._color_profile = profile_list.get_video_stream_profile(1280, 720, OBFormat.MJPG, 30)
        self._config.enable_stream(self._color_profile)
        profile_list = self._pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        assert profile_list is not None
        # self._depth_profile = profile_list.get_default_video_stream_profile()
        self._depth_profile = profile_list.get_video_stream_profile(640, 576, OBFormat.Y16, 30)
        assert self._depth_profile is not None
        self._config.enable_stream(self._depth_profile)
        
        if device_pid == 0x066B:
            # Femto Mega does not support hardware D2C, and it is changed to software D2C
            self._config.set_align_mode(OBAlignMode.SW_MODE)
        else:
            self._config.set_align_mode(OBAlignMode.HW_MODE)

        self._pipeline.enable_frame_sync()

        # # IMU
        # sensor_list = device.get_sensor_list()
        # self._imu_lock = Lock()

        # self._gyro_sensor = sensor_list.get_sensor_by_type(OBSensorType.GYRO_SENSOR)
        # gyro_profile_list = self._gyro_sensor.get_stream_profile_list()
        # self._gyro_profile = gyro_profile_list.get_stream_profile_by_index(0)
        # assert self._gyro_profile is not None
        
        # self._accel_sensor = sensor_list.get_sensor_by_type(OBSensorType.ACCEL_SENSOR)
        # accel_profile_list = self._accel_sensor.get_stream_profile_list()
        # self._accel_profile = accel_profile_list.get_stream_profile_by_index(0)
        # assert self._accel_profile is not None

        # self._ahrs = imufusion.Ahrs()
        # self._imu_ts = None
        # self._gyro_value = None
        # self._accel_value = None
        # self._imu_q = np.zeros(4)
        # self._imu_v = np.zeros(3)
        # self._imu_p = np.zeros(3)
        # self._g = None

    def start(self):
        self._pipeline.start(self._config)
        # self._gyro_sensor.start(self._gyro_profile, self._gyro_callback)
        # self._accel_sensor.start(self._accel_profile, self._accel_callback)

    def get_intrinsics(self):
        camera_param = self._pipeline.get_camera_param()
        camera_matrix = np.array([
            [camera_param.rgb_intrinsic.fx, 0, camera_param.rgb_intrinsic.cx],
            [0, camera_param.rgb_intrinsic.fy, camera_param.rgb_intrinsic.cy],
            [0, 0, 1]
        ])
        distortion_coefficients = np.array([
            camera_param.rgb_distortion.k1,
            camera_param.rgb_distortion.k2,
            camera_param.rgb_distortion.p1,
            camera_param.rgb_distortion.p2,
            camera_param.rgb_distortion.k3
        ])

        return camera_matrix, distortion_coefficients
    
    def get_color_shape(self):
        return self._color_profile.get_height(), self._color_profile.get_width(), 3

    def get_depth_shape(self):
        # return self._depth_profile.get_height(), self._depth_profile.get_width()
        return self.get_color_shape()[:2]
    
    def get_depth_range(self):
        return 0.25, 3.86
    
    def get_depth_scale(self):
        return 1000.0
    
    def wait_for_frames(self, timeout_ms=100):
        frames = self._pipeline.wait_for_frames(timeout_ms)

        if frames is None:
            return False, None, None

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if color_frame is None or depth_frame is None:
            return False, None, None

        color_data = color_frame.get_data()#.reshape(self.get_color_shape())
        # print("COLOR", color_data.shape)

        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        depth_data = depth_data.reshape((
            self._color_profile.get_height(),
            self._color_profile.get_width()
        ))
        # depth_data = cv2.resize(depth_data, (self._depth_profile.get_width(), self._depth_profile.get_height()), interpolation=cv2.INTER_NEAREST)

        return True, color_data, depth_data
    
    def stop(self):
        self._stopping = True
        # self._gyro_sensor.stop()
        # self._accel_sensor.stop()
        self._pipeline.stop()

    def _gyro_callback(self, frame: Frame):
        if self._stopping or frame is None:
            return

        with self._imu_lock:
            gyro_frame = frame.as_gyro_frame()
            
            if gyro_frame is None:
                return
            
            ts = gyro_frame.get_timestamp()

            if self._imu_ts is not None and ts < self._imu_ts:
                return
            
            self._gyro_value = np.array((gyro_frame.get_x(), gyro_frame.get_y(), gyro_frame.get_z()))
        
        self._update_imu(ts)

    def _accel_callback(self, frame: Frame):
        if self._stopping or frame is None:
            return

        with self._imu_lock:
            accel_frame = frame.as_accel_frame()
            
            if accel_frame is None:
                return
            
            ts = accel_frame.get_timestamp()

            if self._imu_ts is not None and ts < self._imu_ts:
                return
            
            self._accel_value = np.array((accel_frame.get_x(), accel_frame.get_y(), accel_frame.get_z()))

        self._update_imu(ts)

    def _update_imu(self, ts: int):
        if self._imu_ts is None:
            self._imu_ts = ts
            return
        
        if self._gyro_value is None or self._accel_value is None:
            return
        
        dt = (ts - self._imu_ts) / 1000
        self._ahrs.update_no_magnetometer(self._gyro_value, self._accel_value, dt)
        # print("EULER", self._ahrs.quaternion.to_euler())

        # print("ACC", a)
        # print("IMU", self._gyro_value, self._accel_value, (ts - self._imu_ts) / 1000)

        # self._imu_q = np.array(
        #     self._ahrs.quaternion.x,
        #     self._ahrs.quaternion.y,
        #     self._ahrs.quaternion.z,
        #     self._ahrs.quaternion.w
        # )

        # a = self._ahrs.linear_acceleration + self._ahrs.earth_acceleration
        if self._g is None:
            self._g = self._accel_value
        else:
            a = self._accel_value - self._g
            self._imu_v += a * dt
            self._imu_p += self._imu_v * dt
            print(a, "V", self._imu_v, "P", self._imu_p)
        
        self._imu_ts = ts
        self._gyro_value = None
        self._accel_value = None
