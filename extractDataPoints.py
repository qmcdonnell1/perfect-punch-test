from collections import deque
import mediapipe as mp
import numpy as np
import threading
import time

import mediapipe as mp
mp_pose = mp.solutions.pose

'''
Args: 
    max_frame (int): maximum number of frames to keep (default 15)
    pose (mp.solutions.pose.Pose | None): Optional externally created MediaPipe Pose instance
'''
class PoseTracker:
    def __init__(self, max_frames=15, pose=None):
        # Accept external pose instance to avoid duplicate MediaPipe initialization
        self.pose = pose if pose is not None else mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self._owns_pose = pose is None  # Track if we created the pose (for cleanup)
        self.frame_buffer = deque(maxlen=max_frames)
        self.coord_buffer = deque(maxlen=max_frames)
        self.frame_index = 0
        self.running = False
        self._selected_landmarks = [
            mp_pose.PoseLandmark.RIGHT_ELBOW,
            mp_pose.PoseLandmark.RIGHT_WRIST,
            mp_pose.PoseLandmark.LEFT_ELBOW,
            mp_pose.PoseLandmark.LEFT_WRIST,
        ]
    
    '''
    Process a single video frame with MediaPipe Pose and record selected landmarks
    
    Parameters:
        frame (numpy.ndarray): HxWx3 image (RGB expected)
    
    Behavior:
        - Increments self.frame_index
        - Calls self.pose.process() on a non-writeable copy of frame
        - If landmarks are detected, extracts pixel (x, y) for RIGHT/LEFT ELBOW and WRIST,
          appends {"frame":<index>, "landmarks":{...}} to self.coord_buffer, and
          appends a copy of frame to self.frame_buffer
        - If no landmakrs are found, buffers are left unchanged
        
    Notes:
        - Not thread-safe; synchronize external access if used concurrently
        - Copies frames; keep max_frames reasonable to avoid high memory usage
    '''
    def process_frame(self, frame):
        self.frame_index += 1
        frame_rgb = np.copy(frame)
        frame_rgb.flags['WRITEABLE'] = False
        results = self.pose.process(frame_rgb)
        frame_rgb.flags['WRITEABLE'] = True

        landmarks_record = {"frame": self.frame_index, "landmarks": None}

        if results.pose_landmarks:
            h, w, _ = frame.shape
            coords = {}

            for landmark_id in self._selected_landmarks:
                lm = results.pose_landmarks.landmark[landmark_id]
                x, y = int(lm.x * w), int(lm.y * h)
                coords[landmark_id.name] = {"x": x, "y": y}

            landmarks_record["landmarks"] = coords

            self.coord_buffer.append(landmarks_record)
            self.frame_buffer.append(np.copy(frame))

    '''
    Behavior:
        - Returns a list of the latest written coords to the coord_buffer of length max_frames
    '''
    def get_last_fifteen_coords(self):
        return list(self.coord_buffer)

    # REMOVED Torch dependency — this now returns None and avoids crashing
    def get_last_frames_tensor(self):
        """Stubbed out: Torch removed from backend."""
        return None

    '''
    Return normalized landmark coordinates for the buffered frames

    Returns:
        - List of dicts: each item is {"frame":int, "landmarks":{<LANDMARK_NAME>:{"X":float, "y":float),...}}}
    
    Behavior:
        - If self.coord_buffer is empty or contains no valid records, returns []
        - Uses the landmark order from self._selected_landmarks
        - Builds and array of pixel coordinates (frames x landmarks x 2), subtracts the first frame's
          coordinates to get relative motion, then scales all relative coordinates by the maximum
          Euclidean distance across the array so results are scale-invariant
        - Converts normalized coordinates to Python floats and preserves the original frame indices
        - Records missing any expected landmark are skipped
    
    Notes:
        - Output is normalized relative to the first buffered frame
        - Not thread-safe; synchronize external access if used concurrently
        - Assumes the stored coordinates are pixel-space integers produced by process_frame
    '''
    def get_last_normalized_coords(self):
        """Return normalized landmark coordinates."""
        if len(self.coord_buffer) == 0:
            return []

        ordered_names = [landmark.name for landmark in self._selected_landmarks]
        frames = []
        coord_rows = []

        for record in self.coord_buffer:
            landmarks = record.get("landmarks")
            if not landmarks:
                continue

            try:
                frame_coords = [[landmarks[name]["x"], landmarks[name]["y"]] for name in ordered_names]
            except KeyError:
                continue

            frames.append(record["frame"])
            coord_rows.append(frame_coords)

        if not coord_rows:
            return []

        coords_array = np.array(coord_rows, dtype=np.float32)
        relative = coords_array - coords_array[0]

        distances = np.linalg.norm(relative, axis=2)
        max_distance = float(distances.max()) if distances.size else 0.0

        if max_distance > 0.0:
            relative /= max_distance

        normalized_records = []
        for frame_index, points in zip(frames, relative):
            normalized_records.append({
                "frame": frame_index,
                "landmarks": {
                    name: {"x": float(coord[0]), "y": float(coord[1])}
                    for name, coord in zip(ordered_names, points)
                }
            })

        return normalized_records


    '''
    Start a polling loop the repeatedly fetches frames and processes them

    Parameters:
        - get_frame_callable (callable): zero-arg function the returns a frame
        - poll_interval (float): seconds to sleep between polls (default 0.03)

    Behavior:
        - Sets self.running = True and enters a blocking loop until self.running becomes False
        - Calls get_frame_callable() each interation; if it returns a frame, calls
          self.process_frame(frame)
        - Sleeps poll_interval between iterations
        - Intended to be run in a background thread or seperate task because it blocks

    Notes:
        - Not thread-safe; coordinate access when calling stop() from another thread
        - get_frame_callable may block or raise; caller should ensure it behaves correctly for the
          polling pattern
    '''
    def run(self, get_frame_callable, poll_interval=0.03):
        self.running = True
        while self.running:
            frame = get_frame_callable()
            if frame is not None:
                self.process_frame(frame)
            time.sleep(poll_interval)

    '''
    Stop the polling loop started by run

    Behavior:
        - Sets self.running = False
        - If run is active in another thread, the loop will exit after the next poll/sleep cycle
        - Non-blocking and does not join threads or perform cleanup of resources
    '''
    def stop(self):
        self.running = False
