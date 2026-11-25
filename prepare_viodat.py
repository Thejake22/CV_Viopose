import os
import glob
import shutil
import subprocess
import numpy as np
import cv2
import librosa
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm
import inference

inference.librosa = librosa

kp2d_dir = r"C:\VioPose\viodat\kp_2d"
kp3d_dir = r"C:\VioPose\viodat\kp_3d"
vid_dir  = r"C:\VioPose\viodat\video"
out_dir  = r"C:\VioPose\data\FullData"
tmp_wav_dir = r"C:\VioPose\tmp_wav"

os.makedirs(out_dir, exist_ok=True)
os.makedirs(tmp_wav_dir, exist_ok=True)

base_options = python.BaseOptions(model_asset_path=r"C:\VioPose\pose_landmarker_full.task")
options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=False)
detector = vision.PoseLandmarker.create_from_options(options)

def extract_mp_2d(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    keypoints2d = []
    while True:
        success, image = cap.read()
        if not success:
            break
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = detector.detect(mp_image)
        if not results.pose_landmarks:
            keypoints2d.append([])
            continue
        lms = results.pose_landmarks[0]
        keypoints2d.append([[lm.x * w, lm.y * h] for lm in lms])
    cap.release()
    keypoints2d = inference.fill_empty(keypoints2d)
    return inference.mediapipe_kp_remap(np.asarray(keypoints2d, dtype=np.float32)), fps

kp_optim_2d_vid, kp_optim_3d_vid, mediapipe_2d, kp_video_fps, audio_data_100FPS = {}, {}, {}, {}, {}

npy_files = sorted(glob.glob(os.path.join(kp3d_dir, "*.npy")))
print(f"Found {len(npy_files)} trials to process.")

for f3d in tqdm(npy_files):
    trial = os.path.splitext(os.path.basename(f3d))[0]
    f2d = os.path.join(kp2d_dir, f"{trial}.npy")
    fvid = os.path.join(vid_dir, f"{trial}.mp4")
    fwav = os.path.join(tmp_wav_dir, f"{trial}.wav")

    if not (os.path.exists(f2d) and os.path.exists(fvid)):
        continue

    kp2d_arr = np.load(f2d)
    kp3d_arr = np.load(f3d)
    mp_2d, fps = extract_mp_2d(fvid)
    if mp_2d is None or len(mp_2d) == 0:
        continue

    subprocess.run(
        ["ffmpeg", "-y", "-i", fvid, "-vn", "-ac", "1", "-ar", "44100", fwav],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    aud_feat = inference.audio_feature_extraction(fwav, fps, mp_2d)
    if os.path.exists(fwav):
        os.remove(fwav)

    kp_optim_2d_vid[trial] = kp2d_arr
    kp_optim_3d_vid[trial] = kp3d_arr
    mediapipe_2d[trial] = mp_2d
    kp_video_fps[trial] = fps
    audio_data_100FPS[trial] = aud_feat

if len(mediapipe_2d) > 0:
    np.savez(
        os.path.join(out_dir, "mmViolin_v1.0.npz"),
        kp_optim_2d_vid=kp_optim_2d_vid,
        kp_optim_3d_vid=kp_optim_3d_vid,
        mediapipe_2d=mediapipe_2d,
        kp_video_fps=kp_video_fps,
    )
    np.savez(
        os.path.join(out_dir, "mmViolin_v1.0_audio.npz"),
        audio_data_100FPS=audio_data_100FPS,
    )
    print(f"Compilation complete! Saved {len(mediapipe_2d)} trials into {out_dir}.")
else:
    print("No trials processed. Check C:\\VioPose\\viodat subfolders.")

shutil.rmtree(tmp_wav_dir, ignore_errors=True)