# Exercise Form Detection

Uses MediaPipe to check if you're doing bicep curls and lateral raises correctly.

## What it does

- Detects body keypoints from video
- Checks your form based on angles and positions
- Shows feedback on each frame
- Saves annotated video with corrections

## Setup

```powershell
pip install -r requirements.txt
```

## How to run

```powershell
python .\main.py .\sample_videos\barbell_biceps_curl_2.mp4 --exercise bicep_curl --mlflow
```

### Options

- `--exercise` - bicep_curl or lateral_raise (default: bicep_curl)
- `--side` - left or right (default: left)
- `--output` - where to save output video
- `--mlflow` - track metrics in MLFlow
- `--no-preview` - disable preview window

## Rules checked

**Bicep curl:**

- Elbow angle between 30-160 degrees
- Elbow stays close to body
- Wrist above elbow at top

**Lateral raise:**

- Arm at shoulder height
- Elbow slightly bent
- Wrist not above shoulder

**Back posture (both):**

- Shoulders level
- Hips level
- Spine straight
- Body balanced

## Files

- `main.py` - runs the detection
- `pose_detector.py` - MediaPipe wrapper
- `posture_checker.py` - form validation rules
- `pose_utils.py` - angle/distance calculations

## Notes

- Works with 3-5 second clips
- Supports MP4, AVI, MOV
- Best with side-view camera angle
- Press 'q' to stop preview

MLFlow UI (if using --mlflow):

```powershell
mlflow ui
```

Then go to http://localhost:5000
