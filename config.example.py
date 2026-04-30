# config.example.py
# Copy this file to config.py and fill in your values.

DROIDCAM_IP = "192.168.29.141"  # ← your DroidCam IP here
DROIDCAM_PORT = 4747

CAMERA_SOURCES = {
    0: 0,  # built-in webcam
    1: f"http://{DROIDCAM_IP}:{DROIDCAM_PORT}/video",  # DroidCam
}

CAMERA_LOCATIONS = {
    0: "Main Entrance",
    1: "Hallway",
}
