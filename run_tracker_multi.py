from core.multi_tracker import MultiCameraTracker

# Camera 0 — laptop webcam
# Camera 1 — iPhone via DroidCam (WiFi)
#   IP shown in DroidCam client: 172.20.10.1
#   If this stops working, open DroidCam client and check the IP there

sources = {
    0: 0,
    1: "http://192.168.0.183:4747/video"
}
tracker = MultiCameraTracker(cam_sources=sources)
tracker.run()