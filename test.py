"""
gpu_check.py — Run this BEFORE your tracker to confirm every model is on GPU.
Usage:  python gpu_check.py
"""

import sys

print("\n" + "="*60)
print("  GPU DIAGNOSTIC — Final Year Project")
print("="*60)

# ── 1. PyTorch ────────────────────────────────────────────────────────────────
try:
    import torch
    cuda_ok = torch.cuda.is_available()
    print(f"\n[PyTorch]  version={torch.__version__}")
    print(f"           CUDA available : {cuda_ok}")
    if cuda_ok:
        print(f"           Device name    : {torch.cuda.get_device_name(0)}")
        print(f"           VRAM total     : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"           VRAM allocated : {torch.cuda.memory_allocated(0) / 1024**2:.0f} MB")
    else:
        print("  ❌  CUDA not available — install torch with CUDA support:")
        print("      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
except ImportError:
    print("[PyTorch]  ❌  Not installed")
    cuda_ok = False

# ── 2. ONNX Runtime ───────────────────────────────────────────────────────────
print()
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"[ONNX RT]  version={ort.__version__}")
    print(f"           Providers: {providers}")
    if "CUDAExecutionProvider" in providers:
        print("           ✅  CUDAExecutionProvider available")
    else:
        print("           ❌  CUDAExecutionProvider NOT available")
        print("               Fix: pip install onnxruntime-gpu")
        print("               Then: pip uninstall onnxruntime  (remove CPU version)")
except ImportError:
    print("[ONNX RT]  ❌  Not installed — InsightFace needs this")

# ── 3. InsightFace ────────────────────────────────────────────────────────────
print()
try:
    import insightface
    from insightface.app import FaceAnalysis
    print(f"[InsightFace] version={insightface.__version__}")

    # Try loading with GPU (ctx_id=0 means GPU 0)
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Check which provider each model is actually using
    for model_name, model in app.models.items():
        provider = getattr(model.session, "get_providers", lambda: ["unknown"])()
        print(f"           Model '{model_name}' → provider={provider[0]}")

    print("           ✅  InsightFace loaded")
except Exception as e:
    print(f"[InsightFace] ❌  Error: {e}")

# ── 4. YOLO (Ultralytics) ─────────────────────────────────────────────────────
print()
try:
    from ultralytics import YOLO
    import torch
    model = YOLO("yolov8n.pt")
    device = next(model.model.parameters()).device
    print(f"[YOLO]     device={device}")
    if "cuda" in str(device):
        print("           ✅  YOLO on GPU")
    else:
        print("           ❌  YOLO on CPU — fix in detector.py (see below)")
except Exception as e:
    print(f"[YOLO]     ❌  Error: {e}")

# ── 5. Torchreid / OSNet ──────────────────────────────────────────────────────
print()
try:
    import torchreid
    print(f"[Torchreid] ✅  installed")
    if cuda_ok:
        print(f"            Will use GPU automatically if model is moved to cuda")
    else:
        print(f"            ❌  No CUDA — will run on CPU")
except ImportError:
    try:
        import torch
        # Check if reid model file exists
        import os
        reid_files = []
        for root, dirs, files in os.walk("."):
            for f in files:
                if "reid" in f.lower() and f.endswith(".pt"):
                    reid_files.append(os.path.join(root, f))
        if reid_files:
            print(f"[ReID]     Found model files: {reid_files}")
        else:
            print("[Torchreid] Not installed (pip install torchreid)")
    except:
        pass

# ── 6. Summary ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  WHAT TO DO IF SOMETHING IS ❌")
print("="*60)
print("""
InsightFace on CPU?
  → pip uninstall onnxruntime
  → pip install onnxruntime-gpu

YOLO on CPU?
  → In models/detector.py, change:
        self.model = YOLO("yolov8n.pt")
    to:
        self.model = YOLO("yolov8n.pt")
        self.model.to("cuda")

PyTorch no CUDA?
  → pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

After fixing, run this script again to verify.
""")