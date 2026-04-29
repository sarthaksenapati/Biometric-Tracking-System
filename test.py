"""
gpu_check.py — Diagnostic for Intel Iris Xe (CPU/OpenVINO setup)
Usage:  python gpu_check.py
"""

import sys

print("\n" + "="*60)
print("  SYSTEM DIAGNOSTIC — Final Year Project")
print("  (Intel Iris Xe — CPU/OpenVINO mode)")
print("="*60)

# ── 1. PyTorch ────────────────────────────────────────────────────────────────
try:
    import torch
    cuda_ok = torch.cuda.is_available()
    print(f"\n[PyTorch]  version={torch.__version__}")
    print(f"           CUDA available : {cuda_ok}")
    if cuda_ok:
        print(f"           ✅  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("           ✅  Running on CPU — OK for Intel Iris Xe")
except ImportError:
    print("[PyTorch]  ❌  Not installed — run: pip install torch torchvision")
    cuda_ok = False

# ── 2. ONNX Runtime ───────────────────────────────────────────────────────────
print()
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"[ONNX RT]  version={ort.__version__}")
    print(f"           Providers: {providers}")
    if "CUDAExecutionProvider" in providers:
        print("           ✅  CUDA provider available")
    elif "OpenVINOExecutionProvider" in providers:
        print("           ✅  OpenVINO provider available (Intel GPU acceleration)")
    elif "CPUExecutionProvider" in providers:
        print("           ✅  CPU provider available — works fine for this project")
    else:
        print("           ❌  No usable provider found")
except ImportError:
    print("[ONNX RT]  ❌  Not installed — run: pip install onnxruntime")

# ── 3. InsightFace ────────────────────────────────────────────────────────────
print()
try:
    import insightface
    from insightface.app import FaceAnalysis
    print(f"[InsightFace] version={insightface.__version__}")

    # CPU-safe initialization — no CUDA required
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))  # ctx_id=-1 = CPU

    for model_name, model in app.models.items():
        try:
            provider = model.session.get_providers()[0]
        except Exception:
            provider = "unknown"
        print(f"           Model '{model_name}' → provider={provider}")

    print("           ✅  InsightFace loaded successfully on CPU")
except Exception as e:
    print(f"[InsightFace] ❌  Error: {e}")
    if "buffalo_l" in str(e) or "download" in str(e).lower():
        print("               → Model not downloaded yet, will auto-download on first run")
    elif "providers" in str(e).lower():
        print("               → Version mismatch. Run: pip install --upgrade insightface")

# ── 4. YOLO ───────────────────────────────────────────────────────────────────
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
        print("           ✅  YOLO on CPU — fine for Intel Iris Xe")
except Exception as e:
    print(f"[YOLO]     ❌  Error: {e}")

# ── 5. Torchreid ─────────────────────────────────────────────────────────────
print()
try:
    import torchreid
    print(f"[Torchreid] ✅  Installed")
    print(f"            Running on CPU — fine for this project")
except ImportError:
    print("[Torchreid] ❌  Not installed — run: pip install torchreid")

# ── 6. Summary ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  SUMMARY & NEXT STEPS")
print("="*60)
print("""
This project is configured for Intel Iris Xe (no NVIDIA GPU).
CPU mode is fully supported — ignore all CUDA warnings.

If anything above shows ❌, run the relevant fix:

  InsightFace error  → pip install --upgrade insightface
  ONNX RT missing    → pip install onnxruntime
  YOLO missing       → pip install ultralytics
  Torchreid missing  → pip install torchreid

DO NOT install:
  ✗ onnxruntime-gpu   (NVIDIA only)
  ✗ torch with cu121  (NVIDIA only)

Optional Intel acceleration:
  pip install openvino onnxruntime-openvino
""")