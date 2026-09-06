import cv2, numpy as np
from ultralytics import YOLO
import onnxruntime as ort

cap = cv2.VideoCapture('people_walking.mp4')
ret, frame = cap.read()
cap.release()
if not ret:
    print("cannot read frame"); raise SystemExit
H, W = frame.shape[:2]
print("frame size", W, H)

# --- ground truth via ultralytics (original image coords) ---
model = YOLO('yolo26s.pt')
res = model(frame, imgsz=640, verbose=False)[0]
gt = []
for b in res.boxes:
    x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
    gt.append((x1, y1, x2, y2, float(b.conf), int(b.cls)))
gt = [g for g in gt if g[4] >= 0.25]
print("GT count:", len(gt))
print("GT sample:", [round(a,1) for a in gt[0]] if gt else "none")

# --- onnx letterbox preprocess (identical to C++ detector) ---
imgsz = 640
r = min(imgsz / W, imgsz / H)
nw, nh = int(W * r), int(H * r)
pad_x = (imgsz - nw) // 2
pad_y = (imgsz - nh) // 2
small = cv2.resize(frame, (nw, nh))
canvas = np.full((imgsz, imgsz, 3), 114, np.uint8)
canvas[pad_y:pad_y+nh, pad_x:pad_x+nw] = small
blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
blob = np.expand_dims(blob, 0)

sess = ort.InferenceSession('YoloTracker/models/yolo26s.onnx', providers=['CPUExecutionProvider'])
out = sess.run(None, {'images': blob})[0][0]  # [300,6]
print("onnx out shape", out.shape, "first row raw:", [round(float(v),1) for v in out[0]])

raw = [g for g in out if g[4] >= 0.25]
mapped = []
for row in raw:
    x1 = (row[0]-pad_x)/r; y1=(row[1]-pad_y)/r; x2=(row[2]-pad_x)/r; y2=(row[3]-pad_y)/r
    mapped.append((x1,y1,x2,y2,row[4],row[5]))
print("ONNX raw[0:2] (640-space):", [[round(float(v),1) for v in r2] for r2 in raw[:2]])
print("ONNX mapped[0:2] (orig-space):", [[round(float(v),1) for v in m] for m in mapped[:2]])

# Decide which matches GT better
def iou(a,b):
    ax1,ay1,ax2,ay2=a[:4]; bx1,by1,bx2,by2=b[:4]
    ix1,iy1,ix2,iy2=max(ax1,bx1),max(ay1,by1),min(ax2,bx2),min(ay2,by2)
    iw=max(0,ix2-ix1); ih=max(0,iy2-iy1); inter=iw*ih
    ua=(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return inter/ua if ua>0 else 0

def best_iou(cands, g):
    best=0
    for c in cands:
        best=max(best, iou(c,g))
    return best

# compare GT against raw (640-space) and against mapped
raw_iou = sum(best_iou(raw, g) for g in gt)/max(1,len(gt))
map_iou = sum(best_iou(mapped, g) for g in gt)/max(1,len(gt))
print(f"avg best-IoU vs GT: raw(640-space)={raw_iou:.3f}  mapped(orig-space)={map_iou:.3f}")
print("VERDICT:", "use MAPPED (apply gain/pad)" if map_iou>raw_iou else "use RAW (no mapping)")
