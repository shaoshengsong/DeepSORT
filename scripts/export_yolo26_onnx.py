#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将官方 yolo26s.pt 权重转换为 ONNX 格式，供 onnxruntime 1.28.2 推理使用。

用法（在本机具有 ultralytics + torch 的环境执行）:
    python export_yolo26_onnx.py

说明:
- 使用 Ultralytics 官方 export 接口，得到标准的检测 ONNX。
- 输出: models/yolo26s.onnx
- 导出后脚本会打印输入/输出张量 shape、opset，并写入 models/yolo26s.onnx.meta.json，
  供 C++ 检测端在需要时进行布局自动识别。

注意:
- YOLO26 检测 ONNX 的标准输出布局通常为 [1, C, N] 或 [1, N, C]，
  其中 C = 4 + num_classes，N 为候选框数量；本工程 C++ 检测器会自动识别两种布局。
- imgsz 默认 640。如需动态尺寸可改 dynamic=True，但固定尺寸对跟踪单帧推理更友好。
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PT = os.path.join(HERE, "..", "yolo26s.pt")  # 仓库根目录的官方权重
if not os.path.exists(MODEL_PT):
    MODEL_PT = os.path.join(HERE, "yolo26s.pt")
OUT_DIR = os.path.join(HERE, "models")
OUT_ONNX = os.path.join(OUT_DIR, "yolo26s.onnx")
IMGSZ = 640


def export():
    try:
        from ultralytics import YOLO
    except Exception as e:  # noqa
        print("[ERROR] 未找到 ultralytics，请先安装: pip install ultralytics onnx onnxruntime")
        print("        异常:", e)
        sys.exit(2)

    if not os.path.exists(MODEL_PT):
        print(f"[ERROR] 未找到权重文件: {MODEL_PT}")
        sys.exit(2)

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[INFO] 加载权重: {MODEL_PT}")
    model = YOLO(MODEL_PT)

    print(f"[INFO] 导出 ONNX (imgsz={IMGSZ}) -> {OUT_ONNX}")
    # simplify=True 可消除冗余节点，便于 onnxruntime 加载；opset 不强制，使用默认。
    success = model.export(
        format="onnx",
        imgsz=IMGSZ,
        dynamic=False,
        simplify=True,
        half=False,
        int8=False,
        device="cpu",
        workspace=None,
    )
    # YOLO.export 返回导出路径
    src = success if isinstance(success, str) else OUT_ONNX
    if src != OUT_ONNX and os.path.exists(src):
        import shutil
        shutil.move(src, OUT_ONNX)

    if not os.path.exists(OUT_ONNX):
        print(f"[ERROR] 导出后未找到 {OUT_ONNX}")
        sys.exit(3)

    _dump_meta(OUT_ONNX)
    print(f"[OK] 转换完成: {OUT_ONNX}")
    print("[INFO] 请将该文件随可执行程序一起部署到 models/ 目录。")


def _dump_meta(onnx_path: str):
    """用 onnxruntime 读取输入输出 shape，写入 meta json 供 C++ 端参考。"""
    try:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        meta = {
            "inputs": [
                {"name": i.name, "shape": [str(s) for s in i.shape]}
                for i in sess.get_inputs()
            ],
            "outputs": [
                {"name": o.name, "shape": [str(s) for s in o.shape]}
                for o in sess.get_outputs()
            ],
        }
    except Exception as e:  # noqa
        meta = {"note": f"无法用 onnxruntime 读取元数据: {e}"}
    meta_path = onnx_path + ".meta.json"
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"[INFO] 元数据已写入: {meta_path}")
        print("[INFO] 输入/输出张量:")
        print(json.dumps(meta, indent=2, ensure_ascii=False))
    except Exception:  # noqa
        pass


if __name__ == "__main__":
    export()
