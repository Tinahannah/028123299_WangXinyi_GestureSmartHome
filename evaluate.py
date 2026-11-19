import os
os.environ.setdefault("MPLCONFIGDIR",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs"))

import numpy as np
from ultralytics import YOLO

MODEL = "add_face.pt"
DATA = "close_eval.yaml"   # 3 类配置（close/open/face）


def main():
    model = YOLO(MODEL)
    print("模型类别:", model.names)

    results = model.val(data=DATA, split="test", verbose=False, plots=False)
    b = results.box

    print("\n===== 整体指标 (test set) =====")
    print(f"mAP@0.5      : {b.map50:.4f}")
    print(f"mAP@0.5:0.95 : {b.map:.4f}")
    print(f"Precision    : {b.mp:.4f}")
    print(f"Recall       : {b.mr:.4f}")

    print("\n===== 每类指标 =====")
    print(f"{'idx':<4}{'name':<14}{'Prec':<9}{'Recall':<9}{'AP50':<9}{'AP50-95':<9}")
    names = model.names
    all_ap = np.asarray(b.all_ap)  # shape (nc, 10)，第0列是 IoU=0.5
    for i in sorted(names):
        ap50 = float(all_ap[i, 0]) if all_ap.ndim == 2 else float('nan')
        print(f"{i:<4}{names[i]:<14}{b.p[i]:<9.4f}{b.r[i]:<9.4f}{ap50:<9.4f}{b.maps[i]:<9.4f}")


if __name__ == "__main__":
    main()
