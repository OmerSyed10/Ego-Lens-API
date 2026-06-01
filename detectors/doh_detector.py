"""
100DOH (Hand-Object Detector) wrapper — Stage 2 of the hand-detection cascade.

Uses the pretrained Shan et al. CVPR 2020 ResNet-101 Faster-RCNN model that
achieves ~90% AP on egocentric ego/100DOH benchmarks and fires on the palm/wrist
even when fingers are occluded (gripping, fist, tool use).

Safe import: DOH_AVAILABLE = False if torch or the vendor repo are missing.
The HandTracker simply skips this tier if DOH_AVAILABLE is False.

Checkpoint layout expected by load_state_dict:
  ckpt['model']  — the trained weights dict
  ckpt['pooling_mode'] (optional)  — 'align' | 'pool'
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

# ── Availability guard ──────────────────────────────────────────────────────
DOH_AVAILABLE = False
_IMPORT_ERROR: Optional[str] = None

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T

    # Vendor lib on sys.path so model.* imports resolve
    _VENDOR_LIB = os.path.join(
        os.path.dirname(__file__),
        "..", "vendors", "hand_object_detector", "lib",
    )
    _VENDOR_LIB = os.path.normpath(_VENDOR_LIB)
    if _VENDOR_LIB not in sys.path:
        sys.path.insert(0, _VENDOR_LIB)

    from model.utils.config import cfg, cfg_from_list
    from model.faster_rcnn.resnet import resnet
    from model.rpn.bbox_transform import clip_boxes, bbox_transform_inv

    DOH_AVAILABLE = True
except Exception as _e:
    _IMPORT_ERROR = str(_e)


# ── Contact state enum values (matches 100DOH training) ─────────────────────
CONTACT_NO_CONTACT    = 0
CONTACT_SELF          = 1
CONTACT_OTHER_PERSON  = 2
CONTACT_PORTABLE_OBJ  = 3
CONTACT_STATIONARY    = 4

CONTACT_NAMES = {
    0: "no_contact",
    1: "self_contact",
    2: "other_person",
    3: "portable_object",
    4: "stationary_object",
}


@dataclass
class DOHResult:
    """One hand detection from 100DOH.

    Attributes
    ----------
    x1, y1, x2, y2 : float  — bounding box in pixels
    score           : float  — detection confidence [0, 1]
    contact_state   : int    — one of CONTACT_* constants above
    is_right        : bool   — True = right hand, False = left
    wrist_xy        : (x, y) — centre of the bottom half of the box (proxy wrist)
    """
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    contact_state: int
    is_right: bool

    @property
    def wrist_xy(self) -> np.ndarray:
        """Approximate wrist position: centre-bottom of bbox."""
        cx = (self.x1 + self.x2) / 2.0
        cy = (self.y1 * 0.25 + self.y2 * 0.75)   # lower-quarter
        return np.array([cx, cy], dtype=np.float32)

    @property
    def contact_name(self) -> str:
        return CONTACT_NAMES.get(self.contact_state, "unknown")


class DOHDetector:
    """
    Wraps the 100DOH ResNet-101 Faster-RCNN model for single-frame inference.

    Usage
    -----
    det = DOHDetector(weights_path="video_qc/vendors/weights/handobj_100K_ego.pth")
    results: list[DOHResult] = det.detect(frame_bgr)
    det.close()

    Raises RuntimeError if DOH_AVAILABLE is False.
    """

    PASCAL_CLASSES = np.asarray(["__background__", "targetobject", "hand"])

    # ImageNet mean/std used by the 100DOH training pipeline
    _PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

    # Short side of the image fed to the network
    _TEST_SCALE = 600
    _TEST_MAX_SIZE = 1000

    def __init__(
        self,
        weights_path: str,
        score_thresh_hand: float = 0.5,
        score_thresh_obj: float = 0.5,
        device: Optional[str] = None,
    ) -> None:
        if not DOH_AVAILABLE:
            raise RuntimeError(
                f"100DOH is not available: {_IMPORT_ERROR}. "
                "Install dependencies or check vendor path."
            )

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights not found: {weights_path}")

        # Force CPU: 100DOH's Faster-RCNN uses custom RPN ops that fall back to CPU
        # under MPS, causing catastrophic synchronization overhead (~230s/frame vs ~7s/frame).
        # roi_align and roi_pool individually work on MPS, but the full forward pass does not.
        if device is None:
            device = "cpu"
        self.device = torch.device(device)

        self.thresh_hand = score_thresh_hand
        self.thresh_obj  = score_thresh_obj

        # Configure 100DOH
        cfg_from_list(["ANCHOR_SCALES", "[8,16,32,64]", "ANCHOR_RATIOS", "[0.5,1,2]"])
        cfg.USE_GPU_NMS = False  # torchvision NMS is always available

        # Build + load model
        self._model = resnet(
            self.PASCAL_CLASSES, 101,
            pretrained=False, class_agnostic=False,
        )
        self._model.create_architecture()

        ckpt = torch.load(weights_path, map_location="cpu")
        self._model.load_state_dict(ckpt["model"])
        if "pooling_mode" in ckpt:
            cfg.POOLING_MODE = ckpt["pooling_mode"]

        self._model.to(self.device)
        self._model.eval()

    # ── Preprocessing ────────────────────────────────────────────────────────
    def _preprocess(self, frame_bgr: np.ndarray):
        """BGR frame → (im_data_tensor, im_info_tensor, im_scale)."""
        im = frame_bgr.astype(np.float32)
        im -= self._PIXEL_MEANS

        h, w = im.shape[:2]
        im_size_min = min(h, w)
        im_size_max = max(h, w)

        scale = float(self._TEST_SCALE) / float(im_size_min)
        if round(scale * im_size_max) > self._TEST_MAX_SIZE:
            scale = float(self._TEST_MAX_SIZE) / float(im_size_max)

        import cv2
        im_scaled = cv2.resize(im, None, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_LINEAR)

        # HWC → CHW, add batch dim
        blob = im_scaled.transpose(2, 0, 1)[np.newaxis, ...]  # (1, C, H', W')
        im_data = torch.from_numpy(blob).to(self.device)
        im_info = torch.FloatTensor([[im_scaled.shape[0], im_scaled.shape[1], scale]]).to(self.device)

        return im_data, im_info, scale

    # ── Inference ────────────────────────────────────────────────────────────
    @torch.no_grad()
    def detect(self, frame_bgr: np.ndarray) -> List[DOHResult]:
        """
        Run 100DOH on one BGR frame.

        Returns a list of DOHResult (one per detected hand).
        Empty list when no hands found above threshold.
        """
        from model.roi_layers import nms as doh_nms

        im_data, im_info, im_scale = self._preprocess(frame_bgr)

        # Dummy tensors expected by the Faster-RCNN forward
        gt_boxes = torch.FloatTensor(1, 1, 5).zero_().to(self.device)
        num_boxes = torch.LongTensor(1).zero_().to(self.device)
        box_info  = torch.FloatTensor(1, 1, 5).zero_().to(self.device)

        (rois, cls_prob, bbox_pred,
         _rpn_cls, _rpn_box, _rcnn_cls, _rcnn_bbox,
         _rois_label, loss_list) = self._model(
            im_data, im_info, gt_boxes, num_boxes, box_info,
        )

        scores = cls_prob.data
        boxes  = rois.data[:, :, 1:5]

        # Extra predictions
        contact_vector = loss_list[0][0]
        lr_vector      = loss_list[2][0].detach()

        _, contact_indices = torch.max(contact_vector, 2)
        contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()

        lr = (torch.sigmoid(lr_vector) > 0.5).squeeze(0).float()

        # BBox regression
        if cfg.TEST.BBOX_REG and cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            box_deltas = bbox_pred.data.view(-1, 4) * \
                torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).to(self.device) + \
                torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).to(self.device)
            box_deltas = box_deltas.view(1, -1, 4 * len(self.PASCAL_CLASSES))
            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            pred_boxes = boxes.repeat(1, 1, scores.shape[-1])

        pred_boxes /= im_scale

        scores     = scores.squeeze()      # (N, num_classes)
        pred_boxes = pred_boxes.squeeze()  # (N, num_classes*4)

        results: List[DOHResult] = []

        # Only care about class index 2 = 'hand'
        hand_cls_idx = 2
        thresh = self.thresh_hand
        inds = torch.nonzero(scores[:, hand_cls_idx] > thresh).view(-1)

        if inds.numel() > 0:
            cls_scores = scores[:, hand_cls_idx][inds]
            cls_boxes  = pred_boxes[inds][:, hand_cls_idx * 4:(hand_cls_idx + 1) * 4]

            _, order = torch.sort(cls_scores, 0, True)
            keep = doh_nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
            keep = keep.view(-1).long()

            final_boxes  = cls_boxes[order][keep].cpu().numpy()
            final_scores = cls_scores[order][keep].cpu().numpy()
            final_contact = contact_indices[inds][order][keep].cpu().numpy().astype(int)
            final_lr      = lr[inds][order][keep].cpu().numpy()

            for i in range(len(final_boxes)):
                x1, y1, x2, y2 = final_boxes[i]
                results.append(DOHResult(
                    x1=float(x1), y1=float(y1),
                    x2=float(x2), y2=float(y2),
                    score=float(final_scores[i]),
                    contact_state=int(final_contact[i].squeeze()),
                    is_right=bool(final_lr[i].squeeze() > 0.5),
                ))

        return results

    def close(self) -> None:
        """Release model from memory."""
        del self._model
        if self.device.type == "mps":
            import torch
            torch.mps.empty_cache()
