from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from .image import transform_preds


def get_pred_depth(depth):
  return depth


def get_alpha(rot):
  # output: (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos,
  #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
  # return rot[:, 0]
  idx = rot[:, 1] > rot[:, 5]
  alpha1 = np.arctan(rot[:, 2] / rot[:, 3]) + (-0.5 * np.pi)
  alpha2 = np.arctan(rot[:, 6] / rot[:, 7]) + ( 0.5 * np.pi)
  return alpha1 * idx + alpha2 * (1 - idx)


def ctdet_post_process(dets_act, c, s, h, w, num_obj_classes, num_act_classes):
  ret_act = []
  for i in range(dets_act.shape[0]):
    top_preds_act = {}
    dets_act[i, :, :2] = transform_preds(
          dets_act[i, :, 0:2], c[i], s[i], (w, h))

    dets_act[i, :, 2:4] = transform_preds(
          dets_act[i, :, 2:4], c[i], s[i], (w, h))

    dets_act[i, :, 4:6] = transform_preds(
          dets_act[i, :, 4:6], c[i], s[i], (w, h))

    # print(dets_act[0])

    classes_act = dets_act[i, :, -1]

    for j in range(num_act_classes):
      inds = (classes_act == j)
      top_preds_act[j + 1] = np.concatenate([
        dets_act[i, inds, :6].astype(np.float32),
        dets_act[i, inds, 6:7].astype(np.float32)], axis=1).tolist()

    ret_act.append(top_preds_act)

  return ret_act
