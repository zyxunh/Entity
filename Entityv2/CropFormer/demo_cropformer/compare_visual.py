import cv2
import numpy
import numpy as np
from unhcv.common import write_im
from unhcv.common.utils import walk_all_files_with_suffix, get_related_path
from unhcv.common.image import concat_differ_size, visual_mask


image_root = "/home/tiger/show/entity_seg/image"
mask_root = "/home/tiger/show/entity_seg/mask"
mask1_root = "/home/tiger/show/entity_seg/mask1"
show_root = "/home/tiger/show/entity_seg/show"
image_names = walk_all_files_with_suffix(image_root)
for image_name in image_names:
    mask1 = cv2.imread(get_related_path(image_name, image_root, mask1_root, ".png"), 0)
    if mask1 is None:
        continue
    image = cv2.imread(image_name)
    mask = cv2.imread(get_related_path(image_name, image_root, mask_root, ".png"))
    mask1_show = visual_mask(image, mask1)[-1]
    mask_ = np.zeros(mask.shape[:2], dtype=np.int64)
    mask_ = mask[..., 0] * 255 **2 + mask[..., 1] * 255 + mask[..., 2]
    mask_show = visual_mask(image, mask_)[-1]
    show = concat_differ_size([mask1_show, mask_show], axis=1)
    write_im(get_related_path(image_name, image_root, show_root), show)
    pass