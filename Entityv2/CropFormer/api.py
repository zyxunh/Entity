import numpy as np
from PIL.Image import Image
from unhcv.common.image import gray2color

from unhcv.common.types import DataDict
from unhcv.common.utils import find_path, obj_load

from .demo_mask2former.predictor import VisualizationDemo
from .demo_mask2former.demo import setup_cfg


def get_parser():
    parser = DataDict()
    parser.opts = []
    return parser

class EntityApi:
    def __init__(self, config_file="configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml",
                 confidence_threshold=0.1):
        args = get_parser()
        args.config_file = config_file
        cfg = setup_cfg(args)
        self.demo = VisualizationDemo(cfg)
        self.confidence_threshold = confidence_threshold

    def __call__(self, image: np.ndarray):
        """
        image: BGR
        """
        if isinstance(image, Image):
            image = image.convert('RGB')
            image = np.array(image)[..., ::-1]
        predictions = self.demo.run_on_image(image)
        return predictions

        ##### color_mask
        pred_masks = predictions["instances"].pred_masks
        pred_scores = predictions["instances"].scores

        # select by confidence threshold
        selected_indexes = (pred_scores >= self.confidence_threshold)
        selected_scores = pred_scores[selected_indexes]
        selected_masks = pred_masks[selected_indexes]
        _, m_H, m_W = selected_masks.shape
        mask_id = np.zeros((m_H, m_W), dtype=np.uint8)
        if len(selected_masks):

            selected_masks_with_score = selected_masks * selected_scores[:, None, None]
            selected_masks_with_score_max, selected_masks_with_score_idx = selected_masks_with_score.max(dim=0)
            selected_scores = selected_scores[selected_masks_with_score_idx].cpu().numpy().tolist()

            selected_masks_with_score_idx += 1
            selected_masks_with_score_idx[selected_masks_with_score_max == 0] = 0

            # rank
            # selected_scores, ranks = torch.sort(selected_scores)
            # ranks = ranks + 1
            # for index in ranks:
            #     mask_id[(selected_masks[index - 1] == 1).cpu().numpy()] = int(index)
            # assert (torch.from_numpy(mask_id).cuda() == selected_masks_with_score_idx).all()

            mask_id = selected_masks_with_score_idx.cpu().numpy()
        predictions['mask_id'] = mask_id
        return predictions


if __name__ == "__main__":
    import cv2
    config_file = find_path("code/Entity/Entityv2/CropFormer/configs/entityv2/entity_segmentation/mask2former_hornet_3x.yaml")
    entity_api = EntityApi(config_file=config_file)
    image = np.array(obj_load("/home/yixing/dataset/tmp/show1.jpg"))[..., ::-1]
    image = np.array(obj_load("/home/yixing/dataset/Adobe_EntitySeg/images_lr/entity_01_11580/coco_000000322352.jpg"))[..., ::-1]
    output = entity_api(image)
    panoptic_seg_color = gray2color(output['panoptic_seg'][0].cpu().numpy())
    cv2.imwrite('/home/yixing/dataset/tmp/mask1_tmp.png', panoptic_seg_color)
    breakpoint()
    pass
