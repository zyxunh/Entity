import os
from dataclasses import dataclass

import numpy as np
import torch
from CropFormer.api import EntityApi
from PIL import Image
from detectron2.utils.memory import retry_if_cuda_oom
from unhcv.common.array import split
from unhcv.datasets.common_datasets.tools.attach_data import ReadData

from .register_entityv2_semseg_150 import ENTITYV2_SEMSEG150_CATEGORIES
from unhcv.common import visual_mask, write_im
from unhcv.common.image import concat_differ_size, gray2color, mask_proportion
from unhcv.common.utils import find_path, attach_home_root, obj_load, ProgressBarTqdm, walk_all_files_with_suffix, \
    replace_str, base_multiprocess_func
from unhcv.datasets.common_datasets.tools import AttachData, AttachDataConfig

@dataclass
class AttachDataConfig(AttachDataConfig):
    config_file: str = find_path("code/Entity/Entityv2/CropFormer/configs/entityv2/entity_segmentation/mask2former_swin_large_3x.yaml")


class AttachData(AttachData):
    def __init__(self, config: AttachDataConfig):
        # self.init_data(config)
        self.init_api(config)

    def init_data(self, config: AttachDataConfig):
        super().__init__(config)
        file2 = find_path("dataset/Adobe_EntitySeg/raw_label_lr/entityseg_val_lr.json")
        data2 = obj_load(file2)

        # 150 category
        id2category = {}
        name2category = {}
        thing_classes = []
        for var in data2['categories']:
            id2category[var['id']] = var
            name2category[var['name']] = var
        id2semseg150_categories = {}
        for i_var, var in enumerate(ENTITYV2_SEMSEG150_CATEGORIES):
            assert i_var == var['id']
            if var['name'] == "buildingstructure_ot":
                var['name'] = 'building_structure_ot'
            if var['name'] == 'porch':
                id2semseg150_categories[var['id']] = dict(name='porch', type='stuff')
            else:
                id2semseg150_categories[var['id']] = name2category[var['name']]
            if id2semseg150_categories[var['id']]['type'] == 'thing':
                thing_classes.append(var['id'])
            elif id2semseg150_categories[var['id']]['type'] == 'stuff':
                pass
            else:
                raise ValueError

        self.id2semseg150_categories = id2semseg150_categories
        self.thing_classes = torch.tensor(thing_classes).cuda()
        self.thing_classes_onehot = torch.zeros(150).bool().cuda()
        self.thing_classes_onehot[self.thing_classes] = 1
        # self.thing_classes_onehot = torch.nn.functional.one_hot(self.thing_classes, num_classes=150).sum(1)

    def init_api(self, config: AttachDataConfig):
        self.entity_api = EntityApi(config_file=config.config_file)
        # self.entity_api1 = EntityApi(config_file=find_path("code/Entity/Entityv2/CropFormer/configs/entityv2/panoptic_segmentation/mask2former_swin_large_w12.yaml"))
        self.entity_api2 = EntityApi(config_file=find_path(
            "code/Entity/Entityv2/CropFormer/configs/entityv2/semantic_segmentation/mask2former_swin_large_w12.yaml"))

    def cal_thing_score(self, mask, sem_seg):
        sem_seg = sem_seg.to(mask.device)
        # thing_seg = sem_seg.softmax(0)[self.thing_classes].sum(0)
        # sem_seg = sem_seg.argmax(0)
        # thing_seg = (sem_seg[None] == self.thing_classes[:, None, None].to(sem_seg)).any(0).to(torch.float)

        thing_classes_onehot = self.thing_classes_onehot.to(mask.device)
        # thing_seg = (sem_seg[thing_classes_onehot].sum(0) > sem_seg[~thing_classes_onehot].sum(0)).to(torch.float)
        thing_seg = sem_seg[thing_classes_onehot].sum(0) / (sem_seg[thing_classes_onehot].sum(0) + sem_seg[~thing_classes_onehot].sum(0)).clamp(min=1e-4)

        mask_one_hot = torch.nn.functional.one_hot(mask.long()).to(torch.float)
        entity_proportion = mask_proportion(mask_one_hot.sum((0, 1)), mask_one_hot.shape[:2])
        score = (thing_seg[..., None] * mask_one_hot).sum((0, 1)) / mask_one_hot.sum((0, 1)).clamp(min=1e-4)

        score[0] = 0
        thing_score = (mask_one_hot * score[None, None]).max(-1)[0]
        entity_proportion[0] = 0
        entity_proportion_mask = (mask_one_hot * entity_proportion[None, None]).max(-1)[0]
        return (thing_score * 255).round().cpu().numpy().astype(np.uint8), (entity_proportion_mask * 255).round().cpu().numpy().astype(np.uint8), score, entity_proportion

    def attach_data_dict(self, data_dict, data_index):
        predictions = self.entity_api(data_dict['image'])
        mask = predictions['panoptic_seg'][0]
        data_dict["mask"] = Image.fromarray(mask.cpu().numpy().astype(np.uint8))

        # data_dict['predictions'] = predictions
        # predictions = self.entity_api1(data_dict['image'])
        # data_dict['predictions1'] = predictions
        predictions = self.entity_api2(data_dict['image'])
        # data_dict['predictions2'] = predictions
        thing_score, entity_proportion_mask, score, entity_proportion = retry_if_cuda_oom(self.cal_thing_score)(mask, predictions['sem_seg'])
        data_index["thing_score"] = tuple(score.cpu().numpy().tolist())
        data_dict["entity_proportion_mask"] = Image.fromarray(entity_proportion_mask)
        data_index["entity_proportion"] = tuple(entity_proportion.cpu().numpy().tolist())
        data_dict["thing_score_mask"] = Image.fromarray(thing_score)
        return data_dict


def main(mp_idx=0, mp_num=1, **kwargs):
    torch.cuda.set_device(mp_idx % torch.cuda.device_count())
    config = AttachDataConfig(
        config_file=find_path(
            "code/Entity/Entityv2/CropFormer/configs/entityv2/entity_segmentation/mask2former_hornet_3x.yaml"),
        data_indexes_path=find_path("dataset/open-images-dataset/train/lmdb/086_wh1.333_num10000_index.bson"),
        data_root=find_path("dataset/open-images-dataset/train/lmdb/086_wh1.333_num10000"),
        save_root=[attach_home_root("dataset/open-images-dataset/train/lmdb_1023/086_wh1.333_num10000")],
        save_keys=('mask', 'thing_score_mask', 'entity_proportion_mask'), remove_keys=("mask_score",),
        save_root_ids=("panoptic",))
    attach_data = AttachData(config)

    original_data_root = find_path("dataset/open-images-dataset/train/lmdb")
    tgt_data_root = attach_home_root("dataset/open-images-dataset/train/lmdb_1025")
    suffix = "_blip_large_catalog.bson"
    data_indexes_paths = walk_all_files_with_suffix(original_data_root, suffix)
    data_indexes_paths = split(data_indexes_paths, mp_num)[mp_idx]
    progress_bar = ProgressBarTqdm(len(data_indexes_paths) * 10000)

    for data_indexes_path in data_indexes_paths:
        show_root = attach_home_root("show/label_panoptic_show3")

        data_root = replace_str(data_indexes_path, suffix, "", position='r')
        save_root = [replace_str(data_root, original_data_root, tgt_data_root, 'l')]
        if os.path.exists(save_root[0]):
            continue
        config = AttachDataConfig(
        data_indexes_path=data_indexes_path,
        data_root=data_root,
        save_root=save_root,
        save_keys=('mask', 'thing_score_mask', 'entity_proportion_mask'), remove_keys=("mask_score",),
        save_root_ids=("panoptic",))
        attach_data.init_data(config)
        # read
        if 1:
            attach_data_iter = iter(attach_data)

            for i in range(len(attach_data)):
                data_dict = next(attach_data_iter)
                progress_bar.update()
                # sem_seg = data_dict['predictions2']['sem_seg'].argmax(0)
                # thing_seg = (sem_seg[None] == attach_data.thing_classes[:, None, None]).any(0).long()
                continue

                # show
                image = np.array(data_dict['image'].convert("RGB"))[..., ::-1]
                mask = np.array(data_dict['mask'])
                thing_score_mask = np.array(data_dict['thing_score_mask'])
                entity_proportion_mask = np.array(data_dict['entity_proportion_mask'])
                mask_show = visual_mask(image, mask)[-1]
                # entity_proportion_mask[entity_proportion_mask != 0] = 255
                thing_score_mask_show = visual_mask(image, thing_score_mask.astype(np.float32) / 255, is_matting=True, contrast_enhance_on=False)[-1]
                thing_score_mask_thres_show = visual_mask(image, (thing_score_mask > 100).astype(np.uint8))[-1]
                entity_proportion_mask_show = visual_mask(image, entity_proportion_mask.astype(np.float32) / 255, is_matting=True, contrast_enhance_on=False)[-1]
                shows = [image, mask_show, thing_score_mask_show, thing_score_mask_thres_show, entity_proportion_mask_show]
                shows = concat_differ_size(shows)
                write_im(os.path.join(show_root, f"{i}.jpg"), shows)
                continue

                panoptic_seg =  data_dict['predictions']['panoptic_seg'][0].cpu().numpy()
                panoptic_seg_show = visual_mask(image, panoptic_seg)[-1]
                mask_id_show = visual_mask(image, data_dict['predictions']['mask_id'])[-1]
                panoptic_seg1 = data_dict['predictions1']['panoptic_seg'][0].cpu().numpy()
                panoptic_seg1_show = visual_mask(image, panoptic_seg1)[-1]
                sem_seg1_show = visual_mask(image, data_dict['predictions1']['sem_seg'].argmax(0).cpu().numpy())[-1]
                sem_seg2_show = visual_mask(image, data_dict['predictions2']['sem_seg'].argmax(0).cpu().numpy())[-1]
                thing_seg2_show = visual_mask(image, thing_seg.cpu().numpy())[-1]
                mask_show = visual_mask(image, mask)[-1]
                shows = [image, mask_show, mask_id_show, panoptic_seg_show, panoptic_seg1_show, sem_seg1_show, sem_seg2_show, thing_seg2_show]
                shows = concat_differ_size(shows)
                write_im(os.path.join(show_root, f"{i}.jpg"), shows)

            attach_data.end_write()

    if 0:
        read_data = ReadData(data_indexes_path=attach_home_root(
            "dataset/open-images-dataset/train/lmdb_1023/086_wh1.333_num10000_catalog.bson"),
            data_root=[find_path("dataset/open-images-dataset/train/lmdb/086_wh1.333_num10000"),
                       find_path("dataset/open-images-dataset/train/lmdb_1023/086_wh1.333_num10000")],
            root_ids=None)  # ["default", "new_mask_1"]

        for i in range(len(read_data)):
            data = read_data.read_data_i(i)

if __name__ == '__main__':
    base_multiprocess_func(main, num_p=8 * 4)