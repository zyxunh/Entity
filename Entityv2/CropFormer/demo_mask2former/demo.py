# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys

from unhcv.common import get_related_path, write_im
from unhcv.common.array import chunk, split
from unhcv.common.utils import walk_all_files_with_suffix, ProgressBarTqdm, obj_dump

sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from predictor import VisualizationDemo
from unhcv.common.image import visual_mask
import torch

# constants
WINDOW_NAME = "mask2former demo"

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        type=str,
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--mp_idx", type=int, default=0)
    parser.add_argument("--mp_num", type=int, default=1)
    return parser

def main(mp_idx=0, mp_num=1, args=None, **kwargs):
    mp.set_start_method("spawn", force=True)
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.output + "_show", exist_ok=True)
    path_mask = os.path.join(args.output, "mask")
    path_score = os.path.join(args.output, "score")

    if args.input:
        filenames = walk_all_files_with_suffix(args.input)
        filenames = split(filenames, mp_num)[mp_idx]
        # filenames = filenames[:100]
        tqdm_bar = ProgressBarTqdm(len(filenames), smoothing=0)
        for path in filenames:
            name_mask = get_related_path(path, args.input, path_mask, suffixs='.png')
            name_score = get_related_path(path, args.input, path_score, suffixs='.png')
            if os.path.exists(name_mask) and os.path.exists(name_score):
                pass
            else:

                # use PIL, to be consistent with evaluation
                img = read_image(path, format="BGR")
                start_time = time.time()
                # with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
                predictions = demo.run_on_image(img)

                ##### color_mask
                pred_masks = predictions["instances"].pred_masks
                pred_scores = predictions["instances"].scores

                # select by confidence threshold
                selected_indexes = (pred_scores >= args.confidence_threshold)
                selected_scores = pred_scores[selected_indexes]
                selected_masks = pred_masks[selected_indexes]
                _, m_H, m_W = selected_masks.shape
                mask_id = np.zeros((m_H, m_W), dtype=np.uint8)

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
                if mask_id.max() > 255:
                    print(f'!!!!!{path} exceed 255')
                write_im(name_mask, mask_id)
                write_im(name_score, (selected_masks_with_score_max * 255).round().cpu().numpy())
                # obj_dump(get_related_path(path, args.input, args.output, suffixs='.bson'), dict(selected_scores=selected_scores))
                # show = visual_mask(img, mask_id)[-1]
                # write_im(get_related_path(path, args.input, args.output + "_show", suffixs='.jpg'), show)
            tqdm_bar.update()
            continue
            continue
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )
            visualized_output.save(os.path.splitext(os.path.join(args.output, os.path.basename(path)))[0] + ".jpg")
            continue
            breakpoint()

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)
        codec, file_ext = (
            ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
        )
        if codec == ".mp4v":
            warnings.warn("x264 codec not available, switching to mp4v")
        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + file_ext
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args.mp_idx, args.mp_num, args)