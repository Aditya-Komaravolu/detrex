# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import sys
import tempfile
import time
import warnings
import cv2
import tqdm
from detectron2.data import MetadataCatalog
from pathlib import Path
sys.path.insert(0, "./")  # noqa

from detectron2.data.datasets import register_coco_instances
register_coco_instances("snaglist_train", {}, "/home/aditya/snaglist_dataset_mar11/annotations/instances_train2017.json", "/home/aditya/snaglist_dataset_mar11/train2017")
register_coco_instances("snaglist_val", {}, "/home/aditya/snaglist_dataset_mar11/annotations/instances_val2017.json", "/home/aditya/snaglist_dataset_mar11/val2017")

from predictors import VisualizationDemo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger


# constants
WINDOW_NAME = "COCO detections"


def setup(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="detrex demo for visualizing customized inputs")
    parser.add_argument(
        "--config-file",
        default="projects/dino/configs/dino_r50_4scale_12ep.py",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--min_size_test",
        type=int,
        default=800,
        help="Size of the smallest side of the image during testing. Set to zero to disable resize in testing.",
    )
    parser.add_argument(
        "--max_size_test",
        type=float,
        default=1333,
        help="Maximum size of the side of the image during testing.",
    )
    parser.add_argument(
        "--img_format",
        type=str,
        default="RGB",
        help="The format of the loading images.",
    )
    parser.add_argument(
        "--metadata_dataset",
        type=str,
        default="coco_2017_val",
        help="The metadata infomation to be used. Default to COCO val metadata.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


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
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup(args)

    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.train.init_checkpoint)

    model.eval()

    if args.metadata_dataset == "snaglist_train":
        # MetadataCatalog.get("snaglist_train").set(thing_classes=["grp1", "cement_slurry", "chipping", "cracks", "exposed_reinforcement", "general_uneven", "honeycomb", "incomplete_deshuttering", "moisture_seepage", "snag_2" ])
        # MetadataCatalog.get("snaglist_train").set(thing_classes=["test", "cement_slurry", "chipping", "honeycomb", "incomplete_deshuttering" ])
        MetadataCatalog.get("snaglist_train").set(thing_classes=["cement_slurry", "honeycomb" ])
        # MetadataCatalog.get("snaglist_train").set(thing_classes=["snag_2", "cement_slurry", "chipping", "exposed_reinforcement", "general_uneven", "honeycomb", "incomplete_deshuttering", "moisture_seepage", "grp1", "cracks" ])

    demo = VisualizationDemo(
        model=model,
        min_size_test=args.min_size_test,
        max_size_test=args.max_size_test,
        img_format=args.img_format,
        metadata_dataset=args.metadata_dataset,
    )

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img, args.confidence_threshold)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                Path(args.output).mkdir(parents=True, exist_ok=True)
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
        try:
            # #validate input paths
            # validate_paths(
            #     args.video_input,
            #     config['odometry']['odom_for_blender'],
            #     config['odometry']['subsampled_odom_for_blender'],
            # )
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
                Path(args.output).mkdir(parents=True, exist_ok=True)
            for predictions,vis_frame,frame_count in tqdm.tqdm(demo.run_on_video(video = orig_video, frame_indexes= filtered_frame_numbers), total=len(filtered_frame_numbers)):
                
                # prediction_list.append(data)
                # print(prediction)
                # print(prediction.shape)
                predictions = predictions["instances"]
                scores = list(predictions.scores.cpu().numpy().astype(np.float)) if predictions.has("scores") else None
                classes = list(predictions.pred_classes.cpu().numpy().astype(np.float)) if predictions.has("pred_classes") else None
                
                if scores is not None:
                    print(predictions.pred_boxes.tensor.cpu().numpy().astype(np.float))
                    bbox = predictions.pred_boxes.tensor.cpu().numpy().astype(np.float).tolist()
                    # centroid of all the bounding boxes
                    centroid_list = []
                    for box in bbox:
                        centroid_list.append([(box[0]+box[2])/2, (box[1]+box[3])/2])
                
                data = { "frame_count" : frame_count , "predictions" : (scores,classes) , "bbox" : bbox, "centroid_list" : centroid_list}
                
                print("FRAME COUNT", frame_count)
                prediction_list.append(data)
                # frame_count = frame_count + 1
                # if frame_count > 1000:
                #     break
                print(frame_count,scores, classes)
                
                if args.output:
                    cv2.imwrite(f"{args.output}/{frame_count}.jpg",vis_frame) # Get the path from the hydra config
                    # output_file.write(vis_frame)
                else:
                    cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                    cv2.imshow(basename, vis_frame)
                    if cv2.waitKey(1) == 27:
                        break  # esc to quit
            with open(config['rgb_defect_pipeline']['file_paths']['detectron_prediction_dump'], "w+") as f:
                json.dump(list(prediction_list), f)
            orig_video.release()

        except Exception as e:
            logger.error(e)
            raise e
