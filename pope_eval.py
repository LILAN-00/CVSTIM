import argparse
import os
import random
from PIL import Image

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.utils import save_image

from pope_loader import POPEDataSet
from minigpt4.common.dist_utils import get_rank
from minigpt4.models import load_preprocess

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from utils.keyw_extract import extract_hallucinated
from utils.direct_focus import direct_img_focus
from utils.locate_obj import locate_obj
from GroundingDINO.groundingdino.util.inference import load_model

MODEL_EVAL_CONFIG_PATH = {
    "instructblip": "eval_configs/instructblip_eval.yaml",
    "llava-1.5": "eval_configs/llava-1.5_eval.yaml",
    "minigpt4": "eval_configs/minigpt4_eval.yaml",
    "shikra": "eval_configs/shikra_eval.yaml",
}

INSTRUCTION_TEMPLATE = {
    "instructblip": "<ImageHere><question>",
    "llava-1.5": "USER: <ImageHere> <question> ASSISTANT:",
    "minigpt4": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "shikra": "USER: <im_start><ImageHere><im_end> <question> ASSISTANT:",
}

POPE_PATH = {
    "random": "pope_coco/coco_pope_random.json",
    "popular": "pope_coco/coco_pope_popular.json",
    "adversarial": "pope_coco/coco_pope_adversarial.json",
}

def parse_args():
    parser = argparse.ArgumentParser(description="POPE evaluation on MLLMs.")
    parser.add_argument("--model", type=str, default='llava-1.5', help="model")
    parser.add_argument("--dino_model", type=str, default='GroundingDINO/', help="dino model path")  
    parser.add_argument("--pope_type", type=str, default='popular', help="model")
    # parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--data_path", type=str, default="dataset/val2014/", help="data path")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="num workers")
    parser.add_argument("--beam", type=int, default=1)
    parser.add_argument("--sample", action='store_true')
    parser.add_argument("--ours", action='store_true')
    args = parser.parse_args()
    return args

def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def print_acc(pred_list, label_list):
    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print('TP\tFP\tTN\tFN\t')
    print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2*precision*recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print('Yes ratio: {}'.format(yes_ratio))


def recorder(out, pred_list):
    NEG_WORDS = ["No", "not", "no", "NO"]
    for line in out:

        line = line.replace('.', '')
        line = line.replace(',', '')
        words = line.split(' ')
        if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
            pred_list.append(0)
        else:
            pred_list.append(1)
    
    return pred_list


def main():

    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    args.cfg_path = MODEL_EVAL_CONFIG_PATH[args.model]
    args.pope_path = POPE_PATH[args.pope_type]
    cfg = Config(args)

    setup_seeds(cfg)
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    print('Initializing Model')
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)
    model.eval()
    vis_processors, txt_processors = load_preprocess(cfg.get_config().preprocess)
    print(vis_processors["eval"].transform)
    dino_model=load_model(f"{args.dino_model}/groundingdino/config/GroundingDINO_SwinB_cfg.py", f"{args.dino_model}/weights/groundingdino_swinb_cogcoor.pth")
    print("Done!")

    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    norm = transforms.Normalize(mean, std)
    # load pope data
    pope_dataset = POPEDataSet(
        pope_path=args.pope_path, 
        data_path=args.data_path, 
        trans=vis_processors["eval"]
    )
    pope_loader = torch.utils.data.DataLoader(
        pope_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        drop_last=False
    )

    print ("load data finished")


    print("Start eval...")
    pred_list, pred_list_s, label_list = [], [], []
    for batch_id, data in tqdm(enumerate(pope_loader), total=len(pope_loader)):
        qu = data["query"]
        label = data["label"]
        label_list = label_list + list(label)
        raw_image=Image.open(data['path'][0]).convert("RGB")
        template = INSTRUCTION_TEMPLATE[args.model]
        qu = [template.replace("<question>", q) for q in qu]
        if args.ours:
            obj_lst=extract_hallucinated(qu[0],None) 
            obj_box=locate_obj(data["path"][0],obj_lst,dino_model)    
            sub_image=direct_img_focus(raw_image,obj_box[0],vis_processors["eval"].transform.crop_size['width'])
        else:
            sub_image=raw_image
        sub_image=pope_dataset.trans(sub_image).unsqueeze(0)
        sub_image=sub_image.to(device)
        label = torch.Tensor(label).to(device)
        with torch.inference_mode():
            with torch.no_grad():
                out = model.generate(
                    {"image": norm(sub_image), "prompt":qu[0]}, 
                    use_nucleus_sampling=args.sample, 
                    num_beams=args.beam,
                    max_new_tokens=10,
                )
                pred_list = recorder(out, pred_list)
                for line in out:
                    print(line)

    if len(pred_list) != 0:
        print_acc(pred_list, label_list)
    if len(pred_list_s) != 0:
        print_acc(pred_list_s, label_list)








if __name__ == "__main__":
    main()