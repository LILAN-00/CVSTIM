import os
from PIL import Image
import torch
from tqdm import tqdm
import random
import numpy as np
import torch.backends.cudnn as cudnn
import argparse
from torchvision import transforms
import json
from itertools import combinations
import spacy

## minigpt4 setting
from minigpt4.common.dist_utils import get_rank
from minigpt4.models import load_preprocess
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from utils.keyw_extract import extract_hallucinated
from GroundingDINO.groundingdino.util.inference import load_model
from utils.locate_obj import locate_obj
from utils.pope_recoder import recorder
from utils.direct_focus import direct_img_focus
from utils.construct_qu import construct_prompt

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

def setup_seeds(config):
    seed = config.run_cfg.seed+get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

parser = argparse.ArgumentParser(description="CHAIR evaluation on MLLMs.")
parser.add_argument("--model", type=str, default='llava-1.5', help="model")  
parser.add_argument("--dino_model", type=str, default='GroundingDINO/', help="dino directory path")  
parser.add_argument("--gpu_id", type=int, default=6, help="specify the gpu to load the model.")
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
parser.add_argument("--object2id", type=str, default='config/object2id.json', help="path for saving object2id")  
parser.add_argument("--co_occurance", type=str, default='config/co_occur_matrix.npy', help="path for saving co_occurance")  
parser.add_argument("--gamma", type=int, default=0.5, help="threshold for co-occurance")  
parser.add_argument("--sample", action='store_true', help='sample strategy')
parser.add_argument("--beam", type=int, default=1)
parser.add_argument("--ours", action='store_true', help='baseline or our method')
args = parser.parse_known_args()[0]
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
args.cfg_path = MODEL_EVAL_CONFIG_PATH[args.model]
cfg = Config(args)
setup_seeds(cfg)
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

print('Initializing Model')
model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to(device)
model.eval()
processor_cfg = cfg.get_config().preprocess
processor_cfg.vis_processor.eval.do_normalize = False
vis_processors, txt_processors = load_preprocess(processor_cfg)
print(vis_processors["eval"].transform)
dino_model=load_model(f"{args.dino_model}/groundingdino/config/GroundingDINO_SwinB_cfg.py", f"{args.dino_model}/weights/groundingdino_swinb_cogcoor.pth")
nlp = spacy.load("en_core_web_sm")
print("Done!")

with open(args.object2id,'r') as f:
    object2id=json.load(f)
id2object = {v: k for k, v in object2id.items()}
co_matrix=np.load(args.co_occurance)
if args.ours:
    my_template="Do NOT mention these objects: {text}."

mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
norm = transforms.Normalize(mean, std)

img_files = os.listdir(args.data_path)
random.shuffle(img_files)
with open(args.data_path + '../annotations/instances_val2014.json', 'r') as f:
    lines = f.readlines()
coco_anns = json.loads(lines[0])

img_dict = {}
categories = coco_anns["categories"]
category_names = [c["name"] for c in categories]
category_dict = {int(c["id"]): c["name"] for c in categories}
for img_info in coco_anns["images"]:
    img_dict[img_info["id"]] = {"name": img_info["file_name"], "anns": []}
for ann_info in coco_anns["annotations"]:
    img_dict[ann_info["image_id"]]["anns"].append(
        category_dict[ann_info["category_id"]]
    )

method_str='ours' if args.ours else 'base'
base_dir=f'./log/{args.model}/chair/{method_str}'
existing_ids=[]
if not os.path.exists(base_dir):
    os.mkdir(base_dir)
save_fp=f'{base_dir}/gamma_{args.gamma}-beam_{args.beam}.jsonl' if args.ours else f'{base_dir}/beam_{args.beam}.jsonl'
if os.path.exists(save_fp):
    with open(save_fp) as f:
        for line in f:
            obj=json.loads(line)
            existing_ids.append(obj['image_id'])

for img_id in tqdm(range(len(img_files))):
    if img_id == 500:
        break
    img_file = img_files[img_id]
    img_id = int(img_file.split(".jpg")[0][-6:])
    if img_id in existing_ids:
        continue
    img_info = img_dict[img_id]
    assert img_info["name"] == img_file
    img_anns = set(img_info["anns"])
    img_save = {}
    img_save["image_id"] = img_id

    image_path = args.data_path + img_file
    raw_image = Image.open(image_path).convert("RGB")
    image = vis_processors["eval"](raw_image).unsqueeze(0)
    image = image.to(device)

    qu = "Please describe this image in detail."
    img_save["input"]=qu
    template = INSTRUCTION_TEMPLATE[args.model]
    qu = template.replace("<question>", qu)
    
    with torch.inference_mode():
        with torch.no_grad():
            out = model.generate(
                {"image": norm(image), "prompt":qu}, 
                use_nucleus_sampling=args.sample,
                num_beams=args.beam,    
                max_new_tokens=512,
            )
    caption1=out[0] 
    img_save["caption"] = caption1
    if not args.ours:
        with open(save_fp, "a+") as f:
            json.dump(img_save, f)
            f.write('\n')
        continue
    word_lst=extract_hallucinated(caption1,nlp,object2id.keys()) 
    id_lst = [object2id[obj] for obj in word_lst]
    hallucination_objs=set()   
    if len(id_lst)==1:
        hallucination_objs.add(id_lst[0])
    else:
        for obj1, obj2 in combinations(id_lst, 2):
            if co_matrix[obj1][obj2] > args.gamma:
                hallucination_objs.update([obj1, obj2])
    if len(hallucination_objs)==0 or len(hallucination_objs)==1:
        pass
    else:
        all_objs=[id2object[obj] for obj in hallucination_objs]
        boxes=locate_obj(image_path,all_objs,dino_model)   
        next_prompt='Is there a/an {obj} in the image?'
        next_prompt = template.replace("<question>", next_prompt)
        all_images=[direct_img_focus(raw_image,box,vis_processors["eval"].transform.crop_size['width']) for box in boxes]
        all_images=[vis_processors["eval"](img).unsqueeze(0).to(device) for img in all_images]
        real_objs=[]
        for next_image,next_obj in zip(all_images,all_objs):
            with torch.inference_mode():
                with torch.no_grad():
                    out = model.generate(
                        {"image": norm(next_image), "prompt":next_prompt.format(obj=next_obj)}, 
                        use_nucleus_sampling=args.sample,  
                        num_beams=args.beam,
                        max_new_tokens=20,
                    )
                    if recorder(out[0]):
                        real_objs.append(next_obj)
        if not len(real_objs)==len(all_objs):
            last_qu='Please describe this image in detail. {text}'
            last_text=construct_prompt(my_template,all_objs,real_objs)
            last_qu=last_qu.format(text=last_text)
            img_save['input']=last_qu
            last_qu = template.replace("<question>", last_qu)
            with torch.inference_mode():
                with torch.no_grad():
                    out = model.generate(
                        {"image": norm(image), "prompt":last_qu}, 
                        use_nucleus_sampling=args.sample,   
                        num_beams=args.beam,  
                        max_new_tokens=512,
                    )
            img_save["caption"] = out[0]
    with open(save_fp, "a+") as f:
        json.dump(img_save, f)
        f.write('\n')