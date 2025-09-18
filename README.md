# ✨CVSTIM

Official implemention of "CVSTIM: Mitigating Object Hallucination in MLLMs via Co-occurrence Guided Visual Stimulation."

## 🔍Key Contributions
* Inspired by biological findings, we propose and experimentally validate that localized visual enhancement can mitigate object hallucination (OH). 
* We introduce CVSTIM, a novel framework that leverages a Co-occurrence Score Matrix to identify high-risk hallucinated objects and applies Visual Stimulus Augmentation (VSA) to enhance their visual details. 
## 💡Methodology
To mitigate OH, we introduce a novel, training-free, post-hoc approach called **C**o-occurrence Guided **V**isual **Stim**ulation (CVSTIM).

Our framework works as follows: Given an image, the MLLM generates the initial description. (1) We construct the Co-occurrence Score Matrix to derive high-risk hallucinated objects. (2) VSA is employed to get the focused images. (3) The focused images and the formulated questions are fed into the MLLM to obtain visual evidence, which is used to revise the initial response. 

🖼️ The overview of our proposed CVSTIM is shown below.

<div align="left">
<img src="fig\framework.png" width="1000px">
</div>

## 📚Environments

To install, run the following commands to install the required packages:

```sh
git clone https://github.com/LILAN-00/CVSTIM
cd CVSTIM
conda env create -f environment.yml
conda activate cvstim
```

We employ [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) to locate high-risk hallucinated objects. Download Grounding DINO and locate it under the repository root path.
```sh
git clone https://github.com/IDEA-Research/GroundingDINO.git
```
Download pre-trained model weights for Grounding DINO:
```sh
cd GroundingDINO
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
cd ../..
```

## 📑Data and Model Preparation
The evaluation requires for MSCOCO 2014 dataset. Please download [here](https://cocodataset.org/#download) and extract it under `./dataset`.

Besides, it needs you to prepare the following models:
* Download [LLaVA-1.5 7B](https://huggingface.co/liuhaotian/llava-v1.5-7b) and specify it at [Line 13](eval_configs/llava-1.5_eval.yaml#L13) of eval_configs/llava-1.5_eval.yaml.
* Download [Vicuna 7B v1.1](https://github.com/lm-sys/FastChat) and specify it at [Line 25](minigpt4/configs/models/blip2_instruct_vicuna7b.yaml#25) of minigpt4/configs/models/blip2_instruct_vicuna7b.yaml.
* Download [Vicuna 7B v0](https://huggingface.co/Vision-CAIR/vicuna-7b/tree/main) and specify it at [Line 18](minigpt4/configs/models/minigpt4_vicuna0.yaml#18) of minigpt4/configs/models/minigpt4_vicuna0.yaml.
* Download [MiniGPT-4 7B pretrained weights](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing) and specify it at [Line 8](eval_configs/minigpt4_eval.yaml#8) of eval_configs/minigpt4_eval.yaml.
* Download [Shikra 7B](https://github.com/shikras/shikra#checkpoint) and specify it at [Line 14](eval_configs/shikra_eval.yaml#14) of eval_configs/shikra_eval.yaml.

## 🔧Key Arguments
| Argument                | Example            | Description                                                  |
| ----------------------- | ------------------ | ------------------------------------------------------------ |
| `--model`               | `llava-1.5`        | Specify the MLLM model, this codebase supports `instructblip`, `llava-1.5`, `minigpt4`, `shikra`. |
| `--dino_model`               | `GroundingDINO/`        | Path to the GroundingDINO directory. |
| `--data-path`           | `dataset/val2014/` | Path to the dataset file or folder. |
| `--object2id`        | `config/object2id.json`  | Path to the `object2id.json` file. |
| `--co_occurance`           | `config/co_occur_matrix.npy` | Path to the `co_occur_matrix.npy` file.     |
| `--gamma` | `0.5`                | The value for the threshold gamma. |
| `--pope-type`           | `random`           | Type for POPE evaluation, supports `random`, `popular`, `adversarial`. |


## ⚙️Preprocess
Construct the co-occurrence frequency matrix $C$ using the base training set.
```sh
python utils/prepare_data/construct_freq.py
```
Produce `./config/object2id.json` file.
```sh
python utils/prepare_data/ex_obj2id.py
```
Construct the Co-occurrence Score Matrix $\mathcal{M}$ in `./config/co_occur_matrix.npy`.
```sh
python utils/prepare_data/ex_matrix.py
```

## 📊CHAIR evaluation
* Generate the MLLM's responses and save them in a jsonl file:
```sh
python chair_eval.py --model MODEL_NAME --dino_model DINO_PATH --gpu-id GPU_ID --data_path /path/to/COCO --object2id ID_FILE --co_occurance MATRIX_FILE --ours
```
Note: Please check out our released results in `log/{MODEL_NAME}/chair/ours` for reproduction.

* Calculate CHAIR using the generated jsonl file:
```sh
python chair_calculate.py --cap_file /path/to/jsonl --image_id_key image_id --caption_key caption --coco_path /path/to/COCO/annotations_trainval2014/annotations/ --save_path /path/to/save/jsonl
```
## 📊POPE evaluation
```sh
python pope_eval.py --model MODEL_NAME --dino_model DINO_PATH --gpu-id GPU_ID --data_path /path/to/COCO --pope-type random --ours
```
