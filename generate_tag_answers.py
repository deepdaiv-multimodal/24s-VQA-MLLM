import os
import sys
import random

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as TS
import matplotlib.pyplot as plt 
from PIL import Image, ImageDraw, ImageFont
import argparse
import re
import json

# current_dir = os.path.dirname(os.path.abspath(__file__))
# groundingdino_dir = os.path.abspath(os.path.join(current_dir, 'GroundingDINO'))
# sys.path.append(groundingdino_dir)

# from GroundingDINO import groundingdino
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

from ram import inference_ram
from ram.models import ram


device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(
        clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model

def get_grounding_output(model, image, caption, box_threshold, text_threshold, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(
            logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases

def draw_box(box, draw, label):
    # random color
    color = tuple(np.random.randint(0, 255, size=3).tolist())
    line_width = int(max(4, min(20, 0.006*max(draw.im.size))))
    draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline=color,  width=line_width)

    if label:
        font_path = os.path.join(
            cv2.__path__[0], 'qt', 'fonts', 'DejaVuSans.ttf')
        font_size = int(max(12, min(60, 0.02*max(draw.im.size))))
        font = ImageFont.truetype(font_path, size=font_size)
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((box[0], box[1]), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (box[0], box[1], w + box[0], box[1] + h)
        draw.rectangle(bbox, fill=color)
        draw.text((box[0], box[1]), str(label), fill="white", font=font)

        draw.text((box[0], box[1]), label, font=font)

def inference(raw_image, specified_tags, tagging_model, grounding_dino_model):

    print(f"Start processing, image size {raw_image.size}")
    raw_image = raw_image.convert("RGB")

    # set threshold
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    iou_threshold = args.iou_threshold

    # run tagging model
    normalize = TS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = TS.Compose([
        TS.Resize((384, 384)),
        TS.ToTensor(),
        normalize
    ])

    image = raw_image.resize((384, 384))
    image = transform(image).unsqueeze(0).to(device)

    # Currently ", " is better for detecting single tags
    # while ". " is a little worse in some case
    res = inference_ram(image, tagging_model)
    tags = res[0].strip(' ').replace('  ', ' ').replace(' |', ',')

  
    # run groundingDINO
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image, _ = transform(raw_image, None)  # 3, h, w

    boxes_filt, scores, pred_phrases = get_grounding_output(
        grounding_dino_model, image, tags, box_threshold, text_threshold, device=device
    )
    print("GroundingDINO finished")

    size = raw_image.size
    H, W = size[1], size[0]

    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()

    # use NMS to handle overlapped boxes
    print(f"Before NMS: {boxes_filt.shape[0]} boxes")
    nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
    boxes_filt = boxes_filt[nms_idx]
    pred_phrases = [pred_phrases[idx] for idx in nms_idx]
    print(f"After NMS: {boxes_filt.shape[0]} boxes")

    # detection_prompt = ", ".join(
    #     [f"{label}: [{box[0]:.3f}, {box[1]:.3f}, {box[2]:.3f}, {box[3]:.3f}]" for box, label in zip(boxes_filt, pred_phrases)]
    # )

    # draw output image
    # image_draw = ImageDraw.Draw(raw_image)

    detection_prompts = []
    for box, label in zip(boxes_filt, pred_phrases):
        detection_prompts.append(f"{label}: [{box[0]:.3f}, {box[1]:.3f}, {box[2]:.3f}, {box[3]:.3f}]")
        # draw_box(box, image_draw, label)

    detection_prompt = ", ".join(detection_prompts)
    
    # out_image = raw_image.convert('RGBA')

    # return
    # return tags.replace(", ", " | "), out_image, detection_prompt
    return detection_prompt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image_path', type=str, default='/content/drive/MyDrive/24s-deep-daiv/ok-vqa/train2014_vqa')
    parser.add_argument('--config_file', type=str, default='/content/drive/MyDrive/24s-deep-daiv/recognize-anything/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py')
    parser.add_argument('--ram_checkpoint', type=str, default='/content/drive/MyDrive/24s-deep-daiv/recognize-anything/pretrained/ram_swin_large_14m.pth')
    parser.add_argument('--grounded_checkpoint', type=str, default='/content/drive/MyDrive/24s-deep-daiv/recognize-anything/Grounded-Segment-Anything/groundingdino_swint_ogc.pth')
    parser.add_argument('--save_image_path', type=str, default='/content/drive/MyDrive/24s-deep-daiv/tagging_image')
    parser.add_argument('--save_json_path', type=str, default='/content/drive/MyDrive/24s-deep-daiv/okvqa_train_tagging.json')

    parser.add_argument('--box_threshold', type=int, default=0.25)
    parser.add_argument('--text_threshold', type=int, default=0.2)
    parser.add_argument('--iou_threshold', type=int, default=0.5)

    # args = parser.parse_args()
    args = parser.parse_args()

    # load RAM
    ram_model = ram(pretrained=args.ram_checkpoint, image_size=384, vit='swin_l')
    ram_model.eval()
    ram_model = ram_model.to(device)

    # load gounding dino
    grounding_dino_model = load_model(args.config_file, args.grounded_checkpoint, device=device)

    def extract_image_id(image_name):
        match = re.search(r'_(\d+)\.jpg$', image_name)
        if match:
            return match.group(1).lstrip('0')  # Strip leading zeros
        return None

    # inference
    in_img_paths = os.listdir(args.input_image_path)
    results = {}

    for in_img_path in in_img_paths:
      in_img = Image.open(os.path.join(args.input_image_path, in_img_path))
      image_id = extract_image_id(in_img_path)

      # ram_tags, ram_out_image, detection_prompt = inference(in_img, None, ram_model, grounding_dino_model)
      detection_prompt = inference(in_img, None, ram_model, grounding_dino_model)
      results[image_id] = detection_prompt

      # print(results)

    with open(args.save_json_path, 'w') as outfile:
      json.dump(results, outfile, indent=4)

    # print(f"Tags: {ram_tags}")
    # print(f"Detection prompt: {detection_prompt}")

    # ram_out_image.thumbnail((500, 500))
    # # display(ram_out_image)
    # plt.imshow(ram_out_image)
    # plt.axis('off')
    # plt.show()