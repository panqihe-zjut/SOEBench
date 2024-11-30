import torch
from safetensors.torch import load_file
from diffusers import AutoencoderKL, AutoPipelineForInpainting
import os
from tqdm import tqdm
import os
import cv2
import torch
import shutil
import random
import numpy as np
from PIL import Image
from typing import Union, Tuple
from PIL import Image, ImageDraw
import random

def get_module_kohya_state_dict(module, prefix: str, dtype: torch.dtype, adapter_name: str = "default"):
    kohya_ss_state_dict = {}
    for peft_key, weight in module.items():
        kohya_key = peft_key.replace("unet.base_model.model", prefix)
        kohya_key = kohya_key.replace("lora_A", "lora_down")
        kohya_key = kohya_key.replace("lora_B", "lora_up")
        kohya_key = kohya_key.replace(".", "_", kohya_key.count(".") - 2)
        kohya_ss_state_dict[kohya_key] = weight.to(dtype)
        # Set alpha parameter
        if "lora_down" in kohya_key:
            alpha_key = f'{kohya_key.split(".")[0]}.alpha'
            kohya_ss_state_dict[alpha_key] = torch.tensor(8).to(dtype)
    return kohya_ss_state_dict

def writeFile(path, info):
    f = open(path, 'w')
    f.write(info)
    f.close()


def prepare_pipe(sd, vae, lora=None):
    vae = AutoencoderKL.from_pretrained(vae, subfolder="vae", revision=None).to(torch.float16)
    pipe = AutoPipelineForInpainting.from_pretrained(sd, revision=None, torch_dtype=torch.float16)
    pipe.vae = vae
    pipe.safety_checker=None
    pipe.vae = vae
    pipe.to('cuda:0')

    if lora is not None:
        lora_weight     = load_file(lora)
        lora_state_dict = get_module_kohya_state_dict(lora_weight, "lora_unet", torch.float16)
        pipe.load_lora_weights(lora_state_dict)
        pipe.fuse_lora()
    return pipe

from PIL import Image, ImageDraw
def box2mask(box, size=(512, 512)):  
    image = Image.new('RGB', size, color='black')
    draw = ImageDraw.Draw(image)
    draw.rectangle([(box[0]*size[0], box[1]*size[1]), (box[2]*size[0], box[3]*size[1])], fill='white')
    return image


def checkThred(input):
    if input<0:
        return 0
    if input>1:
        return 1
    return input

def small2big(box, box_scale=2):
    x1, y1 ,x2, y2 = box[0], box[1], box[2], box[3]
    width = x2-x1
    height = y2-y1
    x11 = checkThred(x1-(box_scale-1)*width/2)
    x22 = checkThred(x2+(box_scale-1)*width/2)
    y11 = checkThred((1-box_scale)*y2+box_scale*y1)
    y22 = checkThred(y2)
    return [x11, y11, x22, y22]

def small2big2(box, box_scale=2):
    x1, y1 ,x2, y2 = box[0], box[1], box[2], box[3]
    thred = min(box_scale, (x2+x1)/(x2-x1), (y2+y1)/(y2-y1), (2-x1-x2)/(x2-x1), (2-y1-y2)/(y2-y1))
    width = x2-x1
    height = y2-y1
    x11 = checkThred(x1-(thred-1)*width/2)
    x22 = checkThred(x2+(thred-1)*width/2)

    y11 = checkThred(y1-(thred-1)*height/2)
    y22 = checkThred(y2+(thred-1)*height/2)


    return [x11, y11, x22, y22]






def disable_grads(model):
    for p in model.parameters():
        p.requires_grad = False

def disable_pipe_grads(pipe):
    pipe.vae.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    # pipe.image_encoder.requires_grad_(False)


def seed_torch(seed=0):
    random.seed(seed)   # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)    # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)   # numpy的随机性
    torch.manual_seed(seed)   # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed)   # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False   # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True   # 选择确定性算法

def vis_one_attn(attention, index, mask=None):


    attn = attention[:, index].sum(0)
    if mask is not None:
        attn = attn * mask
    image = 255* attn/attn.max()
    image = image.unsqueeze(-1).expand(*image.shape, 3)
    image = image.detach().cpu().numpy().astype(np.uint8)
    image = Image.fromarray(image).resize((512, 512))
    image = np.array(image)
    pil_img = Image.fromarray(image)
    

    return pil_img
    

def view_images(images, num_rows=1, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0
    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    # display(pil_img)
    return pil_img
    
    
def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", font_size)
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y ), font, 1, text_color, 2)
    return img



def show_attention(prompt, tokenizer, attnList,save_folder='',  name=None, step=0, iter=None, box=None):
    tokens  = tokenizer.encode(prompt[0])
    decoder = tokenizer.decode

    down_cross_attns, _, _, _, up_cross_attns, _ = attnList
    cross_attns = down_cross_attns + up_cross_attns

    attn_map_res = {}
    for i in cross_attns:
        sub_res = int(i.size()[1]**0.5)
        i = i.reshape( i.size()[0], sub_res, sub_res, i.size()[-1] )
        if sub_res in attn_map_res.keys():
            attn_map_res[sub_res].append(i)
        else:
            attn_map_res[sub_res] = [i]

    for key, value in attn_map_res.items():
        temp = torch.cat(value, dim=0)
        attn_map_res[key] = temp.sum(0)/temp.shape[0]


    for key,  value in attn_map_res.items():
        res_image = []
        for i in range(len(tokens)):
            image = value[:, :, i]
            image = 255 * image / image.max()
            image = image.unsqueeze(-1).expand(*image.shape, 3)
            image = image.detach().cpu().numpy().astype(np.uint8)
            image = Image.fromarray(image).resize((512, 512))
            # image = drawbox(image, box)
            image = np.array(image)
            image = text_under_image(image, decoder(int(tokens[i])))

            res_image.append(image)

        images = view_images(np.stack(res_image, axis=0))

        
        # print(os.path.join('images_p2p', name, str(key)+"_"+str(int(step))+'.jpg'))
        # if iter is not None:    
        #     images.save(os.path.join(bigsmall_folder, str(key)+"_"+str(int(step))+"_"+str(iter)+'.jpg'))
        # else:
        #     images.save(os.path.join(bigsmall_folder, str(key)+"_"+str(int(step))+'.jpg'))

    return images

        