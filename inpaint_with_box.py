import torch
from accelerate.utils import set_seed
from tqdm import tqdm
import utils
import os

import loss_tf

output_type = "pil"
callback = None
callback_steps = 1

# set_seed(1111)
# utils.seed_torch(1111)

def benchGeneration(sd, valdataset, savefolder, subfolder="", param_dict=None ):
    
    loss_dict = {
        'upone':loss_tf.upone,
        'uptwo':loss_tf.uptwo,
        'downone':loss_tf.downone,
        'downtwo':loss_tf.downtwo,
        'updownone':loss_tf.updownone,
        'updowntwo':loss_tf.updowntwo,
    }
    num_inference_steps=param_dict['num_inference_steps']
    box_scale = param_dict['box_scale']
    tf_iters  = param_dict['tf_iters']
    tf_step_s = param_dict['tf_step_s']
    tf_step   = param_dict['tf_step']
    loss_scale= param_dict['loss_scale']
    grad_scale= param_dict['grad_scale']
    labeltype = param_dict['labeltype']
    losstype  = param_dict['losstype']
    res_list  = param_dict['res_list']
    attn_list = param_dict['attn_list']
    loss_num  = param_dict['loss_num']

    pipe = sd
    savefolder = os.path.join(savefolder, subfolder)
    os.makedirs(savefolder, exist_ok=True)
    os.makedirs(os.path.join(savefolder, 'box'), exist_ok=True)
    os.makedirs(os.path.join(savefolder, 'caption'), exist_ok=True)
    os.makedirs(os.path.join(savefolder, 'image_ori'), exist_ok=True)
    os.makedirs(os.path.join(savefolder, 'image_gen'), exist_ok=True)
    os.makedirs(os.path.join(savefolder, 'image_mask'), exist_ok=True)
    os.makedirs(os.path.join(savefolder, 'image_ori_crop'), exist_ok=True)
    os.makedirs(os.path.join(savefolder, 'image_gen_crop'), exist_ok=True)
    f = open(os.path.join(savefolder, 'description.txt'),'w')
    for key, value in param_dict.items():
        f.write(key+":"+str(value)+"\n") 
    f.close()


    for index in tqdm(range(0, len(valdataset)),  ncols=50):
        condition_kwargs = {}
        with torch.no_grad():
            OneInfo = valdataset[index]
            # image_init, text, mask, prompt  = OneInfo['image'], OneInfo['label'], OneInfo['mask'], OneInfo['label']
            # box_small  = OneInfo['box']
            image_init, text, mask, prompt  = OneInfo['croped_image'], OneInfo['croped_label'], OneInfo['croped_mask'], OneInfo['croped_label']
            box_small  = OneInfo['croped_box']

            imageInfo = OneInfo['imageid']
            imageid = imageInfo.split('/')[1].split('.')[0]
            box_big    = utils.small2big2(box_small, box_scale=box_scale)
            mask_small_i = utils.box2mask(box_small)
            mask_big_i   = utils.box2mask(box_big)

            device = pipe._execution_device
            do_classifier_free_guidance = 7.5 > 1.0
            prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
                prompt, device, 1, True, negative_prompt=None,prompt_embeds=None,
                negative_prompt_embeds=None,lora_scale=None,clip_skip=None,)

            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            pipe.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps, num_inference_steps = pipe.get_timesteps(num_inference_steps=num_inference_steps, strength=1.0, device=device)
            latent_timestep = timesteps[:1].repeat(1 * 1)
            is_strength_max = True
            init_image = pipe.image_processor.preprocess(image_init, 512, 512)
            init_image = init_image.to(dtype=torch.float16)
            num_channels_latents = pipe.vae.config.latent_channels
            num_channels_unet = pipe.unet.config.in_channels
            latents_output = pipe.prepare_latents(1, num_channels_latents, 512, 512, prompt_embeds.dtype, device,None, None, image=init_image, 
                                                timestep=latent_timestep,is_strength_max=is_strength_max, return_noise=True, return_image_latents=False)
            latents, noise = latents_output
            latent_smaller  = latents.clone()
            noise_small     = noise.clone()
            latent_bigger   = latents.clone()
            noise_big       = noise.clone()
            mask_condition_small = pipe.mask_processor.preprocess(mask_small_i, 512, 512)
            mask_condition_big   = pipe.mask_processor.preprocess(mask_big_i,   512, 512)
            masked_image_small   = init_image * (mask_condition_small<0.5)
            masked_image_big     = init_image * (mask_condition_big  <0.5)
            mask_small, masked_image_latents_small = pipe.prepare_mask_latents(mask_condition_small, masked_image_small, 1, 512, 512, prompt_embeds.dtype, device, None, True)
            mask_big, masked_image_latents_big     = pipe.prepare_mask_latents(mask_condition_big, masked_image_big, 1, 512, 512, prompt_embeds.dtype, device, None, True)
            extra_step_kwargs = pipe.prepare_extra_step_kwargs(None, 0.0)
            num_warmup_steps  = len(timesteps) - num_inference_steps * pipe.scheduler.order

        for i, t in enumerate(timesteps):

            tf_iter = 1 
            loss = torch.tensor(10000)
            while (((i>=tf_step_s and i<tf_step_s+tf_step) and tf_iter<=tf_iters) ) and loss.item()/loss_scale>0.2:
                tf_iter += 1 
                latent_smaller = latent_smaller.requires_grad_(True)
                latent_model_input_smaller = torch.cat([latent_smaller] * 2) 
                latent_model_input_smaller = pipe.scheduler.scale_model_input(latent_model_input_smaller, t)
                latent_model_input_smaller = torch.cat([latent_model_input_smaller, mask_small, masked_image_latents_small], dim=1)
                noise_pred_smaller, down_cross_attns_small, _, mid_cross_attns_small, _, up_cross_attns_small, _ = pipe.unet(
                        latent_model_input_smaller, t, encoder_hidden_states=prompt_embeds,cross_attention_kwargs=None,return_attn=True)
                noise_pred_smaller = noise_pred_smaller.sample
                small_attn_list = [down_cross_attns_small, mid_cross_attns_small, up_cross_attns_small]
                loss = loss_tf.attention_loss(small_attn_list, [box_small], text, res_list=res_list, attn_list=attn_list, loss_num=loss_num)*loss_scale
                grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latent_smaller])[0]  
                latent_smaller = latent_smaller - grad_cond*grad_scale
            
            with torch.no_grad():
                latent_model_input_smaller = torch.cat([latent_smaller] * 2) if do_classifier_free_guidance else latent_smaller
                latent_model_input_smaller = pipe.scheduler.scale_model_input(latent_model_input_smaller, t)
                latent_model_input_smaller = torch.cat([latent_model_input_smaller, mask_small, masked_image_latents_small], dim=1)
                noise_pred_smaller, down_cross_attns, _, mid_cross_attns, _, up_cross_attns, _ = pipe.unet(
                    latent_model_input_smaller, t, encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=None, return_dict=False,return_attn=True)
                noise_pred_smaller = noise_pred_smaller.sample
                small_attn_list = [down_cross_attns, mid_cross_attns, up_cross_attns]
                noise_pred_uncond, noise_pred_text = noise_pred_smaller.chunk(2)
                noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
                latent_smaller = pipe.scheduler.step(noise_pred, t, latent_smaller, **extra_step_kwargs, return_dict=False)[0]

                if i==len(timesteps)-1:                        
                    utils.writeFile(os.path.join(savefolder, 'caption', str(index)+"_"+imageid+'.txt'), text)
                    utils.writeFile(os.path.join(savefolder, 'box', str(index)+"_"+imageid+'.txt'), 
                                str(box_small[0])+" "+str(box_small[1])+" "+str(box_small[2])+" "+str(box_small[3]))
                    image_gen = pipe.vae.decode(latent_smaller / pipe.vae.config.scaling_factor, return_dict=False, **condition_kwargs)[0]
                    do_denormalize = [True] * image_gen.shape[0]
                    image_gen = pipe.image_processor.postprocess(image_gen, output_type=output_type, do_denormalize=do_denormalize)
                    image_init.save(os.path.join(savefolder, 'image_ori',str(index)+"_"+imageInfo.split('/')[1]))
                    image_gen[0].save(os.path.join(savefolder, 'image_gen',str(index)+"_"+imageInfo.split('/')[1]))
                    mask_small_i.save(os.path.join(savefolder, 'image_mask', str(index)+"_"+imageInfo.split('/')[1]))
                    box_center = [(box_small[2] + box_small[0])/2, (box_small[3]+box_small[1])/2]
                    new_length = max(box_small[2]-box_small[0],box_small[3]-box_small[1])/2*1.1
                    crop_box = [box_center[0]-new_length, box_center[1]-new_length, box_center[0]+new_length, box_center[1]+new_length]
                    image_init.crop((crop_box[0]*512, crop_box[1]*512, crop_box[2]*512, crop_box[3]*512)).save(os.path.join(savefolder, 'image_ori_crop',str(index)+"_"+imageInfo.split('/')[1]))
                    image_gen[0].crop((crop_box[0]*512, crop_box[1]*512, crop_box[2]*512, crop_box[3]*512)).save(os.path.join(savefolder, 'image_gen_crop',str(index)+"_"+imageInfo.split('/')[1]))

 
def getnewsavefolder():
    temps = os.listdir('./exp_baseline')
    folderindex = sorted([int(temp[3:]) for temp in temps if temp.startswith('exp')])[-1]
    newsavefoldername = os.path.join('./exp_baseline', 'exp'+str(folderindex+1))
    return newsavefoldername











    
#1---------------------Training Free Methods---------------------
from sd_inpaint import StableDiffusionInpaintPipeline
from dataset.benchdata import benchdata
sd_weight     = '/home/public/panqihe/checkpoints/stable-diffusion/stable-diffusion-inpainting'
inpaint_model = StableDiffusionInpaintPipeline.from_pretrained(sd_weight, revision=None, torch_dtype=torch.float16)
inpaint_model.to('cuda')
inpaint_model.set_progress_bar_config(disable=True)
utils.disable_pipe_grads(inpaint_model)



param_dict = {
    'num_inference_steps':25 ,
    'box_scale':2,
    'tf_iters' :5,
    'tf_step_s' : 0,
    'tf_step'   : 5,
    'loss_scale' : 1, 
    'grad_scale' : 1,
    'labeltype' :'label',
    'losstype'  : 'updowntwo',
    'res_list'  : [8, 16],
    'attn_list' : ['down', 'mid', 'up'],
    'loss_num'  : 1
}
save_folder   = 'exp_baseline/speedTest'
openimageval  = benchdata(dataname='openimageval' , transformFlag=False ,thred_size=[1/8, 1/6], labeltype=param_dict['labeltype']).dataset
benchGeneration(inpaint_model, openimageval, savefolder = save_folder, subfolder='openimageval',param_dict=param_dict)
cocoval       = benchdata(dataname='cocoval' , transformFlag=False ,thred_size=[1/8, 1/6], labeltype=param_dict['labeltype']).dataset
benchGeneration(inpaint_model, cocoval, savefolder = save_folder, subfolder='cocoval',param_dict=param_dict)



# param_dict = {
#     'num_inference_steps':20 ,
#     'box_scale':2,
#     'tf_iters' :0,
#     'tf_step_s' : 0,
#     'tf_step'   : 0,
#     'loss_scale' : 0, 
#     'grad_scale' : 0,
#     'labeltype' :'colorlabel',
#     'losstype'  : 'updowntwo',
#     'res_list'  : [8, 16, 32],
#     'attn_list' : ['down', 'mid', 'up'],
#     'loss_num'  : 1
# }
# save_folder   = 'exp_baseline/zoomin_box_colorlabel'
# openimageval  = benchdata(dataname='openimageval' , transformFlag=False ,thred_size=[1/8, 1/6], labeltype=param_dict['labeltype']).dataset
# benchGeneration(inpaint_model, openimageval, savefolder = save_folder, subfolder='openimageval',param_dict=param_dict)
# cocoval       = benchdata(dataname='cocoval' , transformFlag=False ,thred_size=[1/8, 1/6], labeltype=param_dict['labeltype']).dataset
# benchGeneration(inpaint_model, cocoval, savefolder = save_folder, subfolder='cocoval',param_dict=param_dict)
