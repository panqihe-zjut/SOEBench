import utils
def boxToPos(box, size=64):
    xmin_px = int(box[0] * size)
    ymin_px = int(box[1] * size)
    xmax_px = int(box[2] * size)
    ymax_px = int(box[3] * size)
    width = xmax_px-xmin_px
    height= ymax_px-ymin_px
    small_newlength = max(width, height)
    # xmin_px = xmax_px-small_newlength
    # ymin_px = ymax_px-small_newlength
    xmax_px = xmin_px+small_newlength
    ymax_px = ymin_px+small_newlength
    big_xmin, big_ymin, big_xmax, big_ymax = xmin_px, ymin_px, xmax_px, ymax_px
    return big_xmin, big_ymin, big_xmax, big_ymax

def resizeAttn(attnlist, resize=None):
    temp = []
    for item in attnlist:
        b, i, j = item.shape
        sub_res = int(math.sqrt(i))
        item = item.reshape(b, sub_res, sub_res, j).permute(3, 0, 1, 2)
        if sub_res <= 64:
            if resize is not None:
                temp.append(F.interpolate(item, resize, mode='bilinear'))
            else:
                temp.append(item)
    return temp


def checkPos(box,  size=64):
    xmin_px = int(box[0] * size)
    ymin_px = int(box[1] * size)
    xmax_px = int(box[2] * size)
    ymax_px = int(box[3] * size)
    width = xmax_px-xmin_px
    height= ymax_px-ymin_px

    if width==0 or height==0:
        return False
    else:
        return True

import math
import torch
import torch.nn.functional as F
import torch.nn as nn
def updowntwo(big_attn_list, small_attn_list, boxes_bigger,  boxes_smaller):

    down_cross_attns_big, down_self_attns_big, mid_cross_attns_big, mid_self_attns_big, up_cross_attns_big, up_self_attns_big             = big_attn_list
    down_cross_attns_small, down_self_attns_small, mid_cross_attns_small, mid_self_attns_small, up_cross_attns_small, up_self_attns_small = small_attn_list 

    resized_big_down_cross_attn = resizeAttn(down_cross_attns_big, resize=None)
    resized_big_up_cross_attn = resizeAttn(up_cross_attns_big, resize=None)
    resized_big_mid_cross_attn = resizeAttn(mid_cross_attns_big, resize=None)
    resized_small_down_cross_attn = resizeAttn(down_cross_attns_small, resize=None)
    resized_small_up_cross_attn = resizeAttn(up_cross_attns_small, resize=None)
    resized_small_mid_cross_attn = resizeAttn(mid_cross_attns_small, resize=None)

    loss = 0
    for index0, (small_box, big_box) in enumerate(zip(boxes_smaller,  boxes_bigger)):
        if checkPos(small_box) and checkPos(big_box):
            pass
        else:
            break
        for index, _ in enumerate(down_cross_attns_big):
            size = resized_big_down_cross_attn[index].size()[2]
            big_cross_attn   = resized_big_down_cross_attn[index]
            small_cross_attn = resized_small_down_cross_attn[index]
            big_xmin, big_ymin, big_xmax, big_ymax         = boxToPos(big_box, size=size)
            samll_xmin, samll_ymin, samll_xmax, samll_ymax = boxToPos(small_box, size=size)
            croped_big_attn   = big_cross_attn[:, :, big_xmin:big_xmax, big_ymin:big_ymax]
            croped_small_attn = small_cross_attn[:, :, samll_xmin:samll_xmax, samll_ymin:samll_ymax]
            try:
                small_size1, small_size2 = croped_small_attn.size()[2], croped_small_attn.size()[3]
                croped_big_to_small_attn = F.interpolate(croped_big_attn, (small_size1, small_size2), mode='bilinear')
                loss += torch.nn.functional.mse_loss(croped_big_to_small_attn, croped_small_attn)
                loss += torch.nn.functional.mse_loss(big_cross_attn, small_cross_attn)*2
            except:
                import pdb
                pdb.set_trace()

        for index, _ in enumerate(up_cross_attns_big):
            size = resized_big_up_cross_attn[index].size()[2]
            big_cross_attn   = resized_big_up_cross_attn[index]
            small_cross_attn = resized_small_up_cross_attn[index]
            big_xmin, big_ymin, big_xmax, big_ymax         = boxToPos(big_box, size=size)
            samll_xmin, samll_ymin, samll_xmax, samll_ymax = boxToPos(small_box, size=size)
            croped_big_attn   = big_cross_attn[:, :, big_xmin:big_xmax, big_ymin:big_ymax]
            croped_small_attn = small_cross_attn[:, :, samll_xmin:samll_xmax, samll_ymin:samll_ymax]
            small_size1, small_size2 = croped_small_attn.size()[2], croped_small_attn.size()[3]
            croped_big_to_small_attn = F.interpolate(croped_big_attn, (small_size1, small_size2), mode='bilinear')
            loss += torch.nn.functional.mse_loss(croped_big_to_small_attn, croped_small_attn)
            loss += torch.nn.functional.mse_loss(big_cross_attn, small_cross_attn) 

    return loss


def updownone(big_attn_list, small_attn_list, boxes_bigger,  boxes_smaller):

    down_cross_attns_big, down_self_attns_big, mid_cross_attns_big, mid_self_attns_big, up_cross_attns_big, up_self_attns_big             = big_attn_list
    down_cross_attns_small, down_self_attns_small, mid_cross_attns_small, mid_self_attns_small, up_cross_attns_small, up_self_attns_small = small_attn_list 

    resized_big_down_cross_attn = resizeAttn(down_cross_attns_big, resize=None)
    resized_big_up_cross_attn = resizeAttn(up_cross_attns_big, resize=None)
    resized_big_mid_cross_attn = resizeAttn(mid_cross_attns_big, resize=None)
    resized_small_down_cross_attn = resizeAttn(down_cross_attns_small, resize=None)
    resized_small_up_cross_attn = resizeAttn(up_cross_attns_small, resize=None)
    resized_small_mid_cross_attn = resizeAttn(mid_cross_attns_small, resize=None)

    loss = 0
    for index0, (small_box, big_box) in enumerate(zip(boxes_smaller,  boxes_bigger)):
        if checkPos(small_box) and checkPos(big_box):
            pass
        else:
            break
        for index, _ in enumerate(down_cross_attns_big):

            size = resized_big_down_cross_attn[index].size()[2]
            big_cross_attn   = resized_big_down_cross_attn[index]
            small_cross_attn = resized_small_down_cross_attn[index]
            big_xmin, big_ymin, big_xmax, big_ymax         = boxToPos(big_box, size=size)
            samll_xmin, samll_ymin, samll_xmax, samll_ymax = boxToPos(small_box, size=size)
            croped_big_attn   = big_cross_attn[:, :, big_xmin:big_xmax, big_ymin:big_ymax]
            croped_small_attn = small_cross_attn[:, :, samll_xmin:samll_xmax, samll_ymin:samll_ymax]
            try:
                small_size1, small_size2 = croped_small_attn.size()[2], croped_small_attn.size()[3]
                croped_big_to_small_attn = F.interpolate(croped_big_attn, (small_size1, small_size2), mode='bilinear')
                loss += torch.nn.functional.mse_loss(croped_big_to_small_attn, croped_small_attn)
            except:
                import pdb
                pdb.set_trace()

        for index, _ in enumerate(up_cross_attns_big):
            size = resized_big_up_cross_attn[index].size()[2]
            big_cross_attn   = resized_big_up_cross_attn[index]
            small_cross_attn = resized_small_up_cross_attn[index]
            big_xmin, big_ymin, big_xmax, big_ymax         = boxToPos(big_box, size=size)
            samll_xmin, samll_ymin, samll_xmax, samll_ymax = boxToPos(small_box, size=size)
            croped_big_attn   = big_cross_attn[:, :, big_xmin:big_xmax, big_ymin:big_ymax]
            croped_small_attn = small_cross_attn[:, :, samll_xmin:samll_xmax, samll_ymin:samll_ymax]
            small_size1, small_size2 = croped_small_attn.size()[2], croped_small_attn.size()[3]
            croped_big_to_small_attn = F.interpolate(croped_big_attn, (small_size1, small_size2), mode='bilinear')
            loss += torch.nn.functional.mse_loss(croped_big_to_small_attn, croped_small_attn)
    return loss





def uptwo(big_attn_list, small_attn_list, boxes_bigger,  boxes_smaller):

    down_cross_attns_big, down_self_attns_big, mid_cross_attns_big, mid_self_attns_big, up_cross_attns_big, up_self_attns_big             = big_attn_list
    down_cross_attns_small, down_self_attns_small, mid_cross_attns_small, mid_self_attns_small, up_cross_attns_small, up_self_attns_small = small_attn_list 

    resized_big_down_cross_attn = resizeAttn(down_cross_attns_big, resize=None)
    resized_big_up_cross_attn = resizeAttn(up_cross_attns_big, resize=None)
    resized_big_mid_cross_attn = resizeAttn(mid_cross_attns_big, resize=None)
    resized_small_down_cross_attn = resizeAttn(down_cross_attns_small, resize=None)
    resized_small_up_cross_attn = resizeAttn(up_cross_attns_small, resize=None)
    resized_small_mid_cross_attn = resizeAttn(mid_cross_attns_small, resize=None)

    loss = 0
    for index0, (small_box, big_box) in enumerate(zip(boxes_smaller,  boxes_bigger)):
        if checkPos(small_box) and checkPos(big_box):
            pass
        else:
            break

        for index, _ in enumerate(up_cross_attns_big):
            size = resized_big_up_cross_attn[index].size()[2]
            big_cross_attn   = resized_big_up_cross_attn[index]
            small_cross_attn = resized_small_up_cross_attn[index]
            big_xmin, big_ymin, big_xmax, big_ymax         = boxToPos(big_box, size=size)
            samll_xmin, samll_ymin, samll_xmax, samll_ymax = boxToPos(small_box, size=size)
            croped_big_attn   = big_cross_attn[:, :, big_xmin:big_xmax, big_ymin:big_ymax]
            croped_small_attn = small_cross_attn[:, :, samll_xmin:samll_xmax, samll_ymin:samll_ymax]
            small_size1, small_size2 = croped_small_attn.size()[2], croped_small_attn.size()[3]
            croped_big_to_small_attn = F.interpolate(croped_big_attn, (small_size1, small_size2), mode='bilinear')
            loss += torch.nn.functional.mse_loss(croped_big_to_small_attn, croped_small_attn)
            loss += torch.nn.functional.mse_loss(big_cross_attn, small_cross_attn) 

    return loss

def upone(big_attn_list, small_attn_list, boxes_bigger,  boxes_smaller):

    down_cross_attns_big, down_self_attns_big, mid_cross_attns_big, mid_self_attns_big, up_cross_attns_big, up_self_attns_big             = big_attn_list
    down_cross_attns_small, down_self_attns_small, mid_cross_attns_small, mid_self_attns_small, up_cross_attns_small, up_self_attns_small = small_attn_list 

    resized_big_down_cross_attn = resizeAttn(down_cross_attns_big, resize=None)
    resized_big_up_cross_attn = resizeAttn(up_cross_attns_big, resize=None)
    resized_big_mid_cross_attn = resizeAttn(mid_cross_attns_big, resize=None)
    resized_small_down_cross_attn = resizeAttn(down_cross_attns_small, resize=None)
    resized_small_up_cross_attn = resizeAttn(up_cross_attns_small, resize=None)
    resized_small_mid_cross_attn = resizeAttn(mid_cross_attns_small, resize=None)

    loss = 0
    for index0, (small_box, big_box) in enumerate(zip(boxes_smaller,  boxes_bigger)):
        if checkPos(small_box) and checkPos(big_box):
            pass
        else:
            break

        for index, _ in enumerate(up_cross_attns_big):
            size = resized_big_up_cross_attn[index].size()[2]
            big_cross_attn   = resized_big_up_cross_attn[index]
            small_cross_attn = resized_small_up_cross_attn[index]
            big_xmin, big_ymin, big_xmax, big_ymax         = boxToPos(big_box, size=size)
            samll_xmin, samll_ymin, samll_xmax, samll_ymax = boxToPos(small_box, size=size)
            croped_big_attn   = big_cross_attn[:, :, big_xmin:big_xmax, big_ymin:big_ymax]
            croped_small_attn = small_cross_attn[:, :, samll_xmin:samll_xmax, samll_ymin:samll_ymax]
            small_size1, small_size2 = croped_small_attn.size()[2], croped_small_attn.size()[3]
            croped_big_to_small_attn = F.interpolate(croped_big_attn, (small_size1, small_size2), mode='bilinear')
            loss += torch.nn.functional.mse_loss(croped_big_to_small_attn, croped_small_attn)

    return loss




def downtwo(big_attn_list, small_attn_list, boxes_bigger,  boxes_smaller):

    down_cross_attns_big, down_self_attns_big, mid_cross_attns_big, mid_self_attns_big, up_cross_attns_big, up_self_attns_big             = big_attn_list
    down_cross_attns_small, down_self_attns_small, mid_cross_attns_small, mid_self_attns_small, up_cross_attns_small, up_self_attns_small = small_attn_list 

    resized_big_down_cross_attn = resizeAttn(down_cross_attns_big, resize=None)
    resized_big_up_cross_attn = resizeAttn(up_cross_attns_big, resize=None)
    resized_big_mid_cross_attn = resizeAttn(mid_cross_attns_big, resize=None)
    resized_small_down_cross_attn = resizeAttn(down_cross_attns_small, resize=None)
    resized_small_up_cross_attn = resizeAttn(up_cross_attns_small, resize=None)
    resized_small_mid_cross_attn = resizeAttn(mid_cross_attns_small, resize=None)

    loss = 0
    for index0, (small_box, big_box) in enumerate(zip(boxes_smaller,  boxes_bigger)):
        if checkPos(small_box) and checkPos(big_box):
            pass
        else:
            break
        for index, _ in enumerate(down_cross_attns_big):
            size = resized_big_down_cross_attn[index].size()[2]
            big_cross_attn   = resized_big_down_cross_attn[index]
            small_cross_attn = resized_small_down_cross_attn[index]
            big_xmin, big_ymin, big_xmax, big_ymax         = boxToPos(big_box, size=size)
            samll_xmin, samll_ymin, samll_xmax, samll_ymax = boxToPos(small_box, size=size)
            croped_big_attn   = big_cross_attn[:, :, big_xmin:big_xmax, big_ymin:big_ymax]
            croped_small_attn = small_cross_attn[:, :, samll_xmin:samll_xmax, samll_ymin:samll_ymax]
            try:
                small_size1, small_size2 = croped_small_attn.size()[2], croped_small_attn.size()[3]
                croped_big_to_small_attn = F.interpolate(croped_big_attn, (small_size1, small_size2), mode='bilinear')
                loss += torch.nn.functional.mse_loss(croped_big_to_small_attn, croped_small_attn)
                loss += torch.nn.functional.mse_loss(big_cross_attn, small_cross_attn)*2
            except:
                import pdb
                pdb.set_trace()

            break


    return loss


def downone(big_attn_list, small_attn_list, boxes_bigger,  boxes_smaller):

    down_cross_attns_big, down_self_attns_big, mid_cross_attns_big, mid_self_attns_big, up_cross_attns_big, up_self_attns_big             = big_attn_list
    down_cross_attns_small, down_self_attns_small, mid_cross_attns_small, mid_self_attns_small, up_cross_attns_small, up_self_attns_small = small_attn_list 

    resized_big_down_cross_attn = resizeAttn(down_cross_attns_big, resize=None)
    resized_big_up_cross_attn = resizeAttn(up_cross_attns_big, resize=None)
    resized_big_mid_cross_attn = resizeAttn(mid_cross_attns_big, resize=None)
    resized_small_down_cross_attn = resizeAttn(down_cross_attns_small, resize=None)
    resized_small_up_cross_attn = resizeAttn(up_cross_attns_small, resize=None)
    resized_small_mid_cross_attn = resizeAttn(mid_cross_attns_small, resize=None)

    loss = 0
    for index0, (small_box, big_box) in enumerate(zip(boxes_smaller,  boxes_bigger)):
        if checkPos(small_box) and checkPos(big_box):
            pass
        else:
            break
        for index, _ in enumerate(down_cross_attns_big):
            size = resized_big_down_cross_attn[index].size()[2]
            big_cross_attn   = resized_big_down_cross_attn[index]
            small_cross_attn = resized_small_down_cross_attn[index]
            big_xmin, big_ymin, big_xmax, big_ymax         = boxToPos(big_box, size=size)
            samll_xmin, samll_ymin, samll_xmax, samll_ymax = boxToPos(small_box, size=size)
            croped_big_attn   = big_cross_attn[:, :, big_xmin:big_xmax, big_ymin:big_ymax]
            croped_small_attn = small_cross_attn[:, :, samll_xmin:samll_xmax, samll_ymin:samll_ymax]
            try:
                small_size1, small_size2 = croped_small_attn.size()[2], croped_small_attn.size()[3]
                croped_big_to_small_attn = F.interpolate(croped_big_attn, (small_size1, small_size2), mode='bilinear')
                loss += torch.nn.functional.mse_loss(croped_big_to_small_attn, croped_small_attn)
            except:
                import pdb
                pdb.set_trace()
            break

    return loss


import math
def attention_loss(attention_list, box_list, text, all_text=False, attn_list=['down', 'mid', 'up'], res_list=[8, 16, 32, 64],loss_num=1):
    loss = 0
    down_cross_attns, mid_cross_attns, up_cross_attns = attention_list[0], attention_list[1], attention_list[2]
    box = box_list[0]

    # if all_text==True:
    #     down_cross_attns = [i[:, :, 1:1+len(text.split(' '))] for i in down_cross_attns]
    #     mid_cross_attns  = [i[:, :, 1:1+len(text.split(' '))] for i in mid_cross_attns]
    #     up_cross_attns   = [i[:, :, 1:1+len(text.split(' '))] for i in up_cross_attns]

    # down loss


    for down_attn in down_cross_attns:
        b, i, j = down_attn.shape
        H = W = int(math.sqrt(i))
        if H not in res_list: continue
        if 'down' not in attn_list: continue
        down_attn = down_attn.permute(0, 2, 1).reshape(b, j, H, W)
        mask = torch.zeros(size=(H, W)).cuda()
        x_min, y_min, x_max, y_max = int(box[0]*W), int(box[1]*H), int(box[2]*W), int(box[3]*H)
        mask[y_min: y_max, x_min: x_max] = 1
        # utils.vis_one_attn(down_attn, index=1, mask=mask).save('0.jpg')
        down_attn = down_attn[:, 1:1+len(text.split(' '))]
        activation_value = (down_attn * mask).reshape(b, -1).sum(dim=-1)/down_attn.reshape(b, -1).sum(dim=-1)
        loss += torch.mean(1-activation_value) **2
    for mid_attn in mid_cross_attns:
        b, i, j = mid_attn.shape
        H = W = int(math.sqrt(i))
        if H not in res_list: continue
        if 'mid' not in attn_list: continue
        mid_attn = mid_attn.permute(0, 2, 1).reshape(b, j, H, W)
        mask = torch.zeros(size=(H, W)).cuda()
        x_min, y_min, x_max, y_max = int(box[0]*W), int(box[1]*H), int(box[2]*W), int(box[3]*H)
        mask[y_min: y_max, x_min: x_max] = 1
        mid_attn = mid_attn[:, 1:1+len(text.split(' '))]
        activation_value = (mid_attn * mask).reshape(b, -1).sum(dim=-1)/mid_attn.reshape(b, -1).sum(dim=-1)
        loss += torch.mean(1-activation_value) **2
    for up_attn in up_cross_attns:
        b, i, j = up_attn.shape
        H = W = int(math.sqrt(i))
        if H not in res_list: continue
        if 'up' not in attn_list: continue
        up_attn = up_attn.permute(0, 2, 1).reshape(b, j, H, W)
        mask = torch.zeros(size=(H, W)).cuda()
        x_min, y_min, x_max, y_max = int(box[0]*W), int(box[1]*H), int(box[2]*W), int(box[3]*H)
        mask[y_min: y_max, x_min: x_max] = 1
        up_attn = up_attn[:, 1:1+len(text.split(' '))]
        activation_value = (up_attn * mask).reshape(b, -1).sum(dim=-1)/up_attn.reshape(b, -1).sum(dim=-1)
        loss += torch.mean(1-activation_value) **2

    if loss_num==2:
        for down_attn in down_cross_attns:
            b, i, j = down_attn.shape
            H = W = int(math.sqrt(i))
            if H not in res_list: continue
            if 'down' not in attn_list: continue
            down_attn = down_attn.permute(0, 2, 1).reshape(b, j, H, W)
            mask = torch.zeros(size=(H, W)).cuda()
            x_min, y_min, x_max, y_max = int(box[0]*W), int(box[1]*H), int(box[2]*W), int(box[3]*H)
            mask[y_min: y_max, x_min: x_max] = 1
            # utils.vis_one_attn(down_attn, index=1, mask=mask).save('0.jpg')
            down_attn = down_attn[:, 1:1+len(text.split(' '))]
            activation_value = (down_attn * (1-mask)).reshape(b, -1).sum(dim=-1)/down_attn.reshape(b, -1).sum(dim=-1)
            loss += torch.mean(activation_value) **2
        for mid_attn in mid_cross_attns:
            b, i, j = mid_attn.shape
            H = W = int(math.sqrt(i))
            if H not in res_list: continue
            if 'mid' not in attn_list: continue
            mid_attn = mid_attn.permute(0, 2, 1).reshape(b, j, H, W)
            mask = torch.zeros(size=(H, W)).cuda()
            x_min, y_min, x_max, y_max = int(box[0]*W), int(box[1]*H), int(box[2]*W), int(box[3]*H)
            mask[y_min: y_max, x_min: x_max] = 1
            mid_attn = mid_attn[:, 1:1+len(text.split(' '))]
            activation_value = (mid_attn * (1-mask)).reshape(b, -1).sum(dim=-1)/mid_attn.reshape(b, -1).sum(dim=-1)
            loss += torch.mean(activation_value) **2
        for up_attn in up_cross_attns:
            b, i, j = up_attn.shape
            H = W = int(math.sqrt(i))
            if H not in res_list: continue
            if 'up' not in attn_list: continue
            up_attn = up_attn.permute(0, 2, 1).reshape(b, j, H, W)
            mask = torch.zeros(size=(H, W)).cuda()
            x_min, y_min, x_max, y_max = int(box[0]*W), int(box[1]*H), int(box[2]*W), int(box[3]*H)
            mask[y_min: y_max, x_min: x_max] = 1
            up_attn = up_attn[:, 1:1+len(text.split(' '))]
            activation_value = (up_attn * (1-mask)).reshape(b, -1).sum(dim=-1)/up_attn.reshape(b, -1).sum(dim=-1)
            loss += torch.mean(activation_value) **2


    return loss