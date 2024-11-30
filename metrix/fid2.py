import os
from PIL import Image
from tqdm import tqdm
import shutil

def merge_folders(savefolder, subfolders, merged_folder_path):
    # 创建目录merged_folder_path
    merge_folder = os.path.join(savefolder, merged_folder_path)
    if not os.path.exists(merge_folder):
        os.makedirs(merge_folder)

    subfolders = [os.path.join(savefolder, i) for i in subfolders]
    
    # 遍历每个文件夹路径
    for  subfolder in subfolders:
        # 获取文件夹中的所有子目录名称
        subdirs = os.listdir(subfolder)
        
        # 遍历子目录
        for subdir in subdirs:
            # 创建目标文件夹路径
            merged_subdir_path = os.path.join(merged_folder_path, subdir)
            
            # 获取源文件夹路径
            source_subdir_path = os.path.join(subfolder, subdir)
            
            # 如果源文件夹路径存在，则合并到目标文件夹中
            if os.path.exists(source_subdir_path):
                merge_folders([source_subdir_path], merged_subdir_path)

def merge_folders(savefolder, subfolders, targetfolder):

    targetfolder = os.path.join(savefolder, targetfolder)
    if not os.path.exists(targetfolder):
        os.makedirs(targetfolder)

    subfolders = [os.path.join(savefolder, i) for i in subfolders]
    
    # 遍历每个文件夹路径
    for folder_path in subfolders:
        # 获取文件夹中的所有子目录名称
        subdirs = os.listdir(folder_path)
        
        # 遍历子目录
        for subdir in subdirs:
            # 创建目标文件夹路径
            merged_subdir_path = os.path.join(targetfolder, subdir)
            # 获取源文件夹路径
            source_subdir_path = os.path.join(folder_path, subdir)

            if os.path.exists(merged_subdir_path)==False:
                os.makedirs(merged_subdir_path, exist_ok=True)
                source_subdir_file = os.listdir(source_subdir_path)
                source_subdir_file = sorted(source_subdir_file, key=lambda x: int(x.split('_')[0]))
                source_file = [os.path.join(source_subdir_path, i) for i in source_subdir_file]
                target_file = [os.path.join(merged_subdir_path, i) for i in source_subdir_file]
                for k in range(0, len(source_file)):
                    shutil.copyfile(source_file[k], target_file[k])
            else:
                source_subdir_file = os.listdir(source_subdir_path)
                source_subdir_file = sorted(source_subdir_file, key=lambda x: int(x.split('_')[0]))
                lentarget          = len(os.listdir(merged_subdir_path))
                source_file = [os.path.join(source_subdir_path, i) for i in source_subdir_file]
                target_file = [os.path.join(merged_subdir_path, str(lentarget+int(i.split('_')[0]))+"_"+i.split('_')[1]) for i in source_subdir_file]
                for k in range(0, len(source_file)):
                    shutil.copyfile(source_file[k], target_file[k])



def box2newbox(box):
    x1,y1,x2,y2 =  box[0],box[1],box[2],box[3]
    
    if x2-x1>=0.25 or  y2-y1>=25:
        return  box

    else:
        centerx = (x1+x2)/2
        centery = (y1+y2)/2
        cropscale  = 0.125
        newsize =min( min((1-centerx), cropscale), min(centerx,cropscale), min((1-centery), cropscale), min(centery, cropscale) )

        # print(newsize)
        crop_x1 = centerx-newsize
        crop_y1 = centery-newsize
        crop_x2 = centerx+newsize
        crop_y2 = centery+newsize

        return [crop_x1, crop_y1, crop_x2, crop_y2]
    
paths = [             
    '/storage/panqihe/projects/SOEDiff/exp_bench/expdebug_colorlabel',
]

for path in paths:
    testclass= os.listdir(path)
    for cls in testclass:
        print("FID of :", cls)
        class_path = os.path.join(path, cls)
        imageids = os.listdir(os.path.join(class_path, 'image_gen'))
        imageids = [i.split('.')[0]  for i in imageids]

        os.makedirs(os.path.join(class_path, 'image_ori_crop_resized'), exist_ok=True)
        os.makedirs(os.path.join(class_path, 'image_gen_crop_resized'), exist_ok=True)

        image_ori_crop_resized = os.path.join(class_path, 'image_ori_crop_resized')
        image_gen_crop_resized = os.path.join(class_path, 'image_gen_crop_resized')
        os.makedirs(image_ori_crop_resized, exist_ok=True)
        os.makedirs(image_gen_crop_resized, exist_ok=True)

        for imageid in tqdm(imageids):
            ori_image_path = os.path.join(class_path, 'image_ori', imageid+'.jpg')
            gen_image_path = os.path.join(class_path, 'image_gen', imageid+'.jpg')
            box_path       = os.path.join(class_path, 'box', imageid+'.txt')

            box            = open(box_path, 'r').readline().split(' ')
            box            = [float(i) for i in box]
            newbox         = box2newbox(box)


            ori_image      = Image.open(ori_image_path).resize((512, 512))
            gen_image      = Image.open(gen_image_path).resize((512, 512))

            ori_image_crop = ori_image.crop((newbox[0]*512, newbox[1]*512, newbox[2]*512, newbox[3]*512)).resize((512, 512))
            gen_image_crop = gen_image.crop((newbox[0]*512, newbox[1]*512, newbox[2]*512, newbox[3]*512)).resize((512, 512))

            ori_image_crop.save(os.path.join(os.path.join(class_path, 'image_ori_crop_resized',imageid+'.jpg' )))
            gen_image_crop.save(os.path.join(os.path.join(class_path, 'image_gen_crop_resized',imageid+'.jpg' )))

        
        os.system("CUDA_VISIBLE_DEVICES=0 python -m pytorch_fid "+"'"+image_ori_crop_resized+"'"+" "+"'"+image_gen_crop_resized+"'" + " "+'--num-workers 16')
        