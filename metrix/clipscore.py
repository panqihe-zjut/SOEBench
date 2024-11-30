import os
def clipscore(bench_folder, image_folder, caption_folder):
    print(bench_folder, image_folder, caption_folder)
    image_path = os.path.join(bench_folder, image_folder)
    text_path  = os.path.join(bench_folder, caption_folder)
    os.system("python -m clip_score "+"'"+image_path+"'"+" "+"'"+text_path+"'")



    

lora_inpaint_list = [
    './debug/folder',
]
for exp in lora_inpaint_list:
    bench_list = os.listdir(exp)
    for bench in bench_list:
        bench_folder = os.path.join(exp, bench)
        clipscore(bench_folder, 'image_gen_crop', 'caption')
print('-'*30)