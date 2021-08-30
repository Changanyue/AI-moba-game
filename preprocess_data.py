import torch
import torchvision
import numpy as np
import os
import json
from PIL import Image
from resnet_utils import myResnet

action_record='../训练数据样本'
if not os.path.exists(action_record):
   os.makedirs(action_record)

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
resnet101=torchvision.models.resnet101(pretrained=True).eval()
resnet101=myResnet(resnet101).cuda(device).requires_grad_(False)
dict_path="./json/词_数表.json"

with open(dict_path, encoding='utf8') as f:
    word_number_dict=json.load(f)

for root, dirs, files in os.walk(action_record):
    if len(dirs)>0:
        break
for dir_name in dirs:
    pathjson = action_record+'/' + dir_name + '/_操作数据.json'
    numpy_array_path= action_record+'/' + dir_name + '/图片_操作预处理数据2.npz'
    if os.path.isfile(numpy_array_path):
        continue

    image_feature = torch.Tensor(0)

    # print(image_feature.shape[0])

    psu_wrod_seq = torch.from_numpy(np.ones((1, 60)).astype(np.int64)).cuda(device).unsqueeze(0)

    action_seq = np.ones((1, 1))
    conut = 0
    print('正在处理{}'.format(dir_name))
    data_col=[]
    with open(pathjson, encoding='gbk') as f:
        move_action='无移动'
        while True:
            df = f.readline()
            if df == "":
                break
            df = json.loads(df)
            data_col.append(df)

    with open(pathjson, encoding='gbk') as f:
        move_action='无移动'
        for i in range(len(data_col)):
            df = data_col[i]

            if image_feature.shape[0] == 0:
                img = Image.open(action_record+'/' + dir_name + '/{}.jpg'.format(df["图片号"]))
                img2 = np.array(img)

                img2 = torch.from_numpy(img2).cuda(device).unsqueeze(0).permute(0, 3, 2, 1) / 255
                _,out = resnet101(img2)
                image_feature = out.reshape(1,6*6*2048)
                move_actiona=df["移动操作"]
                if move_actiona!='无移动':
                    move_action=move_actiona

                action_seq[0, 0] = word_number_dict[move_action + "_" + df["动作操作"]]
            else:
                img = Image.open(action_record+'/' + dir_name + '/{}.jpg'.format(df["图片号"]))
                img2 = np.array(img)

                img2 = torch.from_numpy(img2).cuda(device).unsqueeze(0).permute(0, 3, 2, 1) / 255
                _,out= resnet101(img2)

                image_feature = torch.cat((image_feature, out.reshape(1,6*6*2048)), 0)
                move_actiona=df["移动操作"]
                if move_actiona!='无移动':
                    move_action=move_actiona
                action_seq=np.append(action_seq, word_number_dict[move_action + "_" + df["动作操作"]])

        image_feature_np=image_feature.cpu().numpy()
        action_seq=action_seq.astype(np.int64)
        np.savez(numpy_array_path, image_feature_np=image_feature_np, action_seq=action_seq)

