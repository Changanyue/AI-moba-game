import torch
import torchvision
from PIL import Image
import numpy as np
import time
import json
from config import GPT2Config, TransformerConfig
from Batch import create_masks
from ModelA import get_model
import torch.nn.functional as F
from get_training_data import *
from utils import *
import os
import random

save_path='../训练数据样本'
if not os.path.exists(save_path):
   os.makedirs(save_path)
for root, dirs, files in os.walk('../训练数据样本'):
    if len(dirs)>0:
        break

dict_path="./json/词_数表.json"
score_word_dict_path="./json/数_词表.json"
if os.path.isfile(dict_path) and os.path.isfile(score_word_dict_path):
    word_dict, score__word_dict = read_index(dict_path, score_word_dict_path)
with open(dict_path, encoding='utf8') as f:
    word_number_dict=json.load(f)
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
#
#
config = TransformerConfig()

model = get_model(config,  130)
模型path = 'weights/model_weights'
model = model.cuda(device)
optimizer = torch.optim.Adam(model.parameters(), lr=6.25e-5, betas=(0.9, 0.98), eps=1e-9)


block_length=25
cursor_step=23
branch_length=10  #树枝

conut=0
time_start=time.time()
for j in range(100):
    random.shuffle(dirs)
    for dir_name in dirs:
        preprecee_data = '../训练数据样本/'+dir_name+'/图片_操作预处理数据2.npz'
        if os.path.isfile(preprecee_data):
            npzfile = np.load(preprecee_data, allow_pickle=True)
            image_feature_np, action_seq = npzfile["image_feature_np"], npzfile["action_seq"]
            loop=True
            cursor=0
            action_seq=np.insert(action_seq,0,128)  # 动作的序列

            step_score_d = []
            target_score_d = []
            image_score_d = []

            while loop:
                if cursor + block_length < action_seq.shape[0]:  # 3000
                    step_score = action_seq[cursor:cursor + block_length]
                    target_score = action_seq[cursor + 1:cursor + 1 + block_length]
                    image_score = image_feature_np[cursor:cursor + block_length, :]
                    step_score_d.append(step_score)
                    target_score_d.append(target_score)
                    image_score_d.append(image_score)
                    cursor = cursor + cursor_step
                else:
                    step_score = action_seq[-block_length - 1:-1]
                    target_score = action_seq[-block_length:]

                    image_score = image_feature_np[-block_length:, :]
                    step_score_d.append(step_score)
                    target_score_d.append(target_score)
                    image_score_d.append(image_score)
                    loop = False
            # print(np.array(step_score_d).shape)  # 131,25  131就是bs

            loop=True
            i=0
            while loop:
                if (i+1)*branch_length<len(step_score_d):

                    step_score_branch = np.array(step_score_d[i*branch_length:(i+1)*branch_length])  # [10,25]
                    image_score_branch = np.array(image_score_d[i * branch_length:(i + 1) * branch_length])
                    target_score_branch = np.array(target_score_d[i * branch_length:(i + 1) * branch_length])

                else:
                    step_score_branch = np.array(step_score_d[i * branch_length:len(step_score_d)])
                    image_score_branch = np.array(image_score_d[i * branch_length:len(image_score_d)],dtype=np.float32)
                    target_score_branch = np.array(target_score_d[i * branch_length:len(target_score_d)])
                    loop = False

                step_score_torch=torch.from_numpy(step_score_branch).cuda(device)
                print(step_score_torch)
                image_score_torch = torch.from_numpy(image_score_branch).cuda(device)
                target_score_torch = torch.from_numpy(target_score_branch).cuda(device)


                src_mask, trg_mask = create_masks(step_score_torch, step_score_torch, device)
                if image_score_torch.shape[0]!=step_score_torch.shape[0]:
                    continue

                output_actual_A = model(image_score_torch,step_score_torch ,trg_mask)
                lin = output_actual_A.view(-1, output_actual_A.size(-1))
                optimizer.zero_grad()
                # print(lin.size(),target_score_torch.view(-1).size())
                # [250, 130], [250]
                loss = F.cross_entropy(lin, target_score_torch.contiguous().view(-1), ignore_index=-1)
                if conut % 1 == 0:
                    print('loss:',loss.item())

                    time_end = time.time()
                    time_cost = time_end - time_start

                    _, sample = torch.topk(output_actual_A, k=1, dim=-1) # 10,25,1
                    samplenp = sample.cpu().numpy()
                    print_sample_data(score__word_dict, samplenp[0:1,:,:], target_score_torch[0,:])
                    print("time_cost{} 第{}轮 第{}张 dir_name{}".format(time_cost, j, conut, dir_name))
                if conut % 45060 == 0:
                    print('888')

                loss.backward()

                optimizer.step()
                conut=conut+1
                i=i+1
torch.save(model.state_dict(), 'weights/model_weights')
#torch.save(model.state_dict(), 'weights/model_weights_P{}'.format(str(j)))





