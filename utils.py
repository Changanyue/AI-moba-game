from torch.autograd import Variable
import torch
import numpy as np
def print_sample_data(score_word_dict,data, output_score):
    cur = data[0]
    to_print_str=[score_word_dict[str(cur[i,0])] for i in range(0,cur.shape[0])]
    cur = output_score.cpu().numpy()
    to_print_str2 = [score_word_dict[str(cur[i])] for i in range(0,cur.shape[0])]
    print("sampleoutput",to_print_str)
    print("target", to_print_str2)
    # for i in range(16):
    #     print(score_word_dict[str(cur[i, 0])])

def nopeak_mask(size, device):
    np_mask = np.triu(np.ones((1, size, size)),
    k=1).astype('uint8')
    np_mask =  Variable(torch.from_numpy(np_mask) == 0)

    np_mask = np_mask.cuda(device)
    return np_mask
def print_test_data(score_word_dict,data, input_score,label):
    cur = data[0]
    to_print_str=[score_word_dict[str(cur[i])] for i in range(cur.size)]
    print_str=""
    for i in range(len(to_print_str)):
        print_str=print_str+to_print_str[i]



    cur = input_score.cpu().numpy()[0]
    to_print_str2 = [score_word_dict[str(cur[i])]for i in range(input_score.size(1))]
    # to_print_str2=str(to_print_str2)
    # print("input：", to_print_str2)
    if label==print_str:
        return True
    else:
        print(print_str)
        return False



    print("output：",print_str)

    # for i in range(16):
    #     print(score_word_dict[str(cur[i, 0])])
def print_test_data_A(score_word_dict,data, input_score):
    if data.shape[0]!=0:

        cur = data[0]
        to_print_str=[score_word_dict[str(cur[i])] for i in range(cur.size)]
        print_str=""
        for i in range(len(to_print_str)):
            print_str=print_str+to_print_str[i]



        cur = input_score.cpu().numpy()[0]
        to_print_str2 = [score_word_dict[str(cur[i])]for i in range(input_score.size(1))]
        to_print_str2=str(to_print_str2)
        #print("input：", to_print_str2)
        print("output：",print_str)

