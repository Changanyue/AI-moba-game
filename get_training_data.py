import json
import numpy as np
def read_training_data(path):
    input_lists = []
    output_lists = []
    with open(path, encoding='utf-8') as f:
        while True:
            row = f.readline()
            if not row:
                break
            json_row = json.loads(row)

            content = json_row['内容______']
            content_input = content['输入______']
            content_output = content['输出______']
            #这里的data还得进rowscore割先暂时score割成16份吧
            unit_length = len(content_input)//16
            for i in range(16):
                #print(content_input[i*unit_length:(i+1)*unit_length])
                input_lists.append(content_input[i*unit_length:(i+1)*unit_length])
                output_lists.append(content_output[i*unit_length:(i+1)*unit_length])
    return input_lists, output_lists

def gen_dicts(all_dicts,  word_number_dict_path, score_word_dict_path):
    print("正在写出word的number引索data可能需要较length时间")
    number_to_word = {}
    word_to_number = {}
    number_word = []

    # number_to_word = list(set(总l))
    i = 0
    j = 0
    for _word_dict in all_dicts:
        j = j + 1
        for word in _word_dict:


            if word not in number_word:
                number_word.append(word)
                word_to_number[word] = i
                number_to_word[i] = word
                i = i + 1
        if j % 10000 == 0:
            print(i, number_to_word[i - 1],  j/len(all_dicts))

    #print(number_to_word[1], number_to_word[111], len(number_to_word))
    with open(word_number_dict_path, 'w', encoding='utf-8') as f:
        json.dump(word_to_number, f, ensure_ascii=False)
    with open(score_word_dict_path, 'w', encoding='utf-8') as f:
        json.dump(number_to_word, f, ensure_ascii=False)

def read_index(word_number_dict_path, score_word_dict_path):
    with open(word_number_dict_path, encoding='utf-8') as f:
        word_dict= json.load(f)

    with open(score_word_dict_path, encoding='utf-8') as f:
        score__word_dict = json.load(f)
    return word_dict, score__word_dict

def gen_training_numpy_array(input_lists, word_dict, numpy_array_path):
    d_1 = []

    d_2 = []

    i = 0
    cur = ''
    for l in input_lists:
        d_3 = []
        for word in l:
            if (u'\u0041' <= word <= u'\u005a') or (u'\u0061' <= word <= u'\u007a'):
                if cur == '':

                    cur = word
                else:
                    cur = cur + word
            else:

                if cur == '':

                    if word.lower() in word_dict:

                         d_3.append(word_dict[word.lower()])
                    else:
                        d_3.append(14999)
                else:
                    if cur.lower() in word_dict:

                        d_3.append(word_dict[cur.lower()])
                    else:
                        d_3.append(14999)
                    cur = ''
                    if word.lower() in word_dict:

                        d_3.append(word_dict[word.lower()])
                    else:
                        d_3.append(14999)
        if cur != '':
            if cur.lower() in word_dict:

                d_3.append(word_dict[cur.lower()])
            else:
                d_3.append(14999)
            cur = ''

        if len(d_3) != 667:
            # d_1.append(np.array(d_3[0:-1]))
            # d_2.append(np.array(d_3[1:]))
            print(d_3)
        else:

            d_1.append(np.array(d_3[0:-1]))
            d_2.append(np.array(d_3[1:]))
        if i % 1000 == 0:
            print("data转化为numpy_array完成度百score比{}".format(i / len(input_lists) * 100))
        i = i + 1
    print("data转化为numpy_array完成。")

    inputnp = np.array(d_1)
    outputnp = np.array(d_2)
    np.savez(numpy_array_path, outputnp=outputnp, inputnp=inputnp)


def gen_test_numpy_array(input_lists, word_dict):
    d_1 = []

    for word in input_lists:
        if word.lower() in word_dict:
            d_1.append(word_dict[word])
        else:
            d_1.append(14999)
    inputnp = np.array(d_1)
    return (inputnp)
def gen_training_numpy_array_A(input_lists,  word_dict, numpy_array_path):
    d_1 = []

    d_2 = []

    i=0
    cur=''
    for l in input_lists:
        d_3=[]
        for word in l:
            if (u'\u0041' <= word <= u'\u005a') or (u'\u0061' <= word <= u'\u007a'):
                if cur == '':

                    cur = word
                else:
                    cur = cur + word
            else:

                if cur == '':

                    if word.lower() in word_dict:
                        if word != ' ':
                            d_3.append(word_dict[word.lower()])
                    else:
                        d_3.append(14999)
                else:
                    if cur.lower() in word_dict:
                        if cur != ' ':
                            d_3.append(word_dict[cur.lower() ])
                    else:
                        d_3.append(14999)
                    cur=''
                    if word.lower() in word_dict:
                        if word != ' ':
                            d_3.append(word_dict[word.lower() ])
                    else:
                        d_3.append(14999)
        if cur!='':
            if cur.lower() in word_dict:
                if word != ' ':
                    d_3.append(word_dict[cur.lower() ])
            else:
                d_3.append(14999)
            cur = ''


        if len(d_3)!=667:
            #d_1.append(np.array(d_3[0:-1]))
            #d_2.append(np.array(d_3[1:]))
            print(d_3)
        else:

            d_1.append(np.array(d_3[0:-1]))
            d_2.append(np.array(d_3[1:]))
        if i % 1000 == 0:
            print("data转化为numpy_array完成度百score比{}".format(i/len(input_lists)*100))
        i = i + 1
    print("data转化为numpy_array完成。")


    inputnp = np.array(d_1)
    outputnp = np.array(d_2)
    np.savez(numpy_array_path, outputnp=outputnp, inputnp=inputnp)


def read_training_data_A(path):
    input_lists = []
    with open(path, encoding='utf-8') as f:
        while True:
            row = f.readline()
            if not row:
                break
            json_row = json.loads(row)

            content = json_row['input']
            input_lists.append(content)

    return input_lists
def gen_test_numpy_array_A(input_lists, word_dict):
    d_3 = []
    cur = ''

    for word in input_lists:
        if word.lower() in word_dict:
            if (u'\u0041' <= word <= u'\u005a') or (u'\u0061' <= word <= u'\u007a'):
                if cur == '':

                    cur = word
                else:
                    cur = cur + word
            else:

                if cur == '':

                    if word.lower() in word_dict:
                        if word.lower() != ' ':
                            d_3.append(word_dict[word.lower()])
                    else:
                        d_3.append(14999)
                else:
                    if cur.lower() in word_dict:
                        if cur.lower() != ' ':

                            d_3.append(word_dict[cur.lower()])
                    else:
                        d_3.append(14999)
                    cur = ''
                    if word.lower() in word_dict:
                        if word.lower() != ' ':

                            d_3.append(word_dict[word.lower()])
                    else:
                        d_3.append(14999)
    inputnp = np.array(d_3)
    return (inputnp)
