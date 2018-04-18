# -*- coding: utf-8 -*-
import os
import re
import math
import data_process

cnn_train_dir = '/Users/zhangshaopeng/Downloads/NLP_data/neuralsum/cnn/training'
cnn_validation_dir = '/Users/zhangshaopeng/Downloads/NLP_data/neuralsum/cnn/validation'

dailymail_train_dir = '/Users/zhangshaopeng/Downloads/NLP_data/neuralsum/dailymail/training'
dailymail_validation_dir = '/Users/zhangshaopeng/Downloads/NLP_data/neuralsum/dailymail/validation'

tf_idf_cnn_train_dir = '/Users/zhangshaopeng/Downloads/NLP_data/neuralsum/cnn/tf_idf_cnn_train_dir'

def tf_idf(vocab_path, document_path):
    # 进入到存放各文件词表的指定文件夹下
    os.chdir(vocab_path)
    for root, dirs, files in os.walk(document_path):
        for file in files:
            #得到词表名字，取出tf
            portion = os.path.splitext(file)
            vocab_name = portion[0] + ".txt"
            # 读取词表
            vocab_tf_f = open(vocab_name, 'r')
            a = vocab_tf_f.read()
            vocab_tf = eval(a)
            vocab_tf_f.close()
            #统计一下总词频
            tf_sum=0
            for key in vocab_tf.keys():
                tf_sum += vocab_tf[key]
            #计算idf
            vocab_idf = {}
            D = 1 #从1开始计数而不是从0开始计数，是为了避免log后分子为零
            idf_vocab_file_name = 'idf_vocab.txt'
            '''
            for key in vocab_tf.keys():
                vocab_idf[key] = 0
            for root_temp, dirs_temp, files_temp in os.walk(vocab_path):
                for file_temp in files_temp:
                    #过滤掉文件夹中隐藏文件'.DS_Store'
                    if file_temp!='.DS_Store' and file_temp != idf_vocab_file_name:
                        D = D+1
                        #取出词表
                        vocab_temp_f = open(file_temp,'r')
                        a_temp = vocab_temp_f.read()

                        vocab_temp = eval(a_temp)
                        vocab_temp_f.close()
                        for key in vocab_tf.keys():
                            if key in vocab_temp:
                                vocab_idf[key] +=1
            print D
            '''
            #存表法计算idf
            D = 83569
            vocab_idf_f = open(idf_vocab_file_name, 'r')
            a = vocab_idf_f.read()
            vocab_all_idf = eval(a)
            vocab_idf_f.close()
            for key in vocab_tf.keys():
                vocab_idf[key] = vocab_all_idf[key]

            #计算tf-idf
            vocab_tf_idf = {}
            for key in vocab_tf.keys():
                vocab_tf_idf[key] = float(vocab_tf[key])/float(tf_sum) * math.log(float(D)/float(vocab_idf[key]))
                #print vocab_tf[key],tf_sum,D,vocab_idf[key]

            file_path = os.path.join(root, file)
            #print file_path
            with open(file_path, 'r') as f:
                # part_mark表示当前读到了文档的部分，
                # 0：url of the original article;，1：sentences in the article and their labels，
                # 2：extractable highlights，3：named entity mapping.
                sentences = {}
                highlight = {}
                part_mark = 0
                for line in f:
                    line = line.lower()  # 转换为小写
                    line = re.sub('[,\.\'!:"-]', "", line)#去掉标点符号
                    if line == '\n':
                        part_mark = part_mark + 1
                        #print "############################################"
                        continue
                    if part_mark == 0:
                        continue
                    # 遇到标志性空行
                    if part_mark == 1:
                        #计算句子的tf-idf为各个单词tf-idf的累和
                        #print line
                        sentences[line] = 0
                        line_tf_idf = 0
                        word_num = 1
                        for word in line.split():
                            # 发现非单词，跳过
                            if re.findall('\W', word) or len(word)==1:
                                continue
                            else:
                                line_tf_idf += vocab_tf_idf[word]
                                word_num += 1
                        #print line_tf_idf/word_num
                        sentences[line] += line_tf_idf/word_num
                    if part_mark == 2:
                        #print line
                        highlight[line] = 0
                        line_tf_idf = 0
                        word_num = 1
                        for word in line.split():
                            # 发现非单词，跳过
                            if re.findall('\W', word) or len(word)==1:
                                continue
                            else:
                                line_tf_idf += vocab_tf_idf[word]
                                word_num +=1
                        #print line_tf_idf/word_num
                        highlight[line] += line_tf_idf/word_num
                    if part_mark == 3:
                        continue
                #print vocab
                #依据tf_idf对句子排序，输出前三的句子
                sorted_sentences = data_process.sort_by_count(sentences)
                for sentences_index in range(3):
                    print sorted_sentences[sentences_index][0]
                    print sorted_sentences[sentences_index][1]
                #输出highlight以及他的tf_idf
                for key in highlight.keys():
                    print key
                    print highlight[key]
            break


tf_idf(tf_idf_cnn_train_dir, cnn_train_dir)