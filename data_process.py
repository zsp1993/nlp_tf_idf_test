# -*- coding: utf-8 -*-
import os
import re
import time

cnn_train_dir = '/Users/zhangshaopeng/Downloads/NLP_data/neuralsum/cnn/training'
cnn_validation_dir = '/Users/zhangshaopeng/Downloads/NLP_data/neuralsum/cnn/validation'

dailymail_train_dir = '/Users/zhangshaopeng/Downloads/NLP_data/neuralsum/dailymail/training'
dailymail_validation_dir = '/Users/zhangshaopeng/Downloads/NLP_data/neuralsum/dailymail/validation'


#建单词表，并统计词频
vocab={}
def add_vocab(vocab_t, document_path):
    for root, dirs, files in os.walk(document_path):
        for file in files:
            file_path = os.path.join(root, file)
            #print file_path
            with open(file_path, 'r') as f:
                # part_mark表示当前读到了文档的部分，
                # 0：url of the original article;，1：sentences in the article and their labels，
                # 2：extractable highlights，3：named entity mapping.
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
                        #print line
                        for word in line.split():
                            # 发现非单词，跳过
                            if re.findall('\W', word) or len(word)==1:
                                continue
                            if word not in vocab_t:
                                vocab_t[word] = 0
                            vocab_t[word] += 1
                    if part_mark == 2:
                        #print line
                        for word in line.split():
                            # 发现非单词，跳过
                            if re.findall('\W', word) or len(word)==1:
                                continue
                            if word not in vocab_t:
                                vocab_t[word] = 0
                            vocab_t[word] += 1
                    if part_mark == 3:
                        continue
                #print vocab
            #break

def sort_by_count(d):
    d1 = sorted(d.items(), key=lambda x: -x[1])
    return d1
'''
start = time.time()
add_vocab(vocab, cnn_train_dir)
end = time.time()
print "time is :",end - start
#按照词频排个序，对dict排序后输出是list
sorted_vocab = sort_by_count(vocab)
#print sorted_vocab

for i in xrange(100):
    print sorted_vocab[i][0],sorted_vocab[i][1]
'''

#在cnn_train_dir文件夹试验一下tf—idf的方法，为每一个文件建立一个词频表文件
tf_idf_cnn_train_dir = '/Users/zhangshaopeng/Downloads/NLP_data/neuralsum/cnn/tf_idf_cnn_train_dir'
def build_all_tf_vocab(tf_idf_vocab_path, document_path):
    #进入到存放各文件词表的指定文件夹下
    os.chdir(tf_idf_vocab_path)
    for root, dirs, files in os.walk(document_path):
        D = 0
        for file in files:
            vocab_t={}
            file_path = os.path.join(root, file)
            portion = os.path.splitext(file)
            vocab_name = portion[0] + ".txt"      #新建对应文件的词表文件
            with open(file_path, 'r') as f:
                # part_mark表示当前读到了文档的部分，
                # 0：url of the original article;，1：sentences in the article and their labels，
                # 2：extractable highlights，3：named entity mapping.
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
                        #print line
                        for word in line.split():
                            # 发现非单词，跳过
                            if re.findall('\W', word) or len(word)==1:
                                continue
                            if word not in vocab_t:
                                vocab_t[word] = 0
                            vocab_t[word] += 1
                    if part_mark == 2:
                        #print line
                        for word in line.split():
                            # 发现非单词，跳过
                            if re.findall('\W', word) or len(word)==1:
                                continue
                            if word not in vocab_t:
                                vocab_t[word] = 0
                            vocab_t[word] += 1
                    if part_mark == 3:
                        continue
                #print vocab_t
                #写词表到词表文件
                D +=1
                print D,"_file:",vocab_name
                vocab_t_f = open(vocab_name, 'w')
                vocab_t_f.write(str(vocab_t))
                vocab_t_f.close()
                '''
                # 读取
                vocab_t_f = open(vocab_name, 'r')
                a = vocab_t_f.read()
                vocab_read = eval(a)
                vocab_t_f.close()
                print vocab_read
                '''
            '''   
            if D==12369:
                break
                '''
#build_all_tf_vocab(tf_idf_cnn_train_dir, cnn_train_dir)

def build_idf_vocab(tf_idf_vocab_path):
    #统计所有的词的idf
    idf_vocab_file_name = 'idf_vocab.txt'
    # 进入到存放各文件词表的指定文件夹下
    os.chdir(tf_idf_vocab_path)
    idf_vocab = {}
    for root, dirs, files in os.walk(tf_idf_vocab_path):
        D = 0
        for file in files:
            if file != '.DS_Store' and file != idf_vocab_file_name:
                print D,": ",file
                D = D + 1
                # 读取词表
                vocab_tf_f = open(file, 'r')
                a = vocab_tf_f.read()
                vocab_tf = eval(a)
                vocab_tf_f.close()
                for key in vocab_tf.keys():
                    if key not in idf_vocab:
                        idf_vocab[key] = 0
                    idf_vocab[key] += 1
    #print D

    vocab_t_f = open(idf_vocab_file_name, 'w')
    vocab_t_f.write(str(idf_vocab))
    vocab_t_f.close()

#build_idf_vocab(tf_idf_cnn_train_dir)

def bulid_vocab(document_train_path,document_validation_path):
    #新建一个词典文件，每一行为单词和对应的频数
    vocab = {}
    vocab_file = '/Users/zhangshaopeng/Downloads/NLP_data/neuralsum/cnn/vocab.txt'
    for root, dirs, files in os.walk(document_train_path):
        D = 0
        for file in files:
            D += 1
            file_path = os.path.join(root, file)
            #print file_path
            with open(file_path, 'r') as f:
                # part_mark表示当前读到了文档的部分，
                # 0：url of the original article;，1：sentences in the article and their labels，
                # 2：extractable highlights，3：named entity mapping.
                part_mark = 0
                for line in f:
                    line = line.lower()  # 转换为小写
                    line = re.sub('[,\.\'!:"-]', " ", line)#去掉标点符号
                    if line == '\n':
                        part_mark = part_mark + 1
                        #print "############################################"
                        continue
                    if part_mark == 0:
                        continue
                    # 遇到标志性空行
                    if part_mark == 1:
                        #print line
                        for word in line.split():
                            # 发现非单词，跳过
                            if re.findall('\W', word) or len(word)==1:
                                continue
                            if word not in vocab:
                                vocab[word] = 0
                            vocab[word] += 1
                    if part_mark == 2:
                        #print line
                        for word in line.split():
                            # 发现非单词，跳过
                            if re.findall('\W', word) or len(word)==1:
                                continue
                            if word not in vocab:
                                vocab[word] = 0
                            vocab[word] += 1
                    if part_mark == 3:
                        continue
                #print vocab
            print D," :",file_path
            #break
    for root, dirs, files in os.walk(document_validation_path):
        D = 0
        for file in files:
            D += 1
            file_path = os.path.join(root, file)
            #print file_path
            with open(file_path, 'r') as f:
                # part_mark表示当前读到了文档的部分，
                # 0：url of the original article;，1：sentences in the article and their labels，
                # 2：extractable highlights，3：named entity mapping.
                part_mark = 0
                for line in f:
                    line = line.lower()  # 转换为小写
                    line = re.sub('[,\.\'!:"-]', " ", line)#去掉标点符号
                    if line == '\n':
                        part_mark = part_mark + 1
                        #print "############################################"
                        continue
                    if part_mark == 0:
                        continue
                    # 遇到标志性空行
                    if part_mark == 1:
                        #print line
                        for word in line.split():
                            # 发现非单词，跳过
                            if re.findall('\W', word) or len(word)==1:
                                continue
                            if word not in vocab:
                                vocab[word] = 0
                            vocab[word] += 1
                    if part_mark == 2:
                        #print line
                        for word in line.split():
                            # 发现非单词，跳过
                            if re.findall('\W', word) or len(word)==1:
                                continue
                            if word not in vocab:
                                vocab[word] = 0
                            vocab[word] += 1
                    if part_mark == 3:
                        continue
                #print vocab
            print D," :",file_path
            #break
    sorted_vocab = sort_by_count(vocab)
    output = open(vocab_file, 'w')
    length = len(sorted_vocab)
    for i in range(length):
        print i," :",sorted_vocab[i]
        output.write(sorted_vocab[i][0])
        output.write(' ')
        output.write(str(sorted_vocab[i][1]))
        output.write('\n')
    output.close()

#bulid_vocab(cnn_train_dir, cnn_validation_dir)
