import os
import glob
from tqdm import tqdm

vocab_dict = {}
corpus_list = glob.glob('/home/jylim2/dataset/bert/wiki/corpus_v2/*.txt')
print(len(corpus_list))

for corpus in tqdm(corpus_list):
    f = open(corpus, 'r', encoding='UTF8')
    lines = f.readlines()
    for line in lines:
        for ch in list(line):
            if ch == ' ':
                continue
            
            ch = ch.lower()
            if len(ch) > 1:
                continue
            if ord('가') <= ord(ch) and ord(ch) <= ord('힣'):
                if ch in vocab_dict:
                    vocab_dict[ch] += 1
                else:
                    vocab_dict[ch] = 1
                    
            if ord('a') <= ord(ch) and ord(ch) <= ord('z'):
                if ch in vocab_dict:
                    vocab_dict[ch] += 1
                else:
                    vocab_dict[ch] = 1
            
            if ord('0') <= ord(ch) and ord(ch) <= ord('9'):
                if ch in vocab_dict:
                    vocab_dict[ch] += 1
                else:
                    vocab_dict[ch] = 1
            
            if ch in ['!','@','#','$','%','^','&','*','(',')','[',']','{','}','?','-','+','+','_',',','.',':',';','"','~','*']:
                if ch in vocab_dict:
                    vocab_dict[ch] += 1
                else:
                    vocab_dict[ch] = 1
    f.close()

sort_list = sorted(vocab_dict.items(), key=lambda x:x[1], reverse=True)
with open('vocab.txt', 'w', encoding='UTF8') as fw:
    for k, cnt in sort_list[:-1]:
        fw.write('%s\t%d\n'%(k, cnt))
    k, cnt = sort_list[-1]
    fw.write('%s\t%d'%(k, cnt))