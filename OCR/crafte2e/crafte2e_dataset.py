import os
import glob
from os import replace
import pandas as pd
import numpy as np
import cv2
from PIL import Image

np.random.seed(22)

def four_point_transform(image, pts):

    max_x, max_y = np.max(pts[:, 0]).astype(np.int32), np.max(pts[:, 1]).astype(np.int32)

    dst = np.array([
        [0, 0],
        [image.shape[1] - 1, 0],
        [image.shape[1] - 1, image.shape[0] - 1],
        [0, image.shape[0] - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(dst, pts)
    warped = cv2.warpPerspective(image, M, (max_x, max_y))

    return warped

class Dataset:

    def __init__(self, path, vocab, max_num_boxes, max_num_char) -> None:
        self.image_size = 1440
        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]
        tmp_path_list = glob.glob(os.path.join(path, '*.txt'))
        self.gt_path_list = []
        for path in tmp_path_list:
            if not os.path.exists(path.replace('gts','images').replace('.txt','.jpg')):
                continue
            self.gt_path_list.append(path)
        print('total num dataset:',len(self.gt_path_list))
        self.gt_path_ind = [v for v in range(len(self.gt_path_list))]
        self.vocab = vocab
        self.max_num_boxes = max_num_boxes
        self.max_num_char = max_num_char

        sigma = 10
        spread = 3
        extent = int(spread * sigma)
        self.gaussian_heatmap = np.zeros([2 * extent, 2 * extent], dtype=np.float32)

        for i in range(2 * extent):
            for j in range(2 * extent):
                self.gaussian_heatmap[i, j] = 1 / 2 / np.pi / (sigma ** 2) * np.exp(
                    -1 / 2 * ((i - spread * sigma - 0.5) ** 2 + (j - spread * sigma - 0.5) ** 2) / (sigma ** 2))

        self.gaussian_heatmap = (self.gaussian_heatmap / np.max(self.gaussian_heatmap) * 255).astype(np.uint8)

    def add_affinity(self, image, bbox_1, bbox_2):

        center_1, center_2 = np.mean(bbox_1, axis=0), np.mean(bbox_2, axis=0)
        tl = np.mean([bbox_1[0], bbox_1[1], center_1], axis=0)
        bl = np.mean([bbox_1[2], bbox_1[3], center_1], axis=0)
        tr = np.mean([bbox_2[0], bbox_2[1], center_2], axis=0)
        br = np.mean([bbox_2[2], bbox_2[3], center_2], axis=0)

        affinity = np.array([tl, tr, br, bl])

        self.add_character(image, affinity)

    def add_character(self, image, bbox):
        top_left = np.array([np.min(bbox[:, 0]), np.min(bbox[:, 1])]).astype(np.int32)
        bbox -= top_left[None, :]
        transformed = four_point_transform(self.gaussian_heatmap.copy(), bbox.astype(np.float32))
        transformed[transformed<=0.02] = 0.0
        # print(transformed.shape, top_left[1]+transformed.shape[0], image.shape)
        height, width = image.shape
        v_diff = height - (top_left[1] + transformed.shape[0])
        if v_diff < 0:
            transformed = transformed[:v_diff,:]
        h_diff = width - (top_left[0] + transformed.shape[1])
        if h_diff < 0:
            transformed = transformed[:, h_diff]
        image[top_left[1]:top_left[1]+transformed.shape[0],top_left[0]:top_left[0]+transformed.shape[1]] += transformed
        # return image
    
    def generate(self):
        len_gt_path_ind = len(self.gt_path_ind)
        i = 0
        while True:
            i = i % len_gt_path_ind
            if i == 0:
                np.random.shuffle(self.gt_path_ind)
            yield self.gt_path_ind[i]
            i+=1
        
    def process(self, index):
        gt_path = self.gt_path_list[index]
        df = pd.read_csv(gt_path, sep='\t', header=0, quoting=3, error_bad_lines=False, encoding='UTF8')
        
        image_path = gt_path.replace('gts','images').replace('.txt','.jpg')
        image = Image.open(image_path).convert('RGB')
        
        image = np.array(image)
        height, width = image.shape[:2]
        max_size = max(width, height)
        ratio = float(self.image_size) / max_size
        image = cv2.resize(image, (int(ratio*width), int(ratio*height)))
        height, width = image.shape[:2]

        image = image.astype(np.float32)
        image = ((image / 255.0) - self.mean) / self.std

        image = np.pad(image, [[0, self.image_size-height],
                               [0, self.image_size-width],
                               [0,0]])
        height, width = image.shape[:2]

        target_char = np.zeros([height, width], dtype=np.float32)
        target_affinity = np.zeros([height, width], dtype=np.float32)
        labels = np.zeros([self.max_num_boxes, self.max_num_char], dtype=np.int32)
        bboxes = np.zeros([self.max_num_boxes, 4], dtype=np.float32)
        quad_bboxes = np.zeros([self.max_num_boxes, 4,2], dtype=np.float32)
        labels[:, 0] = self.vocab['<sos>']
        word_id_list = df['word_idx'].unique().tolist()
        if len(word_id_list) > self.max_num_boxes:
            word_id_list = np.random.choice(word_id_list, replace=False, size=self.max_num_boxes)
        for iter_i, word_ind in enumerate(word_id_list):
            word_df = df.loc[df['word_idx'] == word_ind]
            # word_df.sort_values('seq', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
            # print(word_df['label'].values)
            # print(word_df)
            word_list = []
            len_word_df = word_df.shape[0]
            minx1 = 999999
            miny1 = 999999
            maxx2 = 0
            maxy2 = 0
            for row_i, row in enumerate(word_df.itertuples(index=False)):
                ch = row[8].lower()

                bbox = np.array([[float(row[0]),float(row[1])],
                                    [float(row[2]),float(row[3])],
                                    [float(row[4]),float(row[5])],
                                    [float(row[6]),float(row[7])]])
                bbox *= ratio
                minx1 = min(np.min(bbox[:, 0]), minx1)
                miny1 = min(np.min(bbox[:, 1]), miny1)
                maxx2 = max(np.max(bbox[:, 0]), maxx2)
                maxy2 = max(np.max(bbox[:, 1]), maxy2)
                word_list.append(bbox.copy())
                self.add_character(target_char, bbox)
                
                if ch in self.vocab:
                    ch_id = self.vocab[ch]
                else:
                    ch_id = self.vocab['<unk>']
                labels[iter_i, row_i+1] = ch_id
            labels[iter_i, len_word_df+1] = self.vocab['<eos>']
            bboxes[iter_i, :] = [miny1, minx1, maxy2, maxx2]
            
            segmap = target_char[int(miny1):int(maxy2), int(minx1):int(maxx2)] + target_affinity[int(miny1):int(maxy2),int(minx1):int(maxx2)]
            
            # make box
            np_contours = np.roll(np.array(np.where(segmap!=0)),1,axis=0).transpose().reshape(-1,2)
            rectangle = cv2.minAreaRect(np_contours)
            quad_box = cv2.boxPoints(rectangle).astype(np.float32)

            # align diamond-shape
            w, h = np.linalg.norm(quad_box[0] - quad_box[1]), np.linalg.norm(quad_box[1] - quad_box[2])
            box_ratio = max(w, h) / (min(w, h) + 1e-5)
            if abs(1 - box_ratio) <= 0.1:
                l, r = min(np_contours[:,0]), max(np_contours[:,0])
                t, b = min(np_contours[:,1]), max(np_contours[:,1])
                quad_box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

            # make clock-wise order
            startidx = quad_box.sum(axis=1).argmin()
            quad_box = np.roll(quad_box, 4-startidx, 0)
            quad_box[:,0] += minx1
            quad_box[:,1] += miny1
            quad_bboxes[iter_i, :] = np.clip(quad_box, a_min=0, a_max=width)
            
            # for word in text:
            for char_num in range(len(word_list) - 1):
                self.add_affinity(target_affinity,
                                    word_list[char_num],
                                    word_list[char_num + 1])
        target_char = cv2.resize(target_char, (int(self.image_size//2), int(self.image_size//2))) / 255.0
        target_affinity = cv2.resize(target_affinity, (int(self.image_size//2), int(self.image_size//2))) / 255.0
        bboxes /= self.image_size
        quad_bboxes /= self.image_size
        return image.astype(np.float32), target_char.astype(np.float32), target_affinity.astype(np.float32), labels, bboxes, quad_bboxes
    
    

if __name__ == '__main__':
    dataset = Dataset('/home/jylim2/PycharmProjects/hjmoon_github/ocr/synth_images/results/gts')
    for ii, ind in enumerate(dataset.gt_path_ind):
        image, char, affinity = dataset.process(ind)
        cv2.imshow('test',image)
        cv2.imshow('test1',char)
        cv2.imshow('test2',affinity)
        cv2.waitKey(0)
        assert False
