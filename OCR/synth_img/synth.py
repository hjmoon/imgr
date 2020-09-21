from PIL import Image, ImageDraw, ImageFont, ImageOps
import os
import glob
import math
import numpy as np
from config import cfg
from utils import *
from tqdm import tqdm
import cv2

np.random.seed(2222)

kor_corpus = glob.glob('/home/jylim2/dataset/bert/wiki/corpus_v2/*.txt')

# sample_corpus = '가나다라마바사아자차카타파하'
# f = open(kor_corpus[0], 'r', encoding='utf-8')
# lines = f.readlines()
# for line in lines:
#
# f.close()


class CorpusCanvas:

    def __init__(self, font, words, image_filepath):
        self.MAX_SIZE = 1024
        self.kor_font = font
        self.words = words 

        bg_image = Image.open(image_filepath)
        bg_image = bg_image.convert('RGB')
        bg_width, bg_height = bg_image.size
        max_size = max(bg_width, bg_height)
        scale_ratio = self.MAX_SIZE / max_size
        self.bg_image = bg_image.resize((int(bg_width*scale_ratio), int(bg_height*scale_ratio)), Image.ANTIALIAS)
        self.word_canvas_list = []
        self.colors = ['#00ffff', '#000000', '#0000ff', '#ff00ff', '#008000','#808080', '#00ff00', '#800000', '#000080', 
                       '#808000', '#800080', '#ff0000', '#c0c0c0', '#008080', '#fcfcfc', '#ffff00', '#ffa500']

    def compare_rois(self, box, boxes, debug):
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        box_area = (x2-x1) * (y2-y1)
        
        for bb in boxes:
            tx1 = bb[0]#get_left_roi(rois)
            ty1 = bb[1]#get_top_roi(rois)
            tx2 = bb[2]#get_bottom_roi(rois)
            ty2 = bb[3]#get_right_roi(rois)
            twidth = tx2-tx1
            theight = ty2-ty1
            tbox_area = twidth*theight
            overlaps = 1.0
            iw = ( min(tx2, x2) - max(tx1, x1) + 1)
            if iw > 0:
                ih = ( min(ty2, y2) - max(ty1, y1) + 1)
                if ih > 0:
                    ua = float( tbox_area + box_area - (iw * ih) )
                    overlaps = (iw * ih) / ua
                    if overlaps > 0.0:
                        return False
        if debug:
            print(overlaps)
        return True


    def draw_corpus(self):
        bg_width, bg_height = self.bg_image.size

        for word_ind, word in enumerate(self.words):
            if len(self.word_canvas_list) > 20:
                break
            text_type = np.random.choice([cfg.text_type.LINE, cfg.text_type.ROUND])
            tCanvas = TextCanvas(self.kor_font, word, text_type, self.colors)
            width, height = tCanvas.size()
            diff_x = int(bg_width - width)
            if diff_x <= 0:
                continue
            diff_y = int(bg_height - height)
            if diff_y <= 0:
                continue
            
            # print('test:',tCanvas.labels())
            roi = tCanvas.rois()
            left = get_left_roi(roi)
            top = get_top_roi(roi)
        
            tbboxes = np.zeros((len(self.word_canvas_list), 4), np.float32)
            for ii, word in enumerate(self.word_canvas_list):
                rois = word.rois()
                tleft = get_left_roi(rois)
                ttop = get_top_roi(rois)
                tbottom = get_bottom_roi(rois)
                tright = get_right_roi(rois)
                tbboxes[ii] = [tleft, ttop, tright, tbottom]
                
            for _ in range(100):
                random_x = np.random.randint(0, diff_x, size=1)
                random_y = np.random.randint(0, diff_y, size=1)
                _left = left + random_x[0]
                _top = top + random_y[0]
                
                if self.compare_rois(np.array([_left, _top, _left+width, _top+height]), 
                                     tbboxes, 
                                     False) and _left > 0 and _top > 0 and (_left+width) < bg_width and (_top+height) < bg_height:
                    tCanvas.translate_2d(random_x, random_y)
                    self.word_canvas_list.append(tCanvas)
                    tCanvas.paste_image(self.bg_image)
                    break
        # print(len(self.word_canvas_list))
        # assert False
#        for i in range(10000):
        # for text in self.word_canvas_list:
        #    text.paste_image(self.bg_image)
        # print('draw:',bg_width, bg_height)
        # cv2.imshow('test',np.array(self.bg_image))
        # cv2.waitKey(0)

    def store_file(self, gt_file, store_image_filepath):
        gt_file.write('x1\ty1\tx2\ty2\tx3\ty3\tx4\ty4\tlabel\tword_idx\n')
        if len(self.word_canvas_list) > 0:
            self.bg_image.save(store_image_filepath, "JPEG", quality=80, optimize=True, progressive=True)
            word_ind = 0
            for word in self.word_canvas_list:
                image_path_split = image_store_path.split('/')[-2:]
                local_image_path = os.path.join(image_path_split[0], image_path_split[1])
                labels = word.labels()
                rois = word.ch_rois()
                for ll, roi in zip(labels, rois):
                    if ll == ' ':
                        word_ind += 1
                        continue
                    # print(ll, roi)
                    gt_file.write('%0.4f\t%0.4f\t%0.4f\t%0.4f\t%0.4f\t%0.4f\t%0.4f\t%0.4f\t%s\t%d\n'%(
                        roi[0,0],roi[0,1],roi[1,0],roi[1,1],roi[2,0],roi[2,1],roi[3,0],roi[3,1],ll,word_ind))
                word_ind += 1
                # print(len(labels), len(rois))
                # assert False
                
                # labels = ''.join([c for c in word.labels()])
                # rois = ','.join([str(v) for v in word.rois().reshape(-1)])
                # gt_file.write(local_image_path+'\t'+labels+'\t'+rois+'\n')




class TextCanvas:

    def __init__(self, font, text, text_type, colors):
        self.text_type = text_type
        color = np.random.choice(colors)
        word_info, max_height = self._init_pos(font, text, color)
        
        if self.text_type == cfg.text_type.LINE:
            self.bg_left, self.bg_top, self.bg_width, self.bg_height, self.word_info = self._set_pos(word_info, max_height)
        elif self.text_type == cfg.text_type.ROUND:
            self.bg_left, self.bg_top, self.bg_width, self.bg_height, self.word_info = self._set_round_pos(word_info, max_height)


    def pos(self):
        return self.bg_left, self.bg_top

    def size(self):
        return int(self.bg_width), int(self.bg_height)

    def labels(self):
        return [ch['label'] for ch in self.word_info]

    def rois(self):
        return np.concatenate([ch['quad_roi'] for ch in self.word_info])

    def ch_rois(self):
        return [ch['quad_roi'] for ch in self.word_info]

    def translate_2d(self, tx, ty):
        self.bg_left += tx
        self.bg_top += ty
        for ch in self.word_info:
            ch['quad_roi'][:,0] += tx
            ch['quad_roi'][:,1] += ty

    def paste_image(self, image):
        if self.text_type == cfg.text_type.ROUND:
            self._draw_round_text(image, self.word_info)
        else:
            self._draw_text(image, self.word_info)
        if 0:
            bg_draw = ImageDraw.Draw(image)
            for ch in self.word_info:
                xx = ch['quad_roi'][:, 0]
                yy = ch['quad_roi'][:, 1]
                bg_draw.line([(xx[0], yy[0]), (xx[1], yy[1]), (xx[2], yy[2]), (xx[3], yy[3])], fill=(0, 0, 255), width=3)
        #image.paste(text_image, (int(self.bg_left), int(self.bg_top)), text_image)

    def _init_pos(self, font, text, color):
        word = []
        max_height = 0
        font_size = np.random.randint(20,60)
        imgFont = ImageFont.truetype(font, font_size)
        for ch in text:
            width, height = imgFont.getsize(ch)
            max_height = max(max_height, height)
            char = {'angle': 0,
                    'quad_roi': np.array([[0,0],
                                          [width-1,0],
                                          [width-1,height-1],
                                          [0,height-1]], np.float32),
                    'label': ch,
                    'font':imgFont,
                    'color': color}
            word.append(char)
        return word, max_height

    def _set_pos(self, word, max_height):
        for ch_ind in range(1,len(word),1):
            pre = word[ch_ind-1]
            cur = word[ch_ind]
            pre_right = np.max(pre['quad_roi'][:, 0])
            cur_bottom = np.max(cur['quad_roi'][:, 1])
            cur['quad_roi'][:, 0] = cur['quad_roi'][:, 0] + pre_right
            # center
            cur['quad_roi'][:, 1] = cur['quad_roi'][:, 1] + 0.5*(max_height - cur_bottom)
        return self._calc_line_text(word)

    def _set_round_pos(self, word, max_height):
        for ch_ind in range(1,len(word),1):
            pre = word[ch_ind-1]
            cur = word[ch_ind]
            pre_right = get_right_roi(pre['quad_roi'])
            cur_bottom = get_bottom_roi(cur['quad_roi'])
            cur['quad_roi'][:, 0], cur['quad_roi'][:, 1] = translate_pos_2d(cur['quad_roi'],
                                                                            pre_right,
                                                                            0.5*(max_height - cur_bottom))
        max_width = np.max(word[-1]['quad_roi'][:, 0]) + 1

        r = np.random.randint(int(max_width*0.5),int(max_width*1.5))
        angle = math.degrees(max_width / r)
        start_angle = 240 + np.random.randint(-20,+20)
        for ch in word:
            left = np.min(ch['quad_roi'][:, 0])
            if left == 0:
                ch_angle = start_angle
            else:
                ch_angle = start_angle+(angle * float(left) / max_width)
            # print(ch['label'],ch_angle,left)
            ch['angle'] = ch_angle
            cx = math.cos(math.radians(ch_angle))*r
            cy = math.sin(math.radians(ch_angle))*r
            ch['center'] = np.array([cx,cy])

        return self._calc_round_text(word)

    def _calc_round_text(self, word):
        left = 9999999
        top = 9999999
        right = -9999999
        bottom = -9999999
        for ch in word:
            cx = ch['center'][0]
            cy = ch['center'][1]
            quad_roi = ch['quad_roi']
            radians = math.radians(270 - ch['angle'])
            quad_roi[:, 0], quad_roi[:, 1] = translate_pos_2d(quad_roi,
                                                              -get_left_roi(quad_roi),
                                                              0)
            quad_roi[:, 0], quad_roi[:, 1] = translate_pos_2d(quad_roi,
                                                              -get_right_roi(quad_roi) * 0.5,
                                                              -get_bottom_roi(quad_roi) * 0.5)

            # print(quad_roi)
            quad_roi[:, 0], quad_roi[:, 1] = rotate_pos_2d(quad_roi, radians)
            roi_width, roi_height = get_size_roi(quad_roi)
            quad_roi[:, 0], quad_roi[:, 1] = translate_pos_2d(quad_roi,
                                                              cx + (roi_width * 0.5),
                                                              cy + (roi_height * 0.5))
            left = min(left, get_left_roi(quad_roi))
            right = max(right, get_right_roi(quad_roi))
            top = min(top, get_top_roi(quad_roi))
            bottom = max(bottom, get_bottom_roi(quad_roi))

        tx = 0 - left
        ty = 0 - top
        bg_width = right-left
        bg_height = bottom-top
        for ch in word:
            ch['center'][0], ch['center'][1] = translate_pos_2d(np.expand_dims(ch['center'], axis=0), tx, ty)
            ch['quad_roi'][:, 0], ch['quad_roi'][:, 1] = translate_pos_2d(ch['quad_roi'], tx, ty)
        
        left_from_bg = 999999999
        top_from_bg = 999999999
        for ch in word:
            _left = get_left_roi(ch['quad_roi'])
            _top = get_right_roi(ch['quad_roi'])
            left_from_bg = min(_left, left_from_bg)
            top_from_bg = min(_top, top_from_bg)
        return left_from_bg, top_from_bg, bg_width, bg_height, word#bg_image, word

    def _calc_line_text(self, word):
        last_ch = word[-1]
        bg_width = int(np.max(last_ch['quad_roi'][:, 0]) + 1)
        bg_height = int(np.max(last_ch['quad_roi'][:, 1]) + 1)

        return 0, 0, bg_width, bg_height, word#bg_image, word

    def _draw_round_text(self, bg_image, word):
        # bg_image = Image.new('RGBA', (bg_width, bg_height))
        bg_w, bg_h = bg_image.size
        for ch in word:
            #cx, cy = ch['center']
            # cx, cy = translate_pos_2d(np.expand_dims(ch['center'], axis=0), padx, pady)
            imgFont = ch['font']
            _width, _height = imgFont.getsize(ch['label'])
            txt = Image.new('RGBA', (_width, _height))
            d = ImageDraw.Draw(txt)
            d.text((0, 0), ch['label'], font=imgFont, fill=ch['color'])

            # w = txt.rotate(270 - ch['angle'], expand=1, resample=Image.BICUBIC)
            # left = get_left_roi(ch['quad_roi'])
            # top = get_top_roi(ch['quad_roi'])
            # 좌표 순서 - 상단왼쪽 끝, 상단오른쪽 끝, 하단왼쪽 끝, 하단오른쪽 끝
            # print(ch['quad_roi'])
            pts1 = np.float32([[0,0],
                                [_width-1,0],
                                [0,_height-1],
                                [_width-1,_height-1]])
            pts2 = np.float32([ch['quad_roi'][0],
                                    ch['quad_roi'][1],
                                    ch['quad_roi'][3],
                                    ch['quad_roi'][2]])

            # print(pts1)
            # print(pts2)

            M = cv2.getPerspectiveTransform(pts1,pts2)
            mask = Image.fromarray(cv2.warpPerspective(np.array(txt), M, (bg_w, bg_h)))
            # img_result = cv2.warpPerspective(img_original, M, (width,height))
            bg_image.paste(mask, (0, 0), mask)
            # if 0:
            #     xx = ch['quad_roi'][:, 0]
            #     yy = ch['quad_roi'][:, 1]
            #     bg_draw.line([(xx[0], yy[0]), (xx[1], yy[1]), (xx[2], yy[2]), (xx[3], yy[3])], fill=(0, 0, 255), width=3)

        # bg_image = bg_image.convert('RGB')
        # bg_image.show('test')
        # return bg_image

    def _draw_text(self, bg_image, word):
        draw = ImageDraw.Draw(bg_image)
        for ch in word:
            left = int(get_left_roi(ch['quad_roi']))
            top = int(get_top_roi(ch['quad_roi']))
            draw.text((left, top), ch['label'], font=ch['font'], fill=ch['color'])

        # bg_image.show('test')
        # return bg_image
if __name__ == '__main__':
    # font = np.random.choice(kor_fonts)
    # text_type = np.random.choice([cfg.text_type.LINE, cfg.text_type.ROUND])
    # word = WordCanvas(font, , text_type)
    
    kor_fonts = glob.glob('/usr/share/fonts/truetype/nanum/*.ttf')
    bg_images = glob.glob('bg_images/*.jpg')
    corpus = glob.glob('/home/jylim2/dataset/bert/wiki/corpus_v2/*.txt')


    # gt_store_path = os.path.join('./results/gts')
    
    for i in tqdm(range(7000)):
        image_store_path = os.path.join('./train/images','synth_2th_'+str(i)+'.jpg')
        with open(image_store_path.replace('images','gts').replace('jpg','txt'), 'w', encoding='utf8') as wf:
            font = np.random.choice(kor_fonts)
            sample_image = np.random.choice(bg_images)
            sample_corpus_list = np.random.choice(corpus, size=10, replace=False)
            words = []
            for sample_corpus in sample_corpus_list:
                words += get_corpus(sample_corpus, np.random.randint(10,28))
            max_num = min(len(words), 200)
            sample_words = np.random.choice(words, size=np.random.randint(5,max_num), replace=False)
            
            cCanvas = CorpusCanvas(font, sample_words, sample_image)
            cCanvas.draw_corpus()
            
            cCanvas.store_file(wf, image_store_path)
            wf.close()        
    print('finish')
