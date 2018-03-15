<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
# [Leveraging Weakly Annotated Data for Fashion Image Retrieval and Label Prediction](https://arxiv.org/pdf/1709.09426.pdf)

## Abstract
weakly supervised 학습을 기본으로 이커머스의 noisy한 다양한 데이터 셋을 라벨 규칙 없이 학습하려한다.경험적인 표현(description)이 이미지 검색에 적합하다는 것을 증명하려 한다.

## Introduction
소비자의 46% 는 원하는 상품을 고르기위해서 검색 엔진을 사용한다. 31%는 브랜드로 찾는다.(대한민국의 경우는 어떤지 모름..) 검색에서 상품의 설명 정보는 보다 좋은 검색을 위해 중요하다. 그러나 visual 검색엔진에서 이 데이터를 사용하지 않음.

CNN( convolutional neural network ) 덕분에 이미지 특징 찾는게 좋아짐. 그러나 두 가지 문제가 있음.  

    1) ImageNet학습 형식은 데이터셋으로 볼 때 이커머스와 매우 거리가 있다.

    2) 그래서 이커머스의 클래스 셋 구성에서 차이가 있다. 옷의 타입으로 구분되야 됨.( 예. 스커트와 드레스를 구분해야됨 ) 그리고 기장, 텍스쳐, 색상, 모양 등으로 구분되야됨.

학습의 목적은 이미지 검색을 잘하기 위한 좋은 visual feature를 추출하는 것. 

- low level (color, shapes, textures, fabric,...) 에서 high level (style, brand) 까지 다양한 visual semantic 정보를 사용
- clothing types, colors, attributes, textures 등등에 대한 단순 classifier를 학습
- 이미지 사이에 의미있는 유사성을 제공하기 위해, 이미지 context를 사용

그래서 크롤링 한 이미지의 weakly annotation을 사용. 이 annotation은 이미지의 textual description이다.
[9][Learning Visual Features from Large Weakly Supervised Data](https://arxiv.org/pdf/1511.02251.pdf)에서 말한 model presented의 분석에 포커싱 한다.

## Learning Image and Text Embeddings with Weak Supervision
이커머스 fashion dataset 학습의 이슈 중에 하나는 large dataset의 결핍이다. 그리고 unique, clean labeling문제까지 포함해서 이슈가 된다. 데이터 처리의 높은 비용이슈로 visual concept과 noise가 포함된 카탈로그의 description을 사용한다.

### Weakly Supervised Approach
텍스트의 노이즈 데이터 처리. 공통 사용언어 (ex. 'we', 'present'), 형용사 (ex. 'wonderful'), 또는 non visual words (ex. 'xl', 'cm') 등.. text dataset에 대하여 가능한 가볍게 전처리 함.

### Model
k 개 multi label vocabulary를 사용. image embedding은 softmax activation 사용.
두 네트워크를 합침. (dot product)
좀 더 봐야됨..

image embedding is given to a classification layer

<a href="https://www.codecogs.com/eqnedit.php?latex=\hat{y}=softmax(W^{T}&space;,&space;z)" target="_blank"><img src="https://latex.codecogs.com/png.latex?\hat{y}=softmax(W^{T}&space;,&space;z)" title="\hat{y}=softmax(W^{T} , z)" /></a>

- z is **CNN** visual feature
- <a href="https://www.codecogs.com/eqnedit.php?latex=W\in&space;\mathbb{R}^{I\times&space;K}" target="_blank"><img src="https://latex.codecogs.com/png.latex?W\in&space;\mathbb{R}^{I\times&space;K}" title="W\in \mathbb{R}^{I\times K}" /></a>
-  <a href="https://www.codecogs.com/eqnedit.php?latex=K\in&space;[1,K]" target="_blank"><img src="https://latex.codecogs.com/png.latex?K\in&space;[1,K]" title="K\in [1,K]" /></a> ( k labeling )
- <a href="https://www.codecogs.com/eqnedit.php?latex=w_{k}=W[:,k]" target="_blank"><img src="https://latex.codecogs.com/png.latex?w_{k}=W[:,k]" title="w_{k}=W[:,k]" /></a> (  **corresponds** to the embedding of the
k-th word in the vocabular )

### Label imbalance management
text dataset의 non visual ( ex. 'xl', 'cm', 'size'등.. ) imbalance를 다루기 위해 가벼운 전처리 후 uniform sampling 수행. w word중 random으로 단어 선택( 1개?? ) 후 예측. ( predict w given x )

### Loss
각 이미지에서 k vocabulary 중 한 개 라벨을 예측하려 함. 아마 위에서 말한 random uniform sampling 인거 같음. cross-entropy loss를 사용. negative sum of log-probabilities

<a href="https://www.codecogs.com/eqnedit.php?latex=L(\Theta,W,\mathcal{D})=-\frac{1}{N}\sum\limits_{n=1}^N&space;\sum\limits_{k=1}^K&space;y_n^k\log&space;\frac{\exp(w_k^Tf(x_n,\Theta))}{\sum\limits_{I=1}^K\exp(w_i^Tf(x_n,\Theta))}" target="_blank"><img src="https://latex.codecogs.com/png.latex?L(\Theta,W,\mathcal{D})=-\frac{1}{N}\sum\limits_{n=1}^N&space;\sum\limits_{k=1}^K&space;y_n^k\log&space;\frac{\exp(w_k^Tf(x_n,\Theta))}{\sum\limits_{I=1}^K\exp(w_i^Tf(x_n,\Theta))}" title="L(\Theta,W,\mathcal{D})=-\frac{1}{N}\sum\limits_{n=1}^N \sum\limits_{k=1}^K y_n^k\log \frac{\exp(w_k^Tf(x_n,\Theta))}{\sum\limits_{I=1}^K\exp(w_i^Tf(x_n,\Theta))}" /></a> ( k vocab, n dataset, 걍 softmax loss 에 응용인 듯 )

### Implementation details
#### Negative sampling
각 이미지 샘플에서 vocabulary 의 모든 classes 확률을 계산하면 비용이 너무 크기 때문에 negative sampling을 함. negative sampling하는 방법은 각 이미지 샘플에서 정답(positive) label(word)를 선택 후, 정답에 대응되는 negative label(word)를 sampling함. 이렇게 선택된 words 로 score와 softmax를 계산함.
#### Learning
- basenet : resnet50
- dataset size : 약 1,300,000 장
- pre-trained weights : image-net
- last layer weights : random init
- last layer weights training : 20epoch
- solver : SGD
- batch size : 20
- initial lr : 0.1
- num_epochs_per_decay : 10 (향상이 없다면)
- learning_rate_decay_factor : 0.1
- total epoch : 20 (향상이 없다면)
#### Training dataset
여기저기서 crawling했다. label을 title, category name, description등으로 고려해서 만듬. word token NLTK사용. stop word제거, frequent non-relevant words 제거( 웹사이트 이름, 'collection', 'buy', ...) and non alphabetic word. 최소한의 preprocessing을 적용. 그렇게 나온 218,536 words 중 frequent 30,000개를 선택.
###Experiments and evaluation