# Conditional GAN

기존 GAN 에 조건(description)을 추가함. 
[참고](https://arxiv.org/pdf/1411.1784.pdf)

## Related Work
   1. Mutil-model Learning For Image Labelling
      최근에 supervised CNN 학습의 성공. 라벨링에 대한 두 가지 문제제기, 첫 째 예측 너무 큰 카테고리의 수 학습, 두 번째 학습 input과 output 이 one to one으로 mapping 때문에 어렵다는 문제제기. 이 두 문제의 접근 방법 첫 번째 문제. label에 대한 NLP같은 자연어의 추가 정보를 사용하는 방법으로 다양한 카테고리의 학습을 도움. 단순히 이미지 특징에서 워드 특징으로 선형적인 맵핑 분류 성능이 증가한다고 함. 두 번째는 conditional probabilistic generative model을 이용하면 된다고 함.
      
## Conditional Adversarial Nets

   1. GAN
      Generator, Descriptor
      minG maxD V (D, G) = Ex∼pdata(x)[log D(x)] + Ez∼pz(z)[log(1 − D(G(z)))].
   
   2. Conditional Adverasarial Nets
      input에 추가 조건 y 추가함.
      
      minG maxD V (D, G) = Ex∼pdata(x)[log D(x|y)] + Ez∼pz(z)[log(1 − D(G(z|y)))].

