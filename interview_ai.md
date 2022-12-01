# 한양대학교 인공지능 융합대학원 면접 준비

🧐 AI, ML, DL의 차이점
```
```
🧐 지도학습, 비지도학습, 강화학습 정의 및 종류
```
```
🧐 Eigenvector, eigenvalue란
```
```
🧐 PCA 개념 설명
```
```
🧐 Singular value decomposition이란
```
```
🧐 MLE, MAP의 가장 큰 차이점
```
```
🧐 베이즈 정리
```
사건 B가 발생함으로써 사건 A의 확률이 어떻게 변화하는지를 표현한 정리를 말한다.
따라서 베이즈 정리는 새로운 정보가 기존의 추론에 어떤 영향을 미치는지를 나타낸다.

통계적으로 표현하면, 두 확률 변수간 사전확률과 사후확률 사이의 관계를 나타내는 정리라고 할 수 있다.
- 사전확률 : 정보가 없을 때의 확률
- 사후확률 : B라는 정보가 주어졌을 때의 확률
+ 사전확률로부터 사후확률을 구할 수 있다.
```
🧐 SVM이란
```
```
🧐 kernel trick 이란
```
linear 하게 분류선을 긋지 못하는 경우, 차원을 고차원으로 확대하여 linear학 구분선을 긋도록 해주는 작업이다.
즉, 저차원 공간을 고차원공간으로 매핑해주는 작업을 kernel trick 이라고 한다.
```
🧐 FC layer이란
```
Dense layer 라고도 하며 모든 층의 뉴런이 다음 층에 모든 뉴런과 연결된 상태를 말한다.
2차원 벡터를 1차원 벡터로 평탄화할 때, 활성화함수를 사용할 때(relu, sigmoid), softmax로 분류할 때 사용되는 layer 이다.
```
🧐 활성화 함수 종류와 사용하는 이유
```
```
🧐 Backpropagation(역전파)란
```
```
🧐 경사하강법이란
```

```
🧐 Convolution layer란
```
filter, kernel 이라고도 하며, convolution(합성곱)을 이용해 feature를 추출하는 layer 이다.
window 를 일정간격으로 이동하면서 계산한다.
ex) padding 과 pooling
- padding : 입력데이터 주위에 0을 채워 출력크기를 동일하게 적용하는 기법
- pooling : 데이터의 크기를 줄이는 것, sub sampling이라고 부르기도 한다.
  convolution layer와 activation function을 거쳐 나온 output 인 activation feature map 에 대해서 이 기법을 사용한다.
  
  예를 들면, max pooling, average pooling 등이 있다.
  max pooling 은 각 sub sample에 대해서 가장 큰값을 고르는것이고, average는 평균값을 취하는 방식이다.
```
🧐 Batch normalization란
```
```
🧐RNN/LSTM/ATTENTION란
```
```
