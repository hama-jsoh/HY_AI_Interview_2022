# HY_AI_Interview_2022
한양대학교 인공지능 융합대학원 면접준비
  
----
  
[현지]  

### Statistic / Probability

- 🧐 Central Limit Theorem 이란 무엇인가?  
- 🧐 Central Limit Theorem은 어디에 쓸 수 있는가?
- 🧐 큰수의 법칙이란?
- 🧐 확률이랑 통계랑 다른 점은?
- 🧐 Marginal Distribution이란 무엇인가?
- 🧐 Conditional Distribution이란 무엇인가?  
- 🧐 Bias란 무엇인가?  [Answer Post]
- 🧐 Biased/Unbiased estimation의 차이는?  [Answer Post]
- 🧐 Bias, Variance, MSE란? 그리고 그들의 관계는 무엇인가?
- 🧐 Sample Variance란 무엇인가?
- 🧐 Variance를 구할 때, N대신에 N-1로 나눠주는 이유는 무엇인가?
- 🧐 Gaussian Distribution에서 MLE와 Sample Variance 중에 어떤 걸 사용해야 하는가?
- 🧐 Unbiased Estimation은 무조건 좋은가?
- 🧐 Unbiaed Estimation의 장점은 무엇인가?  
- 🧐 Binomial, Bernoulli, Multinomial, Multinoulli 란 무엇인가?
- 🧐 Beta Distribution과 Dirichlet Distribution이란 무엇인가?
- 🧐 Gamma Distribution은 어디에 쓰이는가?
- 🧐 Possion distribution은 어디에 쓰이는가?
- 🧐 Bias and Varaince Trade-Off 란 무엇인가? [Answer Post]
- 🧐 Conjugate Prior란?
- 🧐 Confidence Interval이란 무엇인가?
- 🧐 covariance/correlation 이란 무엇인가?
- 🧐 Total variation 이란 무엇인가?
- 🧐 Explained variation 이란 무엇인가?
- 🧐 Uexplained variation 이란 무엇인가
- 🧐 Coefficient of determination 이란? (결정계수)
- 🧐 Total variation distance이란 무엇인가?
- 🧐 P-value란 무엇인가?
- 🧐 likelihood-ratio test 이란 무엇인가?
  
----
  
### Machine Learning

- 🧐 Frequentist 와 Bayesian의 차이는 무엇인가?
- 🧐 Frequentist 와 Bayesian의 장점은 무엇인가?
- 🧐 차원의 저주란?
- 🧐 Train, Valid, Test를 나누는 이유는 무엇인가?
- 🧐 Cross Validation이란?
- 🧐 (Super-, Unsuper-, Semi-Super) vised learning이란 무엇인가?
  - Supervised Learning
  - Unsupervised Learning
  - Semi-Supervised Learning
- 🧐 Decision Theory란?
- 🧐 Receiver Operating Characteristic Curve란 무엇인가?
- 🧐 Precision Recall에 대해서 설명해보라
- 🧐 Precision Recall Curve란 무엇인가?
- 🧐 Type 1 Error 와 Type 2 Error는?

----
----

[주석]

- 🧐 Entropy란 무엇인가?  
  ```
  entropy 는 무질서도를 의미한다.  
  따라서 엔트로피 값이 클수록 데이터가 혼재되어있어 분류가 잘 되어있지 않은 상태이고  
  그 값이 작을 수록 데이터가 잘 분리되어있다는 뜻이다.  
  즉 불확실성의 정도를 나타내는 수치라고 볼 수 있다.  
  ```
- 🧐 KL-Divergence란 무엇인가?  
  ```
  Kullback Leibler Divergence란 기본적으로 두 개의 서로 다른 확률 분포를 비교하는 방법이다.
  관찰된 데이터 혹은 복잡한 데이터 분포를 이를 나타낼 수 있는 확률 통계 모델로 나타내었을 때,  
  이러한 모델이 실제 관측치 대비 얼마나 정보를 잃어버렸는지 측정한다.  
  ```
- 🧐 Mutual Information이란 무엇 인가?  
  ```
  임의의 두 확률 변수 X 와 Y 가 독립적이라면 결합(Joint)확률 분포는 확률 곱으로 표현 가능합니다.
  
  P(X,Y) = P(X)P(Y)
  
  만약 X,Y 가 서로 독립적이지 않다면 확률 곱과 결합 확률 분포간의 차이를 KL divergence로 측정할 수 있습니다.
  이와 같이 Mutual Information은 두 확률 변수들이 얼마나 서로 dependent 한 지 측정하며 서로 independent한 경우  
  Mutual Information = 0을 만족하고 서로 dependent할 수록 값이 커집니다.
  즉, divergnece가 크다고 해석할 수 있습니다. = 변수 간 의존성이 크다.
  ```
- 🧐 Cross-Entropy란 무엇인가?  
  ```
  크로스 엔트로피는 엔트로피와 상당히 유사한 개념이다.  
  엔트로피가 정답이 나올 확률만을 대상으로 측정한 값이었다면  
  크로스 엔트로피는 모델에서 예측한 확률과 정답확률을 모두 사용해 측정한 값이다.  

  크로스 엔트로피는, 모델에서 예측한 확률값이 실제값과 비교했을 때 틀릴 수 있는 정보량을 의미한다.  
  엔트로피와 마찬가지로 그 값이 적을 수록 모델이 데이터를 더 잘 예측하는 모델임을 의미한다.  

  딥러닝의 손실함수로 많이 사용되는 개념으로  
  딥러닝 모델에선 예측값과 정답값의 크로스 엔트로피 값을 줄이기 위해 가중치와 편향을 업데이트하며 학습을 수행한다.  
  
  즉, 크로스 엔트로피의 의미는 실제 데이터는 분포 P(X) 로부터 생성되지만,  
  분포 Q(X) 를 사용하여 정보량을 측정해서 나타낸 정보량의 기대값을 의미합니다.
  ```
- 🧐 Cross-Entropy loss 란 무엇인가?  
  ```
  머신 러닝의 분류 모델이 얼마나 잘 수행되는지 측정하기 위해 사용되는 지표입니다.
  Loss(또는 Error)는0은 완벽한 모델로0과 1 사이의 숫자로 측정됩니다.  
  일반적인 목표는 모델의 Loss 를 가능한 0에 가깝게 만드는 것입니다.
  
  Cross Entropy Loss은 머신 러닝 분류 모델의 발견된 확률 분포와 예측 분포 사이의 차이를 측정합니다.  
  예측에 대해 가능한 모든 값이 저장되므로, 예를 들어, 동전 던지기에서 특정 값(odd)을 찾는 경우  
  해당 정보가 0.5와 0.5(앞면과 뒷면)에 저장됩니다.
  
  반면Binary Cross Entropy Loss는 하나의 값만 저장합니다.  
  즉, 0.5만 저장하고 다른 0.5는 다른 문제에서 가정하며,  
  첫 번째 확률이 0.7이면 다른 0.5는 0.3)이라고 가정합니다.  
  또한 알고리듬(Log loss)을 사용합니다.
  
  이러한 이유로Binary Cross Entropy Loss (또는Log loss)은 결과값이 두 개뿐인 시나리오에서 사용되며,  
  세 개 이상일 경우 즉시 실패하는 위치를 쉽게 알 수 있습니다. 
  세 가지 이상의 분류 가능성이 있는 모델에서 Cross Entropy Loss가 자주 사용됩니다.
  ```
- 🧐 Generative Model이란 무엇인가?  
  ```
  생성모델은 주어진 학습 데이터를 학습하여 학습데이터의 분포를 따르는 유사한 데이터를 생성하는 모델이다.
  
  여러가지 종류의 생성모델이 있는데,
  - Explicit density : 학습 데이터의 분포를 기반으로 하는 방법
  - Tractable density : 학습데이터의 분포를 직접적으로 구하는 방법 
  - Approximate density : 분포를 단순히 추정하는 방법
  - Generative Adversarial Network, GAN : Generator가 학습 데이터의 분포를 학습하고 이 분포를 재현하여 
    원 데이터의 분포와 차이가 없도록 하여 Discriminator가 실제 데이터인지 생성한 가짜 데이터인지 구별해서
    각각에 대한 확률을 추정하는 방법
  ```
- 🧐 Discriminative Model이란 무엇인가?  
  ```
  Discriminative Model(ex, Logistic regression, Neural Networks)은 
  각 class의 차이에 주목해 바로바로 어떤 class에 들어가야 할지 결정해주는 모델이다.
  
  반면에, Generative Model(ex, Gaussian Mixture Model(GMM))은 
  각 class의 분포에 주목하여 어떤 분포에 들어갈 가능성이 가장 많은지 결정해주는 모델이다.
  ```
- 🧐 Discrinator function이란 무엇인가?  
  ```
  실제 데이터와 Generator 가 생성한 데이터를 구분하는 모델
  ```
- 🧐 Overfitting 이란? [Answer Post]  
  ```
  모델의 파라미터들을 학습 데이터에 너무 가깝게 맞췄을 때 발생하는 현상.
  즉, 학습 데이터가 실제 세계에서 나타나는 방식과 완전히 똑같을것이라고 가정해버리는 것이다.
  
  Overfitting은 너무 세밀하게 학습 데이터 하나하나를 다 설명하려고 하다보니 
  정작 중요한 패턴을 설명할 수 없게 되는 현상을 말한다.
  ```
- 🧐 Underfitting이란? [Answer Post]  
  ```
  ```
- 🧐 Overfitting과 Underfitting은 어떤 문제가 있는가?  
  ```
  Overfitting은 모델 학습 오류가 테스트 데이터의 오류보다 훨씬 작은 경우를 의미하고,
  Underfitting은 모델이 학습 오류를 줄이지 못하는 상황을 의미한다.
  
  즉, Overfitting이 되면 training data에 대한 정확도는 좋지만 실제 test data에 대해서는 에러가 많이 생길 수 있다.
  반면에 Underfitting은 모델이 지나치게 일반화해서 training data에 대해서 학습이 제대로 되지 않는 상태를 말한다.
  ```
- 🧐 Overfitting과 Underfitting을 해결하는 방법은? [Answer Post]  
  ```
  [Overfitting]
  1. 데이터 그룹별 통계치 확인하고 시각화해서 데이터 패턴 확인하기
  2. 애초에 적절하게 수집된 데이터인지 확인하기
  3. Data Augmentation 하기, 다양한 상황의 데이터 만들기
  4. 학습 데이터에 포함될 특성(featureset) 제한하기
  5. 모델의 복잡도 줄이기, 모델이 복잡하면 데이터에 fitting 하기위해서 모델의 계수가 증가하는데
    이렇게되면 값에 민감하게 반응하게되고, 테스트데이터에 대해서 모델이 높은에러를 갖게된다.
    
  [Underfitting]
  1. 학습 횟수 늘리기
  2. 데이터의 특성에 비해 모델이 너무 간단할 때 발생하므로 알맞는 모델 선택
  3. 데이터의 양이 너무 적을때 발생할 수 있는 문제이므로 추가 데이터 수집
  ```
- 🧐 Regularization이란?  
  ```
  모델이 복잡할수록 모델의 계수가 증가하여 입력값에 대해 민감하게 반응하는 경향이 있는데,
  Regularization은 모델의 계수를 0에 가깝게 줄여서
  모델이 data에 대해서 너무 민감하게 반응하지 않게 모델의 복잡도를 줄이는것을 말한다.
  이렇게 되면 모든 데이터에 대해 fitting 되지 않을것이고 그럼 Overfitting 문제를 해결하는데 기여하는 방법이 될 수 있다.
  ```
  추가 질문, Normalization이란?
  ```
  ```
  추가 질문, Standardization이란?
  ```
  ```
  - Ridge
    ```
    ```
  - Lasso  
    ```
    ```
- 🧐 Activation function이란 무엇인가?3가지 Activation function type이 있다.  
  ```
  ```
  - Ridge activation Function
    ```
    ```
  - Radial activation Function
    ```
    ```
  - Folding activation Function  
    ```
    ```
- 🧐 CNN에 대해서 설명해보라  
  ```
  ```
- 🧐 RNN에 대해서 설명해보라  
  ```
  ```
- 🧐 Netwon's method란 무엇인가?  
  ```
  ```
- 🧐 Gradient Descent란 무엇인가?  
  ```
  ```
- 🧐 Stochastic Gradient Descent란 무엇인가?  
  ```
  ```
- 🧐 Local optimum으로 빠지는데 성능이 좋은 이유는 무엇인가?  
  ```
  ```
- 🧐 Internal Covariance Shift 란 무엇인가?  
  ```
  ```
- 🧐 Batch Normalization은 무엇이고 왜 하는가?  
  ```
  ```
- 🧐 Backpropagation이란 무엇인가?  
  ```
  ```
- 🧐 Optimizer의 종류와 차이에 대해서 아는가?  
  ```
  ```
- 🧐 Ensemble이란?  
  ```
  여러개의 Decision Tree를 결합해서 다양성을 획득하여 하나의 Tree보다 더 좋은 성능(=오분류율 감소, 전반적인 예측력 상승)을 내는 머신러닝 기법이다.
  앙상블 학습의 핵심은 여러개의 Week Classifier를 결합해 Strong Classifier를 만들어 정확성을 향상시키는것이다.
  ```
- 🧐 Stacking Ensemble이란?  
  ```
  메타 모델링이라고도 부르는데,
  서로 다른 모델들을 조합해서 최고의 성능을 내는 모델을 생성하는 방법인데, 이렇게만 보면 Ensemble과 다를게 없어보인다.
  하지만, 다른점이라면 각각의 모델이 예측한 데이터를 다시 training data로 사용해서 재학습을 진행한다는 점이다.
  
  개별 알고리즘(모델)의 예측 결과를 training dataset 으로 하여 최종적인 meta dataset을 만들어 별도 머신러닝 알고리즘으로 최종 학습을 수행하는것을 말한다.
  개별 모델의 예측된 dataset을 학습하고 예측하는 방식을 메타 모델이라고 한다.
  
  일반적으로 성능이 비슷한 모델을 결합해 좀 더 나은 성능 향상을 도출하기 위해 적용한다.
  각 모델의 장점은 취하고 약점을 보완하는 방법이라고 할 수 있다.
  ```
- 🧐 Bagging이란?  
  ```
  앙상블 기법 중 하나, bagging은 bootstrap aggregation 의 약자이다.
  샘플을 여러번 뽑아서(= Bootstrap = 복원랜덤샘플링) 조금씩 서로 다른 훈련 데이터를 생성하고,
  각각의 모델을 병렬로 학습시키고 각각의 결과물들을 합쳐서 최종 결과값을 구하는 방식이다.
  
  - categorical data는 voting 방식으로 집계하고, (= 가장 많이 출현한 값)
    .. ex) 6개의 결정 트리 모델이 있다고 합시다. 4개는 A로 예측했고, 2개는 B로 예측했다면 투표에 의해 4개의 모델이 선택한 A를 최종 결과로 예측한다는 것.
  - continuous data는 평균으로 집계한다. (= 말 그대로 평균)
  
  대표적인 예로 Random Forest 가 있다.
  ```
- 🧐 Bootstrapping이란?  
  ```
  머신러닝에서의 Bootstrapping 은 원래의 데이터셋에서 랜덤 샘플링으로 학습데이터를 늘리는 방법을 말한다.
  데이터의 양을 늘릴 수 있고, 분포를 고르게 만들 수 있는 효과가 있다.
  
  이를 이용해 Ensemble을 할 수 있다.
  ```
- 🧐 Boosting이란?  
  ```
  Boosting은 잘못 분류된 객체들에 집중하여(가중치를 두어) 새로운 분류 규칙을 생성하는 단계를 반복하는 순차적 학습 알고리즘이다.
  병렬처리인 Bagging과는 다르게 순차적인것이 특징이다.
  
  예를 들면, Bagging은 각각의 Decision Tree가 서로 독립적으로(=병렬) 결과를 예측하는 반면에,
  Boosting은 처음 Decision Tree가 다음 Decision Tree에 영향을 주어 최종 결과를 예측한다.
  ```
- 🧐 Bagging 과 Boosting의 차이는?  
  ```
  Bagging 은 병렬, Boosting은 앞의 모델이 나중 모델에 영향을 주는 순차적 진행 구조의 모델이다.
  Boosting 은 한번 학습이 끝난 후 결과에 따라 가중치를 부여하는데 부여된 가중치가 다음 모델의 결과 예측에 영향을 준다.
  
  오답에 대해서 높은 가중치를 곱하고, 정답에 대해서는 낮은 가중치를 곱한다. 따라서 오답을 정답으로 맞추기 위해 오답에 더 집중하는 방법이다.
  
  Boosting은 Bagging에 비해 error가 적다. 즉, 성능이 더 좋다.
  하지만, 속도가 느리고 Overfitting이 될 가능성이 있다.
  
  개별적인 Decision Tree의 낮은 성능이 불만이면 Boosting이 적합하고, Overfitting 이 골치라면 Bagging 이 적합하다.
  ```
- 🧐 AdaBoost / Logit Boost / Gradient Boost  
  ```
  ```
- 🧐 Support Vector Machine이란 무엇인가?  
  ```
  ```
- 🧐 Margin을 최대화하면 어떤 장점이 있는가?  
  ```
  ```
  
----
  
### Linear Algebra

- 🧐 Linearly Independent란?  
  ```
  ```
- 🧐 Basis와 Dimension이란 무엇인가?  
  ```
  ```
- 🧐 Null space란 무엇인가?  
  ```
  ```
- 🧐 Symmetric Matrix란?  
  ```
  ```
- 🧐 Possitive-definite란?  
  ```
  ```
- 🧐 Rank 란 무엇인가?  
  ```
  ```
- 🧐 Determinant가 의미하는 바는 무엇인가?  
  ```
  ```
- 🧐 Eigen Vector는 무엇인가?  
  ```
  ```
- 🧐 Eigen Vector는 왜 중요한가?  
  ```
  ```
- 🧐 Eigen Value란?  
  ```
  ```
- 🧐 SVD란 무엇인가?→ 중요한 이유는?  
  ```
  ```
- 🧐 Jacobian Matrix란 무엇인가? 
  ```
  ```
