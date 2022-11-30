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
  ```
- 🧐 Discriminative Model이란 무엇인가?  
  ```
  ```
- 🧐 Discrinator function이란 무엇인가?  
  ```
  ```
- 🧐 Overfitting 이란? [Answer Post]  
  ```
  ```
- 🧐 Underfitting이란? [Answer Post]  
  ```
  ```
- 🧐 Overfitting과 Underfitting은 어떤 문제가 있는가?  
  ```
  ```
- 🧐 Overfitting과 Underfitting을 해결하는 방법은? [Answer Post]  
  ```
  ```
- 🧐 Regularization이란?  
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
  ```
- 🧐 Stacking Ensemble이란?  
  ```
  ```
- 🧐 Bagging이란?  
  ```
  ```
- 🧐 Bootstrapping이란?  
  ```
  ```
- 🧐 Boosting이란?  
  ```
  ```
- 🧐 Bagging 과 Boosting의 차이는?  
  ```
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
