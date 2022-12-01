# HY_AI_Interview_2022
한양대학교 인공지능 융합대학원 면접준비
  
----
  
[현지]  

### Statistic / Probability

🧐 [Central Limit Theorem 이란 무엇인가?](https://blog.naver.com/PostView.naver?blogId=angryking&logNo=222414551159&parentCategoryNo=&categoryNo=22&viewDate=&isShowPopularPosts=true&from=search)
```
데이터의 크기(n)가 일정한 양을 넘으면, 평균의 분포는 정규분포에 근사하게 되며, 표준편차는 모집단의 표준편차를 표본수의 제곱근으로 나눈 값과 근사.
즉 모집단으로부터 무작위로 표본을 여러 번 추출한 다음, 추출된 각각의 표본들의 평균을 분포로 그려보면 정규분포의 형태를 가짐.
주의해야할 점은 표본의 양이 충분하면 표본의 평균이 모집단의 평균과 유사해진다는 뜻이 아니라 표본을 여러 번 추출 했을 때 각각의 표본 평균들의 분포가 정규분포를 이룸.
```
🧐 Central Limit Theorem은 어디에 쓸 수 있는가?
```
중심극한정리는 통계학에 있어 추정과 가설검정을 위한 핵심적인 이론으로 가설검정에 사용됨.
더 나아가 데이터 과학을 위한 예측 모델링 가능
```
🧐 [큰수의 법칙이란?](https://namu.wiki/w/%ED%81%B0%20%EC%88%98%EC%9D%98%20%EB%B2%95%EC%B9%99?__cf_chl_tk=QK.VtsHZHCpJfNb.mzrLeeskXzahWBKOp5M9paBlyAg-1669796077-0-gaNycGzNCdE)
```
경험적 확률과 수학적 확률 사이의 관계를 나타내는 법칙으로 표본집단의 크기가 커지면 그 표본평균이 모평균에 가까워짐을 의미. 
따라서 취합하는 표본의 수가 많을수록 통계적 정확도가 올라감
```
🧐 확률이랑 통계랑 다른 점은?
```
확률은 어떤 사건이 일어날 수 있는 수학적기대치
확률 = 특정사건이 일어날 개수 / 전체 사건이 일어날 개수
통계는 이미 발생한 사건이나 앞으로 발생될 사건에 대해서 수준파악, 예측자료로 사용할 데이터 분석 과정으로 반복횟수가 한번이 아닌 여러 번
```
🧐 Marginal Distribution이란 무엇인가?
```
개별사건의 확률이지만 결합사건들의 합으로 표시될 수 있는 확률
X=0으로 고정할 때 P(X=0,Y=0)+P(X=0,Y=1)=P(X=0) 도출될때 X는 고정되었지만 Y의 값은 계속 변함.
즉 다시말해서 Y=y의 값에 관계없이 X=0인 주변확률이라고 표현할 수있음.
```
<img src='https://user-images.githubusercontent.com/79496166/204947444-1d465c80-a2e6-4c28-a826-1eaa4c6ce689.png'/>

🧐 [Conditional Distribution이란 무엇인가?](https://datalabbit.tistory.com/17#recentComments)
```
조건부확률은 특정한 주어진 조건 하에서 어떤 사건이 발생할 확률을 의미.
즉 어떤사건 A가 일어났다는 전제 하에서 사건 B가 발생할 확률.
조건부확률을 어떠한 사건 A가 일어났다는 전제 하에 확률을 정의하므로 이때의 표본공간은 A의 근원사건 K개로 이루어진 표본공간으로 재정의
공식 : P(B|A) = P(A∩B)/P(A)
```
<img src='https://user-images.githubusercontent.com/79496166/204947774-f9d4fef8-57c8-4832-9c66-a83b1af457a3.png'/>

🧐 Bias란 무엇인가?
```
모델을 통해 얻은 예측값과 실제 정답과의 차이의 평균.
즉 예측값이 실제 정답값과 얼만큼 떨어져 있는지 나타냄.
만약 bias가 높다고 하면 그만큼 예측값과 정답값 간의 차이가 큼
Bias가 높은 게 좋을 수도 있음. 
예를 들면 통계청이 발표한 20대 남성 평균 신장은 174.21. A지역에서 평균을 내보니 175이고 B지역에서는 173이였다면, 평균은 174로, Bias는 0.21. 
그런데 C와 D지역에서는 각각 176과 172였고 이 경우에도 평균은 174로, Bias는 0.21로 동일. 
결국 Bias 안에는 평균의 함정이 숨어있음. 파라미터를 추정했을 때, 추정된 파라미터끼리 차이가 클수도 있고 작을수도 있다는 것

```
🧐 Biased/Unbiased estimation의 차이는? 
```
Unbiased Estimator는 파라미터 추정 평균에 대해서 bias값이 0인 경우
Biased Estimator는 파라미터 추정 평균의 bias 값이 0이 아닌 경우

```

🧐 [Variance, MSE란?](https://gaussian37.github.io/machine-learning-concept-bias_and_variance/)
##### variance 
```
Variance는 다양한 데이터 셋에 대하여 예측값이 얼만큼 변화할 수 있는지에 대한 양의 개념. 
이는 모델이 얼만큼 유동성을 가지는 지에 대한 의미로도 사용되며 분산의 본래 의미와 같이 얼만큼 예측값이 퍼져서 다양하게 출력될 수 있는 정도로 해석할 수있음.
```
<img src='https://user-images.githubusercontent.com/79496166/204948284-82eb5026-3788-4776-8374-a089715ae674.png'/>

##### MSE
```
MSE는 오차의 제곱에 대한 평균을 취한 값으로 통계적 추정의 정확성에 대한 질적인 척도로 많이 사용됨
실제값(관측값)과 추정값의 차이로, 잔차가 얼마인지 알려주는데 많이 사용되는 척도이다.
MSE가 작을수록 추정의 정확성이 높아짐.
```

🧐 Sample Variance란 무엇인가?
```
모집단으로부터 무작위로 n개의 표본을 추출했을 때 이 n개 표본들의 평균과 분산을 각각 표본평균, 표본분산이라고 함.
```

🧐 Confidence Interval이란 무엇인가?
```
신뢰구간은 모수가 실제로 포함될 것으로 예측되는 범위
집단 전체를 연구하는 것을 불가능하므로 샘플링된 데이터를 기반으로 모수의 범위를 추정하기 위해 사용됨 
따라서 신뢰구간은 샘플링된 표본이 연구중인 모집단을 얼마나 잘 대표하는 지 측정하는 방법.
일반적으로 95% 신뢰수준이 사용됨

```
<img src='https://user-images.githubusercontent.com/79496166/204949883-d995dc66-efc4-41eb-9b02-cfc0ee3639ec.png'/>

🧐 [covariance이란 무엇인가?](https://blog.naver.com/sw4r/221025662499)
```
Covariance 즉 공분산은 서로 다른 변수들 사이에 얼마나 의존하는지를 수치적으로 표현하며 그것의 직관적 의미는 어떤 변수(X)가 평균으로부터 증가 또는 감소라는 경향을 보일 때 이러한 경향을 다른 변수(Y 또는 Z 등등)가 따라하는 정도를 수치화 한 것
공분산은 또한 두 변수의 선형관계의 관련성을 측정한다 라고도 할 수 있다.
```
🧐 [correlation이란 무엇인가?](https://otexts.com/fppkr/causality.html)
```
Correlation 즉 상관관계는 상관 분석과 연관됨. 즉 두 변수 간에 어떤 선형적 관계를 갖고 있는 지를 분석하는 방법이 상관 분석인데, 두 변수는 서로 독립적인 관계이거나 상관된 관계일 수 있으며 이 때 두 변수 간의 관계의 강도를 상관관계라고 한다
```
🧐 [Total variation 이란 무엇인가?](https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=lllmanilll&logNo=140184184273)
```
전체분산은 각 표본의 측정치들이 전체 평균으로부터 얼마나 분산되어 있는지를 측정한 것
```
🧐 [Explained variation 이란 무엇인가?](https://dnai-deny.tistory.com/16)
```
설명된 분산은 통계에서 주어진 데이터의 분산을 설명하는 비율을 측정함. 그중에서 설명된 분산의 비율은 전체 고윳값중에서 원하는 고윳값의 비율임.
```
🧐 [Coefficient of determination 이란? (결정계수) r2](https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=xodh16&logNo=220545881424)
```
결정계수란 y의 변화가 x에 의해 몇 % 설명되는지 보여주는 값임. 예를 들면 결정계수가 0.52이면 y의 변화는 x에 의해 52% 설명된다는 뜻이다.
R의 값은 +가 될 수 도있고 -가 될 수도 있지만 r을 제곱하면 무조건 양수이므로 양의 상관관계든 음의 상관관계든 y의 변화가 x에 의해 몇 % 영향을 미친 것인지 설명이 가능함.
결정계수의 범위는 0과 1이며 만약 모든 관찰점이 회귀선상에 위치한다면 결정계수의 값은 1이 됨. 반대로 회귀선에서의 변수들 간 회귀관계가 전혀 없어 추정된 회귀선의 기울기가 0이면 결정계수의 값은 0이 된다.

```
🧐 [P-value란 무엇인가?](https://bodi.tistory.com/entry/%EA%B0%80%EC%84%A4%EA%B2%80%EC%A0%95-P-value%EB%9E%80%EC%96%B4%EB%96%A4-%EC%82%AC%EA%B1%B4%EC%9D%B4-%EC%9A%B0%EC%97%B0%ED%9E%88-%EB%B0%9C%EC%83%9D%ED%95%A0-%ED%99%95%EB%A5%A0)
```
Probability-value의 줄임말로 확률 값을 뜻하며 어떤 사건이 우연히 발생할 확률 
P-value를 어느곳에 적용하는지 알기 위해서는 가설검정에 대한 이해가 선행되어야 함.
가설검정이란 대상집단에 대하여 어떤 가설을 설정하고 검토하는 통계적 추론을 의미함.

```
<img src='https://user-images.githubusercontent.com/79496166/204950540-18263cd2-a649-4092-90f0-1befdf975883.png'/>

```
예를 들어 A회사 B회사로부터 각각 200명씩 표본을 추출하여 고객만족도 조사를 한 결과 아래와 같은 결과가 나타났다고 가정.
A회사 : 80점
B회사 : 90점
그렇다면 A회사의 표본 평균 점수인 80보다 B회사의 표본 평균 점수인 90점이 높은데 실제로 A회사의 모평균인 μa < B회사의 모평균인 μb 인지 아니면 표본의 점수일 뿐 모평균의 점수는 차이가 없는지를 가설을 설정하고 검정해야함
여기서 확인하고 싶은 것은 실제 A회사의 모집단 점수보다 B회사의 점수가 높은가? 이며 확인하고 싶은 부분을 귀무가설과 대립가설로 정의함.

```
<img src='https://user-images.githubusercontent.com/79496166/204950613-79562e92-7ff1-4794-b09d-987024f1f8fd.png'/>

```
이러한 귀무가설을 기각할 수 있는지에 대한 여부 판단 방법 중하나로 P-value(유의확률)를 확인하는 방법이 있음.
P-value의 값이 0.05보다 작다는 것은 어떤 사건이 우연히 일어날 확률이 0.05보다 작다라는 의미이며 우연히 발생할 확률이 5%보다 작다는 것은 이 사건이 우연히 일어났을 가능성이 거의 없다는 것으로 추정 가능.

```
🧐 [likelihood-ratio test 이란 무엇인가?](https://data-scientist-brian-kim.tistory.com/91)
```
우도 비율 검정은 모형 두 개의 우도의 비를 계산해서 두 모형의 우도가 유의하게 차이나는지 비교하는 방법
```
<img src='https://user-images.githubusercontent.com/79496166/204950906-0fdaa8f4-b750-414e-9f4e-7f04c24db5fb.png'/>

```
위의 그림에서 고혈압과 당뇨가 이미 포함된 모형 A에 비만이라는 독립변수를 추가하여 모형 B를 세팅했을 때, 이 두 모형이 통계적으로 유의한 우도의 차를 보인다면 비만은 의미있는 독립변수라고 할 수 있음.
여기서 우도란 어떤 값이 관측되었을 때 해당 관측값이 어떤 확률분포로 나왔는지에 대한 확률

```
<img src='https://user-images.githubusercontent.com/79496166/204951197-2ec69764-936e-411f-8c87-46e6afe734a1.png'/>

----

## Machine Learning

🧐 Frequentist 와 Bayesian의 차이는 무엇인가?
```
보통 통계학에서 한 사건이 장기적으로 일어날 때 발생하는 빈도를 확률이라고 하는데 확률을 사건의 빈도로 보는 것을 빈도주의(Frequentist)라고 하고 확률을 사건 발생에 대한 믿음 또는 척도로 바라보는 관점이 베이지안이라고 한다.
빈도주의와 베이지안은 확률을 해석하는 관점의 차이라고 설명할 수 있다.
빈도주의에서 빈도론자들은 얼만큼 빈번하게 특정한 사건이 반복되어 발생하는 가를 관찰하고 가설을 세우고 모델을 만들어서 검증한다. 
베이지안론자들은 고정된 데이터의 관점에서 파라미터에 대한 신념의 변화를 분석하고 확률은 사건 발생에 대한 믿음 또는 척도라고 봄.

```
<img src='https://user-images.githubusercontent.com/79496166/204951354-d78d6816-8ed9-4db1-ab9b-6f98ade96406.png'/>

🧐 [Frequentist 와 Bayesian의 장점은 무엇인가?](https://bodi.tistory.com/entry/%EA%B0%80%EC%84%A4%EA%B2%80%EC%A0%95-P-value%EB%9E%80%EC%96%B4%EB%96%A4-%EC%82%AC%EA%B1%B4%EC%9D%B4-%EC%9A%B0%EC%97%B0%ED%9E%88-%EB%B0%9C%EC%83%9D%ED%95%A0-%ED%99%95%EB%A5%A0)
```
빈도주의는 여러 번의 실험, 관찰을 통해 알게된 사건의 확률을 검정하므로 사건이 독립적이고 반복적이며 정규 분포형태일 때 사용하는 것이 좋다.
또한 대용량 데이터를 처리 할 수 있다면 계산이 비교적 복잡하지 않기 때문에 쉽게 처리가 가능하다.
베이지안은 확률 모델이 명확히 설정되어 있다면 조건부로 가설을 검증하기 때문에 가설의 타당성이 높아짐

```
🧐 [차원의 저주란?](https://for-my-wealthy-life.tistory.com/40)
```
차원의 저주란 차원이 증가하면서 학습데이터 수가 차원 수보다 적어져서 성능이 저하되는 현상을 일컫는다. 차원이 증가할수록 변수가 증가하고 개별 차원 내에서 학습할 데이터 수가 적어진다.
이때 주의할 점은 변수가 증가한다고 반드시 차원의 저주가 발생하는 것은 아니다. 관측치보다 변수 수가 많아지는 경우에 차원의 저주문제가 발생함

```
<img src='https://user-images.githubusercontent.com/79496166/204951528-e02fbecf-5a6c-4e87-ac12-66463151d0d6.png'/>

```
위 그림에서 보는 것과 같이 차원이 증가할수록 빈 공간이 많아진다.
같은 데이터지만 1차원에서는 데이터 밀도가 촘촘했던 것이 2차원, 3차원으로 차원이 커질수록 점점 데이터 간 거리가 멀어진다. 
이렇게 차원이 증가하면 빈 공간이 생기는데 빈 공간은 컴퓨터에서 0으로 채워진 공간이다. 
즉 정보가 없는 공간이기 때문에 빈 공간이 많을수록 학습 시켰을 때 모델 성능이 저하될 수 밖에 없다.

```
🧐 [Train, Valid, Test를 나누는 이유는 무엇인가?](https://velog.io/@hya0906/2022.03.03-ML-Testvalidtest%EB%82%98%EB%88%84%EB%8A%94-%EC%9D%B4%EC%9C%A0)
```
Train data는 training과정에서 학습을 하기 위한 용도로 사용된다.
validation data는 training과정에서 사용되며 학습을 하는 과정에서 중간평가를 하기 위한 용도이며 train data에서 일부를 떼내서 가져옴
test data는 training 과정이 끝난 후 성능평가를 하기 위해 사용하며 훈련한 모델을 한번도 보지 못한 데이터를 이용해서 평가를 하기 위한 용도이다.
보통 일반적으로 train:validation:test는 6:2:2로 하며 train loss는 낮은데 test loss가 높으면 훈련 데이터에 과대적합(overfitting)이 되었다는 의미이다

```
🧐 [Cross Validation이란?](https://wooono.tistory.com/105)
```
교차검증은 보통 train set으로 모델을 훈련하고 test set으로 모델을 검증함.
그러나 고정된 test set을 통해 모델의 성능을 검증하고 수정하는 과정을 반복하면 결국 내가 만든 모델은 test set에만 잘 동작하는 모델이 된다.
즉 test set에 과적합(overfitting)하게 되므로 다른 실제 데이터를 가져와서 예측을 수행하면 엉망인 결과가 나와버리게 된다.
이를 해결하고자 하는 것이 바로 교차검증(cross validation)이다.
교차검증은 train set을 train set+ validation set으로 분리한 뒤, validation set을 사용해 검증하는 방식이다.
교차검증 기법에는 K-Fold 기법이 있음.
K-Fold는 가장 일반적으로 사용되는 교차 검증 방법이다.
보통 회귀 모델에 사용되며, 데이터가 독립적이고 동일한 분포를 가진 경우에 사용된다.
자세한 K-Fold 교차 검증 과정은 다음과 같다
```
1. 전체 데이터셋을 Training Set과 Test Set으로 나눈다.
2. Training Set를 Traing Set + Validation Set으로 사용하기 위해 k개의 폴드로 나눈다.
3. 첫 번째 폴드를 Validation Set으로 사용하고 나머지 폴드들을 Training Set으로 사용한다.
4. 모델을 Training한 뒤, 첫 번 째 Validation Set으로 평가한다.
5. 차례대로 다음 폴드를 Validation Set으로 사용하며 3번을 반복한다.
6. 총 k 개의 성능 결과가 나오며, 이 k개의 평균을 해당 학습 모델의 성능이라고 한다.
<img src='https://user-images.githubusercontent.com/79496166/204952025-3650e0bd-d8ac-4a8b-ae8f-fd848318fb19.png'/>

🧐 (Super-, Unsuper-, Semi-Super) vised learning이란 무엇인가?

##### Supervised Learning
```
지도학습은 답(레이블이 달린)이 있는 데이터로 학습하는 것으로 입력값이 주어지면 입력값에 대한 label[y data]를 주어 학습 시키는 것
지도학습에는 크게 분류와 회귀가 있음
분류는 이진 분류 즉 True, False로 분류하는 것이며 다중분류는 여러값으로 분류하는 것
회귀는 어떤 데이터들의 특징을 토대로 값을 예측하는 것이다. 결과 값은 실수 값을 가짐.

```

##### Unsupervised Learning
```
비지도 학습은 정답을 따로 알려주지 않고 비슷한 데이터 들을 군집화하는 것이다. 일종의 그룹핑 알고리즘으로 볼 수 있다.
라벨링 되어있지 않은 데이터로부터 패턴이나 형태를 찾아야 하기 때문에 지도학습보다는 조금 더 난이도가 있다. 
실제로 지도 학습에서 적절한 피처를 찾아내기 위한 전처리 방법으로 비지도 학습을 이용하기도 한다.
대표적인 종류는 클러스터링, Dimentiionality Reduction, Hidden Markov Model 등을 사용한다.

```
<img src='https://user-images.githubusercontent.com/79496166/204952684-e6a706f7-195d-4214-845e-787b79f891c2.png'/>

##### Semi-Supervised Learning
```
ㄴㅇㄹ
```
[강화학습](https://bangu4.tistory.com/96)
```
강화학습은 분류할 수 있는 데이터가 존재하지 않고 데이터가 있어도 정답이 따로 정해져 있지 않으며 자신이 한 행동에 대해 보상을 받으며 학습하는 것을 말함.
게임을 예로들면 게임의 규칙을 따로 입력하지 않고 자신이 게임 환경에서 현재 상태에서 높은 점수를 얻는 방법을 찾아가며 행동하는 학습 방법으로 특정 학습 횟수를 초과하면 높은 점수를 획득할 수 있는 전략이 형성되게 됨.

```
🧐 [Receiver Operating Characteristic Curve란 무엇인가?](https://bioinfoblog.tistory.com/221)
```
ROC는 FPR(False positive rate)과 TPR(True Positive Rate)을 각각 x, y축으로 놓은 그래프이다.
TPR(True Positive Rate)는 1인 케이스에 대해 1로 바르게 예측하는 비율(Sensitivity)로 암 환자에 대해 암이라고 진단하는 경우를 뜻함.
FPR(False positive rate)는 0인 케이스에 대해 1로 틀리게 예측하는 비율(1-Specificity)로 정상에 대해 암이라고 진단하는 경우를 뜻함
ROC curve는 모델의 판단 기준을 연속적으로 바꾸면서 측정했을 때 FPR 과 TPR 의 변화를 나타낸 것으로 (0,0)과 (1,1)을 잇는 곡선이다.
ROC curve는 어떤 모델이 좋은 성능을 보이는 지 판단할 때 사용할 수 있다. 
즉 높은 sensitivitiy와 높은 specifity를 보이는 모델을 고르기 위해 다양한 모델에 대해 ROC curve를 그릴 때 좌상단으로 가장 많이 치우친 그래프를 갖는 모델이 가장 높은 성능을 보인다고 말할 수 있다.
```
<img src='https://user-images.githubusercontent.com/79496166/204953013-9b6f4216-b6e0-4c67-a25c-8206e8c50106.png'/>

🧐 Accuracy,  Recall, Precision, f1-score에 대해서 설명해보라
##### Accuracy
```
Accuracy는 올바르게 예측된 데이터의 수를 전체 데이터의 수로 나눈 값

Accuracy는 데이터에 따라 매우 잘못된 통계를 나타낼 수 도있음.
예를들어, 내일 눈이 내릴지 아닐지를 예측하는 모델이 있다고 가정해볼때 
항상 False로 예측하는 모델의 경우 눈이 내리는 날은 그리 많지 않기떄문에  굉장히 높은 Accuracy를 가짐. 
높은 정확도를 가짐에도 해당모델은 쓸모 없음.
```
<img src='https://user-images.githubusercontent.com/79496166/204954875-75bfc367-8ae8-42f0-bc21-7fc687a90072.png'/>
<accuracy 수식>
<img src='https://user-images.githubusercontent.com/79496166/204954975-f777583e-87b4-4342-afb0-f19c2d63c2de.png'/>

##### Recall
```
Accuracy 문제를 해결하기 위해 재현율 사용.
Recall 즉 재현율은 실제로 True인 데이터를 모델이 True라고 인식한 데이터수.
만약 항상 False로 예측하는 모델의 경우는 Recall은 0이 됨

그러나 recall도 완벽하지는 않음.
예를들어 눈내림 예측기에서 항상 True라고 예측할 경우 accuracy는 낮겠지만 모델이 모든 날을 눈이 내릴 것이라
예측하기 때문에 recall은 1이됨.
해당 모델은 recall이 1이지만 쓸모 없는 모델임.
```
<img src='https://user-images.githubusercontent.com/79496166/204956193-9b0d43ce-338d-4957-a50c-f36019118593.png'/>
<recall 수식>

##### Precision
```
recall의 문제를 해결하기위해 Precision 사용.
Precision은 모델이 True로 예측한 데이터 중 실제로 True인 데이터의 수이다.
예시로 Precision은 실제로 눈이 내린 날의 수를 모델이 눈이 내릴거라 예측한 날의 수로 나눈 값이다.
```
<img src='https://user-images.githubusercontent.com/79496166/204957228-70351214-0910-4c2d-823e-ffa67aed7539.png'/>
<Precision 수식>
**Note: Precision과 recall은 서로 trade-off되는 관계가 있음.**

[F1 score](https://eunsukimme.github.io/ml/2019/10/21/Accuracy-Recall-Precision-F1-score/)
```
모델의 성능이 얼마나 효과적인지 설명할 수 있는 지표.
F1score는 Precision과 recall의 조화평균이다.
F1score는 Precision과 recall을 조합하여 하나의 통계치를 반환한다.
여기서 일반적인 평균이 아닌 조화 평균을 계산하였는데 그 이유는 Precision과 recall이 0에 가까울수록 F1score도 동일하게 낮은 값을 갖도록 하기 위함 이다.

예를들면 recall =1 이고 precision = 0.01로 측정된 모델은 Precision이 매우 낮기때문에 F1score에도 영향을 미치게 된다.
만약 일반적인 평균을 구하게 된다면 다음과 같다.
```
<img src='https://user-images.githubusercontent.com/79496166/204957659-2c0c1dc4-7af3-4854-b6e0-ce142fa1cd8d.png'/>

```
일반적으로 평균을 계산하면 높은 값이 나옴. 그러나 조화평균으로 계산하면 다음과 같은 결과를 얻음.
```
<img src='https://user-images.githubusercontent.com/79496166/204958000-120cbeee-7dc5-4fc7-94ea-8b564fe24c4f.png'/>

```
F1score가 매우 낮게 계산된 것이 확인됨.
```
🧐 [Precision Recall Curve란 무엇인가?](https://ardentdays.tistory.com/20)
```
Precision-Recall Curves는 Parameter인 Threshold를 변화시키면서 Precision과 Recall을 Plot 한 Curve. Precision-Recall Curves는 X축으로는 Recall을, Y축으로는 Precision을 가짐. 
Precision-Recall Curve는 단조함수가 아니기 때문에 이러한 이유로 ROC Curve보다 직관적이지 못하다는 단점을 가짐.

단조함수 ? 주어진 순서를 보존하는 함수
```
🧐 [Type 1 Error 와 Type 2 Error는?](https://angeloyeo.github.io/2021/01/26/types_of_errors.html)
```
가설검정 이론에서 1종 오류와 2종오류는 각각 귀무가설을 잘못 기각하는 오류와 귀무가설을 잘못 채택하는 오류이다.

```
##### 1종오류
```
귀무가설이 실제로 참이지만 이에 불구하고 귀무가설을 기각하는 오류이다. 
즉 실제 음성인 것을 양성으로 판정하는 경우이다. 거짓 양성 또는 알파 오류라고도 한다.

예를들면 아파트에 불이 나지 않았음에도 화재 경보 알람이 울린 경우를 말하게 된다.

```
##### 2종오류
```
귀무가설이 거짓인데도 기각하지 않아서 생기는 오류 
즉 실제 양성인 것을 음성으로 판정하는 경우이다. 

예를들면 아파트에 불이 났음에도 화재경보 알람이 울리지 않고 그대로 지나간 경우를 말하게 된다.
```

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
- 🧐 [Overfitting 이란?](https://jrc-park.tistory.com/271)
  ```
  모델의 파라미터들을 학습 데이터에 너무 가깝게 맞췄을 때 발생하는 현상.
  즉, 학습 데이터가 실제 세계에서 나타나는 방식과 완전히 똑같을것이라고 가정해버리는 것이다.
  
  Overfitting은 너무 세밀하게 학습 데이터 하나하나를 다 설명하려고 하다보니 
  정작 중요한 패턴을 설명할 수 없게 되는 현상을 말한다.
  ```
- 🧐 [Underfitting이란?](https://jrc-park.tistory.com/271)
  ```
  링크 참조
  ```
- 🧐 Overfitting과 Underfitting은 어떤 문제가 있는가?  
  ```
  Overfitting은 모델 학습 오류가 테스트 데이터의 오류보다 훨씬 작은 경우를 의미하고,
  Underfitting은 모델이 학습 오류를 줄이지 못하는 상황을 의미한다.
  
  즉, Overfitting이 되면 training data에 대한 정확도는 좋지만 실제 test data에 대해서는 에러가 많이 생길 수 있다.
  반면에 Underfitting은 모델이 지나치게 일반화해서 training data에 대해서 학습이 제대로 되지 않는 상태를 말한다.
  ```
- 🧐 [Overfitting과 Underfitting을 해결하는 방법은?](https://jrc-park.tistory.com/272)
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
  
  weight를 조정할 때 규제(제약)을 거는 기법이다.
  위에서 설명한것과 같이 Overfitting을 막기 위해 사용한다.
  L1 regularization, L2 regularization 등의 종류가 있다.
  - L1 : Lasso, 마름모
  - L2 : Ridge, 원
  ```
  추가 질문, Normalization이란?
  ```
  값의 범위(scale)을 0~1 사이의 값으로 바꾸는것을 말한다.
  주로 학습 전에 scaling 하는 것이고
  - 머신러닝에서는 scale이 큰 feature의 영향이 비대해지는 것을 방지할때 사용하고,
  - 딥러닝에서는 Local minimum(minima)에 빠질 위험을 감소시킬때 사용한다. 이로 인해 학습속도 향상을 기대할 수 있다.

  예를 들면, scikit-learn에서 MinMaxScaler 가 있다.
  ```
  추가 질문, Standardization이란?
  ```
  값의 범위(scale)을 평균 0, 분산 1이 되도록 변환하는것을 말한다.
  주로 학습 전에 scaling 하는 것이고 (Normalization 과 동일)
  - 머신러닝에서는 scale이 큰 feature의 영향이 비대해지는 것을 방지할때 사용하고,
  - 딥러닝에서는 Local minimum(minima)에 빠질 위험을 감소시킬때 사용한다. 이로 인해 학습속도 향상을 기대할 수 있다.

  정규분포를 표준정규분포로 변환하는 것과 같다.
  표준화로 번역하기도 한다.
  
  예를 들면, scikit-learn에서 StandardScaler 가 있다.
  ```
  ![lp-norm](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FrZ5Ys%2FbtqWsHx2sRa%2FK9NVok5LaMRDmk6YSi2jy1%2Fimg.png)
  
  추가 참고 : https://sooho-kim.tistory.com/85
  
  l1-norm, l2-norm 의 컨셉을 가져와 cost-function에 추가하여 weight를 조절하는 기법이다.
  일반적으로 lp-norm이라 하고 p값에 따라 위 그래프처럼 그래프가 변화한다.
  p가 1인 norm이 l1-norm, p가 2인 norm이 l2-norm이다.
  
  l1-norm : 두 벡터간의 거리를 절대 값으로 구하는 것
  l2-norm : 유클리디안 거리. 즉, 최단거리.. 피타고라스정리를 생각하면 쉽다.
  
  따라서, `l1-loss`는 타겟값과 예측값의 차를 절대값으로 구한것이고,
  `l2-loss`는 타겟값과 예측값의 차를 제곱한 값으로 구한것이다.
  
  - Ridge
    ```
    L1 regularization 이며 cost function에 L1-norm 을 추가한 형태이다.
    변수간 상관관계가 높아도 좋은 성능을 낸다.
    크기가 큰 변수를 우선적으로 줄일때 사용한다.
    ```
  - Lasso  
    ```
    L2 regularization 이며 cost function에 L2-norm 을 추가한 형태이다.
    변수간 상관관계가 높으면 오히려 성능이 떨어진다.
    비중요 변수를 우선적으로 줄일때 사용한다.
    ```
- 🧐 Activation function이란 무엇인가?3가지 Activation function type이 있다.  
  ```
  각 노드에서 가중치-편향 연산을 거친 입력값을 다음 단계로 줄지 말지, 주면 어떻게 줄지 결정하는 일종의 문지기 역할을 하는 함수이다.
  
  1. 가중치-편향 연산의 결과값(가중합)을 그대로 내보내면 너무 크거나 작을 수 있다.
     활성화 함수는 이를 0에서 1 또는 -1에서 1 사이의 값 등으로 바꿔준다(활성화 함수 유형에 따라 다름. 전부 다 상하한의 제한이 있는 것은 아니다)

  2. 가중치-편향 연산 결과에 비선형성을 부여한다(비선형적인 활성화 함수인 경우. 특별한 경우가 아니라면 비선형성을 가진 활성화 함수만 사용한다.
  ```
  - Ridge activation Function
    ```
    차원의 저주에 영향을 받지 않는 함수이다.
    일반적인 선형 모델 및 Neural Network의 활성화 함수로 활용된다.
    ```
  - Radial activation Function
    ```
    PASS
    ```
  - Folding activation Function  
    ```
    PASS
    ```
- 🧐 [CNN에 대해서 설명해보라](https://youngq.tistory.com/40)  
  추가 참고 : https://velog.io/@kim_haesol/CNN-%EA%B8%B0%EC%B4%88%EC%84%A4%EB%AA%85
  ```
  Convolution Neural Network 의 약자인데,
  이미지를 분석하기 위한 패턴을 찾는데 유용한 알고리즘이고,
  말 그대로 Convolution 연산을 수행하는 Neural Network이다.
  
  기존에는 DNN을 사용해서 문제를 해결했는데 DNN은 기본적으로 1차원 데이터를 다룬다.
  때문에 이미지가 입력값이 되는 경우 이것을 flatten 시켜서 1차원 데이터로 만드는데 이 과정에서
  지역적/공간적 정보가 손실되게 된다. 또한 추상화 과정 없이 바로 연산과정으로 넘어가 버리기 때문에
  학습시간과 능률의 효율성이 저하되는 문제가 발생했다.
  
  이러한 문제들을 해결한것이 바로 CNN이다.
  이미지를 날 것 그대로 받음으로써 지역적/공간적 정보를 유지한 채 feature들의 layer를 빌드업 할 수 있게되었다.
  CNN의 중요 포인트는 이미지 전체보다는 부분을 보는 것, 그리고 이미지의 한 픽셀과 주변 픽셀들의 연관성을 살리는 것이다.
  ```
- 🧐 [RNN에 대해서 설명해보라](https://byumm315.tistory.com/entry/RNN%EC%9D%84-%EC%95%8C%EC%95%84%EB%B4%85%EC%8B%9C%EB%8B%A4)
  추가 참고1 : https://hwi-doc.tistory.com/entry/cs231n-Recurrent-Neural-Network-RNN
  추가 참고2 : https://techblog-history-younghunjo1.tistory.com/481
  ```
  RNN은 주로 시계열 데이터, 텍스트 데이터를 다룰때 사용하는데,
  이름 그대로 Recurrent Neural Network, 순환하는 신경망이다.
  hidden layer에서 나온 결과값이 다시 hidden layer로 돌아가 새로운 입력값과 연산을 수행하는 순환 알고리즘이다.
  
  요약하면, FC layer나 CNN은 hidden layer에서 나온 결과값이 output layer 방향으로 흐르는데,
  RNN은 hidden layer로 되돌아가 순환한다는 점에서 큰 차이가 있다.
  
  RNN은 역전파(back-propagation)과정에서 기울기가 폭발(끝없이 발산)하거나 소실(0으로 수렴)되는 문제가 발생한다.
  layer가 많을수록(깊어질수록), 오차 역전파가 진행될수록 기울기가 소실되는 고질적인 문제를 안고 있다.
  
  이 기울기 소실문제를 개선한 RNN모델, LSTM이 있다.
  LSTM은 정보를 기억(보존)하는 셀(또는 게이트)를 두어 기울기 소실 문제를 `개선`했다. (완전히 없애지는 못했다, 예방의 느낌?)
  ```
- 🧐 [Netwon's method란 무엇인가?](https://darkpgmr.tistory.com/58)
  ```
  방정식 f(x)=0 의 해를 근사적으로 찾을 때 사용하는 방법이다.
  인수분해도 안되고 달리 정상적인 방법으로는 해를 구하기 힘들 때 사용할 수 있는 방법이 바로 Netwon's method 이다.
  
  기본적으로 f(a) 가 x=a 에서의 접선의 기울기라는 미분의 기하학적 해석을 이용한다.
  현재 x값에서 접선을 그린다음 접선이 x축과 만나는 지점으로 x를 이동시켜 가면서 점진적으로 해를 찾는 방법이다.
  ```
- 🧐 Local optimum으로 빠지는데 성능이 좋은 이유는 무엇인가?  
  ```
  어떤 파라미터가 모든 경우를 통틀어 최적인 경우를 global optimum이라 하고, 주변의 다른 파라미터들보다 더 나은 경우를 local optimum이라고 부른다.
  global optimum이든 local optimum이든 최적점에서는 항상 기울기가 0이 된다.
  따라서 local optimum에 빠지면 더이상 파라미터를 개선할 수 없고, 학습이 거기서 종료된다.
  
  선형회귀에서의 local optimum은 곧 global optimum이기 때문에 항상 오차가 적은 파라미터를 찾을 수 있다.
  
  (딥러닝에서의 local optimum은 global optimum이라는 보장이 없기 때문에 오히려 local optimum에 빠지지 않기 위한 여러가지 방법을 동원한다.)
  ```
- 🧐 [Internal Covariance Shift 란 무엇인가?](https://data-newbie.tistory.com/356)
  ```
  링크 참조
  ```
- 🧐 [Batch Normalization은 무엇이고 왜 하는가?](https://eehoeskrap.tistory.com/430)
  추가 참고 : https://wooono.tistory.com/227
  ```
  Batch Normalization이란, 신경망 내부에서 학습 시 평균과 분산을 조정하여 변형된 분포가 나오지 않도록 하는것이다.
  
  [하는 이유]
  - 기울기 폭발, 소실을 예방할 수 있고 학습속도를 향상시킬 수 있다.
  - 배치 정규화를 사용하면 시그모이드 함수나 하이퍼볼릭탄젠트 함수를 사용하더라도 기울기 소실 문제를 크게 개선할 수 있다.
  - 가중치 초기화에 훨씬 덜 민감해진다.
  - 훨씬 큰 learning rate를 사용할 수 있어 학습 속도를 개선할 수 있다.
  - 미니 배치 마다 평균과 표준편차를 계산하여 사용하므로 훈련 데이터에 일종의 잡음 주입의 부수효과로 과적합을 방지하는 효과가 있다.
  - 드롭아웃과 비슷한 효과를 내며, 드롭아웃과 함께 사용하면 좋다.
  - 배치정규화를 하면 모델을 복잡하게하고 추가 계산을 하는 하는 것이므로 test data에 대한 예측 시에 실행시간이 느려진다.
    - 서비스 속도를 고려하는 관점에서는 배치 정규화가 꼭 필요한지 고민이 필요하다.
  ```
- 🧐 [Backpropagation이란 무엇인가?](https://box-world.tistory.com/19)
  ```
  내가 뽑고자 하는 target 값과 실제 모델이 계산한 output 이 얼마나 차이가 나는지 구한 후
  그 오차값을 다시 뒤로 전파해가면서 각 노드가 가지고 있는 변수들을 갱신하는 알고리즘이다.
  ```
- 🧐 [Optimizer의 종류와 차이에 대해서 아는가?](https://yeomko.tistory.com/39)
  ```
  - Gradient Descent
    기울기의 반대 방향으로 일정 크기만큼 이동하는 것을 반복하며 Loss function의 값을 최소화 하는 특정 파라미터값을 찾는 방법이다.
      > 추가 참고 : https://mangkyu.tistory.com/62
    Neural Network의 Weight를 조정하는 과정에서 보통 이 방법을 사용한다.
    네트워크에서 내놓는 결과값과 실제값 사이의 차이를 정의하는 Loss function의 값을 최소화하기 위해 기울기를 이용하는 방법이다.
    
    이 방법은 함수의 최소값을 찾는 문제에서 활용된다.
    가령, 미분계수가 0인 지점을 찾으면 되지 않느냐? 라고 반문할수도 있는데, 그렇지 않고 굳이 이 방법을 사용하는 이유는
    실무에서 맞닥뜨리는 문제(함수)의 형태가 복잡해(비선형) 미분계수와 그 근을 계산하기 어려운 경우가 많고,
    실제로 미분계수를 구하는것보다 gradient descent가 computational cost(연산량)이 더 적고 효율적이기 때문이다.
    
  - Stochastic Gradient Descent
    Gradient Descent 는 Loss function을 계산할때 전체 데이터에 대해 연산하기 때문에 상대적으로 많은 연산량을 필요로 한다.
    이를 방지하고자 Loss function을 계산할 때, 전체 데이터가(배치) 아닌 일부 데이터(미니배치)를 사용해 Loss를 계산한다.
    계산속도가 훨씬 빠르기 때문에 같은 시간에 더 많은 step을 갈 수 있고, 여러번 반복할 경우 Batch 처리한 결과로 수렴한다.
    또한, Local minima에 빠지지 않고 더 좋은 방향으로 수렴할 가능성이 높다는 장점도 갖고 있다.
    
    이를 변형한 알고리즘으로 Momentum, Adagrad, RMSProp 등이 있다.
    
  - Momentum : 
  - AdaGrad : 
  - RMSProp : 
  - Adam : 
  - AdaMax : 
  - Nadam : 
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
  - [AdaBoost](https://bkshin.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-14-AdaBoost)
    ```
    링크 참조
    ```
  - Logit Boost
    ```
    PASS
    ```
  - [Gradient Boost](https://bkshin.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-15-Gradient-Boost)
    ```
    링크 참조
    ```
- 🧐 [Support Vector Machine이란 무엇인가?](https://velog.io/@shlee0125/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EC%A0%95%EB%A6%AC-Support-Vector-Machine-05.-Why-does-SVM-maximize-margin)
  <p align="left">
    <img src="https://i.ibb.co/tYFd14p/svm04.webp", height="300x">
  </p> 
    
  ```
  SVM은 분류를 위한 기준선(결정 경계, Decision Boundary)을 정의하는 모델이다.
  결정 경계(Decision Boundary)를 통해 어느쪽에 속하는지 판단하는 모델로 선형이나 비선형 분류, 회귀, 이상치 탐색에 사용할 수 있다.
  최적의 결정 경계는 마진을 최대화한다. 점선으로부터 결정 경계(기준선)까지의 거리가 바로 `마진(margin)`이다.
  
  서포트 벡터는 Margin을 결정하는 각 클래스의 샘플을 말하기도 하고,
  새로운 데이터가 들어왔을 때 클래스를 구분하는 기준이 되는 샘플을 말하기도 한다.
  
  SVM은 데이터셋이 작을 때 효과적이다.
  ```
- 🧐 [Margin을 최대화하면 어떤 장점이 있는가?](https://velog.io/@shlee0125/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EC%A0%95%EB%A6%AC-Support-Vector-Machine-05.-Why-does-SVM-maximize-margin)
  ```
  Margin을 최대화하는 것으로 현재 데이터로부터 Decision Boundary를 가능한한 멀리 떨어트려놓음으로써 Overfitting의 위험을 최소화 할 수 있다.
  ```
  
----
  
### [Linear Algebra](https://mole-starseeker.tistory.com/79)

- 🧐 Linearly Independent란?  
  ```
  n개의 벡터 선형 결합에 쓰인 스케일링 팩터(factor)가 모두 0일때만 선형 결합의 결과가 0으로 나오는 경우를 선형 독립이라고 한다. 
  즉, n개의 벡터 중 어느 한 벡터라도 다른 벡터들로 표현할 수 없을 때를 선형 독립이라고 한다.
  ```
- 🧐 Basis와 Dimension이란 무엇인가?  
  ```
  기저벡터 라고 번역되며
  어떤 벡터 공간을 구성하는 벡터들의 기본재료, 기준을 뜻한다. 
  n차원에 존재하는 다른 모든 벡터들을 n개의 기저벡터에 적절한 스칼라 값을 곱함으로써 얻을 수 있다. 
  (ex. xy 좌표계, 즉 2차원의 모든 벡터들은 해당 좌표계의 기저벡터 (1,0), (0,1)로 표현할 수 있음)
  ```
- 🧐 Null space란 무엇인가?  
  ```
  영공간이라고 번역되며 Kernel(커널)이라고도 부른다.
  원점으로 이동하는 벡터들의 집합을 그 행렬의 null space 또는 kernel이라고 한다.
  다시 말해, null이 되는 모든 벡터의 공간이다. null space 안에 있는 모든 벡터를 원점으로 뭉겐다는 뜻이다.
  수식적으로는 Ax = 0을 만족하는 벡터 x들로 span된 벡터 공간을 뜻한다.
  ```
- 🧐 Symmetric Matrix란?  
  ```
  대칭 행렬이라고 번역되며
  정사각 nxn 행렬 A가 그것의 transpose 형태인 A^T와 같으면, 즉, A = A^T가 성립하면 A는 symmetric 행렬이다.
  ```
- 🧐 Possitive-definite란?  
  ```
  양의 정부호 행렬이라고 번역되며
  정사각 행렬 A가 symmetric일 때, 0이 아닌 모든 벡터 x에 대해 x^TAx ＞ 0이면 A를 positive definite 행렬이다. 
  x^TAx ≥ 0이면 A는 positive semidefinite 행렬이다. 이때 symmetric이란 말은 붙이거나 붙이지 않는다. 
  만약 x^TAx ＜ 0이면 A는 negative definite 행렬이다.
  ```
- 🧐 Rank 란 무엇인가?  
  ```
  행렬로 인해 변환된 공간의 차원 수를 의미한다.
  즉, 행렬 A에 의해 공간이 변환된다면, 그 행렬 A의 기저벡터 수를 의미한다.
  nxn 행렬의 경우 최대 rank는 n인데, 즉, 기저 벡터들의 선형 결합으로 온전한 n차원 공간을 만든다는 의미이다.
  이를 full rank라고 하고, full rank일 경우 역행렬이 존재하며, 그러므로 행렬에 의해 변환된 공간을 다시 원래 차원의 공간으로 되돌릴 수 있게 된다.

  만약 full rank가 아니라면, 즉 기저 벡터가 n개가 아니라면, 행렬 A에 의해 공간 변환이 수행되었을 때 n보다 작은 차원으로 공간이 수축한다는 뜻이다.
  그러면 다시 원래 차원의 공간으로 되돌릴 수 없다는 것이므로 역행렬이 없다.
  그리고 이것은 결국 공간 변화 factor인 det(A)의 값도 0이라는 뜻이다.
  ```
- 🧐 Determinant가 의미하는 바는 무엇인가?  
  ```
  ```
- 🧐 Eigen Vector는 무엇인가?  
  ```
  고유벡터라고 번역되며
  행렬에 의해 공간의 선형 변환이 일어나면 해당 벡터 공간 상의 벡터들은 대부분 크기와 방향이 모두 바뀌는데,
  어떤 벡터들은 크기만 바뀐다. 그런 벡터들을 고유벡터라 한다.
  ```
- 🧐 Eigen Vector는 왜 중요한가?  
  ```
  ```
- 🧐 Eigen Value란?  
  ```
  고유값이라고 번역되며
  행렬에 의해 공간의 선형 변환이 일어나면 해당 벡터 공간 상의 벡터들은 대부분 크기와 방향이 모두 바뀌는데,
  어떤 벡터들은 크기만 바뀐다. 크기가 바뀌는 정도를 고유값이라고 한다.
  ```
- 🧐 SVD란 무엇인가?→ 중요한 이유는?  
  ```
  ```
- 🧐 Jacobian Matrix란 무엇인가? 
  ```
  ```
