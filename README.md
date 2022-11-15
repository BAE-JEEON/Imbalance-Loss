## Imbalance Loss with the Statistical Properties of Pseudo-labels in Class Imbalanced Semi-Supervised Learning

- Imbalance Loss with the Statistical Properties of Pseudo-labels in Class Imbalanced Semi-Supervised Learning (KSC 2022)
- 지도 교수: 배성호
- 주저자: 배제언

## 주요 내용
- 준 지도 학습은 라벨링 되지 않은 많은 양의 데이터를 이용하여 라벨링의 어려움을 해결했을 뿐 아니라 지
도 학습 못지않은 우수한 성능을 보였다. 특히 준지도 학습의 state-of-the-art 알고리즘인 FixMatch는
Pseudo-label을 알고리즘을 사용할 때, 라벨이 없는 데이터에 대해 예측한 Confidence가 사전에 설정된
Threshold 미만이면 해당 Pseudo-label은 학습하지 않는 방법을 사용하여 큰 성능 향상을 보였다. 그러나
FIxMatch를 포함한 SSL 알고리즘들은 클래스의 분포가 불균형하면 성능이 크게 저하되는 문제점이 있다. 이
러한 문제를 해결하기 위해 DARP, CReST와 같은 해결 방법이 제시되었지만, 해당 방법들은 라벨이 없는 데
이터의 통계적 특성이 라벨링된 데이터와 같다는 엄격한 가정이 필요하다. 이에 본 논문은 라벨링된 데이터의
통계적 특성을 이용하지 않고 생성된 Pseudo-label과 Threshold를 통과한 Pseudo-label의 통계적 특성만을
이용하여 클래스 불균형 준지도 학습의 성능 개선 방법을 제안하였다. 결과적으로, CIFAR10-LT에서 제안 방
법은 베이스라인 FIxMatch 알고리즘과 비교하여 불균형 정도에 따라 최대 1.7%의 성능 향상을 보였다.

## 실험 결과

- 데이터셋:CIFAR10-LT
- 라벨링 데이터의 비율: 20%
- 사용한 준지도 학습 알고리즘: FixMatch
- 모델: Wide ResNet-28-2
- 기타: 다른 하이퍼파라미터 값은 FixMatch 논문에서 사용한 최적 값을 그대로 사용하였다. 

- 불균형 정도에 따른 FixMatch와 성능 비교

|                |    imb ratio= 50   |   imb ratio= 100     |    imb ratio= 200    |
|:--------------:|     :---------:    |     :---------:      | :-----------------:  |
|    FixMatch    |        82.06       |       73.17          |         66.37        |   
|      Ours      |        83.01       |       74.86          |         68.11        |  

## 실행 방법
```
python train.py --dataset cifar10 --num-labeled 4000 --arch wideresnet --batch-size 64 --lr 0.03 --expand-labels --seed 5 --label_ratio 20 --num_max 1000  --imb_ratio 100 --imbalancetype long --out results/New_loss_imb100  --threshold 0.99 --imbalance_loss_weight 5
```

## 레퍼 런스
- https://github.com/kekmodel/FixMatch-pytorch
- https://github.com/google-research/fixmatch
- https://github.com/ildoonet/pytorch-randaugment
- https://github.com/LeeHyuck/ABC

## Requirements
- python 3.6+
- torch 1.4
- torchvision 0.5
- tensorboard
- numpy
- tqdm
- apex (optional)
