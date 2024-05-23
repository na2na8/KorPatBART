# KorPatBART: A Korean Patent Pre-trained Model for Generating Text from Patent Documents with BART
## KorPatBART: 특허 문서 텍스트 생성을 위한 한국어 특허 도메인 사전학습 BART 모델

논문 : [한국정보과학회 2023 KSC(한국소프트웨어종합학술대회)](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11705184)

### Abstract
특허 문서는 일반 문서와 사용하는 단어, 문법이 다르기 때문에 일반 도메인에서 학습 언어 모델은 특허 도메인의 자연어 처리 태스크를 수행하기 어렵다. KorPatELECTRA는 특허 도메인 문서로 사전학습하여 특허 도메인의 자연어 처리 태스크에서 우수한 성능을 달성하였다. 그러나 인코더 모델 기반 특성상 해당 모델로는 특허 문서 생성이 어렵다는 문제점이 있다. 본 연구에서는 이를 해결하기 위해 생성 성능이 뛰어난 인코더-디코더 기반 한국어 언어 모델인 KoBART를 특허명세서 데이터로 사전학습한 특허 도메인 자연어 처리 모델인 KorPatBART를 제안한다. 제안하는 방법은 특허 도메인에서 생성 태스크인 요약과 번역에서 우수한 성능을 보여주었다. 

### KorPatBART
KorPatBART는 특허 도메인에서의 문서 생성을 최적화하기 위해 한국어 BART 모델인 KoBART를 특허명세서 데이터로 사전학습한 언어모델이다.

#### Pre-training Dataset
2013년~2021년도 특허명세서 데이터 약 135만 건

<img width="292" alt="image" src="https://github.com/na2na8/KorPatBART/assets/32005272/8a8c831f-b3ec-4306-bb9d-6fc9d45b71ad">

#### Experiments
- [산업정보 연계 주요국 특허 영-한 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=563)
- [특허 분야 자동분류 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=547)

<img width="351" alt="image" src="https://github.com/na2na8/KorPatBART/assets/32005272/6f4a5d76-d90b-4d9e-bd5d-daef193428f3"><br/>    
<img width="350" alt="image" src="https://github.com/na2na8/KorPatBART/assets/32005272/c60e6bd2-7705-4e5e-8c2d-c4a0569d1cf3"><br/>    
<img width="350" alt="image" src="https://github.com/na2na8/KorPatBART/assets/32005272/17b9dd87-2a31-4dac-977f-2bc496ce5787">    


