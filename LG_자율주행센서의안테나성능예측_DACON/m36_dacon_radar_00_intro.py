# 8/12 주말 과제 ; 데이콘 자율주행 센서의 안테나 성능 예측 AI 경진대회
# https://dacon.io/competitions/official/235927/


# 데이터 설명

# 1. 학습(Train) 데이터셋 (39607개)

# 파일명: train.csv
# 설명: ID, X Feature(56개), Y Feature(14개)


# 2. 테스트(Test) 데이터셋 (39608개)

# 파일명: test.csv
# 설명: ID, X Feature(56개)


# 3. sample_submission.csv (제출양식)

# 설명 : ID, 예측한 Y Feature(14개)


# 4. ./meta/x_feature_info.csv

# 설명: 비식별화된 X Feature에 대한 세부 설명 자료


# 5. ./meta/y_feature_info.csv

# 설명 : 비식별화된 Y Feature에 대한 세부 설명 자료


# 6. ./meta/y_feature_spec_info.csv

# 설명 : 각 샘플의 정상, 불량을 판정할 수 있는 Y Feature 별 스펙 기준 자료
# 실제 공정 과정에서의 데이터로, 대회 진행을 위해 해당 스펙 값들은 임의 조정된 상태입니다.
# 대회 진행 중 공개되는 스펙 값으로부터 정상/불량 판정 결과가 실제 공정에서의 정상/불량률과는 차이가 있습니다.



# 토크 게시판 ; 질문 및 답변
# Q. X_10 / X_11 방열재료 2, 3의 무게값이 0이 많은데 결측치인지 실제로 방열재료 2, 3 이 공정에 사용되지 않았는지?

# 	A. 실제 방열재료 2, 3은 공정에 사용되고 있으며 0으로 표현된 값은 결측치라고 생각하시면 될 것 같습니다.

 

# 	Q. X인자가 숫자가 작을수록 앞에서 일어나는 공정인지?

# 	A. 순서가 작을수록 앞에서 일어나는 공정이 맞습니다

 

# 	Q. X_12 커넥터 위치 기준좌표값이 (X,Y)의 형태가 아니라 하나의 값으로 나와있는데 어떤 의미인지?

# 	A. 세부설명을 공개하는 과정에서 명칭의 변화가 있었는데 장비의 원점으로부터 거리 값이라고 생각하시면 될 것 같습니다.

 

# 	Q. X_19 ~ X_22 / X_30 ~ X_33 같은 스크류 삽입깊이로 나와있는 인자의 의미가 무엇인지?

# 	A. 두가지 인자 그룹은 스크류 삽입 깊이를 측정하기 위해서 수집되는 인자가 맞습니다.

# 		다만 서로 다른 장비로 측정하게 되고 이때 장비가 측정하는 위치가 다르기 때문에 값의 차이가 다소 있을 수 있습니다.

# 		자세한 공정 설명은 보안상의 이유로 공개하지 못하는 부분 양해 부탁드립니다.



# 	Q. 무게와 면적 관련 인자들 사이의 단위가 서로 다른 경우가 있는지? (22.08.10 추가)

# 	A. 연속된 무게나 면적 항목은 같은 장비로 측정한 것으로 같은 단위를 가지지만 떨어져 있는 항목들 사이의 단위는 다를 수 있습니다.

 

# 	Q. X_34, X_35, X_36, X_37 가 스크류 체결 시 분당 회전수인데, 스크류 4개를 체결하면서 각각 같은 장비로 측정한 회전수인지, 4개의 측정장비로 측정한 회전수인지? (22.08.10 추가)

# 	A. 같은 장비로 측정한 회전 수 입니다.

