# 데이콘 문제풀기
# https://dacon.io/competitions/official/235959/


# 데이터 설명

# 1. train.csv : 학습 데이터

# id : 샘플 아이디
# Age : 나이
# TypeofContact : 고객의 제품 인지 방법 (회사의 홍보 or 스스로 검색)
# CityTier : 주거 중인 도시의 등급. (인구, 시설, 생활 수준 기준) (1등급 > 2등급 > 3등급) 
# DurationOfPitch : 영업 사원이 고객에게 제공하는 프레젠테이션 기간
# Occupation : 직업
# Gender : 성별
# NumberOfPersonVisiting : 고객과 함께 여행을 계획 중인 총 인원
# NumberOfFollowups : 영업 사원의 프레젠테이션 후 이루어진 후속 조치 수
# ProductPitched : 영업 사원이 제시한 상품
# PreferredPropertyStar : 선호 호텔 숙박업소 등급
# MaritalStatus : 결혼여부
# NumberOfTrips : 평균 연간 여행 횟수
# Passport : 여권 보유 여부 (0: 없음, 1: 있음)
# PitchSatisfactionScore : 영업 사원의 프레젠테이션 만족도
# OwnCar : 자동차 보유 여부 (0: 없음, 1: 있음)
# NumberOfChildrenVisiting : 함께 여행을 계획 중인 5세 미만의 어린이 수
# Designation : (직업의) 직급
# MonthlyIncome : 월 급여
# ProdTaken : 여행 패키지 신청 여부 (0: 신청 안 함, 1: 신청함)


# 2. test.csv : 테스트 데이터

# id : 샘플 아이디
# Age : 나이
# TypeofContact : 고객의 제품 인지 방법 (회사의 홍보 or 스스로 검색)
# CityTier : 주거 중인 도시의 등급. (인구, 시설, 생활 수준 기준) (1등급 > 2등급 > 3등급) 
# DurationOfPitch : 영업 사원이 고객에게 제공하는 프레젠테이션 기간
# Occupation : 직업
# Gender : 성별
# NumberOfPersonVisiting : 고객과 함께 여행을 계획 중인 총 인원
# NumberOfFollowups : 영업 사원의 프레젠테이션 후 이루어진 후속 조치 수
# ProductPitched : 영업 사원이 제시한 상품
# PreferredPropertyStar : 선호 호텔 숙박업소 등급
# MaritalStatus : 결혼여부
# NumberOfTrips : 평균 연간 여행 횟수
# Passport : 여권 보유 여부 (0: 없음, 1: 있음)
# PitchSatisfactionScore : 영업 사원의 프레젠테이션 만족도
# OwnCar : 자동차 보유 여부 (0: 없음, 1: 있음)
# NumberOfChildrenVisiting : 함께 여행을 계획 중인 5세 미만의 어린이 수
# Designation : (직업의) 직급
# MonthlyIncome : 월 급여


# 3. sample_submission.csv : 제출 양식

# id : 샘플 아이디
# ProdTaken : 여행 패키지 신청 여부 (0: 신청 안 함, 1: 신청함)

