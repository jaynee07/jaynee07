## Hi there 👋

<!--
**jaynee07/jaynee07** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->

### 구내식당 중식 선택 비율 예측
메뉴 텍스트만으로 그날 한식 vs 일품 중 어느 쪽을 더 많이 선택할지 비율을 예측하는 프로젝트. 식수(인원) 자체는 요일·근무 인원에 좌우되므로 제외하고, 메뉴가 바꾸는 부분인 "두 코너 간 선택 비율"만 타깃으로 설계했다.

- **접근**: 메뉴를 키워드(면류/국·탕·찌개/튀김/매운맛 등)로 벡터화하고, 한식−일품의 키워드 차이를 Bradley–Terry 이항 GLM으로 학습 — 매일을 "한식 vs 일품 대결"로 모델링
- **검증**: 미래를 보지 않는 TimeSeriesSplit 5-fold + 식수 가중 MAE(WMAE)로 평가
- **결과**: 베타회귀·XGBoost·MLP·BGE-M3 임베딩 모델과 비교 실험한 결과, 표본이 작은(~170일) 상황에서 단순 선형 모델인 Bradley–Terry GLM이 가장 낮은 오차(WMAE 0.066)를 기록
- **스택**: Python, pandas, statsmodels(GLM), XGBoost, scikit-learn, BAAI/bge-m3

🔗 [Repo: lunch_prediction](https://github.com/jaynee07/lunch_prediction)

### Predicting Cafeteria Lunch Choice Share
Predicts, from menu text alone, what share of diners will choose the **Korean set** vs. the **single-plate special** on a given day. Total headcount is driven by weekday/staffing and left out of scope — the target is only the part the menu actually moves: how the choice splits between the two counters.

- **Approach**: Menus are vectorized into keyword groups (noodles, soups/stews, fried food, spicy dishes, etc.); the Korean−special keyword difference feeds a Bradley–Terry binomial GLM, treating each day as a head-to-head "Korean vs. special" contest
- **Validation**: 5-fold `TimeSeriesSplit` (never trains on the future) scored with headcount-weighted MAE (WMAE)
- **Result**: Benchmarked against beta regression, XGBoost, MLP, and BGE-M3 sentence embeddings — with a small sample (~170 days), the simple linear Bradley–Terry GLM won out with the lowest error (WMAE 0.066)
- **Stack**: Python, pandas, statsmodels (GLM), XGBoost, scikit-learn, BAAI/bge-m3

🔗 [Repo: lunch_prediction](https://github.com/<your-username>/lunch_prediction)
