# Taiwan Stock Analysis Framework

基於機器學習的台股分析框架，整合技術分析指標與隨機森林演算法，提供基礎數據處理和預測模型建構功能。

## 功能

- 自動下載台股歷史數據
- 計算基礎技術指標（RSI、KD、MACD等）
- 機器學習模型（RandomForest）預測
- 支援自定義策略開發

## 技術架構

### 資料處理
- 使用 yfinance 下載歷史股價
- 資料清理與特徵工程
- 技術指標計算

### 預測模型
- 模型：隨機森林分類器（RandomForestClassifier）
- 特徵：
  - 移動平均（MA5、MA20、MA60）
  - MACD 指標
  - RSI 指標
  - KD 指標
  - 波動率
  - 成交量變化率
- 標準化：StandardScaler

## 安裝需求

- Python 3.8+
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- yfinance >= 0.2.0

安裝所需套件：
```bash
pip install -r requirements.txt
```

## 基本使用

```python
from stock_predictor import StockPredictor

# 初始化預測器
predictor = StockPredictor()

# 訓練模型
predictor.train(['2330', '2317'])  # 使用特定股票進行訓練

# 預測單一股票
result = predictor.predict('2330')
print(result)
```

## 自定義策略

您可以通過繼承 `StockPredictor` 類別來實現自己的策略：

```python
class MyPredictor(StockPredictor):
    def _calculate_market_signal(self, df):
        # 實現您的市場判斷邏輯
        pass

    def _calculate_future_return(self, df, period=120):
        # 實現您的報酬率計算邏輯
        pass
```

## 模型調整建議

1. 特徵選擇：
   - 可根據實際需求增減技術指標
   - 考慮加入基本面指標
   - 可自行設計新特徵

2. 模型參數：
   - n_estimators: 決策樹數量
   - max_depth: 樹的最大深度
   - min_samples_leaf: 葉節點最小樣本數
   - 建議使用交叉驗證尋找最佳參數

## 注意事項

- 本框架僅提供基礎分析功能，投資決策需自行評估風險
- 歷史數據不代表未來表現
- 建議在實際應用前進行充分的回測和驗證
- 參數和策略邏輯需要根據實際情況調整

## 資料來源

股市數據使用 [yfinance](https://github.com/ranaroussi/yfinance) 套件從 Yahoo Finance 獲取。

## 授權條款

本專案採用 MIT 授權條款 - 詳見 [LICENSE](LICENSE) 文件