import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import traceback

def get_stock_data(symbol, years=3):
    """獲取股票歷史數據"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years*365)
        
        print(f"下載 {symbol} 的數據...")
        df = yf.download(f"{symbol}.TWO", start=start_date, end=end_date)
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        if df.empty:
            print(f"警告: {symbol} 沒有數據")
            return None
            
        print(f"處理 {symbol} 的技術指標...")
        df = df.sort_index()
        
        result = pd.DataFrame(index=df.index)
        result['Close'] = df['Close']
        result['High'] = df['High']
        result['Low'] = df['Low']
        result['Volume'] = df['Volume']
        
        # 基本指標計算
        result['Returns'] = result['Close'].pct_change()
        result['Volume_Returns'] = result['Volume'].pct_change()
        
        # 移動平均
        for period in [5, 20, 60]:
            result[f'MA{period}'] = result['Close'].rolling(window=period, min_periods=1).mean()
            result[f'Volume_MA{period}'] = result['Volume'].rolling(window=period, min_periods=1).mean()
        
        # MACD
        exp1 = result['Close'].ewm(span=12, adjust=False).mean()
        exp2 = result['Close'].ewm(span=26, adjust=False).mean()
        result['MACD'] = exp1 - exp2
        result['Signal_Line'] = result['MACD'].ewm(span=9, adjust=False).mean()
        
        # 波動率
        result['Volatility'] = result['Returns'].rolling(window=20).std()
        
        # RSI
        delta = result['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        result['RSI'] = 100 - (100 / (1 + rs))
        
        # KD
        high_14 = result['High'].rolling(window=14).max()
        low_14 = result['Low'].rolling(window=14).min()
        result['K'] = (result['Close'] - low_14) / (high_14 - low_14) * 100
        result['D'] = result['K'].rolling(window=3).mean()
        
        print(f"完成 {symbol} 的數據處理")
        return result
        
    except Exception as e:
        print(f"處理 {symbol} 時發生錯誤: {str(e)}")
        print(traceback.format_exc())
        return None
    
class StockPredictor:
    def __init__(self):
        self.trend_model = None
        self.return_model = None
        self.scaler = StandardScaler()
        self.feature_cols = [
            'MA5', 'MA20', 'MA60',
            'MACD', 'Signal_Line',
            'Volatility', 'RSI',
            'K', 'D',
            'Volume_Returns'
        ]
    
    def _calculate_market_signal(self, df):
        """
        市場趨勢信號計算 - 由使用者自行實現
        建議根據自己的交易策略來定義多空判斷條件
        """
        # 這裡僅提供一個基礎範例
        signals = pd.DataFrame(index=df.index)
        signals['Market_Signal'] = 0  # 預設為中性
        
        # 使用者需要自行實現判斷邏輯
        return signals['Market_Signal']
    
    def _calculate_future_return(self, df, period=120):
        """
        未來報酬計算 - 由使用者自行實現
        建議根據自己的投資目標來定義高報酬標準
        """
        returns = pd.DataFrame(index=df.index)
        future_price = df['Close'].shift(-period)
        current_price = df['Close']
        returns['Future_Return'] = (future_price - current_price) / current_price
        
        # 使用者需要自行定義高報酬標準
        return returns['Future_Return'] > 0
        
    def prepare_features(self, df):
        """準備特徵和標籤"""
        if df is None or df.empty:
            return None, None, None
            
        # 基本特徵處理
        features = df[self.feature_cols].copy()
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # 計算市場信號和未來報酬
        market_signal = self._calculate_market_signal(df)
        future_return = self._calculate_future_return(df)
        
        # 移除無效數據
        valid_mask = (
            features.notna().all(axis=1) & 
            market_signal.notna() & 
            future_return.notna()
        )
        
        if not valid_mask.any():
            return None, None, None
            
        return (
            features[valid_mask],
            market_signal[valid_mask],
            future_return[valid_mask]
        )
    
    def train(self, symbols=['2330', '2317']):  # 預設使用兩個示例股票
        """訓練模型"""
        all_features = []
        all_market_signals = []
        all_return_signals = []
        
        for symbol in symbols:
            print(f"\n開始處理 {symbol} 的數據...")
            df = get_stock_data(symbol)
            if df is not None:
                features, market_signal, return_signal = self.prepare_features(df)
                if features is not None and len(features) > 0:
                    all_features.append(features)
                    all_market_signals.append(market_signal)
                    all_return_signals.append(return_signal)
        
        if not all_features:
            raise ValueError("沒有足夠的訓練數據")
            
        X = pd.concat(all_features)
        y_market = pd.concat(all_market_signals)
        y_return = pd.concat(all_return_signals)
        
        if len(X) < 100:
            raise ValueError("訓練數據太少")
            
        X_scaled = self.scaler.fit_transform(X)
        
        # 訓練模型
        self.trend_model = RandomForestClassifier(random_state=42)
        self.return_model = RandomForestClassifier(random_state=42)
        
        self.trend_model.fit(X_scaled, y_market)
        self.return_model.fit(X_scaled, y_return)
        
        print("\n模型訓練完成！")
    
    def predict(self, symbol):
        """預測特定股票的趨勢和報酬機率"""
        try:
            df = get_stock_data(symbol)
            if df is None:
                return f"無法獲取 {symbol} 的數據"
                
            features, _, _ = self.prepare_features(df)
            if features is None or len(features) == 0:
                return "無法生成預測所需的特徵"
                
            latest_features = features.iloc[-1:]
            scaled_features = self.scaler.transform(latest_features)
            
            trend_prob = self.trend_model.predict_proba(scaled_features)[0][1]
            return_prob = self.return_model.predict_proba(scaled_features)[0][1]
            
            return {
                'trend': '看多' if trend_prob > 0.5 else '看空',
                'trend_probability': trend_prob,
                'return_probability': return_prob,
                'indicators': {
                    'RSI': df['RSI'].iloc[-1],
                    'K': df['K'].iloc[-1],
                    'D': df['D'].iloc[-1]
                }
            }
            
        except Exception as e:
            print(traceback.format_exc())
            return f"預測錯誤: {str(e)}"

def main():
    predictor = StockPredictor()
    
    print("正在初始化模型...")
    try:
        predictor.train()
        
        while True:
            command = input("\n請輸入指令 (1: 預測, Q: 退出): ")
            
            if command.upper() == 'Q':
                break
            
            if command == '1':
                symbol = input("請輸入股票代碼: ")
                result = predictor.predict(symbol)
                if isinstance(result, dict):
                    print(f"\n分析結果:")
                    print(f"趨勢: {result['trend']}")
                    print(f"看多機率: {result['trend_probability']:.2%}")
                    print(f"高報酬機率: {result['return_probability']:.2%}")
                    print("\n技術指標:")
                    print(f"RSI: {result['indicators']['RSI']:.2f}")
                    print(f"K值: {result['indicators']['K']:.2f}")
                    print(f"D值: {result['indicators']['D']:.2f}")
                else:
                    print(result)
            else:
                print("無效的指令！")
                
    except Exception as e:
        print(f"程式執行錯誤: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()