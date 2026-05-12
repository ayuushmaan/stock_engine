import pandas as pd
import os

def generate_prediction_report():
    print("Loading Data Files")
    
    # 1. Load the Price Data (from daily_loader.py)
    # Check if file exists to avoid "FileNotFoundError"
    if not os.path.exists("daily_prices.csv"):
        print("daily_prices.csv not found.")
        return
    
    prices = pd.read_csv("daily_prices.csv")
    
    # 2. Load the Sentiment Data (from Scanner.py)
    # Note: Scanner.py saves to the main folder, not 'data/' folder
    if not os.path.exists("nifty_sentiment_results.csv"):
        print("nifty_sentiment_results.csv not found.")
        return

    sentiment = pd.read_csv("nifty_sentiment_results.csv")
    
    # 3. Merge them on 'Symbol'
    # inner join = only show stocks that exist in BOTH files
    merged = pd.merge(prices, sentiment, on="Symbol", how="inner")
    
    if merged.empty:
        print("⚠️ No common stocks found between Prices and Sentiment files.")
        print("Check if Scanner.py actually found any news.")
        return

    # 4. Calculate Accuracy
    print("\n" + "="*75)
    print(f"{'SYMBOL':<15} | {'PREDICTION':<10} | {'ACTUAL MOVE':<12} | {'RESULT':<10}")
    print("-" * 75)
    
    correct_count = 0
    total_valid = 0
    
    for index, row in merged.iterrows():
        # FIX: Use 'Direction' because that is what Scanner.py saved
        prediction = row['Direction'] 
        actual = row['Actual_Direction']
        
        # We only judge if the prediction was BULLISH or BEARISH (Skip Neutral)
        if prediction == "BULLISH":
            pred_val = "UP"
        elif prediction == "BEARISH":
            pred_val = "DOWN"
        else:
            pred_val = "NEUTRAL"
            
        # Check against reality
        if pred_val == "NEUTRAL":
            result = "SKIPPED"
        elif pred_val == actual:
            result = "CORRECT"
            correct_count += 1
            total_valid += 1
        else:
            result = "WRONG"
            total_valid += 1
            
        print(f"{row['Symbol']:<15} | {prediction:<10} | {actual:<12} | {result:<10}")

    print("="*75)
    
    if total_valid > 0:
        accuracy = (correct_count / total_valid) * 100
        print(f"Final Model Accuracy: {accuracy:.2f}%")
    else:
        print("No valid predictions made (All were Neutral or Data is missing).")

if __name__ == "__main__":
    generate_prediction_report()