# Stalk Market: Learn Investing Through Growth

## About Our Project
Financial markets are complex and intimidating, making it hard for beginners to learn how investing works.  
**StockGarden** simplifies financial education by turning investing into an interactive and visual experience — where **stocks grow like plants**.  
Our platform helps users learn key financial principles safely, through simulation, gamification, and guided experimentation.

At the core of our system is a **hybrid predictive model** combining traditional algorithmic forecasting with AI-based prediction:
- **Algorithm-Based Model:** Uses macroeconomic data (CPI, unemployment rate, etc.) and the Geometric Brownian Motion (GBM) model from the **FRED** dataset to estimate stock movements.
- **AI-Based Model:** Trained with 10 years of historical stock data from **Yahoo Finance**, using transfer learning on a modified **GPT-2** model.  
  - GPT-2 is adapted to handle **numerical sequences**: previous price fluctuations as input, and predicted price changes as output.  
- **Final Price:** Computed as  
  **`0.8 × (Algorithm Prediction) + 0.2 × (AI Prediction)`**

To make learning engaging and realistic, we also introduce:
- **Random Market Events:** These simulate real-world unpredictability — affecting both crop (stock) growth and prices.  
- **Education Mode:** An extra ML layer discourages risky investing by adjusting market volatility. If a user “puts all eggs in one basket,” the model reduces volatility to show why diversification matters.

Our platform is designed for **accessibility and fun**. Users track their portfolio as a garden — with plants that grow, bear fruit, or wither — visually showing how decisions and market conditions impact performance. Fonts are large, navigation is simple, and visuals are engaging enough for all ages.

## Running the Project

### Step 1. Navigate to the Backend Folder
Go to the main backend directory:
```bash
cd .../StalkMarket/main/Code
```
### Step 2. Run the Python File
```bash
python -m uvicorn main:api --reload --port 8000
```

## Accomplishments
* Developed a working hybrid AI + algorithmic stock simulation
* Built a stable and accessible web platform suitable for all learners
* Education mode to reinforce responsible investing

## What's Next
* Add personalized learning paths that adapt to user choices and skill levels
* Introduce social and collaborative features — e.g., classroom mode or friendly competitions
* Expand random event types (economic news, weather, policy shifts)
