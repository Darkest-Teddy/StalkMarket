# Stalk Market: Learn Investing Through Growth

## About Our Project
Financial markets are complex and intimidating, making it hard for beginners to learn how investing works.  
**Stalk Market** simplifies financial education by turning investing into an interactive and visual experience — where **stocks grow like plants**.  
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

### Option A: Local development
```bash
cd path/to/StalkMarket
python -m uvicorn main:app --reload --port 8000
```
Frontend assets live in `public/`; open `public/index.html` or run a static server (e.g., `python -m http.server 5173`).

### Option B: Render (backend) + Vercel (frontend)
1. **Render backend**
   - Push this repo (including `Dockerfile` and `requirements.txt`) to GitHub/GitLab.
   - Create a Render Web Service → choose Docker → point to the repo.
   - Render runs `uvicorn main:app --host 0.0.0.0 --port 8000` automatically.
   - Add env vars like `FRED_API_KEY` in the Render dashboard.
   - Note the service URL, e.g. `https://your-render-service.onrender.com`.
2. **Vercel frontend**
   - `vercel.json` already rewrites `/api/*` to the Render URL. Replace `https://your-render-service.onrender.com` with your actual Render endpoint, commit, and redeploy on Vercel.
   - Vercel serves everything in `public/` as static assets; API calls are proxied to Render.
3. **Seed a season after deployment**
   - The backend stores data in memory, so call the demo seed endpoint after each fresh deployment:
     ```bash
     curl -X POST https://your-vercel-domain.vercel.app/api/demo_seed
     ```
   - This populates the default season so the `/api/season/...` routes have data.
4. **Verify**
   - Visit your Vercel URL, open dev tools → Network, and confirm `/api/...` requests return 200 responses.

Once Render is running, you no longer need to start uvicorn manually—Vercel routes all API traffic to the Render service.

## Accomplishments
* Developed a working hybrid AI + algorithmic stock simulation
* Built a stable and accessible web platform suitable for all learners
* Education mode to reinforce responsible investing

## What's Next
* Add personalized learning paths that adapt to user choices and skill levels
* Introduce social and collaborative features — e.g., classroom mode or friendly competitions
* Expand random event types (economic news, weather, policy shifts)
