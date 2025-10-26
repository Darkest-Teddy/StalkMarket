"""
main.py — Unified FarmFinance FastAPI backend

Combines:
 - RealMLAIEngine (sklearn-based models trained on simulated portfolios, training runs in background)
 - Optional StockLLM wrapper (if `stock_llm` module + torch + model.pt provided)
 - FRED macro integration (optional)
 - Event engine, simulate, forecast, montecarlo, report endpoints
 - CORS middleware and optional StaticFiles frontend serve
"""

import os
import math
import time
import random
import copy
import threading
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Optional dependencies
try:
    import pandas as pd
except Exception:
    pd = None

# sklearn (may be missing in some environments)
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False
    print("[WARN] scikit-learn not available - ML features disabled")

# statsmodels Holt-Winters optional
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_HW = True
except Exception:
    HAS_HW = False

# fredapi optional
try:
    from fredapi import Fred
    HAS_FRED = True
except Exception:
    HAS_FRED = False

# torch / StockLLM optional
HAS_TORCH = False
try:
    import torch
    HAS_TORCH = True
except Exception:
    torch = None

StockLLM = None
if HAS_TORCH:
    try:
        from stock_llm import StockLLM
    except Exception:
        StockLLM = None

# Logging
logger = logging.getLogger("main")
logging.basicConfig(level=logging.INFO)

EPS = 1e-12


# ------------------------------
# Utilities
# ------------------------------
def seasonal_multiplier(t: int, T: int, amplitude: float, phase: float = 0.0) -> float:
    return amplitude * math.sin(2 * math.pi * (t / max(T, 1)) + phase)


# ------------------------------
# Optional LLM wrapper (torch + user-provided StockLLM)
# ------------------------------
class LLMWrapper:
    def __init__(self, model_path: str = "model.pt"):
        self.model = None
        self.available = False
        if StockLLM is None or not HAS_TORCH:
            logger.info("StockLLM or torch not available; LLM disabled")
            return
        try:
            self.model = StockLLM()
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
                self.model.eval()
                self.available = True
                logger.info("LLM loaded from %s", model_path)
            else:
                logger.warning("LLM model file %s not found; LLM disabled", model_path)
        except Exception as e:
            logger.exception("Failed to initialize LLM: %s", e)
            self.model = None
            self.available = False

    def predict_multiplier(self, features: List[float]) -> float:
        """Return a multiplicative factor near 1.0 to adjust price"""
        if not self.available:
            return 1.0
        try:
            arr = torch.tensor([features], dtype=torch.float32)
            with torch.no_grad():
                out = self.model(arr)
            val = float(np.array(out.cpu().numpy()).flatten()[0])
            # map to a reasonable range around 1.0
            return float(max(0.5, min(1.5, 0.98 + 0.1 * val)))
        except Exception as e:
            logger.exception("LLM predict error: %s", e)
            return 1.0


LLM = LLMWrapper(model_path="model.pt")


# ------------------------------
# Price step function — stochastic + behavioral + optional LLM
# ------------------------------
def step_price(P: float, mu: float, sigma: float, dt: float, seasonality_t: float,
               lam: float = 0.0, mu_j: float = 0.0, sig_j: float = 0.0,
               use_llm: bool = True, llm_features: Optional[List[float]] = None) -> float:
    # Geometric Brownian motion-like step with seasonality and jumps
    Z = np.random.normal()
    log_ret = (mu - 0.5 * sigma**2) * dt + sigma * (dt**0.5) * Z
    log_ret += seasonality_t * dt
    if lam > 0.0:
        jump_count = np.random.poisson(lam * dt)
        if jump_count > 0:
            log_ret += np.random.normal(mu_j, sig_j) * jump_count
    P_stochastic = P * math.exp(log_ret)

    # Behavioral dampening: placeholder risk proxy
    risk_score = random.uniform(0.0, 1.0)
    vol_factor = 1.0 - 0.5 * risk_score
    P_adjusted = P + (P_stochastic - P) * vol_factor

    # LLM-driven multiplier if available
    if use_llm and LLM.available:
        feat = llm_features if llm_features is not None else [P, mu, sigma, dt, seasonality_t, lam, mu_j, sig_j, risk_score]
        try:
            mult = LLM.predict_multiplier(feat)
            P_final = P_adjusted * mult
        except Exception:
            P_final = P_adjusted
    else:
        P_final = P_adjusted

    return max(P_final, EPS)


# ------------------------------
# Portfolio Simulator — used to generate training data for ML
# ------------------------------
class PortfolioSimulator:
    def __init__(self):
        self.crop_params = {
            'wheat': {'mu': 0.03, 'sigma': 0.08, 'seas': 0.05, 'jump_lam': 0.00},
            'corn': {'mu': 0.07, 'sigma': 0.16, 'seas': 0.07, 'jump_lam': 0.02},
            'berries': {'mu': 0.12, 'sigma': 0.32, 'seas': 0.10, 'jump_lam': 0.05},
            'truffle': {'mu': 0.09, 'sigma': 0.22, 'seas': 0.06, 'jump_lam': 0.03},
        }
        self.crop_ids = list(self.crop_params.keys())

    def simulate_portfolio(self, weights: np.ndarray, steps: int = 52,
                          dt: float = 1.0/52.0, seed: Optional[int] = None) -> dict:
        if seed is not None:
            np.random.seed(seed)
        prices = {cid: 100.0 for cid in self.crop_ids}
        portfolio_values = [100.0]
        for t in range(steps):
            for i, cid in enumerate(self.crop_ids):
                params = self.crop_params[cid]
                seas = seasonal_multiplier(t, steps, params['seas'])
                prices[cid] = step_price(prices[cid], mu=params['mu'], sigma=params['sigma'],
                                         dt=dt, seasonality_t=seas, lam=params['jump_lam'],
                                         mu_j=-0.01, sig_j=0.05, use_llm=False)
            port_val = sum(weights[i] * prices[cid] for i, cid in enumerate(self.crop_ids))
            portfolio_values.append(port_val)
        portfolio_values = np.array(portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        realized_vol = np.std(returns) * math.sqrt(52)
        cummax = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - cummax) / cummax
        max_drawdown = abs(np.min(drawdowns))
        rf = 0.02
        sharpe = (np.mean(returns) * 52 - rf) / (realized_vol + 1e-10)
        downside_returns = returns[returns < 0]
        downside_vol = np.std(downside_returns) * math.sqrt(52) if len(downside_returns) > 0 else 0.0
        n_negative = np.sum(returns < 0)
        pct_negative = n_negative / len(returns) if len(returns) > 0 else 0
        return {
            'total_return': float(total_return),
            'realized_vol': float(realized_vol),
            'max_drawdown': float(max_drawdown),
            'sharpe': float(sharpe),
            'downside_vol': float(downside_vol),
            'pct_negative': float(pct_negative),
            'final_value': float(portfolio_values[-1])
        }


# ------------------------------
# Real ML AI Engine (scikit-learn) — trains in background
# ------------------------------
class RealMLAIEngine:
    def __init__(self, n_simulations: int = 2000):
        self.enabled = HAS_SKLEARN
        self.is_trained = False
        self.training_metrics = {}
        self.simulator = PortfolioSimulator()
        if not self.enabled:
            logger.warning("scikit-learn not available; ML engine disabled")
            return

        # Models
        self.risk_classifier = RandomForestClassifier(n_estimators=100, max_depth=10,
                                                      min_samples_split=5, random_state=42)
        self.drawdown_predictor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        self.volatility_predictor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        self.return_predictor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        self.portfolio_clusterer = KMeans(n_clusters=5, random_state=42, n_init=10)
        self.scaler = StandardScaler()

        # Start background training thread
        t = threading.Thread(target=self._train_from_simulations, kwargs={"n_simulations": n_simulations}, daemon=True)
        t.start()
        logger.info("ML engine started background training thread (%d simulations)", n_simulations)

    def _extract_features_from_weights(self, weights: np.ndarray) -> np.ndarray:
        hhi = np.sum(weights**2)
        max_weight = np.max(weights) if len(weights) else 0.0
        n_positions = np.sum(weights > 0.01)
        entropy = -np.sum(weights[weights > 0] * np.log(weights[weights > 0] + 1e-10))
        weight_std = np.std(weights)
        vols = np.array([0.08, 0.16, 0.32, 0.22])
        port_vol_est = np.sqrt(np.sum((weights * vols) ** 2))
        mus = np.array([0.03, 0.07, 0.12, 0.09])
        port_mu_est = np.sum(weights * mus)
        # individual weights (pad if needed)
        w = np.zeros(4)
        for i in range(min(4, len(weights))):
            w[i] = weights[i]
        high_vol_weight = float(w[2] + w[3])
        low_vol_weight = float(w[0] + w[1])
        return np.array([hhi, max_weight, n_positions, entropy, weight_std,
                         port_vol_est, port_mu_est, w[0], w[1], w[2], w[3], high_vol_weight, low_vol_weight])

    def _train_from_simulations(self, n_simulations: int = 5000):
        if not self.enabled:
            return
        logger.info("[ML-AI] Generating training data from %d portfolio simulations...", n_simulations)
        X_features = []
        y_risk = []
        y_drawdown = []
        y_volatility = []
        y_return = []
        np.random.seed(42)
        for i in range(n_simulations):
            n_positions = np.random.randint(1, 5)
            weights = np.random.dirichlet(np.ones(4))
            if n_positions < 4:
                zero_indices = np.random.choice(4, 4 - n_positions, replace=False)
                weights[zero_indices] = 0
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                continue
            features = self._extract_features_from_weights(weights)
            outcome = self.simulator.simulate_portfolio(weights, steps=52, seed=i)
            X_features.append(features)
            y_drawdown.append(outcome['max_drawdown'])
            y_volatility.append(outcome['realized_vol'])
            y_return.append(outcome['total_return'])
            # risk label from realized drawdown
            dd = outcome['max_drawdown']
            if dd < 0.10:
                risk = 0
            elif dd < 0.20:
                risk = 1
            elif dd < 0.35:
                risk = 2
            elif dd < 0.50:
                risk = 3
            else:
                risk = 4
            y_risk.append(risk)

        X = np.array(X_features)
        y_risk = np.array(y_risk)
        y_drawdown = np.array(y_drawdown)
        y_volatility = np.array(y_volatility)
        y_return = np.array(y_return)

        if len(X) < 10:
            logger.warning("Not enough samples to train ML models")
            return

        X_train, X_test, y_risk_train, y_risk_test = train_test_split(X, y_risk, test_size=0.2, random_state=42)
        _, _, y_dd_train, y_dd_test = train_test_split(X, y_drawdown, test_size=0.2, random_state=42)
        _, _, y_vol_train, y_vol_test = train_test_split(X, y_volatility, test_size=0.2, random_state=42)
        _, _, y_ret_train, y_ret_test = train_test_split(X, y_return, test_size=0.2, random_state=42)

        # scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        logger.info("[ML-AI] Training risk classifier...")
        self.risk_classifier.fit(X_train_scaled, y_risk_train)
        risk_acc = self.risk_classifier.score(X_test_scaled, y_risk_test)

        logger.info("[ML-AI] Training drawdown predictor...")
        self.drawdown_predictor.fit(X_train_scaled, y_dd_train)
        dd_r2 = self.drawdown_predictor.score(X_test_scaled, y_dd_test)
        dd_mae = mean_absolute_error(y_dd_test, self.drawdown_predictor.predict(X_test_scaled))

        logger.info("[ML-AI] Training volatility predictor...")
        self.volatility_predictor.fit(X_train_scaled, y_vol_train)
        vol_r2 = self.volatility_predictor.score(X_test_scaled, y_vol_test)

        logger.info("[ML-AI] Training return predictor...")
        self.return_predictor.fit(X_train_scaled, y_ret_train)
        ret_r2 = self.return_predictor.score(X_test_scaled, y_ret_test)

        logger.info("[ML-AI] Training portfolio clusterer...")
        self.portfolio_clusterer.fit(X_train_scaled)

        self.is_trained = True
        self.training_metrics = {
            "n_samples": len(X),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "risk_classifier_accuracy": float(risk_acc),
            "drawdown_predictor_r2": float(dd_r2),
            "drawdown_predictor_mae": float(dd_mae),
            "volatility_predictor_r2": float(vol_r2),
            "return_predictor_r2": float(ret_r2),
        }
        logger.info("[ML-AI] Training complete: acc=%.3f drawdown_r2=%.3f", risk_acc, dd_r2)

    def extract_features(self, holdings: Dict[str, float], prices: Dict[str, float]) -> Tuple[np.ndarray, dict]:
        total_value = sum(holdings.get(cid, 0) * prices.get(cid, 0) for cid in ['wheat', 'corn', 'berries', 'truffle'])
        if total_value <= 0:
            weights = np.zeros(4)
        else:
            weights = np.array([(holdings.get(cid, 0) * prices.get(cid, 0)) / total_value for cid in ['wheat', 'corn', 'berries', 'truffle']])
        features = self._extract_features_from_weights(weights)
        metadata = {
            "hhi": float(features[0]),
            "max_weight": float(features[1]),
            "n_positions": int(features[2]),
            "entropy": float(features[3]),
            "weights": {'wheat': float(weights[0]), 'corn': float(weights[1]), 'berries': float(weights[2]), 'truffle': float(weights[3])}
        }
        return features, metadata

    def analyze_portfolio(self, holdings: Dict[str, float], prices: Dict[str, float]) -> dict:
        if not self.enabled or not self.is_trained:
            return {"ml_enabled": False, "message": "ML unavailable or still training"}
        features, metadata = self.extract_features(holdings, prices)
        if metadata["n_positions"] == 0:
            return {"ml_enabled": True, "empty_portfolio": True, "message": "Build a portfolio", "training_metrics": self.training_metrics}
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        risk_class = int(self.risk_classifier.predict(features_scaled)[0])
        risk_probs = self.risk_classifier.predict_proba(features_scaled)[0]
        predicted_drawdown = float(self.drawdown_predictor.predict(features_scaled)[0])
        predicted_volatility = float(self.volatility_predictor.predict(features_scaled)[0])
        predicted_return = float(self.return_predictor.predict(features_scaled)[0])
        cluster = int(self.portfolio_clusterer.predict(features_scaled)[0])
        risk_levels = ["Safe", "Moderate", "Elevated", "High", "Extreme"]
        archetypes = ["Well-Diversified Balanced", "Conservative Low-Risk", "Growth-Oriented Aggressive", "Moderately Concentrated", "Highly Concentrated High-Risk"]
        insights = self._generate_insights(metadata, risk_class, predicted_drawdown, predicted_volatility, predicted_return, risk_probs)
        recommendations = self._generate_recommendations(metadata, risk_class, predicted_drawdown, predicted_volatility)
        return {
            "ml_enabled": True,
            "empty_portfolio": False,
            "features": metadata,
            "predictions": {
                "risk_level": risk_levels[risk_class],
                "risk_class": int(risk_class),
                "risk_probabilities": risk_probs.tolist(),
                "max_drawdown": float(np.clip(predicted_drawdown, 0, 1)),
                "expected_volatility": float(np.clip(predicted_volatility, 0, 1)),
                "expected_return": float(predicted_return),
                "sharpe_estimate": float(predicted_return / (predicted_volatility + 0.01))
            },
            "portfolio_archetype": {"cluster": int(cluster), "name": archetypes[cluster]},
            "ml_insights": insights,
            "recommendations": recommendations,
            "training_metrics": self.training_metrics
        }

    def _generate_insights(self, metadata, risk_class, pred_dd, pred_vol, pred_ret, risk_probs):
        risk_levels = ["Safe", "Moderate", "Elevated", "High", "Extreme"]
        risk_name = risk_levels[risk_class]
        confidence = float(risk_probs[risk_class] * 100)
        insight = f"🤖 ML Prediction: {risk_name} (confidence: {confidence:.1f}%). "
        insight += f"Predicted {pred_dd*100:.1f}% max drawdown, {pred_vol*100:.1f}% vol, {pred_ret*100:.1f}% return. "
        if risk_class >= 3:
            insight += "⚠️ HIGH RISK"
        insight += f" Portfolio has {metadata['n_positions']} position(s) with HHI={metadata['hhi']:.3f}."
        return insight

    def _generate_recommendations(self, metadata, risk_class, pred_dd, pred_vol):
        recs = []
        if risk_class >= 3:
            recs.append("⚠️ ML ALERT: High risk predicted.")
        if metadata["hhi"] > 0.5:
            recs.append("⚠️ Concentration detected. Diversify.")
        if metadata["n_positions"] < 3:
            recs.append("📊 Consider adding positions to improve diversification.")
        if pred_vol > 0.25:
            recs.append(f"🌊 High volatility predicted ({pred_vol*100:.1f}%). Consider lower-volatility crops.")
        if not recs:
            recs.append("✅ Portfolio looks balanced.")
        return recs

    def calculate_market_adjustment(self, analysis: dict, educational_mode: bool) -> dict:
        if not analysis.get("ml_enabled"):
            return {"mu_penalty": 0.0, "vol_multiplier": 1.0, "apply_adjustments": False}
        if analysis.get("empty_portfolio"):
            return {"mu_penalty": 0.0, "vol_multiplier": 1.0, "apply_adjustments": False}
        risk_class = analysis["predictions"]["risk_class"]
        pred_vol = analysis["predictions"]["expected_volatility"]
        penalties = [(0.00, 1.0), (0.05, 1.1), (0.10, 1.3), (0.15, 1.5), (0.20, 1.8)]
        mu_penalty, vol_mult = penalties[risk_class]
        if pred_vol > 0.30:
            vol_mult += 0.2
        return {"mu_penalty": float(mu_penalty), "vol_multiplier": float(vol_mult), "apply_adjustments": risk_class >= 2, "ml_driven": True}


# instantiate ML engine (background training)
ML_AI = RealMLAIEngine(n_simulations=2000)


# ------------------------------
# Shared Storage, Crops, Events
# ------------------------------
class Storage:
    def __init__(self):
        self.players: Dict[str, dict] = {}
        self.seasons: Dict[str, dict] = {}
        self.crops: Dict[str, dict] = {}
        self.prices: Dict[str, List[dict]] = {}
        self.events: Dict[str, List[dict]] = {}
        self.event_engines: Dict[str, "EventEngine"] = {}
        self.season_params: Dict[str, List[dict]] = {}

DB = Storage()

DEFAULT_CROPS = [
    dict(id="wheat",   name="Wheat",   cls="bond_like",  base_mu=0.03, base_sigma=0.08, seas=0.05, kappa=0.0, jump_lam=0.00, j_mu=0.0,  j_sig=0.00),
    dict(id="corn",    name="Corn",    cls="blue_chip",  base_mu=0.07, base_sigma=0.16, seas=0.07, kappa=0.0, jump_lam=0.02, j_mu=-0.01,j_sig=0.05),
    dict(id="berries", name="Berries", cls="growth",     base_mu=0.12, base_sigma=0.32, seas=0.10, kappa=0.0, jump_lam=0.05, j_mu=-0.02,j_sig=0.09),
    dict(id="truffle", name="Truffle", cls="alt",        base_mu=0.09, base_sigma=0.22, seas=0.06, kappa=0.0, jump_lam=0.03, j_mu=-0.01,j_sig=0.07),
]
for c in DEFAULT_CROPS:
    DB.crops[c["id"]] = c


@dataclass
class EventRule:
    id: str
    level: str
    target: Optional[str]
    name: str
    base_prob: float
    impact_mu: float
    impact_sigma: float
    duration_steps: int = 1


@dataclass
class ActiveEvent:
    rule_id: str
    remaining: int
    affected: List[str]


class EventEngine:
    def __init__(self, rules: List[EventRule]):
        self.rules = rules
        self.active: List[ActiveEvent] = []

    def step(self, crop_ids: List[str], macro: dict) -> List[dict]:
        events_out: List[dict] = []
        for r in self.rules:
            prob = r.base_prob
            if macro.get("recession", False) and r.id in {"bear_storm", "bug_attack", "supply_shock"}:
                prob *= 1.5
            if macro.get("term_spread", 1.0) < 0 and r.id == "vol_spike":
                prob *= 1.4
            if random.random() < prob:
                if r.level == "global":
                    affected = list(crop_ids)
                elif r.level == "class":
                    affected = [cid for cid in crop_ids if DB.crops[cid]["cls"] == r.target]
                else:
                    affected = [r.target]
                self.active.append(ActiveEvent(r.id, r.duration_steps, affected))
                events_out.append({"type": r.id, "name": r.name, "affected": affected, "note": "event triggered",
                                   "duration": r.duration_steps, "level": r.level, "target": r.target})
        # decay active events
        for ev in list(self.active):
            ev.remaining -= 1
            if ev.remaining <= 0:
                self.active.remove(ev)
        return events_out

    def log_return_shock(self, crop_id: str) -> float:
        shock = 0.0
        for ev in self.active:
            rule = next((r for r in self.rules if r.id == ev.rule_id), None)
            if rule and crop_id in ev.affected:
                shock += np.random.normal(rule.impact_mu, rule.impact_sigma)
        return shock


DEFAULT_EVENT_RULES = [
    EventRule("bull_sun", "global", None, "Sunny Bull Season", 0.03, +0.01, 0.005, 2),
    EventRule("bear_storm", "global", None, "Stormy Bear Front", 0.03, -0.012, 0.008, 2),
    EventRule("vol_spike", "global", None, "Volatility Spike", 0.02, 0.0, 0.015, 1),
    EventRule("bug_attack", "class", "growth", "Pest Outbreak (Growth)", 0.04, -0.015, 0.010, 1),
    EventRule("bountiful_rain", "class", "bond_like", "Bountiful Rains (Bond-like)", 0.025, +0.006, 0.003, 1),
    EventRule("supply_shock", "crop", "corn", "Supply Chain Shock (Corn)", 0.02, -0.02, 0.012, 1),
]


# ------------------------------
# FRED helpers + macro computation
# ------------------------------
def _fred():
    if not HAS_FRED:
        raise RuntimeError("fredapi not installed")
    key = os.getenv("FRED_API_KEY", "")
    if not key:
        raise RuntimeError("FRED_API_KEY not set")
    return Fred(api_key=key)


def compute_macro_context(start: str = "2024-01-01", end: str = "2024-12-31") -> dict:
    fallback = {"inflation_ann": 0.02, "rf_rate_ann": 0.02, "recession": False, "term_spread": 1.0, "vol_mult": 1.0, "asof": None}
    if not HAS_FRED:
        return fallback
    try:
        fred = _fred()
        cpi_raw = fred.get_series("CPIAUCSL", observation_start="2023-01-01", observation_end=end).dropna()
        ff_raw = fred.get_series("FEDFUNDS", observation_start="2023-01-01", observation_end=end).dropna()
        rec_raw = fred.get_series("USREC", observation_start="2023-01-01", observation_end=end).dropna()
        spr_raw = fred.get_series("T10Y3M", observation_start="2023-01-01", observation_end=end).dropna()
        if pd is None:
            raise RuntimeError("pandas required for FRED integration")
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        cpi = cpi_raw.loc[(cpi_raw.index >= start_dt) & (cpi_raw.index <= end_dt)]
        ff = ff_raw.loc[(ff_raw.index >= start_dt) & (ff_raw.index <= end_dt)]
        rec = rec_raw.loc[(rec_raw.index >= start_dt) & (rec_raw.index <= end_dt)]
        spr = spr_raw.loc[(spr_raw.index >= start_dt) & (spr_raw.index <= end_dt)]
        asof = str(end_dt.date())
        if len(cpi) and len(cpi_raw) >= 13:
            last = cpi.iloc[-1]
            one_year_prior = cpi.index[-1] - pd.DateOffset(years=1)
            hist_window = cpi_raw.loc[:one_year_prior]
            if len(hist_window):
                prev_year = hist_window.iloc[-1]
                infl_ann = float(last / prev_year - 1.0)
            else:
                infl_ann = fallback["inflation_ann"]
        else:
            infl_ann = fallback["inflation_ann"]
        rf_ann = float(ff.iloc[-1]) / 100.0 if len(ff) else fallback["rf_rate_ann"]
        recession = bool(rec.iloc[-1]) if len(rec) else False
        term_spread = float(spr.iloc[-1]) if len(spr) else fallback["term_spread"]
        penalty = max(0.0, -term_spread)
        vol_mult = float(np.clip(1.0 + 0.35 * penalty, 0.75, 1.4))
        return dict(inflation_ann=float(infl_ann), rf_rate_ann=float(rf_ann), recession=recession, term_spread=float(term_spread), vol_mult=vol_mult, asof=asof)
    except Exception as exc:
        logger.warning("compute_macro_context fallback: %s", exc)
        return fallback


# ------------------------------
# API Schemas
# ------------------------------
class CropParams(BaseModel):
    crop_id: str
    mu: float
    sigma: float
    seasonality_strength: float = Field(0.2, ge=0.0, le=1.0)
    mean_revert_kappa: float = 0.0
    jump_lam: float = 0.0
    jump_mu: float = 0.0
    jump_sig: float = 0.0


class AdvanceRequest(BaseModel):
    steps: int = Field(12, ge=1, le=260)
    seed: Optional[int] = None
    diversification_hhi: Optional[float] = Field(None, ge=0.0, le=1.0)


class SimulateRequest(BaseModel):
    season_id: str
    seed: int = 42
    steps: int = 52
    dt: float = 1.0 / 52.0
    crop_params: List[CropParams]
    start_prices: Dict[str, float] = Field(default_factory=dict)


class SimulateResponse(BaseModel):
    season_id: str
    prices: List[dict]
    events: List[dict]
    macro: dict


class ForecastRequest(BaseModel):
    season_id: str
    horizon: int = 12
    crop_ids: Optional[List[str]] = None


class ForecastResponse(BaseModel):
    season_id: str
    horizon: int
    forecasts: Dict[str, dict]


class MonteCarloRequest(BaseModel):
    prices_now: Dict[str, float]
    weights: Dict[str, float]
    crop_params: List[CropParams]
    horizon_steps: int = 12
    dt: float = 1.0 / 52.0
    N: int = 2000


class MonteCarloResponse(BaseModel):
    percentiles: Dict[str, List[float]]
    var_es: Dict[str, float]


class Choice(BaseModel):
    outcomes_A: List[float]
    probs_A: List[float]
    outcomes_B: List[float]
    probs_B: List[float]
    picked: str


class FitRiskRequest(BaseModel):
    player_id: str
    choices: List[Choice]


class FitRiskResponse(BaseModel):
    player_id: str
    gamma: float
    beta: float


class ReportRequest(BaseModel):
    season_id: str
    rf_rate: float = 0.01


class ReportResponse(BaseModel):
    season_id: str
    metrics: dict
    diversification: dict
    market_exposure: dict
    behavior: dict
    tips: List[str]
    counterfactual: dict
    macro: dict


# ------------------------------
# FastAPI app & routes
# ------------------------------
api = FastAPI(title="FarmFinance Unified API", version="1.0.0")
api.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Serve frontend if "frontend" directory exists
if os.path.isdir("frontend"):
    api.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


@api.get("/health")
def health():
    return {
        "ok": True,
        "time": time.time(),
        "ml_enabled": ML_AI.enabled,
        "ml_trained": ML_AI.is_trained if ML_AI.enabled else False,
        "llm_available": LLM.available
    }


@api.get("/macro")
def macro_snapshot():
    m = compute_macro_context(start="2024-01-01", end="2024-12-31")
    return {"ok": True, "macro": m}


@api.post("/ai/analyze")
def ai_analyze_portfolio(req: dict):
    holdings = req.get("holdings", {})
    prices = req.get("prices", {})
    analysis = ML_AI.analyze_portfolio(holdings, prices) if ML_AI.enabled else {"ml_enabled": False}
    adjustment = ML_AI.calculate_market_adjustment(analysis, educational_mode=True) if ML_AI.enabled else {"mu_penalty": 0.0, "vol_multiplier": 1.0}
    return {"analysis": analysis, "market_adjustment": adjustment}


@api.get("/ai/training_metrics")
def get_training_metrics():
    if not ML_AI.enabled or not ML_AI.is_trained:
        return {"error": "ML not available or still training"}
    return ML_AI.training_metrics


@api.post("/simulate", response_model=SimulateResponse)
def simulate(req: SimulateRequest):
    np.random.seed(req.seed)
    season_id = req.season_id
    macro = compute_macro_context(start="2024-01-01", end="2024-12-31")
    engine = EventEngine(copy.deepcopy(DEFAULT_EVENT_RULES))
    DB.event_engines[season_id] = engine
    DB.season_params[season_id] = [cp.dict() for cp in req.crop_params]
    DB.seasons[season_id] = dict(
        id=season_id,
        inflation=macro.get("inflation_ann", 0.02),
        rf=macro.get("rf_rate_ann", 0.02),
        recession=macro.get("recession", False),
        asof=macro.get("asof"),
        macro=macro,
        dt=req.dt,
        period=req.steps,
    )
    DB.prices[season_id] = []
    DB.events[season_id] = []

    P = {cp.crop_id: req.start_prices.get(cp.crop_id, 100.0) for cp in req.crop_params}
    adj = []
    for cp in req.crop_params:
        mu_nom = cp.mu + macro.get("inflation_ann", 0.0)
        sig_adj = cp.sigma * macro.get("vol_mult", 1.0)
        adj.append((cp.crop_id, mu_nom, sig_adj, cp))

    crop_ids = [cp.crop_id for cp in req.crop_params]

    for t in range(req.steps):
        seas = {cp.crop_id: seasonal_multiplier(t, req.steps, cp.seasonality_strength) for cp in req.crop_params}
        new_events = engine.step(crop_ids=crop_ids, macro=macro)
        DB.events[season_id].extend([{**e, "ts": t} for e in new_events])
        for (cid, mu_nom, sig_adj, cp) in adj:
            base_next = step_price(P[cid], mu_nom, sig_adj, req.dt, seas[cid],
                                   lam=cp.jump_lam * (1.3 if macro.get("recession", False) else 1.0),
                                   mu_j=cp.jump_mu, sig_j=cp.jump_sig, use_llm=True)
            shock = engine.log_return_shock(cid)
            P[cid] = float(base_next * math.exp(shock))
            DB.prices[season_id].append({"ts": t, "crop_id": cid, "price": float(P[cid])})

    return SimulateResponse(season_id=season_id, prices=DB.prices[season_id], events=DB.events[season_id], macro=macro)


@api.get("/season/{season_id}/prices")
def get_prices(season_id: str):
    if season_id not in DB.prices:
        raise HTTPException(404, "season not found")
    return {"season_id": season_id, "prices": DB.prices[season_id], "events": DB.events.get(season_id, [])}


@api.post("/season/{season_id}/advance")
def advance_season(season_id: str, req: AdvanceRequest):
    if season_id not in DB.prices:
        raise HTTPException(404, "season not found")
    params = DB.season_params.get(season_id)
    if not params:
        raise HTTPException(400, "season parameters unavailable")
    crop_params = [CropParams(**p) for p in params]
    crop_ids = [cp.crop_id for cp in crop_params]
    season_meta = DB.seasons.get(season_id, {})
    macro = season_meta.get("macro") or compute_macro_context(start="2024-01-01", end="2024-12-31")
    dt = float(season_meta.get("dt", 1.0 / 52.0))
    period = int(season_meta.get("period", max(52, len(DB.prices[season_id]) // max(1, len(crop_ids)))))

    engine = DB.event_engines.get(season_id)
    if engine is None:
        engine = EventEngine(copy.deepcopy(DEFAULT_EVENT_RULES))
        DB.event_engines[season_id] = engine

    latest_by_crop: Dict[str, dict] = {}
    current_ts = -1
    for row in DB.prices[season_id]:
        cid = row["crop_id"]
        ts = row["ts"]
        if cid not in latest_by_crop or ts > latest_by_crop[cid]["ts"]:
            latest_by_crop[cid] = {"ts": ts, "price": float(row["price"])}
        if ts > current_ts:
            current_ts = ts
    if current_ts < 0:
        raise HTTPException(400, "season has no price history")

    latest_prices = {cid: info["price"] for cid, info in latest_by_crop.items()}
    new_prices: List[dict] = []
    new_events: List[dict] = []

    seed = req.steps  # simple; not necessary
    if req.seed is not None:
        seed = req.seed
    np.random.seed(seed)

    div_hhi = float(req.diversification_hhi) if req.diversification_hhi is not None else 0.0
    penalty = max(0.0, min(1.0, div_hhi - 0.35))
    mu_penalty = 0.10 * penalty
    vol_multiplier = 1.0 + 0.35 * penalty

    for step in range(req.steps):
        t = current_ts + 1 + step
        seas = {cp.crop_id: seasonal_multiplier(t, period, cp.seasonality_strength) for cp in crop_params}
        triggered = engine.step(crop_ids=crop_ids, macro=macro)
        stamped_events = [{**e, "ts": t} for e in triggered]
        if stamped_events:
            DB.events.setdefault(season_id, []).extend(stamped_events)
            new_events.extend(stamped_events)
        for cp in crop_params:
            cid = cp.crop_id
            prev_price = latest_prices.get(cid, 100.0)
            mu_nom = cp.mu + macro.get("inflation_ann", 0.0)
            sig_adj = cp.sigma * macro.get("vol_mult", 1.0) * vol_multiplier
            mu_effective = mu_nom - mu_penalty
            base_next = step_price(prev_price, mu_effective, sig_adj, dt, seas[cid],
                                   lam=cp.jump_lam * (1.3 if macro.get("recession", False) else 1.0),
                                   mu_j=cp.jump_mu, sig_j=cp.jump_sig, use_llm=True)
            if penalty > 0.0:
                damp = max(0.6, 1.0 - 0.6 * penalty)
                base_next *= damp
            shock = engine.log_return_shock(cid)
            price = float(base_next * math.exp(shock))
            latest_prices[cid] = price
            rec = {"ts": t, "crop_id": cid, "price": price}
            DB.prices[season_id].append(rec)
            new_prices.append(rec)

    return {"season_id": season_id, "prices": new_prices, "events": new_events}


@api.post("/montecarlo", response_model=MonteCarloResponse)
def montecarlo(req: MonteCarloRequest):
    np.random.seed()
    idx = {cp.crop_id: i for i, cp in enumerate(req.crop_params)}
    P0 = np.array([req.prices_now[c] for c in idx], dtype=float)
    w = np.array([req.weights.get(c, 0.0) for c in idx], dtype=float)
    w = w / (w.sum() + EPS)
    params = [dict(mu=cp.mu, sigma=cp.sigma, lam=cp.jump_lam, mu_j=cp.jump_mu, sig_j=cp.jump_sig) for cp in req.crop_params]
    wealth = np.zeros(req.N)
    for i in range(req.N):
        P = P0.copy()
        for t in range(req.horizon_steps):
            seas = [seasonal_multiplier(t, req.horizon_steps, req.crop_params[j].seasonality_strength) for j in range(len(req.crop_params))]
            for j in range(len(P)):
                P[j] = step_price(P[j], mu=params[j]["mu"], sigma=params[j]["sigma"], dt=req.dt, seasonality_t=seas[j], lam=params[j]["lam"], mu_j=params[j]["mu_j"], sig_j=params[j]["sig_j"], use_llm=False)
        wealth[i] = float(np.dot(w, P / (P0 + EPS)))
    pcts = np.percentile(wealth, [5, 25, 50, 75, 95]).tolist()
    losses = 1.0 - wealth
    var5 = float(np.percentile(losses, 95))
    es5 = float(losses[losses >= var5].mean()) if np.any(losses >= var5) else float(losses.mean())
    return MonteCarloResponse(percentiles={"p5,p25,p50,p75,p95": pcts}, var_es={"VaR_5": var5, "ES_5": es5})


@api.post("/fit_risk", response_model=FitRiskResponse)
def fit_risk(req: FitRiskRequest):
    def U(w, g):
        return math.log(max(w, EPS)) if abs(g - 1.0) < 1e-9 else ((max(w, EPS) ** (1 - g) - 1) / (1 - g))
    gammas = np.linspace(0.0, 4.0, 81)
    betas = np.linspace(0.5, 10.0, 20)
    best = (-1e18, 1.0, 1.0)
    for g in gammas:
        for b in betas:
            ll = 0.0
            for c in req.choices:
                EU_A = sum(p * U(1 + x, g) for x, p in zip(c.outcomes_A, c.probs_A))
                EU_B = sum(p * U(1 + x, g) for x, p in zip(c.outcomes_B, c.probs_B))
                prA = 1.0 / (1.0 + math.exp(-b * (EU_A - EU_B)))
                prA = min(max(prA, 1e-6), 1 - 1e-6)
                ll += math.log(prA if c.picked == 'A' else (1 - prA))
            if ll > best[0]:
                best = (ll, float(g), float(b))
    return FitRiskResponse(player_id=req.player_id, gamma=best[1], beta=best[2])


@api.post("/report", response_model=ReportResponse)
def report(req: ReportRequest):
    if req.season_id not in DB.prices:
        raise HTTPException(404, "season not found")
    crop_ids = list({r["crop_id"] for r in DB.prices[req.season_id]})
    steps = max(r["ts"] for r in DB.prices[req.season_id]) + 1
    prices = {cid: np.zeros(steps) for cid in crop_ids}
    for r in DB.prices[req.season_id]:
        prices[r["crop_id"]][r["ts"]] = r["price"]
    rets = {cid: (prices[cid][1:] - prices[cid][:-1]) / (prices[cid][:-1] + EPS) for cid in crop_ids}
    port_rets = np.mean(np.vstack([rets[cid] for cid in crop_ids]), axis=0)
    def perf_summary(returns: np.ndarray, rf: float = 0.0) -> dict:
        r = returns
        mean = float(r.mean()); vol = float(r.std(ddof=1) + EPS)
        sharpe = float((mean - rf) / vol)
        downside = r[r < 0]; sortino = float((mean - rf) / (downside.std(ddof=1) + EPS)) if downside.size else float("inf")
        cum = np.cumprod(1 + r); peak = np.maximum.accumulate(cum); dd = 1 - cum / (peak + EPS)
        mdd = float(dd.max()); wealth = float(cum[-1])
        return dict(mean=mean, vol=vol, sharpe=sharpe, sortino=sortino, wealth=wealth, mdd=mdd)
    season_meta = DB.seasons.get(req.season_id, {})
    rf_annual = float(season_meta.get("rf", req.rf_rate)); rf_weekly = rf_annual / 52.0
    metrics = perf_summary(port_rets, rf=rf_weekly)
    weights = np.ones(len(crop_ids)) / len(crop_ids)
    HHI = float(np.sum(weights ** 2)); N_eff = float(1.0 / (HHI + EPS))
    M = np.vstack([rets[cid] for cid in crop_ids]); C = np.corrcoef(M) if M.shape[1] > 1 else np.eye(len(crop_ids))
    mkt = port_rets; X = np.vstack([np.ones_like(mkt), mkt]).T
    coef = np.linalg.lstsq(X, port_rets, rcond=None)[0]; alpha, beta = float(coef[0]), float(coef[1])
    tips = []
    if metrics["mdd"] > 0.2: tips.append("Your money curve dipped over 20%. Mix more low-correlation crops to smooth swings.")
    if HHI > 0.4: tips.append("Most value was concentrated. Try keeping any single crop ≤ 40% and HHI < 0.40.")
    if metrics["sharpe"] < 0.2: tips.append("Returns were small relative to risk. Consider harvesting earlier during storms.")
    if not tips: tips.append("Strong balance between growth and safety. Keep experimenting with diversification!")
    prices_now = {cid: float(prices[cid][-1]) for cid in crop_ids}
    cps = [CropParams(crop_id=cid, mu=0.06, sigma=float(DB.crops.get(cid, {}).get("base_sigma", 0.2)),
                      seasonality_strength=float(DB.crops.get(cid, {}).get("seas", 0.2)),
                      jump_lam=float(DB.crops.get(cid, {}).get("jump_lam", 0.02)),
                      jump_mu=float(DB.crops.get(cid, {}).get("j_mu", 0.0)),
                      jump_sig=float(DB.crops.get(cid, {}).get("j_sig", 0.05))) for cid in crop_ids]
    mc = montecarlo(MonteCarloRequest(prices_now=prices_now, weights={cid: float(w) for cid, w in zip(crop_ids, weights)},
                                      crop_params=cps, horizon_steps=6, N=1000))
    macro = dict(inflation_yoY=season_meta.get("inflation", None), rf_annual=rf_annual, recession=season_meta.get("recession", False), asof=season_meta.get("asof", None))
    counterfactual = dict(horizon_steps=6, wealth_percentiles=mc.percentiles, risk=mc.var_es, note="If you waited a bit, median outcome improves but worst-case risk widens.")
    return ReportResponse(season_id=req.season_id, metrics=metrics,
                          diversification=dict(HHI=HHI, N_eff=N_eff, corr=C.tolist(), crops=crop_ids),
                          market_exposure=dict(alpha=alpha, beta=beta), behavior=dict(), tips=tips,
                          counterfactual=counterfactual, macro=macro)


@api.post("/demo_seed")
def demo_seed(season_id: str = "S1"):
    cps = [CropParams(crop_id=c["id"], mu=c["base_mu"], sigma=c["base_sigma"],
                      seasonality_strength=c["seas"], mean_revert_kappa=c["kappa"],
                      jump_lam=c["jump_lam"], jump_mu=c["j_mu"], jump_sig=c["j_sig"]) for c in DEFAULT_CROPS]
    res = simulate(SimulateRequest(season_id=season_id, seed=123, steps=52, crop_params=cps, start_prices={c["id"]: 100.0 for c in DEFAULT_CROPS}))
    return {"ok": True, "season_id": season_id, "n_prices": len(res.prices), "macro": res.macro}


# Run with uvicorn main:api --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:api", host="0.0.0.0", port=8000, reload=True)
