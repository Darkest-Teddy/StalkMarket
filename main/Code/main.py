# from google import genai

"""
FarmFinance — FastAPI backend (main.py)
Includes: FRED macro integration, event engine, simulation, forecast, monte carlo,
risk fitting, season prices endpoint, and report.
"""
# from __future__ import annotations
from stock_llm import StockLLM 
import os, math, time, random, copy
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    import pandas as pd
except Exception:
    pd = None

# Optional ETS
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_HW = True
except Exception:
    HAS_HW = False

# FRED integration
try:
    from fredapi import Fred
    HAS_FRED = True
except Exception:
    HAS_FRED = False

EPS = 1e-12

def seasonal_multiplier(t: int, T: int, amplitude: float, phase: float = 0.0) -> float:
    import math as _m
    return amplitude * _m.sin(2 * _m.pi * (t / max(T, 1)) + phase)


model = StockLLM()
model.load_state_dict(torch.load("model.pt"))
model.eval()


def step_price(P: float, mu: float, sigma: float, dt: float, seasonality_t: float,
               lam: float = 0.0, mu_j: float = 0.0, sig_j: float = 0.0) -> float:
    # --- Original stochastic component ---
    Z = np.random.normal()
    log_ret = (mu - 0.5 * sigma**2) * dt + sigma * (dt**0.5) * Z
    log_ret += seasonality_t * dt
    if lam > 0.0:
        jump_count = np.random.poisson(lam * dt)
        if jump_count > 0:
            log_ret += np.random.normal(mu_j, sig_j) * jump_count
    P_stochastic = P * math.exp(log_ret)

    # --- Behavioral adjustment ---
    # Damp volatility if the user is risky
    compute_risk_score = lambda: random.uniform(0.0, 1.0)  # Placeholder for actual risk score computation
    risk_score = compute_risk_score()  # ∈ [0.0, 1.0]
    alpha = 0.9  # weight for stochastic vs AI prediction
    ai_input = torch.tensor([[P, mu, sigma, dt, seasonality_t,
                                lam, mu_j, sig_j, risk_score]], dtype=torch.float32)
    vol_factor = 1.0 - 0.5 * risk_score  # reduces fluctuations when risk_score high
    P_adjusted = P + (P_stochastic - P) * vol_factor
    ai_pred = None
    try:
        with torch.no_grad():
            ai_output = model(ai_input).numpy().flatten()
    except IndexError as e:
        print("AI input shape:", ai_input.shape)
        print("AI input max value:", ai_input.max())
        raise e

    ai_pred = ( 0.98 + float(ai_output[0])) * P
    # --- AI prediction ---
    if ai_pred is not None:
        # Combine AI prediction and stochastic+behavioral
        P_final =  ai_pred * (1 - alpha) + P_adjusted * alpha
        # print(f"AI Prediction: {ai_pred}, Stochastic Adjusted: {P_adjusted}")
    else:
        P_final = P_adjusted
        # print(f"Stochastic Adjusted Price: {P_adjusted}")

    return max(P_final, EPS)

# ---------- FRED helpers ----------
def _fred():
    if not HAS_FRED:
        raise RuntimeError("fredapi not installed")
    key = os.getenv("FRED_API_KEY", "")
    if not key:
        raise RuntimeError("FRED_API_KEY not set")
    return Fred(api_key=key)

def weekly_series_from_fred(series_id: str, start: str = "2024-01-01", end: str = "2024-12-31"):
    fred = _fred()
    s = fred.get_series(series_id, observation_start="2023-01-01", observation_end=end)
    if pd is None:
        raise RuntimeError("pandas required for FRED integration")
    df = s.to_frame(name=series_id)
    df.index = pd.to_datetime(df.index)
    mask = (df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))
    return df.loc[mask]

def compute_macro_context(start: str = "2024-01-01", end: str = "2024-12-31") -> dict:
    fallback = {"inflation_ann": 0.02, "rf_rate_ann": 0.02, "recession": False,
                "term_spread": 1.0, "vol_mult": 1.0, "asof": None}
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

        return dict(
            inflation_ann=float(infl_ann),
            rf_rate_ann=float(rf_ann),
            recession=recession,
            term_spread=float(term_spread),
            vol_mult=vol_mult,
            asof=asof,
        )
    except Exception as exc:
        print("[WARN] compute_macro_context fallback:", exc)
        return fallback
    
def compute_risk_score(portfolio, market):
    weights = np.array(list(portfolio.values()))
    hhi = np.sum(weights**2)

    vol = np.std([market[c]['volatility'] for c in portfolio])

    leverage = portfolio.get('leverage', 1.0)

    macro_factor = 1.2 if market.get('recession', False) else 1.0

    raw_score = 0.5*hhi + 0.3*vol + 0.2*leverage
    risk_score = np.clip(raw_score * macro_factor, 0.0, 1.0)
    return risk_score


def behavioral_adjustment(risk_score: float, macro: dict) -> dict:
    """
    Adjust macroeconomic variables based on user risk tolerance.
    risk_score ∈ [-1.0, +1.0]:
      -1.0 = highly risk-averse (lower returns, lower vol)
      +1.0 = risk-seeking (higher returns, higher vol)
    """
    adj = macro.copy()
    adj["inflation_ann"] *= (1 + 0.2 * risk_score)  # higher inflation for risk seekers
    adj["vol_mult"] *= (1 + 0.3 * risk_score)       # amplify volatility
    adj["rf_rate_ann"] *= (1 - 0.1 * risk_score)    # risk-seekers rely less on risk-free returns
    return adj

# ---------- Storage ----------
class Storage:
    def __init__(self):
        self.players: Dict[str, dict] = {}
        self.seasons: Dict[str, dict] = {}
        self.crops: Dict[str, dict] = {}
        self.prices: Dict[str, List[dict]] = {}
        self.events: Dict[str, List[dict]] = {}
        self.event_engines: Dict[str, EventEngine] = {}
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

# ---------- Events ----------
@dataclass
class EventRule:
    id: str
    level: str                  # 'global' | 'class' | 'crop'
    target: Optional[str]       # None | class | crop id
    name: str
    base_prob: float            # per-tick probability
    impact_mu: float            # mean log-return shock
    impact_sigma: float         # std of log-return shock
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
            if macro.get("recession", False) and r.id in {"bear_storm","bug_attack","supply_shock"}:
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
                events_out.append({
                    "type": r.id,
                    "name": r.name,
                    "affected": affected,
                    "note": "event triggered",
                    "duration": r.duration_steps,
                    "level": r.level,
                    "target": r.target,
                })
        # decay
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
    EventRule("bull_sun","global",None,"Sunny Bull Season",0.03,+0.01,0.005,2),
    EventRule("bear_storm","global",None,"Stormy Bear Front",0.03,-0.012,0.008,2),
    EventRule("vol_spike","global",None,"Volatility Spike",0.02,0.0,0.015,1),
    EventRule("bug_attack","class","growth","Pest Outbreak (Growth)",0.04,-0.015,0.010,1),
    EventRule("bountiful_rain","class","bond_like","Bountiful Rains (Bond-like)",0.025,+0.006,0.003,1),
    EventRule("supply_shock","crop","corn","Supply Chain Shock (Corn)",0.02,-0.02,0.012,1),
]

# ---------- API Schemas ----------
class CropParams(BaseModel):
    crop_id: str
    mu: float
    sigma: float
    seasonality_strength: float = Field(0.2, ge=0.0, le=1.0)
    mean_revert_kappa: float = 0.0
    jump_lam: float = 0.0
    jump_mu: float = 0.0
    jump_sig: float = 0.0

class SimulateRequest(BaseModel):
    season_id: str
    seed: int = 42
    steps: int = 52
    dt: float = 1.0/52.0
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
    dt: float = 1.0/52.0
    N: int = 2000

class MonteCarloResponse(BaseModel):
    percentiles: Dict[str, List[float]]
    var_es: Dict[str, float]

class AdvanceRequest(BaseModel):
    steps: int = Field(12, ge=1, le=260)
    seed: Optional[int] = None
    diversification_hhi: Optional[float] = Field(None, ge=0.0, le=1.0)

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

from fastapi.middleware.cors import CORSMiddleware
api = FastAPI(title="FarmFinance API", version="0.3.0")

# CORS (dev)
api.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@api.get("/health")
def health(): return {"ok": True, "time": time.time()}

@api.get("/macro")
def macro_snapshot():
    m = compute_macro_context(start="2024-01-01", end="2024-12-31")
    return {"ok": True, "macro": m}

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
    dt = float(season_meta.get("dt", 1.0/52.0))
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

    seed = req.seed if req.seed is not None else random.randint(0, 1_000_000_000)
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
                                   mu_j=cp.jump_mu, sig_j=cp.jump_sig)
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
        inflation=macro.get("inflation_ann",0.02),
        rf=macro.get("rf_rate_ann",0.02),
        recession=macro.get("recession",False),
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
        mu_nom = cp.mu + macro.get("inflation_ann",0.0)
        sig_adj = cp.sigma * macro.get("vol_mult",1.0)
        adj.append((cp.crop_id, mu_nom, sig_adj, cp))

    crop_ids = [cp.crop_id for cp in req.crop_params]

    for t in range(req.steps):
        seas = {cp.crop_id: seasonal_multiplier(t, req.steps, cp.seasonality_strength) for cp in req.crop_params}
        # events
        new_events = engine.step(crop_ids=crop_ids, macro=macro)
        DB.events[season_id].extend([{**e, "ts": t} for e in new_events])
        for (cid, mu_nom, sig_adj, cp) in adj:
            base_next = step_price(P[cid], mu_nom, sig_adj, req.dt, seas[cid],
                                   lam=cp.jump_lam * (1.3 if macro.get("recession",False) else 1.0),
                                   mu_j=cp.jump_mu, sig_j=cp.jump_sig)
            shock = engine.log_return_shock(cid)
            P[cid] = float(base_next * math.exp(shock))
            DB.prices[season_id].append({"ts": t, "crop_id": cid, "price": float(P[cid])})
    return SimulateResponse(season_id=season_id, prices=DB.prices[season_id], events=DB.events[season_id], macro=macro)

@api.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    if req.season_id not in DB.prices:
        raise HTTPException(404, "season not found")
    crop_ids = req.crop_ids or list({r["crop_id"] for r in DB.prices[req.season_id]})
    df = pd.DataFrame(DB.prices[req.season_id]) if pd is not None else None
    forecasts: Dict[str, dict] = {}
    for cid in crop_ids:
        if df is not None:
            series = df[df.crop_id == cid].sort_values("ts")["price"].values
        else:
            series = [r["price"] for r in DB.prices[req.season_id] if r["crop_id"] == cid]
        series = np.asarray(series, dtype=float)
        if len(series) < 8:
            mu = float(np.mean(series)); std = float(np.std(series) + 1.0)
            mean_path = [mu] * req.horizon
            lo = (np.array(mean_path) - 1.28*std).clip(min=EPS).tolist()
            hi = (np.array(mean_path) + 1.28*std).tolist()
            forecasts[cid] = {"mean": mean_path, "p10": lo, "p90": hi}; continue
        if HAS_HW and pd is not None:
            try:
                s = pd.Series(series)
                slen = max(6, min(52, len(series)//2))
                model = ExponentialSmoothing(s, trend="add", seasonal="add", seasonal_periods=slen).fit(optimized=True)
                pred = model.forecast(req.horizon)
                resid = s - model.fittedvalues
                sd = np.std(resid.values) if len(resid) > 1 else np.std(s.values)
                mean_path = pred.values.clip(min=EPS).tolist()
                lo = (pred.values - 1.28*sd).clip(min=EPS).tolist()
                hi = (pred.values + 1.28*sd).clip(min=EPS).tolist()
                forecasts[cid] = {"mean": mean_path, "p10": lo, "p90": hi}; continue
            except Exception:
                pass
        k = max(6, min(52, len(series)//2))
        hist = np.array(series[-k:])
        mean_path = hist.mean().repeat(req.horizon).tolist()
        sd = float(hist.std() + 1.0)
        lo = (np.array(mean_path) - 1.28*sd).clip(min=EPS).tolist()
        hi = (np.array(mean_path) + 1.28*sd).tolist()
        forecasts[cid] = {"mean": mean_path, "p10": lo, "p90": hi}
    return ForecastResponse(season_id=req.season_id, horizon=req.horizon, forecasts=forecasts)

@api.post("/montecarlo", response_model=MonteCarloResponse)
def montecarlo(req: MonteCarloRequest):
    np.random.seed()
    idx = {cp.crop_id: i for i, cp in enumerate(req.crop_params)}
    P0 = np.array([req.prices_now[c] for c in idx], dtype=float)
    w = np.array([req.weights.get(c,0.0) for c in idx], dtype=float); w = w/(w.sum()+EPS)
    params = [dict(mu=cp.mu, sigma=cp.sigma, lam=cp.jump_lam, mu_j=cp.jump_mu, sig_j=cp.jump_sig) for cp in req.crop_params]
    wealth = np.zeros(req.N)
    for i in range(req.N):
        P = P0.copy()
        for t in range(req.horizon_steps):
            seas = [seasonal_multiplier(t, req.horizon_steps, req.crop_params[j].seasonality_strength) for j in range(len(req.crop_params))]
            for j in range(len(P)):
                P[j] = step_price(P[j], seasonality_t=seas[j], dt=req.dt, **params[j])
        wealth[i] = float(np.dot(w, P/(P0+EPS)))
    pcts = np.percentile(wealth, [5,25,50,75,95]).tolist()
    losses = 1.0 - wealth
    var5 = float(np.percentile(losses, 95))
    es5 = float(losses[losses >= var5].mean()) if np.any(losses >= var5) else float(losses.mean())
    return MonteCarloResponse(percentiles={"p5,p25,p50,p75,p95": pcts}, var_es={"VaR_5": var5, "ES_5": es5})

@api.post("/fit_risk", response_model=FitRiskResponse)
def fit_risk(req: FitRiskRequest):
    def U(w, g):
        return math.log(max(w, EPS)) if abs(g-1.0) < 1e-9 else ((max(w,EPS)**(1-g)-1)/(1-g))
    gammas = np.linspace(0.0, 4.0, 81)
    betas  = np.linspace(0.5, 10.0, 20)
    best = (-1e18, 1.0, 1.0)
    for g in gammas:
        for b in betas:
            ll = 0.0
            for c in req.choices:
                EU_A = sum(p*U(1+x,g) for x,p in zip(c.outcomes_A, c.probs_A))
                EU_B = sum(p*U(1+x,g) for x,p in zip(c.outcomes_B, c.probs_B))
                prA = 1.0/(1.0+math.exp(-b*(EU_A-EU_B)))
                prA = min(max(prA,1e-6),1-1e-6)
                ll += math.log(prA if c.picked=='A' else (1-prA))
            if ll > best[0]: best = (ll, float(g), float(b))
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
    rets = {cid: (prices[cid][1:] - prices[cid][:-1])/(prices[cid][:-1]+EPS) for cid in crop_ids}
    port_rets = np.mean(np.vstack([rets[cid] for cid in crop_ids]), axis=0)
    def perf_summary(returns: np.ndarray, rf: float = 0.0) -> dict:
        r = returns
        mean = float(r.mean()); vol = float(r.std(ddof=1)+EPS)
        sharpe = float((mean - rf)/vol)
        downside = r[r<0]; sortino = float((mean-rf)/(downside.std(ddof=1)+EPS)) if downside.size else float("inf")
        cum = np.cumprod(1+r); peak = np.maximum.accumulate(cum); dd = 1 - cum/(peak+EPS)
        mdd = float(dd.max()); wealth = float(cum[-1])
        return dict(mean=mean, vol=vol, sharpe=sharpe, sortino=sortino, wealth=wealth, mdd=mdd)
    season_meta = DB.seasons.get(req.season_id, {})
    rf_annual = float(season_meta.get("rf", req.rf_rate)); rf_weekly = rf_annual/52.0
    metrics = perf_summary(port_rets, rf=rf_weekly)
    weights = np.ones(len(crop_ids))/len(crop_ids)
    HHI = float(np.sum(weights**2)); N_eff = float(1.0/(HHI+EPS))
    M = np.vstack([rets[cid] for cid in crop_ids]); C = np.corrcoef(M) if M.shape[1]>1 else np.eye(len(crop_ids))
    mkt = port_rets; X = np.vstack([np.ones_like(mkt), mkt]).T
    coef = np.linalg.lstsq(X, port_rets, rcond=None)[0]; alpha, beta = float(coef[0]), float(coef[1])
    tips = []
    if metrics["mdd"] > 0.2: tips.append("Your money curve dipped over 20%. Mix more low‑correlation crops to smooth swings.")
    if HHI > 0.4: tips.append("Most value was concentrated. Try keeping any single crop ≤ 40% and HHI < 0.40.")
    if metrics["sharpe"] < 0.2: tips.append("Returns were small relative to risk. Consider harvesting earlier during storms.")
    if not tips: tips.append("Strong balance between growth and safety. Keep experimenting with diversification!")
    # MC counterfactual from last prices
    prices_now = {cid: float(prices[cid][-1]) for cid in crop_ids}
    cps = [CropParams(crop_id=cid, mu=0.06, sigma=float(DB.crops.get(cid,{}).get("base_sigma",0.2)),
                      seasonality_strength=float(DB.crops.get(cid,{}).get("seas",0.2)),
                      jump_lam=float(DB.crops.get(cid,{}).get("jump_lam",0.02)),
                      jump_mu=float(DB.crops.get(cid,{}).get("j_mu",0.0)),
                      jump_sig=float(DB.crops.get(cid,{}).get("j_sig",0.05))) for cid in crop_ids]
    mc = montecarlo(MonteCarloRequest(prices_now=prices_now, weights={cid: float(w) for cid,w in zip(crop_ids,weights)},
                                      crop_params=cps, horizon_steps=6, N=1000))
    macro = dict(inflation_yoY=season_meta.get("inflation",None), rf_annual=rf_annual,
                 recession=season_meta.get("recession",False), asof=season_meta.get("asof",None))
    counterfactual = dict(horizon_steps=6, wealth_percentiles=mc.percentiles, risk=mc.var_es,
                          note="If you waited a bit, median outcome improves but worst‑case risk widens.")
    return ReportResponse(season_id=req.season_id, metrics=metrics,
                          diversification=dict(HHI=HHI, N_eff=N_eff, corr=C.tolist(), crops=crop_ids),
                          market_exposure=dict(alpha=alpha, beta=beta), behavior=dict(), tips=tips,
                          counterfactual=counterfactual, macro=macro)

@api.post("/demo_seed")
def demo_seed(season_id: str = "S1"):
    cps = [CropParams(crop_id=c["id"], mu=c["base_mu"], sigma=c["base_sigma"],
                      seasonality_strength=c["seas"], mean_revert_kappa=c["kappa"],
                      jump_lam=c["jump_lam"], jump_mu=c["j_mu"], jump_sig=c["j_sig"]) for c in DEFAULT_CROPS]
    res = simulate(SimulateRequest(season_id=season_id, seed=123, steps=52, crop_params=cps,
                                   start_prices={c["id"]: 100.0 for c in DEFAULT_CROPS}))
    return {"ok": True, "season_id": season_id, "n_prices": len(res.prices), "macro": res.macro}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:api", host="0.0.0.0", port=8000, reload=True)
