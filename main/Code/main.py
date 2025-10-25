from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional
import threading
import time
import math
import random
from datetime import datetime
from collections import defaultdict

app = FastAPI(title="STALK Market API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PlantConfig(BaseModel):
    id: str
    name: str
    base_price: float
    volatility: float
    drift: float
    emoji: str

class Price(BaseModel):
    plant_id: str
    price: float
    change_percent: float

class Holding(BaseModel):
    plant_id: str
    quantity: int
    avg_buy_price: float
    total_invested: float

class Portfolio(BaseModel):
    cash: float
    total_value: float
    holdings: List[Holding]

class Transaction(BaseModel):
    id: int
    timestamp: str
    action: str
    plant_id: str
    plant_name: str
    quantity: int
    price: float
    total: float

class BuyRequest(BaseModel):
    plant_id: str
    quantity: int

class SellRequest(BaseModel):
    plant_id: str
    quantity: int

class PricePoint(BaseModel):
    timestamp: str
    price: float

PLANT_CONFIGS = [
    PlantConfig(
        id="tomato",
        name="Tomato Tech",
        base_price=10.0,
        volatility=0.15,
        drift=0.08,
        emoji="🍅"
    ),
    PlantConfig(
        id="sunflower",
        name="Sunflower Solar",
        base_price=25.0,
        volatility=0.25,
        drift=0.12,
        emoji="🌻"
    ),
    PlantConfig(
        id="cactus",
        name="Cactus Crypto",
        base_price=50.0,
        volatility=0.40,
        drift=0.18,
        emoji="🌵"
    ),
    PlantConfig(
        id="fern",
        name="Fern Finance",
        base_price=5.0,
        volatility=0.08,
        drift=0.05,
        emoji="🌿"
    ),
]

class MemoryStorage:
    def __init__(self):
        self.prices: Dict[str, float] = {}
        self.initial_prices: Dict[str, float] = {}
        self.price_history: Dict[str, List[Dict]] = defaultdict(list)
        self.cash = 10000.0
        self.holdings: Dict[str, Holding] = {}
        self.transactions: List[Transaction] = []
        self.transaction_counter = 0
        
        for config in PLANT_CONFIGS:
            self.prices[config.id] = config.base_price
            self.initial_prices[config.id] = config.base_price
            self.price_history[config.id] = [{
                "timestamp": datetime.now().isoformat(),
                "price": config.base_price
            }]
    
    def add_price_to_history(self, plant_id: str, price: float):
        self.price_history[plant_id].append({
            "timestamp": datetime.now().isoformat(),
            "price": price
        })
        if len(self.price_history[plant_id]) > 1000:
            self.price_history[plant_id] = self.price_history[plant_id][-1000:]

storage = MemoryStorage()

def box_muller_transform():
    u1 = random.random()
    u2 = random.random()
    z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return z0

def update_prices():
    dt = 3.0 / (365.0 * 24.0 * 3600.0)
    
    for config in PLANT_CONFIGS:
        current_price = storage.prices[config.id]
        
        z = box_muller_transform()
        
        drift_term = (config.drift - 0.5 * config.volatility ** 2) * dt
        diffusion_term = config.volatility * math.sqrt(dt) * z
        
        new_price = current_price * math.exp(drift_term + diffusion_term)
        
        min_price = config.base_price * 0.1
        new_price = max(new_price, min_price)
        
        storage.prices[config.id] = new_price
        storage.add_price_to_history(config.id, new_price)

def price_update_loop():
    while True:
        time.sleep(3)
        update_prices()

price_thread = threading.Thread(target=price_update_loop, daemon=True)
price_thread.start()

@app.get("/api/market", response_model=List[Price])
def get_market():
    prices = []
    for config in PLANT_CONFIGS:
        current_price = storage.prices[config.id]
        initial_price = storage.initial_prices[config.id]
        change_percent = ((current_price - initial_price) / initial_price) * 100
        
        prices.append(Price(
            plant_id=config.id,
            price=current_price,
            change_percent=change_percent
        ))
    
    return prices

@app.get("/api/portfolio", response_model=Portfolio)
def get_portfolio():
    holdings = list(storage.holdings.values())
    
    total_holdings_value = sum(
        h.quantity * storage.prices[h.plant_id]
        for h in holdings
    )
    
    total_value = storage.cash + total_holdings_value
    
    return Portfolio(
        cash=storage.cash,
        total_value=total_value,
        holdings=holdings
    )

@app.get("/api/price-history/{plant_id}", response_model=List[PricePoint])
def get_price_history(plant_id: str):
    if plant_id not in storage.price_history:
        raise HTTPException(status_code=404, detail="Plant not found")
    
    history = storage.price_history[plant_id]
    return [
        PricePoint(timestamp=h["timestamp"], price=h["price"])
        for h in history[-100:]
    ]

@app.get("/api/transactions", response_model=List[Transaction])
def get_transactions():
    return list(reversed(storage.transactions[-50:]))

@app.post("/api/buy")
def buy_plant(request: BuyRequest):
    if request.plant_id not in storage.prices:
        raise HTTPException(status_code=404, detail="Plant not found")
    
    if request.quantity <= 0:
        raise HTTPException(status_code=400, detail="Quantity must be positive")
    
    current_price = storage.prices[request.plant_id]
    total_cost = current_price * request.quantity
    
    if total_cost > storage.cash:
        raise HTTPException(status_code=400, detail="Insufficient funds")
    
    storage.cash -= total_cost
    
    if request.plant_id in storage.holdings:
        holding = storage.holdings[request.plant_id]
        new_quantity = holding.quantity + request.quantity
        new_total_invested = holding.total_invested + total_cost
        new_avg_price = new_total_invested / new_quantity
        
        storage.holdings[request.plant_id] = Holding(
            plant_id=request.plant_id,
            quantity=new_quantity,
            avg_buy_price=new_avg_price,
            total_invested=new_total_invested
        )
    else:
        storage.holdings[request.plant_id] = Holding(
            plant_id=request.plant_id,
            quantity=request.quantity,
            avg_buy_price=current_price,
            total_invested=total_cost
        )
    
    plant_name = next(c.name for c in PLANT_CONFIGS if c.id == request.plant_id)
    
    transaction = Transaction(
        id=storage.transaction_counter,
        timestamp=datetime.now().isoformat(),
        action="buy",
        plant_id=request.plant_id,
        plant_name=plant_name,
        quantity=request.quantity,
        price=current_price,
        total=total_cost
    )
    storage.transactions.append(transaction)
    storage.transaction_counter += 1
    
    return {"success": True, "transaction": transaction}

@app.post("/api/sell")
def sell_plant(request: SellRequest):
    if request.plant_id not in storage.holdings:
        raise HTTPException(status_code=400, detail="No holdings for this plant")
    
    holding = storage.holdings[request.plant_id]
    
    if request.quantity <= 0:
        raise HTTPException(status_code=400, detail="Quantity must be positive")
    
    if request.quantity > holding.quantity:
        raise HTTPException(status_code=400, detail="Insufficient holdings")
    
    current_price = storage.prices[request.plant_id]
    total_revenue = current_price * request.quantity
    
    storage.cash += total_revenue
    
    new_quantity = holding.quantity - request.quantity
    
    if new_quantity == 0:
        del storage.holdings[request.plant_id]
    else:
        proportion_sold = request.quantity / holding.quantity
        new_total_invested = holding.total_invested * (1 - proportion_sold)
        
        storage.holdings[request.plant_id] = Holding(
            plant_id=request.plant_id,
            quantity=new_quantity,
            avg_buy_price=holding.avg_buy_price,
            total_invested=new_total_invested
        )
    
    plant_name = next(c.name for c in PLANT_CONFIGS if c.id == request.plant_id)
    
    transaction = Transaction(
        id=storage.transaction_counter,
        timestamp=datetime.now().isoformat(),
        action="sell",
        plant_id=request.plant_id,
        plant_name=plant_name,
        quantity=request.quantity,
        price=current_price,
        total=total_revenue
    )
    storage.transactions.append(transaction)
    storage.transaction_counter += 1
    
    return {"success": True, "transaction": transaction}

@app.get("/api/plant-configs", response_model=List[PlantConfig])
def get_plant_configs():
    return PLANT_CONFIGS

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
