/* STALK Market Frontend – connects to FarmFinance FastAPI
 * Gameplay loop: plan -> simulate (season) -> trade -> Monte Carlo peek -> report
 * State is client-side (localStorage) for hackathon simplicity.
 */

const API = () => window.API_BASE || 'http://localhost:8000';

// ---------- Global State ----------
const state = {
  seasonId: null,
  prices: {},     // {cropId: [prices...]}
  crops: [],      // cropIds
  events: [],
  macro: {},
  cash: 10000,
  holdings: {},   // {cropId: qty}
  shorts: {},     // {cropId: qty}
  txns: [],
  charts: {},
  fullHistory: {},   // complete price history per crop
  currentStep: 0,
  maxStep: 0,
  tickIntervalMs: 10000,
  timerId: null,
  timerRunning: false,
  costBasis: {},
  shortBasis: {},
  extending: false,
  timelineComplete: false,
  eventPopups: [],
  educationalMode: false,
  starting: false,
  gardenSprites: {},
};

const CROPS_META = {
  wheat: {
    name: 'Golden Wheat',
    emoji: '🌾',
    tagline: 'Steady bond-like staple',
  },
  corn: {
    name: 'Sunrise Corn',
    emoji: '🌽',
    tagline: 'Blue-chip harvest with supply swings',
  },
  berries: {
    name: 'Berry Patch',
    emoji: '🫐',
    tagline: 'High-growth seasonal favorite',
  },
  truffle: {
    name: 'Truffle Grove',
    emoji: '🍄',
    tagline: 'Alt delicacy with rare windfalls',
  },
};

const GARDEN_SPRITE_OVERRIDES = {
  wheat: 'wheat',
  corn: 'pumpkin',
  berries: 'tomato',
  truffle: 'parsnip',
};
const GARDEN_SPRITE_POOL = ['wheat','pumpkin','tomato','potato','carrot','radish','beet','jalapeno','califlower'];
const MAX_GARDEN_COLUMNS = 8;
const BASE_GARDEN_ROWS = 4;
const SPRITE_BASE = '../Objects';
const TILE_BASE = '../Tiles';

// ---------- Utils ----------
const fmt = (n) => '$' + (n || 0).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
const fmtSigned = (n) => {
  const sign = n >= 0 ? '+' : '-';
  const mag = Math.abs(n).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
  return `${sign}$${mag}`;
};
const pct = (x) => ((x >= 0 ? '+' : '') + (100 * x).toFixed(2) + '%');

function toast(msg) {
  const t = document.getElementById('toast');
  document.getElementById('toast-message').textContent = msg;
  t.classList.remove('hidden');
  setTimeout(() => t.classList.add('hidden'), 2600);
}

function sum(arr){ return arr.reduce((a,b)=>a+b,0); }

// Estimate mu/sigma from history (simple, weekly)
function estimateParams(prices){
  const rets = [];
  for(let i=1;i<prices.length;i++){
    rets.push((prices[i]-prices[i-1])/(prices[i-1]||1));
  }
  const m = rets.length ? rets.reduce((a,b)=>a+b,0)/rets.length : 0.001;
  const v = rets.length ? Math.sqrt(rets.map(x=> (x-m)**2).reduce((a,b)=>a+b,0)/Math.max(1,rets.length-1)) : 0.02;
  return {mu: m, sigma: v, seasonality_strength: 0.2};
}

// Convert portfolio to weights (by market value)
function portfolioWeights() {
  const latest = latestPrices();
  let total = 0;
  const w = {};
  for (const cid of state.crops) {
    const qtyLong = state.holdings[cid] || 0;
    const qtyShort = state.shorts[cid] || 0;
    const p = latest[cid] || 0;
    const exposure = qtyLong * p - qtyShort * p;
    total += Math.abs(exposure);
    w[cid] = exposure;
  }
  if(total > 0){
    for (const cid of state.crops) {
      w[cid] = w[cid] / total;
    }
  } else {
    for (const cid of state.crops) {
      w[cid] = 0;
    }
  }
  return w;
}

function latestPrices(){
  const last = {};
  for(const cid of state.crops) {
    const arr = state.prices[cid] || [];
    last[cid] = arr.length ? arr[arr.length-1] : 0;
  }
  return last;
}

function titleize(id){
  if(!id) return '';
  return id.replace(/[_-]/g, ' ').replace(/\b\w/g, (c)=>c.toUpperCase());
}

function cropMeta(cid){
  const meta = CROPS_META[cid] || {};
  return {
    name: meta.name || titleize(cid),
    emoji: meta.emoji || '🪴',
    tagline: meta.tagline || 'Versatile seasonal grower',
  };
}

function computeDiversificationHHI(){
  const prices = latestPrices();
  const weights = {};
  let total = 0;
  for(const cid of state.crops){
    const netQty = (state.holdings[cid] || 0) - (state.shorts[cid] || 0);
    const exposure = Math.abs(netQty * (prices[cid] || 0));
    if(exposure > 0){
      weights[cid] = exposure;
      total += exposure;
    }
  }
  if(total <= 0){
    return 0.0;
  }
  let hhi = 0;
  for(const cid of Object.keys(weights)){
    const share = weights[cid] / total;
    hhi += share * share;
  }
  return Math.min(1, Math.max(0, hhi));
}

function applyEducationalMode(enabled){
  state.educationalMode = enabled;
  localStorage.setItem('STALK_EDU_MODE', enabled ? '1' : '0');
  const btn = document.getElementById('edu-mode-button');
  if(btn){
    if(enabled){
      btn.textContent = 'Educational Mode: ON';
      btn.classList.remove('bg-gray-200','text-gray-800');
      btn.classList.add('bg-emerald-100','text-emerald-700');
    }else{
      btn.textContent = 'Educational Mode: OFF';
      btn.classList.remove('bg-emerald-100','text-emerald-700');
      btn.classList.add('bg-gray-200','text-gray-800');
    }
  }
}

function setupIntroOverlay(){
  const overlay = document.getElementById('intro-overlay');
  const startBtn = document.getElementById('start-button');
  const eduBtn = document.getElementById('edu-mode-button');
  const saved = localStorage.getItem('STALK_EDU_MODE') === '1';
  applyEducationalMode(saved);

  if(eduBtn){
    eduBtn.addEventListener('click', ()=>{
      applyEducationalMode(!state.educationalMode);
      if(!overlay || overlay.classList.contains('hidden')){
        toast(state.educationalMode ? 'Educational Mode active. Diversification pressure OFF.' : 'Educational Mode OFF. Diversification pressure ON.');
      }
    });
  }

  if(startBtn){
    startBtn.addEventListener('click', async ()=>{
      if(overlay){
        overlay.classList.add('opacity-0','pointer-events-none');
        setTimeout(()=> overlay.classList.add('hidden'), 400);
      }
      await newSeason();
    });
  }

  renderGarden();
}

function resetTimeline(){
  if(state.timerId){
    clearTimeout(state.timerId);
    state.timerId = null;
  }
  state.timerRunning = false;
  state.currentStep = 0;
  state.maxStep = 0;
  state.timelineComplete = false;
}

function initTimeline(prices){
  resetTimeline();
  state.fullHistory = {};
  state.prices = {};
  let max = 0;
  Object.entries(prices).forEach(([cid, series])=>{
    const arr = Array.isArray(series) ? [...series] : [];
    state.fullHistory[cid] = arr;
    state.prices[cid] = arr.length ? [arr[0]] : [];
    if(arr.length){
      max = Math.max(max, arr.length - 1);
    }
  });
  state.currentStep = 0;
  state.maxStep = max;
  renderClock();
}

function renderClock(){
  const stepEl = document.getElementById('stat-current-step');
  if(stepEl){
    stepEl.textContent = state.seasonId ? `Day ${state.currentStep}` : 'Day —';
  }
  const statusEl = document.getElementById('stat-clock-status');
  if(statusEl){
    let label = 'Paused';
    let cls = 'text-gray-500';
    if(state.timelineComplete){
      label = 'Complete';
      cls = 'text-amber-600';
    }else if(state.timerRunning){
      label = 'Running';
      cls = 'text-green-600';
    }
    statusEl.textContent = label;
    statusEl.className = `text-xs font-semibold ${cls}`;
  }
  const btn = document.getElementById('btn-toggle-clock');
  if(btn){
    if(state.timelineComplete){
      btn.textContent = 'Start New Season';
    }else{
      btn.textContent = state.timerRunning ? 'Pause Clock' : 'Resume Clock';
    }
  }
}

function hydrateEventPopups(){
  const now = state.currentStep;
  state.eventPopups = [];
  (state.events || []).forEach(ev=>{
    const duration = ev.duration || 1;
    const end = ev.ts + duration;
    if(now >= ev.ts && now < end){
      state.eventPopups.push({...ev, endStep: end});
    }
  });
  renderEventPopups();
}

function triggerEventPopup(ev){
  const duration = ev.duration || 1;
  const end = ev.ts + duration;
  state.eventPopups = state.eventPopups.filter(x => !(x.type === ev.type && x.ts === ev.ts));
  state.eventPopups.push({...ev, endStep: end});
  renderEventPopups();
}

function renderEventPopups(){
  const container = document.getElementById('event-stack');
  if(!container) return;
  const now = state.currentStep;
  state.eventPopups = state.eventPopups.filter(ev => ev.endStep > now);
  container.innerHTML = '';
  state.eventPopups
    .slice()
    .sort((a,b)=> (b.endStep - a.endStep))
    .forEach(ev=>{
      const el = document.createElement('div');
      el.className = 'bg-white/95 border border-emerald-200 shadow-lg rounded-lg p-4 space-y-2 pointer-events-auto';
      const name = ev.name || titleize(ev.type);
      const remaining = Math.max(0, ev.endStep - now);
      const affects = (ev.affected || []).length ? ev.affected.join(', ') : 'All crops';
      el.innerHTML = `
        <div class="flex items-center justify-between">
          <span class="font-semibold text-green-800">${name}</span>
          <span class="text-xs font-mono text-gray-500">Day ${ev.ts}</span>
        </div>
        <p class="text-xs text-gray-600">${ev.note || 'Seasonal conditions in effect.'}</p>
        <p class="text-xs text-gray-500">Affects: ${affects}</p>
        <p class="text-xs text-emerald-700 font-semibold">Expires in ${remaining} day${remaining === 1 ? '' : 's'}</p>
      `;
      container.appendChild(el);
    });
}

function scheduleNextTick(){
  if(!state.timerRunning) return;
  if(state.timerId){
    clearTimeout(state.timerId);
  }
  state.timerId = setTimeout(async ()=>{
    state.timerId = null;
    await advanceTick();
    scheduleNextTick();
  }, state.tickIntervalMs);
}

function startClock(){
  if(state.timerRunning || state.timelineComplete) return;
  state.timerRunning = true;
  renderClock();
  scheduleNextTick();
}

function pauseClock(){
  state.timerRunning = false;
  if(state.timerId){
    clearTimeout(state.timerId);
    state.timerId = null;
  }
  renderClock();
}

function toggleClock(){
  if(state.timerRunning){
    pauseClock();
  }else{
    if(state.timelineComplete){
      toast('Growing a fresh season...');
      newSeason();
      return;
    }
    startClock();
  }
}

// ---------- API Calls ----------
async function getMacro() {
  try{
    const r = await fetch(`${API()}/macro`);
    if(!r.ok) throw new Error(`macro ${r.status}`);
    const j = await r.json();
    return j.macro || j;
  }catch(err){
    console.error('macro fetch failed', err);
    return {inflation_ann: 0.02, rf_rate_ann: 0.02, recession: false, term_spread: 1.0, vol_mult: 1.0, asof: 'Offline'};
  }
}

async function newSeason() {
  if(state.starting) return;
  state.starting = true;
  try{
    const r = await fetch(`${API()}/demo_seed`, {method: 'POST'});
    const j = await r.json();
    if(!j.ok){ throw new Error('demo seed failed'); }
    const seasonId = j.season_id || 'S1';
    document.getElementById('season-id').textContent = seasonId;

    const {prices, crops, events} = await fetchSeasonSnapshot(seasonId);
    const macro = await getMacro();

    state.seasonId = seasonId;
    state.crops = crops;
    state.events = (events || []).slice().sort((a,b)=>a.ts - b.ts);
    state.macro = macro;
    state.cash = 10000;
    state.holdings = {};
    state.shorts = {};
    state.txns = [];
    state.costBasis = {};
    state.shortBasis = {};
    state.gardenSprites = {};
    initTimeline(prices);

    renderMacro();
    renderMarket();
    renderPortfolio();
    renderTxns();
    renderClock();
    state.eventPopups = [];
    hydrateEventPopups();
    startClock();
    toast('New season started!');
  }catch(err){
    console.error(err);
    toast('Season start failed. Check backend server.');
  }finally{
    state.starting = false;
  }
}

async function fetchSeasonSnapshot(seasonId){
  const r = await fetch(`${API()}/season/${encodeURIComponent(seasonId)}/prices`);
  if(!r.ok){
    throw new Error(`season snapshot failed: ${r.status}`);
  }
  const j = await r.json();
  const grouped = {};
  for(const row of j.prices || []){
    if(!grouped[row.crop_id]) grouped[row.crop_id] = [];
    grouped[row.crop_id].push({ts: row.ts, price: row.price});
  }
  const prices = {};
  Object.keys(grouped).forEach((cid)=>{
    const series = grouped[cid].sort((a,b)=>a.ts-b.ts).map((pt)=>pt.price);
    prices[cid] = series;
  });
  const crops = Object.keys(prices).sort();
  return {prices, crops, events: j.events || []};
}

async function runMonteCarlo(){
  const pricesNow = latestPrices();
  const weights = portfolioWeights();
  const cropParams = state.crops.map(cid => {
    const est = estimateParams(state.prices[cid] || []);
    return { crop_id: cid, mu: est.mu, sigma: est.sigma, seasonality_strength: est.seasonality_strength, jump_lam: 0.02, jump_mu: 0.0, jump_sig: 0.05 };
  });
  const body = { prices_now: pricesNow, weights, crop_params: cropParams, horizon_steps: 12, N: 1000 };
  const r = await fetch(`${API()}/montecarlo`, { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  const j = await r.json();
  if(j.percentiles){
    const p = j.percentiles["p5,p25,p50,p75,p95"];
    toast(`Monte Carlo wealth x-multiple: p5=${p[0].toFixed(2)}, p50=${p[2].toFixed(2)}, p95=${p[4].toFixed(2)}`);
  } else {
    toast('Monte Carlo failed');
  }
}

async function extendSeason(steps = 12){
  if(!state.seasonId) return null;
  const diversification = state.educationalMode ? 0 : computeDiversificationHHI();
  const r = await fetch(`${API()}/season/${encodeURIComponent(state.seasonId)}/advance`, {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({steps, diversification_hhi: diversification})
  });
  if(!r.ok){
    throw new Error(`extend failed: ${r.status}`);
  }
  return await r.json();
}

async function ensureFutureSteps(){
  if(!state.seasonId || state.extending) return;
  const remaining = state.maxStep - state.currentStep;
  if(remaining > 1) return;
  try{
    state.extending = true;
    const data = await extendSeason(12);
    if(!data) return;
    const grouped = {};
    (data.prices || []).forEach(rec=>{
      (grouped[rec.crop_id] ||= []).push(rec);
    });
    Object.entries(grouped).forEach(([cid, rows])=>{
      rows.sort((a,b)=>a.ts - b.ts);
      const target = state.fullHistory[cid] || [];
      rows.forEach(row=>{
        target.push(row.price);
      });
      state.fullHistory[cid] = target;
      if(!state.prices[cid]){
        state.prices[cid] = target.length ? [target[0]] : [];
      }
      if(!state.crops.includes(cid)){
        state.crops.push(cid);
        state.crops.sort();
      }
    });
    if(data.prices && data.prices.length){
      const newMax = Math.max(...data.prices.map(p=>p.ts));
      state.maxStep = Math.max(state.maxStep, newMax);
      state.timelineComplete = false;
    }
    if(data.events && data.events.length){
      state.events = [...state.events, ...data.events].sort((a,b)=>a.ts - b.ts);
    }
    renderClock();
    hydrateEventPopups();
  }catch(err){
    console.error(err);
    toast('Unable to extend market timeline. Start a new season.');
    state.timelineComplete = true;
  }finally{
    state.extending = false;
  }
}

async function advanceTick(manual = false){
  if(!state.seasonId) return false;
  if(state.currentStep >= state.maxStep){
    await ensureFutureSteps();
  }
  if(state.currentStep >= state.maxStep){
    if(!state.timelineComplete){
      toast('Season timeline exhausted. Start a new season to keep growing.');
      state.timelineComplete = true;
    }
    if(!manual){
      pauseClock();
    }
    return false;
  }
  const penaltyRaw = state.educationalMode ? 0 : Math.max(0, computeDiversificationHHI() - 0.35);
  const dampFactor = penaltyRaw > 0 ? Math.max(0.55, 1 - 0.45 * penaltyRaw) : 1;
  state.currentStep += 1;
  state.crops.forEach(cid=>{
    const hist = state.fullHistory[cid] || [];
    let next = hist[state.currentStep];
    if(next === undefined) return;
    if(!state.educationalMode && penaltyRaw > 0){
      const prev = hist[state.currentStep-1] ?? next;
      next = prev + (next - prev) * dampFactor;
      state.fullHistory[cid][state.currentStep] = next;
    }
    const visible = state.prices[cid] || [];
    visible.push(next);
    if(visible.length > 180){
      visible.shift();
    }
    state.prices[cid] = visible;
  });
  renderMarket();
  renderPortfolio();
  renderClock();
  renderEventPopups();
  const modal = document.getElementById('trade-modal');
  if(modal && !modal.classList.contains('hidden') && currentCrop){
    await buildModalChart(currentCrop);
    updateTradeTotal();
  }
  const eventsThisStep = (state.events || []).filter(e => e.ts === state.currentStep);
  if(eventsThisStep.length){
    eventsThisStep.forEach(triggerEventPopup);
  }
  return true;
}

// ---------- Rendering ----------
function renderMacro(){
  const m = state.macro || {};
  const el = document.getElementById('macro-card');
  el.innerHTML = `
    <p><span class="font-semibold">Inflation (YoY):</span> ${(100*(m.inflation_ann||0)).toFixed(1)}%</p>
    <p><span class="font-semibold">Fed Funds (annual):</span> ${(100*(m.rf_rate_ann||0)).toFixed(2)}%</p>
    <p><span class="font-semibold">Term Spread:</span> ${(m.term_spread||0).toFixed(2)}%</p>
    <p><span class="font-semibold">Recession:</span> ${m.recession ? 'Yes' : 'No'}</p>
    <p class="text-xs text-gray-500">As of ${m.asof||'—'}. Volatility multiplier applied: ${(m.vol_mult||1).toFixed(2)}×</p>
  `;
}

function renderMarket(){
  const grid = document.getElementById('plant-cards');
  grid.innerHTML = '';
  if(!state.crops.length){
    grid.innerHTML = `<div class="col-span-full bg-white border border-dashed border-gray-300 rounded-lg p-6 text-center text-gray-500">
      No plants yet. Start a season to populate the market garden.
    </div>`;
    return;
  }
  state.crops.forEach((cid, idx)=>{
    const arr = state.prices[cid] || [];
    const last = arr[arr.length-1] || 100;
    const prev = arr[arr.length-2] || last;
    const change = (last - prev) / (prev || 1);
    const color = change > 0 ? 'price-up' : (change < 0 ? 'price-down' : 'price-neutral');
    const meta = cropMeta(cid);
    const qty = state.holdings[cid] || 0;
    const basis = state.costBasis[cid] || 0;
    const shortQty = state.shorts[cid] || 0;
    const shortBasis = state.shortBasis[cid] || 0;
    let positionHtml = '';
    if(qty > 0){
      const delta = basis > 0 ? (last - basis) / basis : 0;
      const basisClass = delta >= 0 ? 'text-green-600' : 'text-red-600';
      positionHtml += `<p class="text-xs ${basisClass}">Long ${qty} @ ${fmt(basis)} (${(delta*100).toFixed(1)}%)</p>`;
    }
    if(shortQty > 0){
      const deltaShort = shortBasis > 0 ? (shortBasis - last) / shortBasis : 0;
      const basisClassShort = deltaShort >= 0 ? 'text-green-600' : 'text-red-600';
      positionHtml += `<p class="text-xs ${basisClassShort}">Short ${shortQty} @ ${fmt(shortBasis)} (${(deltaShort*100).toFixed(1)}%)</p>`;
    }
    const card = document.createElement('div');
    card.className = 'plant-card bg-white rounded-lg p-6 shadow-sm border border-gray-200 fade-in';
    card.innerHTML = `
      <div class="flex items-start justify-between">
        <div>
          <div class="mb-2 flex justify-center">
            <img class="market-card-image" src="${SPRITE_BASE}/${ensureGardenSprite(cid, idx)}_0.png" alt="${meta.name} seed">
          </div>
          <h4 class="text-lg font-semibold">${meta.name}</h4>
          <p class="text-sm text-gray-500">${meta.tagline}</p>
          ${positionHtml}
        </div>
        <div class="text-right">
          <div class="text-2xl font-mono font-semibold">${fmt(last)}</div>
          <div class="${color} text-sm">${pct(change)}</div>
        </div>
      </div>
      <div class="mt-4">
        <button class="w-full bg-green-600 text-white px-3 py-2 rounded-lg hover:bg-green-700" data-crop="${cid}">Trade</button>
      </div>
    `;
    card.querySelector('button').addEventListener('click', ()=> openTradeModal(cid));
    grid.appendChild(card);
  });
}

function renderShorts(){
  const grid = document.getElementById('short-cards');
  if(!grid) return;
  const active = state.crops.filter(cid => (state.shorts[cid] || 0) > 0);
  if(!active.length){
    grid.innerHTML = `<div class="col-span-full bg-white border border-dashed border-gray-300 rounded-lg p-6 text-center text-gray-500">
      No borrowed seeds. Use Borrow & Short to profit from falling prices.
    </div>`;
    return;
  }
  grid.innerHTML = '';
  active.forEach(cid => {
    const qty = state.shorts[cid] || 0;
    const basis = state.shortBasis[cid] || 0;
    const arr = state.prices[cid] || [];
    const last = arr[arr.length-1] || 100;
    const prev = arr[arr.length-2] || last;
    const change = (last - prev) / (prev || 1);
    const pnl = qty * (basis - last);
    const pnlClass = pnl >= 0 ? 'text-green-600' : 'text-red-600';
    const meta = cropMeta(cid);
    const card = document.createElement('div');
    card.className = 'plant-card bg-white rounded-lg p-6 shadow-sm border border-gray-200 fade-in';
    card.innerHTML = `
      <div class="flex items-start justify-between">
        <div>
          <div class="plant-emoji mb-2">${meta.emoji}</div>
          <h4 class="text-lg font-semibold">${meta.name}</h4>
          <p class="text-sm text-gray-500">Borrowed ${qty} @ ${fmt(basis)}</p>
          <p class="text-sm font-mono ${pnlClass}">${fmtSigned(pnl)} unrealized</p>
        </div>
        <div class="text-right">
          <div class="text-2xl font-mono font-semibold">${fmt(last)}</div>
          <div class="${change>0?'price-up':(change<0?'price-down':'price-neutral')} text-sm">${pct(change)}</div>
        </div>
      </div>
      <div class="mt-4">
        <button class="w-full bg-blue-600 text-white px-3 py-2 rounded-lg hover:bg-blue-700" data-crop="${cid}">Manage</button>
      </div>
    `;
    card.querySelector('button').addEventListener('click', ()=> openTradeModal(cid));
    grid.appendChild(card);
  });
}

function renderPortfolio(){
  // Compute total value
  const last = latestPrices();
  let plants = 0;
  let value = state.cash;
  let growth = 0;
  for(const cid of state.crops){
    const qty = state.holdings[cid] || 0;
    const shortQty = state.shorts[cid] || 0;
    plants += qty;
    plants -= shortQty;
    value += qty * (last[cid]||0);
    value -= shortQty * (last[cid]||0);
    const series = state.prices[cid] || [];
    if(series.length > 1){
      const delta = series[series.length-1] - series[series.length-2];
      growth += qty * delta;
      growth -= shortQty * delta;
    }
  }
  document.getElementById('header-cash').textContent = fmt(state.cash);
  document.getElementById('header-portfolio-value').textContent = fmt(value);
  document.getElementById('stat-total-value').textContent = fmt(value);
  document.getElementById('stat-cash').textContent = fmt(state.cash);
  document.getElementById('stat-plants-owned').textContent = plants.toString();
  const growthEl = document.getElementById('stat-today-growth');
  if(growthEl){
    growthEl.textContent = fmtSigned(growth);
    growthEl.className = `text-2xl font-mono font-semibold ${growth >= 0 ? 'text-green-600' : 'text-red-600'}`;
  }

  renderGarden();
  renderShorts();
}

function ensureGardenSprite(cid, idx){
  state.gardenSprites = state.gardenSprites || {};
  if(state.gardenSprites[cid]) return state.gardenSprites[cid];
  const override = GARDEN_SPRITE_OVERRIDES[cid];
  if(override){
    state.gardenSprites[cid] = override;
    return override;
  }
  const used = new Set(Object.values(state.gardenSprites));
  const pool = GARDEN_SPRITE_POOL;
  let sprite = pool.find(name => !used.has(name));
  if(!sprite){
    sprite = pool[idx % pool.length] || pool[0];
  }
  state.gardenSprites[cid] = sprite;
  return sprite;
}

function createGardenPlot(sprite, stage){
  const plot = document.createElement('div');
  plot.className = 'garden-plot';
  plot.style.backgroundImage = `url("${TILE_BASE}/tdfarmtiles_dirt.png")`;
  if(sprite != null && stage != null){
    const img = document.createElement('img');
    img.src = `${SPRITE_BASE}/${sprite}_${stage}.png`;
    img.alt = sprite;
    img.className = 'garden-plant';
    plot.appendChild(img);
  } else {
    plot.classList.add('garden-plot--empty');
  }
  return plot;
}

function renderGarden(){
  const grid = document.getElementById('garden-grid');
  if(!grid) return;
  grid.innerHTML = '';
  const crops = state.crops.slice(0, MAX_GARDEN_COLUMNS);
  for(let i=0; i<MAX_GARDEN_COLUMNS; i++){
    const cid = crops[i];
    const column = document.createElement('div');
    column.className = 'garden-column';
    const header = document.createElement('div');
    header.className = 'garden-column-header';
    if(cid){
      const meta = cropMeta(cid);
      const qty = Math.max(0, state.holdings[cid]||0);
      header.innerHTML = `<span class="garden-name">${meta.name}</span><span class="garden-qty">${qty} seeds</span>`;
    } else {
      header.innerHTML = `<span class="garden-name text-gray-400">Empty Plot</span>`;
    }
    column.appendChild(header);

    const plots = document.createElement('div');
    plots.className = 'garden-plots';
    const plotData = [];
    if(cid){
      const qty = Math.max(0, state.holdings[cid]||0);
      const sprite = ensureGardenSprite(cid, i);
      const fullCount = Math.floor(qty / 20);
      const remainder = qty % 20;
      for(let j=0;j<fullCount;j++){
        plotData.push({sprite, stage: 4});
      }
      if(remainder > 0){
        const stage = Math.min(3, Math.max(0, Math.floor((remainder / 20) * 4)));
        plotData.push({sprite, stage});
      }
    }
    while(plotData.length < BASE_GARDEN_ROWS){
      plotData.push(null);
    }
    plotData.forEach(data => {
      if(data){
        plots.appendChild(createGardenPlot(data.sprite, data.stage));
      }else{
        plots.appendChild(createGardenPlot(null, null));
      }
    });
    column.appendChild(plots);
    grid.appendChild(column);
  }
}

function updateHeader(){ renderPortfolio(); }

// ---------- Trade Modal ----------
let modalChart = null;
let currentCrop = null;

function openTradeModal(cid){
  currentCrop = cid;
  const modal = document.getElementById('trade-modal');
  modal.classList.remove('hidden');
  const meta = cropMeta(cid);
  document.getElementById('modal-plant-name').textContent = meta.name;
  document.getElementById('modal-plant-id').textContent = cid;
  const arr = state.prices[cid]||[100,100.5,101];
  const last = arr[arr.length-1], prev = arr[arr.length-2]||last;
  const change = (last-prev)/(prev||1);
  document.getElementById('modal-current-price').textContent = fmt(last);
  const pc = document.getElementById('modal-price-change');
  pc.textContent = pct(change);
  pc.className = `text-sm font-medium ${change>0?'price-up':(change<0?'price-down':'price-neutral')}`;
  document.getElementById('modal-available-cash').textContent = fmt(state.cash);
  const owned = state.holdings[cid]||0;
  const basis = state.costBasis[cid] || 0;
  const borrowed = state.shorts[cid] || 0;
  const borrowBasis = state.shortBasis[cid] || 0;
  const hi = document.getElementById('holdings-info');
  if(owned>0){
    hi.classList.remove('hidden');
    document.getElementById('modal-holdings').textContent = `${owned} plants`;
    const basisEl = document.getElementById('modal-holdings-basis');
    if(basisEl){
      basisEl.textContent = `${fmt(basis)} avg cost`;
    }
  } else {
    hi.classList.add('hidden');
  }
  const si = document.getElementById('short-info');
  if(borrowed>0){
    si.classList.remove('hidden');
    document.getElementById('modal-short-qty').textContent = `${borrowed} plants`;
    document.getElementById('modal-short-basis').textContent = `${fmt(borrowBasis)} avg borrow`;
  } else {
    si.classList.add('hidden');
  }
  document.getElementById('trade-quantity').value = 1;
  updateTradeTotal();

  // Build chart with history + forecast
  buildModalChart(cid);
}

function closeTradeModal(){
  const modal = document.getElementById('trade-modal');
  modal.classList.add('hidden');
  if(modalChart){ modalChart.destroy(); modalChart = null; }
}

function updateTradeTotal(){
  const qty = parseInt(document.getElementById('trade-quantity').value || '0',10);
  const price = (state.prices[currentCrop]||[]).slice(-1)[0] || 0;
  document.getElementById('trade-total').textContent = fmt(qty*price);
}

async function buildModalChart(cid){
  const ctx = document.getElementById('modal-chart').getContext('2d');
  const hist = (state.fullHistory[cid]||[]).slice(0, state.currentStep + 1).slice(-50);
  // fetch forecast
  let fc = {mean:[], p10:[], p90:[]};
  try{
    const fr = await fetch(`${API()}/forecast`, {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({season_id: state.seasonId, crop_ids: [cid], horizon: 12})
    });
    if(fr.ok){
      const fj = await fr.json();
      fc = (fj.forecasts && fj.forecasts[cid]) || fc;
    }
  }catch(err){
    console.warn('forecast fetch failed', err);
  }

  const labels = [...hist.map((_,i)=>`t-${hist.length-i}`), ...fc.mean.map((_,i)=>`+${i+1}`)];
  const histData = hist;
  let forecastData = fc.mean ? [...fc.mean] : [];
  let bandLow = fc.p10 ? [...fc.p10] : [];
  let bandHigh = fc.p90 ? [...fc.p90] : [];
  if(hist.length && forecastData.length){
    const anchor = hist[hist.length-1];
    const offset = forecastData[0] - anchor;
    forecastData = forecastData.map(v => v - offset);
    bandLow = bandLow.map(v => v - offset);
    bandHigh = bandHigh.map(v => v - offset);
  }

  if(modalChart){ modalChart.destroy(); }
  modalChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {label: 'History', data: histData, borderWidth: 2, tension: 0.2},
        {label: 'Forecast (mean)', data: [...Array(hist.length).fill(null), ...forecastData], borderWidth: 2, borderDash: [6,4], tension: 0.2},
        {label: 'p10', data: [...Array(hist.length).fill(null), ...bandLow], borderWidth: 1, borderDash: [2,2], tension: 0.2},
        {label: 'p90', data: [...Array(hist.length).fill(null), ...bandHigh], borderWidth: 1, borderDash: [2,2], tension: 0.2},
      ]
    },
    options: {
      plugins: { legend: { display: true } },
      scales: { y: { beginAtZero: false } }
    }
  });
}

function executeBuy(){
  const qty = parseInt(document.getElementById('trade-quantity').value || '0',10);
  if(!qty || qty < 1) return;
  const price = (state.prices[currentCrop]||[]).slice(-1)[0] || 0;
  const cost = qty * price;
  if(cost > state.cash){ toast('Not enough cash'); return; }
  state.cash -= cost;
  const prevQty = state.holdings[currentCrop] || 0;
  const prevBasis = state.costBasis[currentCrop] || 0;
  const newQty = prevQty + qty;
  const totalCost = prevQty * prevBasis + cost;
  state.holdings[currentCrop] = newQty;
  state.costBasis[currentCrop] = newQty ? totalCost / newQty : 0;
  state.txns.unshift({t: Date.now(), type:'BUY', cid: currentCrop, qty, price});
  renderPortfolio();
  renderTxns();
  updateTradeTotal();
  toast(`Bought ${qty} ${currentCrop}`);
}

function executeSell(){
  const qty = parseInt(document.getElementById('trade-quantity').value || '0',10);
  if(!qty || qty < 1) return;
  const owned = state.holdings[currentCrop]||0;
  if(qty > owned){ toast('Not enough holdings'); return; }
  const price = (state.prices[currentCrop]||[]).slice(-1)[0] || 0;
  const proceeds = qty * price;
  state.cash += proceeds;
  state.holdings[currentCrop] = owned - qty;
  if(state.holdings[currentCrop] <= 0){
    state.holdings[currentCrop] = 0;
    state.costBasis[currentCrop] = 0;
  }
  state.txns.unshift({t: Date.now(), type:'SELL', cid: currentCrop, qty, price});
  renderPortfolio();
  renderTxns();
  updateTradeTotal();
  toast(`Sold ${qty} ${currentCrop}`);
}

function executeShort(){
  const qty = parseInt(document.getElementById('trade-quantity').value || '0',10);
  if(!qty || qty < 1) return;
  const price = (state.prices[currentCrop]||[]).slice(-1)[0] || 0;
  const proceeds = qty * price;
  state.cash += proceeds;
  const prevQty = state.shorts[currentCrop] || 0;
  const prevBasis = state.shortBasis[currentCrop] || 0;
  const newQty = prevQty + qty;
  const totalBorrow = prevQty * prevBasis + qty * price;
  state.shorts[currentCrop] = newQty;
  state.shortBasis[currentCrop] = newQty ? totalBorrow / newQty : 0;
  state.txns.unshift({t: Date.now(), type:'SHORT', cid: currentCrop, qty, price});
  renderPortfolio();
  renderTxns();
  updateTradeTotal();
  toast(`Borrowed ${qty} ${currentCrop} and sold short`);
}

function executeCover(){
  const qty = parseInt(document.getElementById('trade-quantity').value || '0',10);
  if(!qty || qty < 1) return;
  const borrowed = state.shorts[currentCrop] || 0;
  if(qty > borrowed){ toast('Not enough borrowed seeds'); return; }
  const price = (state.prices[currentCrop]||[]).slice(-1)[0] || 0;
  const cost = qty * price;
  if(cost > state.cash){ toast('Not enough cash to cover'); return; }
  state.cash -= cost;
  state.shorts[currentCrop] = borrowed - qty;
  if(state.shorts[currentCrop] <= 0){
    state.shorts[currentCrop] = 0;
    state.shortBasis[currentCrop] = 0;
  }
  state.txns.unshift({t: Date.now(), type:'COVER', cid: currentCrop, qty, price});
  renderPortfolio();
  renderTxns();
  updateTradeTotal();
  toast(`Covered ${qty} ${currentCrop}`);
}

function renderTxns(){
  const list = document.getElementById('transactions-list');
  if(!state.txns.length){
    list.innerHTML = `<div class="p-8 text-center text-gray-500"><p>No transactions yet.</p></div>`;
    return;
  }
  list.innerHTML = state.txns.slice(0,10).map(x=>{
    let color = 'text-gray-700';
    if(x.type === 'BUY') color = 'text-green-700';
    else if(x.type === 'SELL') color = 'text-red-700';
    else if(x.type === 'SHORT') color = 'text-yellow-600';
    else if(x.type === 'COVER') color = 'text-blue-700';
    const meta = cropMeta(x.cid);
    return `<div class="flex justify-between items-center px-4 py-2 border-b border-gray-100">
      <div class="${color} font-semibold">${x.type}</div>
      <div class="font-mono">${meta.name}</div>
      <div class="font-mono">qty ${x.qty}</div>
      <div class="font-mono">${fmt(x.price)}</div>
    </div>`;
  }).join('');
}

// ---------- Wire buttons ----------
document.getElementById('btn-new-season').addEventListener('click', newSeason);
document.getElementById('btn-report').addEventListener('click', async ()=>{
  if(!state.seasonId){ toast('Start a season first'); return; }
  const r = await fetch(`${API()}/report`, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({season_id: state.seasonId})});
  const j = await r.json();
  if(j && j.metrics){
    const m = j.metrics;
    toast(`Report → Sharpe ${(m.sharpe||0).toFixed(2)}, MDD ${(100*(m.mdd||0)).toFixed(1)}%, Wealth x${(m.wealth||1).toFixed(2)}`);
  } else {
    toast('Report failed');
  }
});
document.getElementById('btn-mc').addEventListener('click', runMonteCarlo);
document.getElementById('btn-toggle-clock').addEventListener('click', toggleClock);
document.getElementById('btn-step').addEventListener('click', async ()=>{
  if(state.timelineComplete){
    await newSeason();
    return;
  }
  pauseClock();
  await advanceTick(true);
});

// ---------- Boot ----------
(async function boot(){
  setupIntroOverlay();
  try{
    const macro = await getMacro();
    state.macro = macro || {};
    renderMacro();
  }catch(e){
    console.error('macro fetch failed', e);
  }
})();
