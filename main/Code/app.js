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
  txns: [],
  charts: {},
  fullHistory: {},   // complete price history per crop
  currentStep: 0,
  maxStep: 0,
  tickIntervalMs: 10000,
  timerId: null,
  timerRunning: false,
  costBasis: {},
  extending: false,
  timelineComplete: false,
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
    const qty = state.holdings[cid] || 0;
    const p = latest[cid] || 0;
    total += qty * p;
  }
  for (const cid of state.crops) {
    const qty = state.holdings[cid] || 0;
    const p = latest[cid] || 0;
    w[cid] = total > 0 ? (qty * p) / total : 0;
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
  const r = await fetch(`${API()}/macro`);
  const j = await r.json();
  return j.macro || j;
}

async function newSeason() {
  try{
    const r = await fetch(`${API()}/demo_seed`, {method: 'POST'});
    const j = await r.json();
    if(!j.ok){ toast('Failed to seed demo'); return; }
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
    state.txns = [];
    state.costBasis = {};
    initTimeline(prices);

    renderMacro();
    renderMarket();
    renderPortfolio();
    renderEvents();
    renderTxns();
    renderClock();
    startClock();
    toast('New season started!');
  }catch(err){
    console.error(err);
    toast('Season start failed. Check backend server.');
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
  const r = await fetch(`${API()}/season/${encodeURIComponent(state.seasonId)}/advance`, {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({steps})
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
  state.currentStep += 1;
  state.crops.forEach(cid=>{
    const hist = state.fullHistory[cid] || [];
    const next = hist[state.currentStep];
    if(next === undefined) return;
    const visible = state.prices[cid] || [];
    visible.push(next);
    if(visible.length > 180){
      visible.shift();
    }
    state.prices[cid] = visible;
  });
  renderMarket();
  renderPortfolio();
  renderEvents();
  renderClock();
  const modal = document.getElementById('trade-modal');
  if(modal && !modal.classList.contains('hidden') && currentCrop){
    await buildModalChart(currentCrop);
    updateTradeTotal();
  }
  const eventsThisStep = (state.events || []).filter(e => e.ts === state.currentStep);
  if(eventsThisStep.length){
    const names = eventsThisStep.map(e=>e.name || titleize(e.type));
    toast(`New event: ${names.join(', ')}`);
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

function renderEvents(){
  const el = document.getElementById('event-timeline');
  const visible = (state.events || []).filter(e => typeof e.ts === 'number' ? e.ts <= state.currentStep : true);
  if(!visible.length){
    el.innerHTML = '<p>No events yet.</p>';
    return;
  }
  el.innerHTML = visible.slice(-20).map(e => {
    const emoji = e.type.includes('bull') ? '☀️' : e.type.includes('bear') ? '⛈️' : e.type.includes('bug') ? '🐛' : '⚡';
    return `<div class="flex items-center space-x-2"><span>${emoji}</span><span class="font-mono text-xs">t=${e.ts}</span><span>${e.name||e.type}</span></div>`;
  }).join('');
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
  for(const cid of state.crops){
    const arr = state.prices[cid] || [];
    const last = arr[arr.length-1] || 100;
    const prev = arr[arr.length-2] || last;
    const change = (last - prev) / (prev || 1);
    const color = change > 0 ? 'price-up' : (change < 0 ? 'price-down' : 'price-neutral');
    const meta = cropMeta(cid);
    const qty = state.holdings[cid] || 0;
    const basis = state.costBasis[cid] || 0;
    let basisHtml = '';
    if(qty > 0 && basis){
      const delta = basis > 0 ? (last - basis) / basis : 0;
      const basisClass = delta >= 0 ? 'text-green-600' : 'text-red-600';
      basisHtml = `<p class="text-xs ${basisClass}">Basis ${fmt(basis)} (${(delta*100).toFixed(1)}%)</p>`;
    }
    const card = document.createElement('div');
    card.className = 'plant-card bg-white rounded-lg p-6 shadow-sm border border-gray-200 fade-in';
    card.innerHTML = `
      <div class="flex items-start justify-between">
        <div>
          <div class="plant-emoji mb-2 ${change>0?'plant-thriving':'plant-wilting'}">${meta.emoji}</div>
          <h4 class="text-lg font-semibold">${meta.name}</h4>
          <p class="text-sm text-gray-500">${meta.tagline}</p>
          ${basisHtml}
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
  }
}

function renderPortfolio(){
  // Compute total value
  const last = latestPrices();
  let plants = 0;
  let value = state.cash;
  let growth = 0;
  for(const cid of state.crops){
    const qty = state.holdings[cid] || 0;
    plants += qty;
    value += qty * (last[cid]||0);
    const series = state.prices[cid] || [];
    if(series.length > 1){
      const delta = series[series.length-1] - series[series.length-2];
      growth += qty * delta;
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

  // Render holdings table
  const container = document.getElementById('portfolio-holdings');
  if(plants === 0){
    container.innerHTML = `<div class="p-8 text-center text-gray-500"><p class="text-lg mb-2">Your garden is empty!</p><p>Start investing in plants to grow your wealth.</p></div>`;
    return;
  }
  const rows = state.crops.filter(cid => (state.holdings[cid]||0) > 0).map(cid=>{
    const qty = state.holdings[cid]||0;
    const p = last[cid]||0;
    const val = qty*p;
    const meta = cropMeta(cid);
    const basis = state.costBasis[cid] || 0;
    const plValue = qty ? qty * (p - basis) : 0;
    const plPct = basis > 0 ? ((p - basis) / basis) * 100 : 0;
    const plClass = plValue >= 0 ? 'text-green-600' : 'text-red-600';
    return `<tr>
      <td class="px-4 py-2">${meta.name}</td>
      <td class="px-4 py-2 text-right font-mono">${qty}</td>
      <td class="px-4 py-2 text-right font-mono">${fmt(p)}</td>
      <td class="px-4 py-2 text-right font-mono">${fmt(basis)}</td>
      <td class="px-4 py-2 text-right font-mono">${fmt(val)}</td>
      <td class="px-4 py-2 text-right font-mono ${plClass}">${fmtSigned(plValue)} (${plPct.toFixed(1)}%)</td>
    </tr>`;
  }).join('');
  container.innerHTML = `
    <table class="min-w-full text-sm">
      <thead class="bg-gray-50">
        <tr>
          <th class="px-4 py-2 text-left">Crop</th>
          <th class="px-4 py-2 text-right">Qty</th>
          <th class="px-4 py-2 text-right">Price</th>
          <th class="px-4 py-2 text-right">Avg Cost</th>
          <th class="px-4 py-2 text-right">Value</th>
          <th class="px-4 py-2 text-right">P/L</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>`;
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
  const forecastData = fc.mean ? fc.mean : [];
  const bandLow = fc.p10 || [];
  const bandHigh = fc.p90 || [];

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

function renderTxns(){
  const list = document.getElementById('transactions-list');
  if(!state.txns.length){
    list.innerHTML = `<div class="p-8 text-center text-gray-500"><p>No transactions yet.</p></div>`;
    return;
  }
  list.innerHTML = state.txns.slice(0,10).map(x=>{
    const color = x.type === 'BUY' ? 'text-green-700' : 'text-red-700';
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
  try{
    const macro = await getMacro();
    state.macro = macro || {};
    renderMacro();
  }catch(e){
    // ignore
  }
  // Auto-start one season on load for demo
  await newSeason();
})();
