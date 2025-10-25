const API_BASE = window.location.origin;
const UPDATE_INTERVAL = 3000;

let currentPlantId = null;
let plantConfigs = [];
let modalChart = null;
let portfolioData = null;

async function fetchAPI(endpoint) {
    const response = await fetch(`${API_BASE}${endpoint}`);
    if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
    }
    return response.json();
}

async function postAPI(endpoint, data) {
    const response = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    });
    
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Request failed');
    }
    
    return response.json();
}

function formatCurrency(amount) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    }).format(amount);
}

function formatPercent(value) {
    const sign = value >= 0 ? '+' : '';
    return `${sign}${value.toFixed(2)}%`;
}

function getPlantEmoji(plantId) {
    const config = plantConfigs.find(c => c.id === plantId);
    return config ? config.emoji : '🌱';
}

function getPlantName(plantId) {
    const config = plantConfigs.find(c => c.id === plantId);
    return config ? config.name : plantId;
}

function getPlantState(changePercent) {
    if (changePercent > 10) return 'thriving';
    if (changePercent < -10) return 'wilting';
    return 'stable';
}

async function loadPlantConfigs() {
    try {
        plantConfigs = await fetchAPI('/api/plant-configs');
    } catch (error) {
        console.error('Failed to load plant configs:', error);
    }
}

async function updateMarket() {
    try {
        const prices = await fetchAPI('/api/market');
        
        const container = document.getElementById('plant-cards');
        container.innerHTML = '';
        
        prices.forEach(plant => {
            const emoji = getPlantEmoji(plant.plant_id);
            const name = getPlantName(plant.plant_id);
            const state = getPlantState(plant.change_percent);
            const priceClass = plant.change_percent >= 0 ? 'price-up' : 'price-down';
            
            const card = document.createElement('div');
            card.className = 'plant-card bg-white rounded-lg shadow-sm border border-gray-200 p-6 cursor-pointer fade-in';
            card.onclick = () => openTradeModal(plant.plant_id);
            
            card.innerHTML = `
                <div class="text-center mb-4">
                    <div class="plant-emoji ${state === 'thriving' ? 'plant-thriving' : ''} ${state === 'wilting' ? 'plant-wilting' : ''}">${emoji}</div>
                </div>
                <h4 class="text-lg font-semibold text-gray-900 mb-2">${name}</h4>
                <div class="space-y-1">
                    <p class="text-2xl font-mono font-semibold text-gray-900">${formatCurrency(plant.price)}</p>
                    <p class="text-sm font-medium ${priceClass}">${formatPercent(plant.change_percent)}</p>
                </div>
                <div class="mt-4 flex space-x-2">
                    <button class="flex-1 bg-green-600 text-white px-4 py-2 rounded-lg text-sm font-semibold hover:bg-green-700 transition-colors" onclick="event.stopPropagation(); openTradeModal('${plant.plant_id}')">
                        Buy
                    </button>
                    <button class="flex-1 bg-gray-100 text-gray-700 px-4 py-2 rounded-lg text-sm font-semibold hover:bg-gray-200 transition-colors" onclick="event.stopPropagation(); openTradeModal('${plant.plant_id}')">
                        Details
                    </button>
                </div>
            `;
            
            container.appendChild(card);
        });
    } catch (error) {
        console.error('Failed to update market:', error);
    }
}

async function updatePortfolio() {
    try {
        portfolioData = await fetchAPI('/api/portfolio');
        
        document.getElementById('header-cash').textContent = formatCurrency(portfolioData.cash);
        document.getElementById('header-portfolio-value').textContent = formatCurrency(portfolioData.total_value);
        document.getElementById('stat-total-value').textContent = formatCurrency(portfolioData.total_value);
        document.getElementById('stat-cash').textContent = formatCurrency(portfolioData.cash);
        
        const totalHoldings = portfolioData.holdings.reduce((sum, h) => sum + h.quantity, 0);
        document.getElementById('stat-plants-owned').textContent = totalHoldings;
        
        const totalGrowth = portfolioData.total_value - 10000;
        const growthElement = document.getElementById('stat-today-growth');
        growthElement.textContent = formatCurrency(totalGrowth);
        growthElement.className = `text-2xl font-mono font-semibold ${totalGrowth >= 0 ? 'price-up' : 'price-down'}`;
        
        const container = document.getElementById('portfolio-holdings');
        
        if (portfolioData.holdings.length === 0) {
            container.innerHTML = `
                <div class="p-8 text-center text-gray-500">
                    <p class="text-lg mb-2">Your garden is empty!</p>
                    <p>Start investing in plants to grow your wealth.</p>
                </div>
            `;
        } else {
            const prices = await fetchAPI('/api/market');
            const priceMap = Object.fromEntries(prices.map(p => [p.plant_id, p.price]));
            
            let html = '<table class="w-full"><thead class="bg-gray-50"><tr>';
            html += '<th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Plant</th>';
            html += '<th class="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">Quantity</th>';
            html += '<th class="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">Avg Price</th>';
            html += '<th class="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">Current</th>';
            html += '<th class="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">Total Value</th>';
            html += '<th class="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">P/L</th>';
            html += '<th class="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">Action</th>';
            html += '</tr></thead><tbody class="divide-y divide-gray-200">';
            
            portfolioData.holdings.forEach(holding => {
                const currentPrice = priceMap[holding.plant_id] || 0;
                const currentValue = holding.quantity * currentPrice;
                const profitLoss = currentValue - holding.total_invested;
                const profitLossPercent = (profitLoss / holding.total_invested) * 100;
                const plClass = profitLoss >= 0 ? 'price-up' : 'price-down';
                
                html += '<tr class="hover:bg-gray-50">';
                html += `<td class="px-6 py-4"><div class="flex items-center space-x-3"><span class="text-2xl">${getPlantEmoji(holding.plant_id)}</span><span class="font-medium">${getPlantName(holding.plant_id)}</span></div></td>`;
                html += `<td class="px-6 py-4 text-right font-mono">${holding.quantity}</td>`;
                html += `<td class="px-6 py-4 text-right font-mono">${formatCurrency(holding.avg_buy_price)}</td>`;
                html += `<td class="px-6 py-4 text-right font-mono">${formatCurrency(currentPrice)}</td>`;
                html += `<td class="px-6 py-4 text-right font-mono font-semibold">${formatCurrency(currentValue)}</td>`;
                html += `<td class="px-6 py-4 text-right font-mono font-semibold ${plClass}">${formatCurrency(profitLoss)} (${formatPercent(profitLossPercent)})</td>`;
                html += `<td class="px-6 py-4 text-right"><button onclick="openTradeModal('${holding.plant_id}')" class="text-sm bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700 transition-colors">Sell</button></td>`;
                html += '</tr>';
            });
            
            html += '</tbody></table>';
            container.innerHTML = html;
        }
    } catch (error) {
        console.error('Failed to update portfolio:', error);
    }
}

async function updateTransactions() {
    try {
        const transactions = await fetchAPI('/api/transactions');
        
        const container = document.getElementById('transactions-list');
        
        if (transactions.length === 0) {
            container.innerHTML = `
                <div class="p-8 text-center text-gray-500">
                    <p>No transactions yet.</p>
                </div>
            `;
        } else {
            let html = '<div class="divide-y divide-gray-200">';
            
            transactions.forEach(tx => {
                const date = new Date(tx.timestamp);
                const actionClass = tx.action === 'buy' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800';
                const actionText = tx.action === 'buy' ? 'Bought' : 'Sold';
                
                html += `
                    <div class="px-6 py-4 hover:bg-gray-50">
                        <div class="flex items-center justify-between">
                            <div class="flex items-center space-x-4">
                                <span class="text-2xl">${getPlantEmoji(tx.plant_id)}</span>
                                <div>
                                    <div class="flex items-center space-x-2">
                                        <span class="font-semibold">${tx.plant_name}</span>
                                        <span class="px-2 py-1 rounded-full text-xs font-medium ${actionClass}">${actionText}</span>
                                    </div>
                                    <p class="text-sm text-gray-500">${date.toLocaleString()}</p>
                                </div>
                            </div>
                            <div class="text-right">
                                <p class="font-mono font-semibold">${formatCurrency(tx.total)}</p>
                                <p class="text-sm text-gray-500">${tx.quantity} × ${formatCurrency(tx.price)}</p>
                            </div>
                        </div>
                    </div>
                `;
            });
            
            html += '</div>';
            container.innerHTML = html;
        }
    } catch (error) {
        console.error('Failed to update transactions:', error);
    }
}

async function openTradeModal(plantId) {
    currentPlantId = plantId;
    
    const prices = await fetchAPI('/api/market');
    const plant = prices.find(p => p.plant_id === plantId);
    
    if (!plant) return;
    
    document.getElementById('modal-plant-name').textContent = getPlantName(plantId);
    document.getElementById('modal-plant-id').textContent = plantId;
    document.getElementById('modal-current-price').textContent = formatCurrency(plant.price);
    
    const changeClass = plant.change_percent >= 0 ? 'price-up' : 'price-down';
    const changeElement = document.getElementById('modal-price-change');
    changeElement.textContent = formatPercent(plant.change_percent);
    changeElement.className = `text-sm font-medium ${changeClass}`;
    
    document.getElementById('modal-available-cash').textContent = formatCurrency(portfolioData.cash);
    document.getElementById('trade-quantity').value = 1;
    
    const holding = portfolioData.holdings.find(h => h.plant_id === plantId);
    if (holding) {
        document.getElementById('holdings-info').classList.remove('hidden');
        document.getElementById('modal-holdings').textContent = `${holding.quantity} plants`;
        document.getElementById('sell-button').disabled = false;
    } else {
        document.getElementById('holdings-info').classList.add('hidden');
        document.getElementById('sell-button').disabled = true;
    }
    
    updateTradeTotal();
    
    await loadPriceChart(plantId);
    
    document.getElementById('trade-modal').classList.remove('hidden');
}

function closeTradeModal() {
    document.getElementById('trade-modal').classList.add('hidden');
    currentPlantId = null;
    
    if (modalChart) {
        modalChart.destroy();
        modalChart = null;
    }
}

async function loadPriceChart(plantId) {
    try {
        const history = await fetchAPI(`/api/price-history/${plantId}`);
        
        const ctx = document.getElementById('modal-chart').getContext('2d');
        
        if (modalChart) {
            modalChart.destroy();
        }
        
        const labels = history.map(h => new Date(h.timestamp).toLocaleTimeString());
        const data = history.map(h => h.price);
        
        modalChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Price',
                    data: data,
                    borderColor: 'rgb(34, 197, 94)',
                    backgroundColor: 'rgba(34, 197, 94, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    borderWidth: 2,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        callbacks: {
                            label: (context) => formatCurrency(context.parsed.y)
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        ticks: {
                            callback: (value) => formatCurrency(value)
                        }
                    },
                    x: {
                        ticks: {
                            maxTicksLimit: 8
                        }
                    }
                }
            }
        });
    } catch (error) {
        console.error('Failed to load price chart:', error);
    }
}

function updateTradeTotal() {
    const quantity = parseInt(document.getElementById('trade-quantity').value) || 0;
    const prices = document.getElementById('modal-current-price').textContent;
    const price = parseFloat(prices.replace(/[$,]/g, ''));
    const total = quantity * price;
    
    document.getElementById('trade-total').textContent = formatCurrency(total);
}

async function executeBuy() {
    const quantity = parseInt(document.getElementById('trade-quantity').value);
    
    if (!quantity || quantity <= 0) {
        showToast('Please enter a valid quantity', 'error');
        return;
    }
    
    try {
        await postAPI('/api/buy', {
            plant_id: currentPlantId,
            quantity: quantity
        });
        
        showToast(`Successfully bought ${quantity} ${getPlantName(currentPlantId)}!`, 'success');
        closeTradeModal();
        
        await updatePortfolio();
        await updateTransactions();
    } catch (error) {
        showToast(error.message, 'error');
    }
}

async function executeSell() {
    const quantity = parseInt(document.getElementById('trade-quantity').value);
    
    if (!quantity || quantity <= 0) {
        showToast('Please enter a valid quantity', 'error');
        return;
    }
    
    try {
        await postAPI('/api/sell', {
            plant_id: currentPlantId,
            quantity: quantity
        });
        
        showToast(`Successfully sold ${quantity} ${getPlantName(currentPlantId)}!`, 'success');
        closeTradeModal();
        
        await updatePortfolio();
        await updateTransactions();
    } catch (error) {
        showToast(error.message, 'error');
    }
}

function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    const messageEl = document.getElementById('toast-message');
    
    messageEl.textContent = message;
    toast.className = `fixed bottom-4 right-4 rounded-lg shadow-lg border p-4 max-w-sm z-50 ${
        type === 'success' ? 'bg-green-50 border-green-200 text-green-800' :
        type === 'error' ? 'bg-red-50 border-red-200 text-red-800' :
        'bg-white border-gray-200'
    }`;
    
    toast.classList.remove('hidden');
    
    setTimeout(() => {
        toast.classList.add('hidden');
    }, 3000);
}

async function init() {
    await loadPlantConfigs();
    await updateMarket();
    await updatePortfolio();
    await updateTransactions();
    
    setInterval(async () => {
        await updateMarket();
        await updatePortfolio();
    }, UPDATE_INTERVAL);
}

document.addEventListener('DOMContentLoaded', init);

document.getElementById('trade-modal').addEventListener('click', (e) => {
    if (e.target.id === 'trade-modal') {
        closeTradeModal();
    }
});
