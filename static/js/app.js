// Sri Lanka Tourism Recommender - Frontend Application

// Global state
let map = null;
let markers = [];
let destinations = [];
let users = [];

// Type icons mapping
const typeIcons = {
    'beach': '🏖️',
    'cultural': '🏛️',
    'nature': '🌿',
    'adventure': '🧗',
    'urban': '🏙️',
    'wildlife': '🦁',
    'historical': '🏰',
    'religious': '🛕',
    'default': '📍'
};

// Initialize the application
document.addEventListener('DOMContentLoaded', async () => {
    initMap();
    await loadModelInfo();
    await loadUsers();
    await loadDestinations();
});

// Initialize Leaflet map
function initMap() {
    // Sri Lanka center coordinates
    map = L.map('map').setView([7.8731, 80.7718], 8);
    
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
    }).addTo(map);
    
    // Add click handler to set location
    map.on('click', (e) => {
        document.getElementById('latitude').value = e.latlng.lat.toFixed(4);
        document.getElementById('longitude').value = e.latlng.lng.toFixed(4);
        updateUserMarker(e.latlng.lat, e.latlng.lng);
    });
}

// Update user location marker
let userMarker = null;
function updateUserMarker(lat, lng) {
    if (userMarker) {
        map.removeLayer(userMarker);
    }
    userMarker = L.marker([lat, lng], {
        icon: L.divIcon({
            className: 'user-marker',
            html: '📍',
            iconSize: [30, 30],
            iconAnchor: [15, 30]
        })
    }).addTo(map);
}

// Load model information
async function loadModelInfo() {
    try {
        const response = await fetch('/api/model-info');
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Update status badge
        const statusBadge = document.getElementById('model-status');
        statusBadge.textContent = 'Ready';
        statusBadge.className = 'status-badge ready';
        
        // Update model info panel
        const modelInfo = document.getElementById('model-info');
        modelInfo.innerHTML = `
            <div class="model-item">
                <h4>Collaborative Filter</h4>
                <p>${data.models.collaborative_filter.type}</p>
                <div class="model-stat">
                    <span>Factors: ${data.models.collaborative_filter.n_factors}</span>
                    <span>${data.models.collaborative_filter.size_mb} MB</span>
                </div>
            </div>
            <div class="model-item">
                <h4>Content-Based Filter</h4>
                <p>${data.models.content_based_filter.type}</p>
                <div class="model-stat">
                    <span>Features: ${data.models.content_based_filter.max_features}</span>
                    <span>${data.models.content_based_filter.size_mb} MB</span>
                </div>
            </div>
            <div class="model-item">
                <h4>Context-Aware Engine</h4>
                <p>${data.models.context_aware_engine.type}</p>
                <div class="model-stat">
                    <span>Max Depth: ${data.models.context_aware_engine.max_depth}</span>
                    <span>${data.models.context_aware_engine.size_mb} MB</span>
                </div>
            </div>
            <div class="model-item" style="background: #d1fae5;">
                <h4>Total Size</h4>
                <div class="model-stat">
                    <span>${data.num_destinations} destinations</span>
                    <span><strong>${data.total_size_mb} MB</strong></span>
                </div>
            </div>
        `;
    } catch (error) {
        console.error('Failed to load model info:', error);
        const statusBadge = document.getElementById('model-status');
        statusBadge.textContent = 'Error';
        statusBadge.className = 'status-badge error';
    }
}

// Load users for dropdown
async function loadUsers() {
    try {
        const response = await fetch('/api/users');
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        users = data.users;
        const select = document.getElementById('user-select');
        
        users.forEach(user => {
            const option = document.createElement('option');
            option.value = user.user_id;
            const typeLabel = user.is_cold_start ? '❄️' : (user.visit_count >= 10 ? '⭐' : '👤');
            option.textContent = `${typeLabel} ${user.user_id.substring(0, 20)}... (${user.visit_count} visits)`;
            select.appendChild(option);
        });
        
        // Add change handler
        select.addEventListener('change', updateUserInfo);
    } catch (error) {
        console.error('Failed to load users:', error);
    }
}

// Update user info display
function updateUserInfo() {
    const userId = document.getElementById('user-select').value;
    const userInfo = document.getElementById('user-info');
    const typeBadge = document.getElementById('user-type-badge');
    const visitsSpan = document.getElementById('user-visits');
    
    if (userId === 'anonymous') {
        userInfo.classList.add('hidden');
        return;
    }
    
    const user = users.find(u => u.user_id === userId);
    if (user) {
        userInfo.classList.remove('hidden');
        
        if (user.is_cold_start) {
            typeBadge.textContent = 'Cold Start';
            typeBadge.className = 'badge cold-start';
        } else if (user.visit_count >= 10) {
            typeBadge.textContent = 'Frequent';
            typeBadge.className = 'badge frequent';
        } else {
            typeBadge.textContent = 'Regular';
            typeBadge.className = 'badge regular';
        }
        
        visitsSpan.textContent = user.visit_count;
    }
}

// Load destinations
async function loadDestinations() {
    try {
        const response = await fetch('/api/destinations');
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        destinations = data.destinations;
        
        // Add destination markers to map
        destinations.forEach(dest => {
            if (dest.latitude && dest.longitude) {
                const icon = typeIcons[dest.type] || typeIcons.default;
                const marker = L.marker([dest.latitude, dest.longitude], {
                    icon: L.divIcon({
                        className: 'dest-marker',
                        html: `<span style="font-size: 20px;">${icon}</span>`,
                        iconSize: [25, 25],
                        iconAnchor: [12, 12]
                    })
                });
                
                marker.bindPopup(`
                    <div class="popup-title">${dest.name}</div>
                    <div class="popup-meta">
                        ${dest.city} • ${dest.type}<br>
                        ⭐ ${dest.avg_rating.toFixed(1)} (${dest.review_count} reviews)
                    </div>
                `);
                
                marker.addTo(map);
                markers.push(marker);
            }
        });
    } catch (error) {
        console.error('Failed to load destinations:', error);
    }
}

// Use current location
function useCurrentLocation() {
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
            (position) => {
                const lat = position.coords.latitude;
                const lng = position.coords.longitude;
                document.getElementById('latitude').value = lat.toFixed(4);
                document.getElementById('longitude').value = lng.toFixed(4);
                map.setView([lat, lng], 10);
                updateUserMarker(lat, lng);
            },
            (error) => {
                alert('Unable to get your location. Please enter coordinates manually.');
            }
        );
    } else {
        alert('Geolocation is not supported by your browser.');
    }
}

// Get recommendations
async function getRecommendations() {
    const listContainer = document.getElementById('recommendations-list');
    listContainer.innerHTML = '<div class="loading-spinner"></div>';
    
    // Gather form data
    const requestData = {
        user_id: document.getElementById('user-select').value,
        latitude: parseFloat(document.getElementById('latitude').value),
        longitude: parseFloat(document.getElementById('longitude').value),
        weather: document.getElementById('weather').value,
        season: document.getElementById('season').value,
        is_holiday: document.getElementById('is-holiday').checked,
        is_peak_season: document.getElementById('is-peak-season').checked,
        travel_style: document.getElementById('travel-style').value || null,
        top_k: parseInt(document.getElementById('top-k').value),
        voting_strategy: document.getElementById('voting-strategy').value
    };
    
    // Add optional filters
    const maxDistance = document.getElementById('max-distance').value;
    if (maxDistance) requestData.max_distance = parseFloat(maxDistance);
    
    const budgetMin = document.getElementById('budget-min').value;
    const budgetMax = document.getElementById('budget-max').value;
    if (budgetMin) requestData.budget_min = parseFloat(budgetMin);
    if (budgetMax) requestData.budget_max = parseFloat(budgetMax);
    
    try {
        const response = await fetch('/api/recommend', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Update results count and inference time
        document.getElementById('results-count').textContent = `${data.count} results`;
        document.getElementById('inference-time').textContent = `⚡ ${data.inference_time_ms}ms`;
        
        // Update weight display based on context
        updateWeightDisplay(data.user_type, data.context);
        
        // Render recommendations
        renderRecommendations(data.recommendations);
        
        // Update map to show recommended destinations
        highlightRecommendations(data.recommendations);
        
    } catch (error) {
        console.error('Failed to get recommendations:', error);
        listContainer.innerHTML = `
            <div class="empty-state">
                <span class="empty-icon">❌</span>
                <p>Error: ${error.message}</p>
            </div>
        `;
    }
}

// Update weight display based on context
function updateWeightDisplay(userType, context) {
    let cfWeight = 35;
    let cbWeight = 25;
    let caWeight = 25;
    
    // Apply adjustments
    if (userType === 'cold_start') {
        cfWeight -= 20;
        cbWeight += 20;
    }
    
    if (context.weather === 'rainy' || context.weather === 'stormy') {
        caWeight += 15;
    }
    
    if (context.is_peak_season) {
        cfWeight += 10;
        cbWeight -= 10;
    }
    
    // Update bars
    document.getElementById('cf-weight').style.width = `${cfWeight}%`;
    document.getElementById('cb-weight').style.width = `${cbWeight}%`;
    document.getElementById('ca-weight').style.width = `${caWeight}%`;
    
    // Update values
    const weightBars = document.querySelectorAll('.weight-bar');
    weightBars[0].querySelector('.weight-value').textContent = `${cfWeight}%`;
    weightBars[1].querySelector('.weight-value').textContent = `${cbWeight}%`;
    weightBars[2].querySelector('.weight-value').textContent = `${caWeight}%`;
}

// Render recommendations list
function renderRecommendations(recommendations) {
    const listContainer = document.getElementById('recommendations-list');
    
    if (recommendations.length === 0) {
        listContainer.innerHTML = `
            <div class="empty-state">
                <span class="empty-icon">🔍</span>
                <p>No recommendations found matching your criteria.</p>
            </div>
        `;
        return;
    }
    
    listContainer.innerHTML = recommendations.map((rec, index) => {
        const dest = destinations.find(d => d.id === rec.destination_id) || {};
        const icon = typeIcons[dest.type] || typeIcons.default;
        const rankClass = index < 3 ? 'top-3' : '';
        
        return `
            <div class="recommendation-card" onclick="focusDestination('${rec.destination_id}')">
                <div class="rank-badge ${rankClass}">${index + 1}</div>
                <div class="card-content">
                    <div class="card-title">
                        <span class="type-icon">${icon}</span>
                        ${rec.name}
                    </div>
                    <div class="card-meta">
                        <span>${dest.city || 'Sri Lanka'}</span>
                        <span>${dest.type || 'destination'}</span>
                        ${rec.distance_km ? `<span>📍 ${rec.distance_km} km</span>` : ''}
                    </div>
                    <div class="card-explanation">${rec.explanation}</div>
                </div>
                <div class="card-score">
                    <div class="score-value">${(rec.score * 100).toFixed(0)}</div>
                    <div class="score-label">Score</div>
                </div>
            </div>
        `;
    }).join('');
}

// Highlight recommended destinations on map
function highlightRecommendations(recommendations) {
    // Clear existing highlight markers
    markers.forEach(marker => {
        marker.setOpacity(0.3);
    });
    
    // Highlight recommended destinations
    recommendations.forEach((rec, index) => {
        const dest = destinations.find(d => d.id === rec.destination_id);
        if (dest && dest.latitude && dest.longitude) {
            const marker = L.marker([dest.latitude, dest.longitude], {
                icon: L.divIcon({
                    className: 'rec-marker',
                    html: `<div style="background: #2563eb; color: white; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 12px; border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.3);">${index + 1}</div>`,
                    iconSize: [28, 28],
                    iconAnchor: [14, 14]
                }),
                zIndexOffset: 1000 - index
            });
            
            marker.bindPopup(`
                <div class="popup-title">#${index + 1} ${rec.name}</div>
                <div class="popup-meta">
                    Score: ${(rec.score * 100).toFixed(0)}%<br>
                    ${rec.explanation}
                </div>
            `);
            
            marker.addTo(map);
        }
    });
    
    // Fit map to show all recommendations
    if (recommendations.length > 0) {
        const bounds = [];
        recommendations.forEach(rec => {
            const dest = destinations.find(d => d.id === rec.destination_id);
            if (dest && dest.latitude && dest.longitude) {
                bounds.push([dest.latitude, dest.longitude]);
            }
        });
        if (bounds.length > 0) {
            map.fitBounds(bounds, { padding: [50, 50] });
        }
    }
}

// Focus on a specific destination
function focusDestination(destId) {
    const dest = destinations.find(d => d.id === destId);
    if (dest && dest.latitude && dest.longitude) {
        map.setView([dest.latitude, dest.longitude], 12);
    }
}
