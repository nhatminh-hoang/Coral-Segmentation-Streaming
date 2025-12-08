/**
 * CoralScapes Production Monitor
 * Main Application JavaScript
 */

// ==============================
// State Management
// ==============================
const AppState = {
    ws: null,
    labels: [],
    activeLabels: new Set(),
    currentPeriod: 'day',
    language: 'Tiếng Việt',
    skipFrames: 3,
    isConnected: false,
    chart: null,
    timeseriesChart: null,  // Line chart for trends
    timeseriesData: { timestamps: [], datasets: [] },  // Real-time data buffer
    reconnectAttempts: 0,
    maxReconnectAttempts: 5,
    // Hover label data
    predMap: null,
    predShape: [0, 0],
    predScale: 4,
    id2label: {},  // Will be populated from labels
    // Fullscreen state
    fullscreenMode: null  // null, 'stream', or 'stats'
};

// ==============================
// DOM Elements
// ==============================
const Elements = {
    streamImage: document.getElementById('stream-image'),
    streamOverlay: document.getElementById('stream-overlay'),
    streamContainer: document.querySelector('.stream-container'),
    streamSection: document.getElementById('stream-section'),
    rightPanel: document.getElementById('right-panel'),
    mainContent: document.querySelector('.main-content'),
    streamFullscreenBtn: document.getElementById('stream-fullscreen-btn'),
    statsFullscreenBtn: document.getElementById('stats-fullscreen-btn'),
    timeseriesFullscreenBtn: document.getElementById('timeseries-fullscreen-btn'),
    connectionStatus: document.getElementById('connection-status'),
    fpsValue: document.getElementById('fps-value'),
    labelsGrid: document.getElementById('labels-grid'),
    statsChart: document.getElementById('stats-chart'),
    timeseriesChart: document.getElementById('timeseries-chart'),
    settingsBtn: document.getElementById('settings-btn'),
    settingsPanel: document.getElementById('settings-panel'),
    settingsOverlay: document.getElementById('settings-overlay'),
    closeSettings: document.getElementById('close-settings'),
    skipSlider: document.getElementById('skip-slider'),
    skipValue: document.getElementById('skip-value'),
    cameraUrl: document.getElementById('camera-url'),
    applyCameraUrl: document.getElementById('apply-camera-url'),
    selectAll: document.getElementById('select-all'),
    selectNone: document.getElementById('select-none'),
    selectCoral: document.getElementById('select-coral'),
    periodBtns: document.querySelectorAll('.period-btn')
};

// ==============================
// Initialization
// ==============================
async function init() {
    console.log('🪸 Initializing CoralScapes Monitor...');

    // Load labels
    await loadLabels();

    // Initialize charts
    initChart();
    initTimeseriesChart();

    // Load initial statistics and timeseries
    loadStats(AppState.currentPeriod);
    loadTimeseries();

    // Setup event listeners
    setupEventListeners();

    // Setup hover tooltip
    setupHoverTooltip();

    // Setup fullscreen toggle
    setupFullscreenToggle();

    // Connect WebSocket
    connectWebSocket();
}

// ==============================
// API Functions
// ==============================
async function loadLabels() {
    try {
        const response = await fetch('/api/labels');
        const data = await response.json();
        AppState.labels = data.labels;

        // Initialize active labels and id2label map
        AppState.labels.forEach(label => {
            if (label.active) {
                AppState.activeLabels.add(label.name);
            }
            AppState.id2label[label.id] = label;
        });

        renderLabelButtons();
    } catch (error) {
        console.error('Failed to load labels:', error);
    }
}

async function loadStats(period) {
    try {
        const response = await fetch(`/api/stats/${period}`);
        const data = await response.json();
        updateChart(data);
    } catch (error) {
        console.error('Failed to load stats:', error);
    }
}

async function toggleLabel(labelName) {
    try {
        const response = await fetch(`/api/toggle_label/${encodeURIComponent(labelName)}`, {
            method: 'POST'
        });
        const data = await response.json();

        if (data.active) {
            AppState.activeLabels.add(labelName);
        } else {
            AppState.activeLabels.delete(labelName);
        }

        // Also send to WebSocket for real-time update
        if (AppState.ws && AppState.ws.readyState === WebSocket.OPEN) {
            AppState.ws.send(JSON.stringify({
                type: 'toggle_label',
                label: labelName
            }));
        }

        updateLabelButtonState(labelName);

        // Refresh charts to apply filter
        loadStats(AppState.currentPeriod);
        loadTimeseries();
    } catch (error) {
        console.error('Failed to toggle label:', error);
    }
}

// ==============================
// WebSocket Connection
// ==============================
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/stream`;

    console.log('Connecting to WebSocket:', wsUrl);
    updateConnectionStatus('connecting');

    AppState.ws = new WebSocket(wsUrl);

    AppState.ws.onopen = () => {
        console.log('✅ WebSocket connected');
        AppState.isConnected = true;
        AppState.reconnectAttempts = 0;
        updateConnectionStatus('online');
    };

    AppState.ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleWebSocketMessage(data);
        } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
        }
    };

    AppState.ws.onclose = () => {
        console.log('❌ WebSocket disconnected');
        AppState.isConnected = false;
        updateConnectionStatus('offline');

        // Attempt reconnection
        if (AppState.reconnectAttempts < AppState.maxReconnectAttempts) {
            AppState.reconnectAttempts++;
            console.log(`Reconnecting... (attempt ${AppState.reconnectAttempts})`);
            setTimeout(connectWebSocket, 2000);
        }
    };

    AppState.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        updateConnectionStatus('offline');
    };
}

function handleWebSocketMessage(data) {
    switch (data.type) {
        case 'frame':
            // Update stream image
            Elements.streamImage.src = `data:image/jpeg;base64,${data.image}`;
            Elements.streamOverlay.classList.add('hidden');

            // Update FPS
            Elements.fpsValue.textContent = data.fps.toFixed(1);

            // Store prediction map for hover labels
            if (data.pred_map) {
                // Decode base64 prediction map
                const binaryString = atob(data.pred_map);
                const bytes = new Uint8Array(binaryString.length);
                for (let i = 0; i < binaryString.length; i++) {
                    bytes[i] = binaryString.charCodeAt(i);
                }
                AppState.predMap = bytes;
                AppState.predShape = data.pred_shape;
                AppState.predScale = data.pred_scale;
            }
            break;

        case 'status':
            console.log('Status:', data.message);
            break;

        case 'error':
            console.error('Stream error:', data.message);
            break;
    }
}

function updateConnectionStatus(status) {
    const statusDot = Elements.connectionStatus.querySelector('.status-dot');
    const statusText = Elements.connectionStatus.querySelector('.status-text');

    statusDot.className = 'status-dot';

    switch (status) {
        case 'online':
            statusDot.classList.add('online');
            statusText.textContent = 'Connected';
            break;
        case 'offline':
            statusDot.classList.add('offline');
            statusText.textContent = 'Disconnected';
            break;
        case 'connecting':
            statusText.textContent = 'Connecting...';
            break;
    }
}

// ==============================
// UI Rendering
// ==============================
function renderLabelButtons() {
    Elements.labelsGrid.innerHTML = '';

    AppState.labels.forEach(label => {
        const btn = document.createElement('button');
        btn.className = 'label-btn';
        btn.dataset.label = label.name;

        if (AppState.activeLabels.has(label.name)) {
            btn.classList.add('active');
            btn.style.color = label.color;
        }

        btn.innerHTML = `
            <span class="color-dot" style="background-color: ${label.color}"></span>
            <span class="label-name">${AppState.language === 'English' ? label.name : label.name_vn}</span>
        `;

        btn.addEventListener('click', () => toggleLabel(label.name));
        btn.addEventListener('mouseenter', () => highlightLabel(label.name, true));
        btn.addEventListener('mouseleave', () => highlightLabel(label.name, false));

        Elements.labelsGrid.appendChild(btn);
    });
}

function updateLabelButtonState(labelName) {
    const btn = Elements.labelsGrid.querySelector(`[data-label="${labelName}"]`);
    if (!btn) return;

    const label = AppState.labels.find(l => l.name === labelName);
    if (!label) return;

    if (AppState.activeLabels.has(labelName)) {
        btn.classList.add('active');
        btn.style.color = label.color;
    } else {
        btn.classList.remove('active');
        btn.style.color = '';
    }
}

function updateLiveLabels(visibleLabels) {
    Elements.liveLabels.innerHTML = '';

    visibleLabels.forEach(item => {
        const div = document.createElement('div');
        div.className = 'live-label-item';
        div.innerHTML = `
            <span class="live-label-color" style="background-color: ${item.color}"></span>
            <span class="live-label-name">${AppState.language === 'English' ? item.label : getLabelVietnamese(item.label)}</span>
        `;
        Elements.liveLabels.appendChild(div);
    });
}

function getLabelVietnamese(englishName) {
    const label = AppState.labels.find(l => l.name === englishName);
    return label ? label.name_vn : englishName;
}

function formatCount(count) {
    if (count >= 1000000) return (count / 1000000).toFixed(1) + 'M';
    if (count >= 1000) return (count / 1000).toFixed(1) + 'K';
    return count.toString();
}

function highlightLabel(labelName, highlight) {
    // Could be used for visual feedback when hovering
    // For now, just log it
    // console.log(`Highlight ${labelName}: ${highlight}`);
}

// ==============================
// Chart Functions
// ==============================
function initChart() {
    const ctx = Elements.statsChart.getContext('2d');

    AppState.chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Pixel Count',
                data: [],
                backgroundColor: [],
                borderRadius: 4,
                borderSkipped: false
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
                    backgroundColor: 'rgba(26, 35, 50, 0.95)',
                    titleColor: '#fff',
                    bodyColor: '#a0aec0',
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    borderWidth: 1,
                    cornerRadius: 8,
                    padding: 12,
                    callbacks: {
                        label: function (context) {
                            return `Count: ${formatCount(context.raw)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        color: '#6b7280',
                        font: {
                            size: 10
                        },
                        maxRotation: 45,
                        minRotation: 45
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    },
                    ticks: {
                        color: '#6b7280',
                        font: {
                            size: 10
                        },
                        callback: function (value) {
                            return formatCount(value);
                        }
                    }
                }
            }
        }
    });
}

function updateChart(data) {
    if (!AppState.chart || !data.labels || !data.values) return;

    // Filter to only show active labels
    const filteredIndices = [];
    const filteredLabels = [];
    const filteredValues = [];

    data.labels.forEach((labelName, index) => {
        if (AppState.activeLabels.has(labelName)) {
            filteredIndices.push(index);
            filteredLabels.push(labelName);
            filteredValues.push(data.values[index]);
        }
    });

    // Get colors for each filtered label
    const colors = filteredLabels.map(labelName => {
        const label = AppState.labels.find(l => l.name === labelName);
        return label ? label.color : '#6b7280';
    });

    // Translate labels if needed
    const displayLabels = filteredLabels.map(labelName => {
        if (AppState.language === 'English') return labelName;
        const label = AppState.labels.find(l => l.name === labelName);
        return label ? label.name_vn : labelName;
    });

    AppState.chart.data.labels = displayLabels;
    AppState.chart.data.datasets[0].data = filteredValues;
    AppState.chart.data.datasets[0].backgroundColor = colors;
    AppState.chart.update('none');
}

// ==============================
// Time-Series Line Chart
// ==============================
// Track highlighted dataset for legend interaction
let highlightedDatasetIndex = null;  // null = no highlight, number = locked highlight

function setDatasetHighlight(chart, highlightIndex) {
    // Set opacity for all datasets based on highlight state
    chart.data.datasets.forEach((dataset, index) => {
        if (highlightIndex === null) {
            // No highlight - restore all to full opacity
            dataset.borderColor = dataset._originalColor || dataset.borderColor;
            dataset.backgroundColor = (dataset._originalColor || dataset.borderColor) + '20';
            dataset.borderWidth = 2;
        } else if (index === highlightIndex) {
            // This is the highlighted dataset - full opacity and thicker
            dataset.borderColor = dataset._originalColor || dataset.borderColor;
            dataset.backgroundColor = (dataset._originalColor || dataset.borderColor) + '40';
            dataset.borderWidth = 3;
        } else {
            // Other datasets - fade out
            dataset.borderColor = (dataset._originalColor || dataset.borderColor) + '30';
            dataset.backgroundColor = (dataset._originalColor || dataset.borderColor) + '10';
            dataset.borderWidth = 1;
        }
    });
    chart.update('none');
}

function initTimeseriesChart() {
    if (!Elements.timeseriesChart) {
        console.warn('Timeseries chart canvas not found');
        return;
    }

    const ctx = Elements.timeseriesChart.getContext('2d');

    AppState.timeseriesChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: []
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 200  // Fast updates for real-time
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: '#a0aec0',
                        usePointStyle: true,
                        pointStyle: 'circle',
                        padding: 12,
                        font: {
                            size: 10
                        }
                    },
                    onHover: function (event, legendItem, legend) {
                        // Only apply hover highlight if no locked highlight
                        if (highlightedDatasetIndex === null) {
                            const chart = legend.chart;
                            setDatasetHighlight(chart, legendItem.datasetIndex);
                        }
                        // Change cursor to pointer
                        event.native.target.style.cursor = 'pointer';
                    },
                    onLeave: function (event, legendItem, legend) {
                        // Only restore if no locked highlight
                        if (highlightedDatasetIndex === null) {
                            const chart = legend.chart;
                            setDatasetHighlight(chart, null);
                        }
                    },
                    onClick: function (event, legendItem, legend) {
                        const chart = legend.chart;
                        const clickedIndex = legendItem.datasetIndex;

                        if (highlightedDatasetIndex === clickedIndex) {
                            // Clicking same label again - unlock highlight
                            highlightedDatasetIndex = null;
                            setDatasetHighlight(chart, null);
                        } else {
                            // Lock highlight to this dataset
                            highlightedDatasetIndex = clickedIndex;
                            setDatasetHighlight(chart, clickedIndex);
                        }
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(26, 35, 50, 0.95)',
                    titleColor: '#fff',
                    bodyColor: '#a0aec0',
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    borderWidth: 1,
                    cornerRadius: 8,
                    padding: 12,
                    callbacks: {
                        label: function (context) {
                            return `${context.dataset.label}: ${formatCount(context.raw)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    display: true,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    },
                    ticks: {
                        color: '#6b7280',
                        font: { size: 9 },
                        maxTicksLimit: 8,
                        maxRotation: 0
                    }
                },
                y: {
                    display: true,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    },
                    ticks: {
                        color: '#6b7280',
                        font: { size: 9 },
                        callback: function (value) {
                            return formatCount(value);
                        }
                    },
                    beginAtZero: true
                }
            },
            interaction: {
                mode: 'nearest',
                axis: 'x',
                intersect: false
            }
        }
    });

    console.log('✅ Time-series chart initialized');
}

async function loadTimeseries() {
    try {
        const response = await fetch('/api/timeseries?limit=50&labels_limit=20');
        const data = await response.json();
        updateTimeseriesChart(data);
    } catch (error) {
        console.error('Failed to load timeseries:', error);
    }
}

function updateTimeseriesChart(data) {
    if (!AppState.timeseriesChart || !data.timestamps) return;

    // Reset highlight when datasets change
    highlightedDatasetIndex = null;

    // Filter datasets to only show active labels
    const filteredDatasets = data.datasets.filter(ds =>
        AppState.activeLabels.has(ds.label)
    );

    // Build Chart.js datasets from filtered API data
    const datasets = filteredDatasets.map(ds => {
        const displayLabel = AppState.language === 'English' ? ds.label : ds.label_vn;
        return {
            label: displayLabel,
            data: ds.data,
            borderColor: ds.color,
            backgroundColor: ds.color + '20',  // 20% opacity for fill
            _originalColor: ds.color,  // Store original color for highlight restore
            tension: 0.3,
            fill: false,
            pointRadius: 2,
            pointHoverRadius: 4,
            borderWidth: 2
        };
    });

    AppState.timeseriesChart.data.labels = data.timestamps;
    AppState.timeseriesChart.data.datasets = datasets;
    AppState.timeseriesChart.update('none');
}

function addTimeseriesDataPoint(frameStats) {
    // Add real-time data point from current frame
    if (!AppState.timeseriesChart || !frameStats || Object.keys(frameStats).length === 0) return;

    const now = new Date();
    const timestamp = now.toTimeString().slice(0, 8);  // HH:MM:SS

    // Get current labels from chart
    const currentDatasets = AppState.timeseriesChart.data.datasets;
    if (currentDatasets.length === 0) return;

    // Add timestamp
    AppState.timeseriesChart.data.labels.push(timestamp);

    // Add data point for each dataset
    currentDatasets.forEach(ds => {
        // Find matching stat by label
        const matchingLabel = AppState.labels.find(l =>
            l.name === ds.label || l.name_vn === ds.label
        );
        const count = matchingLabel ? (frameStats[matchingLabel.name] || 0) : 0;
        ds.data.push(count);
    });

    // Keep only last 50 points
    const maxPoints = 50;
    if (AppState.timeseriesChart.data.labels.length > maxPoints) {
        AppState.timeseriesChart.data.labels.shift();
        currentDatasets.forEach(ds => ds.data.shift());
    }

    AppState.timeseriesChart.update('none');
}

// ==============================
// Event Listeners
// ==============================
function setupEventListeners() {
    // Settings panel
    Elements.settingsBtn.addEventListener('click', openSettings);
    Elements.closeSettings.addEventListener('click', closeSettings);
    Elements.settingsOverlay.addEventListener('click', closeSettings);

    // Skip slider
    Elements.skipSlider.addEventListener('input', (e) => {
        const value = parseInt(e.target.value);
        Elements.skipValue.textContent = value;
        AppState.skipFrames = value;

        // Send to WebSocket
        if (AppState.ws && AppState.ws.readyState === WebSocket.OPEN) {
            AppState.ws.send(JSON.stringify({
                type: 'set_skip',
                skip: value
            }));
        }
    });

    // Language selection
    document.querySelectorAll('input[name="language"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            AppState.language = e.target.value;
            renderLabelButtons();
            loadStats(AppState.currentPeriod);
        });
    });

    // Period buttons
    Elements.periodBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            Elements.periodBtns.forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            AppState.currentPeriod = e.target.dataset.period;
            loadStats(AppState.currentPeriod);
        });
    });

    // Label action buttons
    Elements.selectAll.addEventListener('click', () => selectAllLabels(true));
    Elements.selectNone.addEventListener('click', () => selectAllLabels(false));
    Elements.selectCoral.addEventListener('click', selectCoralLabels);

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            closeSettings();
        }
    });
}

function openSettings() {
    Elements.settingsPanel.classList.add('open');
    Elements.settingsOverlay.classList.add('visible');
}

function closeSettings() {
    Elements.settingsPanel.classList.remove('open');
    Elements.settingsOverlay.classList.remove('visible');
}

async function selectAllLabels(active) {
    const labels = {};
    AppState.labels.forEach(label => {
        labels[label.name] = active;
        if (active) {
            AppState.activeLabels.add(label.name);
        } else {
            AppState.activeLabels.delete(label.name);
        }
    });

    try {
        await fetch('/api/set_labels', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(labels)
        });
        renderLabelButtons();

        // Refresh charts to apply filter
        loadStats(AppState.currentPeriod);
        loadTimeseries();
    } catch (error) {
        console.error('Failed to set labels:', error);
    }
}

function selectCoralLabels() {
    const coralKeywords = ['coral', 'branching', 'massive', 'meandering', 'acropora', 'pocillopora', 'stylophora', 'millepora'];

    AppState.labels.forEach(label => {
        const isCoral = coralKeywords.some(keyword =>
            label.name.toLowerCase().includes(keyword)
        );

        if (isCoral) {
            AppState.activeLabels.add(label.name);
        } else {
            AppState.activeLabels.delete(label.name);
        }
    });

    // Update server
    const labels = {};
    AppState.labels.forEach(label => {
        labels[label.name] = AppState.activeLabels.has(label.name);
    });

    fetch('/api/set_labels', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(labels)
    }).then(() => {
        renderLabelButtons();

        // Refresh charts to apply filter
        loadStats(AppState.currentPeriod);
        loadTimeseries();
    });
}

// ==============================
// Hover Label Tooltip
// ==============================
function setupHoverTooltip() {
    // Create tooltip element
    const tooltip = document.createElement('div');
    tooltip.id = 'hover-tooltip';
    tooltip.className = 'hover-tooltip';
    tooltip.style.cssText = `
        position: fixed;
        padding: 8px 12px;
        background: rgba(10, 15, 26, 0.95);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 8px;
        color: white;
        font-size: 14px;
        pointer-events: none;
        z-index: 1000;
        opacity: 0;
        transition: opacity 0.15s ease;
        backdrop-filter: blur(8px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    `;
    document.body.appendChild(tooltip);

    Elements.streamImage.addEventListener('mousemove', (e) => {
        if (!AppState.predMap || !AppState.predShape) return;

        const rect = Elements.streamImage.getBoundingClientRect();
        const imgWidth = Elements.streamImage.naturalWidth || rect.width;
        const imgHeight = Elements.streamImage.naturalHeight || rect.height;

        // Calculate relative position on image
        const relX = (e.clientX - rect.left) / rect.width;
        const relY = (e.clientY - rect.top) / rect.height;

        // Scale to prediction map coordinates
        const predH = AppState.predShape[0];
        const predW = AppState.predShape[1];
        const scale = AppState.predScale;

        const px = Math.floor(relX * predW);
        const py = Math.floor(relY * predH);

        if (px >= 0 && px < predW && py >= 0 && py < predH) {
            const idx = py * predW + px;
            const classId = AppState.predMap[idx];

            const label = AppState.id2label[classId];
            if (label) {
                const displayName = AppState.language === 'English' ? label.name : label.name_vn;
                tooltip.innerHTML = `
                    <span style="display: inline-block; width: 12px; height: 12px; background: ${label.color}; border-radius: 3px; margin-right: 8px;"></span>
                    ${displayName}
                `;
                tooltip.style.opacity = '1';
                tooltip.style.left = (e.clientX + 15) + 'px';
                tooltip.style.top = (e.clientY + 15) + 'px';
            }
        }
    });

    Elements.streamImage.addEventListener('mouseleave', () => {
        tooltip.style.opacity = '0';
    });
}

// ==============================
// Fullscreen Toggle
// ==============================
function setupFullscreenToggle() {
    const streamBtn = Elements.streamFullscreenBtn;
    const statsBtn = Elements.statsFullscreenBtn;
    const mainContent = Elements.mainContent;

    if (!streamBtn || !statsBtn || !mainContent) {
        console.warn('Fullscreen toggle elements not found');
        return;
    }

    // Toggle stream fullscreen (make stream bigger, stats smaller)
    streamBtn.addEventListener('click', () => {
        if (AppState.fullscreenMode === 'stream') {
            // Exit fullscreen
            mainContent.classList.remove('stream-fullscreen');
            streamBtn.classList.remove('active');
            AppState.fullscreenMode = null;
        } else {
            // Enter stream fullscreen
            mainContent.classList.remove('stats-fullscreen');
            mainContent.classList.add('stream-fullscreen');
            streamBtn.classList.add('active');
            statsBtn.classList.remove('active');
            AppState.fullscreenMode = 'stream';
        }

        // Resize charts after transition
        setTimeout(() => {
            if (AppState.chart) AppState.chart.resize();
            if (AppState.timeseriesChart) AppState.timeseriesChart.resize();
        }, 300);
    });

    // Toggle stats fullscreen (make stats bigger, stream smaller)
    statsBtn.addEventListener('click', () => {
        if (AppState.fullscreenMode === 'stats') {
            // Exit fullscreen
            mainContent.classList.remove('stats-fullscreen');
            statsBtn.classList.remove('active');
            AppState.fullscreenMode = null;
        } else {
            // Enter stats fullscreen
            mainContent.classList.remove('stream-fullscreen');
            mainContent.classList.add('stats-fullscreen');
            statsBtn.classList.add('active');
            streamBtn.classList.remove('active');
            AppState.fullscreenMode = 'stats';
        }

        // Resize charts after transition
        setTimeout(() => {
            if (AppState.chart) AppState.chart.resize();
            if (AppState.timeseriesChart) AppState.timeseriesChart.resize();
        }, 300);
    });

    // Toggle timeseries fullscreen (also makes stats bigger, same as stats fullscreen)
    const timeseriesBtn = Elements.timeseriesFullscreenBtn;
    if (timeseriesBtn) {
        timeseriesBtn.addEventListener('click', () => {
            if (AppState.fullscreenMode === 'stats') {
                // Exit fullscreen
                mainContent.classList.remove('stats-fullscreen');
                timeseriesBtn.classList.remove('active');
                statsBtn.classList.remove('active');
                AppState.fullscreenMode = null;
            } else {
                // Enter stats fullscreen (expands both charts)
                mainContent.classList.remove('stream-fullscreen');
                mainContent.classList.add('stats-fullscreen');
                timeseriesBtn.classList.add('active');
                statsBtn.classList.add('active');
                streamBtn.classList.remove('active');
                AppState.fullscreenMode = 'stats';
            }

            // Resize charts after transition
            setTimeout(() => {
                if (AppState.chart) AppState.chart.resize();
                if (AppState.timeseriesChart) AppState.timeseriesChart.resize();
            }, 300);
        });
    }

    console.log('✅ Fullscreen toggle initialized');
}

// ==============================
// Camera URL Handler
// ==============================
function setupCameraUrlHandler() {
    if (Elements.applyCameraUrl) {
        Elements.applyCameraUrl.addEventListener('click', async () => {
            const url = Elements.cameraUrl.value.trim();

            try {
                await fetch(`/api/camera_url?url=${encodeURIComponent(url)}`, {
                    method: 'POST'
                });

                // Also notify via WebSocket to restart stream
                if (AppState.ws && AppState.ws.readyState === WebSocket.OPEN) {
                    AppState.ws.send(JSON.stringify({
                        type: 'set_camera_url',
                        url: url || null
                    }));
                }

                console.log('Camera URL applied:', url || 'Using sample videos');
            } catch (error) {
                console.error('Failed to set camera URL:', error);
            }
        });
    }
}

// ==============================
// Start Application
// ==============================
document.addEventListener('DOMContentLoaded', () => {
    init();
    setupCameraUrlHandler();
});
