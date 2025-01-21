// Dashboard de IA - Admin Kiosk

let socket = null;
let realtimeData = {
    predictions: [],
    anomalies: 0,
    totalPredictions: 0
};

document.addEventListener('DOMContentLoaded', function() {
    // Inicializar componentes
    initializeDashboard();
    initializeWebSocket();
    
    // Event listeners
    document.getElementById('refreshData').addEventListener('click', refreshData);
    document.getElementById('modelVersion').addEventListener('change', refreshData);
    document.getElementById('startDate').addEventListener('change', refreshData);
    document.getElementById('endDate').addEventListener('change', refreshData);
});

function initializeDashboard() {
    // Configurar fechas iniciales
    const endDate = new Date();
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - 7);
    
    document.getElementById('startDate').value = startDate.toISOString().split('T')[0];
    document.getElementById('endDate').value = endDate.toISOString().split('T')[0];
    
    // Cargar datos iniciales
    refreshData();
}

function initializeWebSocket() {
    socket = io();
    
    socket.on('connect', () => {
        console.log('WebSocket conectado');
        updateConnectionStatus(true);
    });
    
    socket.on('disconnect', () => {
        console.log('WebSocket desconectado');
        updateConnectionStatus(false);
    });
    
    socket.on('kiosk_prediction', (prediction) => {
        // Actualizar datos en tiempo real
        updateRealtimeData(prediction);
        
        // Actualizar UI
        updateRealtimeMetrics();
        updateRealtimePredictionsList(prediction);
        
        // Si es anomalía, mostrar alerta
        if (prediction.is_anomaly) {
            showAnomalyAlert(prediction);
        }
    });
}

function updateConnectionStatus(connected) {
    const statusElement = document.getElementById('connectionStatus');
    if (statusElement) {
        statusElement.className = connected ? 'status-connected' : 'status-disconnected';
        statusElement.textContent = connected ? 'Conectado' : 'Desconectado';
    }
}

function updateRealtimeData(prediction) {
    // Mantener solo las últimas 100 predicciones
    realtimeData.predictions.unshift(prediction);
    if (realtimeData.predictions.length > 100) {
        realtimeData.predictions.pop();
    }
    
    // Actualizar contadores
    realtimeData.totalPredictions++;
    if (prediction.is_anomaly) {
        realtimeData.anomalies++;
    }
}

function updateRealtimeMetrics() {
    // Actualizar contadores en tiempo real
    const totalElement = document.getElementById('totalPredictions');
    const anomaliesElement = document.getElementById('totalAnomalies');
    const rateElement = document.getElementById('anomalyRate');
    
    if (totalElement) {
        totalElement.textContent = realtimeData.totalPredictions;
    }
    if (anomaliesElement) {
        anomaliesElement.textContent = realtimeData.anomalies;
    }
    if (rateElement && realtimeData.totalPredictions > 0) {
        const rate = (realtimeData.anomalies / realtimeData.totalPredictions * 100).toFixed(2);
        rateElement.textContent = `${rate}%`;
    }
}

function updateRealtimePredictionsList(prediction) {
    const container = document.getElementById('realtimePredictions');
    if (!container) return;
    
    // Crear elemento para la nueva predicción
    const element = document.createElement('div');
    element.className = `prediction-item ${prediction.is_anomaly ? 'anomaly' : 'normal'}`;
    element.innerHTML = `
        <div class="prediction-header">
            <span class="kiosk-id">Kiosk #${prediction.kiosk_id}</span>
            <span class="timestamp">${new Date(prediction.timestamp).toLocaleTimeString()}</span>
        </div>
        <div class="prediction-metrics">
            <div>CPU: ${prediction.metrics.cpu_usage.toFixed(1)}%</div>
            <div>Memoria: ${prediction.metrics.memory_usage.toFixed(1)}%</div>
            <div>Latencia: ${prediction.metrics.network_latency.toFixed(0)}ms</div>
        </div>
        <div class="prediction-confidence">
            Confianza: ${(prediction.confidence * 100).toFixed(1)}%
        </div>
    `;
    
    // Agregar al inicio de la lista
    container.insertBefore(element, container.firstChild);
    
    // Mantener solo los últimos 10 elementos
    while (container.children.length > 10) {
        container.removeChild(container.lastChild);
    }
}

function showAnomalyAlert(prediction) {
    const alertElement = document.createElement('div');
    alertElement.className = 'anomaly-alert';
    alertElement.innerHTML = `
        <strong>¡Anomalía Detectada!</strong><br>
        Kiosk #${prediction.kiosk_id}<br>
        CPU: ${prediction.metrics.cpu_usage.toFixed(1)}%<br>
        Memoria: ${prediction.metrics.memory_usage.toFixed(1)}%<br>
        Latencia: ${prediction.metrics.network_latency.toFixed(0)}ms
    `;
    
    document.body.appendChild(alertElement);
    
    // Remover alerta después de 5 segundos
    setTimeout(() => {
        alertElement.remove();
    }, 5000);
}

async function refreshData() {
    const params = {
        model_version: document.getElementById('modelVersion').value,
        start_date: document.getElementById('startDate').value,
        end_date: document.getElementById('endDate').value
    };
    
    try {
        const response = await fetch('/ai/api/metrics?' + new URLSearchParams(params));
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        
        if (data.error) {
            showError(data.error);
            return;
        }
        
        updateDashboard(data);
    } catch (error) {
        console.error('Error al cargar datos:', error);
        showError('Error al cargar los datos del dashboard');
    }
}

function updateDashboard(data) {
    // Actualizar métricas principales
    updateMainMetrics(data.metrics);
    
    // Actualizar gráficos
    updateCharts(data);
    
    // Actualizar tabla de errores
    updateErrorCasesTable(data.error_cases);
}

function updateMainMetrics(metrics) {
    // Actualizar valores y tendencias
    document.getElementById('overallAccuracy').textContent = `${(metrics.accuracy * 100).toFixed(2)}%`;
    document.getElementById('rocAuc').textContent = metrics.roc_auc.toFixed(3);
    document.getElementById('prAuc').textContent = metrics.pr_auc.toFixed(3);
    document.getElementById('meanConfidence').textContent = `${(metrics.mean_confidence * 100).toFixed(2)}%`;
}

function updateCharts(data) {
    // ROC Curve
    const rocCtx = document.getElementById('rocCurve').getContext('2d');
    new Chart(rocCtx, {
        type: 'line',
        data: {
            labels: data.metrics.roc_curve.fpr.map(v => v.toFixed(2)),
            datasets: [{
                label: 'ROC Curve',
                data: data.metrics.roc_curve.tpr,
                borderColor: '#2c3e50',
                fill: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'False Positive Rate'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'True Positive Rate'
                    }
                }
            }
        }
    });
    
    // PR Curve
    const prCtx = document.getElementById('prCurve').getContext('2d');
    new Chart(prCtx, {
        type: 'line',
        data: {
            labels: data.metrics.pr_curve.recall.map(v => v.toFixed(2)),
            datasets: [{
                label: 'PR Curve',
                data: data.metrics.pr_curve.precision,
                borderColor: '#2c3e50',
                fill: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Recall'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Precision'
                    }
                }
            }
        }
    });
}

function updateErrorCasesTable(cases) {
    const tbody = document.getElementById('errorCases');
    tbody.innerHTML = cases.map(errorCase => 
        `<tr>
            <td>${errorCase.id}</td>
            <td>${errorCase.predicted}</td>
            <td>${errorCase.actual}</td>
            <td>${(errorCase.confidence * 100).toFixed(2)}%</td>
            <td>${(errorCase.error_margin * 100).toFixed(2)}%</td>
        </tr>`
    ).join('');
}

function showError(message) {
    // Implementar sistema de notificaciones de error
    alert(message);
} 