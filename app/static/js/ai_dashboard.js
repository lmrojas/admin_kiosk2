// Dashboard de IA - Admin Kiosk

document.addEventListener('DOMContentLoaded', function() {
    // Inicializar componentes
    initializeDashboard();
    
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

async function refreshData() {
    const params = {
        model_version: document.getElementById('modelVersion').value,
        start_date: document.getElementById('startDate').value,
        end_date: document.getElementById('endDate').value
    };
    
    try {
        const response = await fetch('/api/ai/metrics?' + new URLSearchParams(params));
        const data = await response.json();
        
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
    
    // Actualizar análisis de drift
    updateDriftAnalysis(data.drift);
    
    // Actualizar matriz de confusión
    updateConfusionMatrix(data.metrics.confusion_matrix);
    
    // Actualizar análisis de errores
    updateErrorAnalysis(data.errors);
}

function updateMainMetrics(metrics) {
    // Actualizar valores y tendencias
    document.getElementById('overallAccuracy').textContent = `${(metrics.accuracy * 100).toFixed(2)}%`;
    document.getElementById('rocAuc').textContent = metrics.roc_auc.toFixed(3);
    document.getElementById('prAuc').textContent = metrics.pr_auc.toFixed(3);
    document.getElementById('meanConfidence').textContent = `${(metrics.mean_confidence * 100).toFixed(2)}%`;
    
    // Actualizar tendencias
    updateTrendIndicator('accuracy', metrics.accuracy_trend);
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
    
    // Confidence Distribution
    const confidenceData = data.metrics.prob_distribution;
    Plotly.newPlot('confidenceDistribution', [{
        x: Object.keys(confidenceData),
        y: Object.values(confidenceData),
        type: 'bar'
    }], {
        title: 'Distribución de Confianza',
        xaxis: { title: 'Confianza' },
        yaxis: { title: 'Frecuencia' }
    });
}

function updateDriftAnalysis(drift) {
    // Distribution Shift
    const distCtx = document.getElementById('distributionChart').getContext('2d');
    new Chart(distCtx, {
        type: 'bar',
        data: {
            labels: Object.keys(drift.distribution_shift.distribution_period2),
            datasets: [
                {
                    label: 'Período 1',
                    data: Object.values(drift.distribution_shift.distribution_period1),
                    backgroundColor: 'rgba(44, 62, 80, 0.6)'
                },
                {
                    label: 'Período 2',
                    data: Object.values(drift.distribution_shift.distribution_period2),
                    backgroundColor: 'rgba(52, 152, 219, 0.6)'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false
        }
    });
    
    // Feature Drift Heatmap
    const featureDrift = drift.feature_drift;
    const features = Object.keys(featureDrift);
    const driftScores = features.map(f => featureDrift[f].drift_score);
    
    Plotly.newPlot('featureDriftHeatmap', [{
        z: [driftScores],
        x: features,
        type: 'heatmap',
        colorscale: 'RdYlBu'
    }], {
        title: 'Feature Drift Scores',
        height: 200
    });
    
    // Performance Trend
    const trendCtx = document.getElementById('performanceTrendChart').getContext('2d');
    new Chart(trendCtx, {
        type: 'line',
        data: {
            labels: drift.performance_decay.window_metrics.map(w => `Ventana ${w.window}`),
            datasets: [{
                label: 'Accuracy',
                data: drift.performance_decay.window_metrics.map(w => w.accuracy),
                borderColor: '#2c3e50',
                fill: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false
        }
    });
    
    // Actualizar alertas
    updateDriftAlerts(drift.alerts);
}

function updateConfusionMatrix(matrix) {
    Plotly.newPlot('confusionMatrix', [{
        z: matrix,
        type: 'heatmap',
        colorscale: 'Viridis'
    }], {
        title: 'Matriz de Confusión',
        height: 400
    });
}

function updateErrorAnalysis(errors) {
    // Error Distribution
    const errorDistCtx = document.getElementById('errorDistribution').getContext('2d');
    new Chart(errorDistCtx, {
        type: 'bar',
        data: {
            labels: errors.distribution.labels,
            datasets: [{
                label: 'Errores',
                data: errors.distribution.values,
                backgroundColor: 'rgba(231, 76, 60, 0.6)'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false
        }
    });
    
    // Errors by Confidence
    const errorConfCtx = document.getElementById('errorsByConfidence').getContext('2d');
    new Chart(errorConfCtx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Errores vs Confianza',
                data: errors.by_confidence.map(e => ({
                    x: e.confidence,
                    y: e.error_rate
                })),
                backgroundColor: 'rgba(231, 76, 60, 0.6)'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Confianza'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Tasa de Error'
                    }
                }
            }
        }
    });
    
    // Actualizar tabla de casos problemáticos
    updateErrorCasesTable(errors.cases);
}

function updateDriftAlerts(alerts) {
    const alertsContainer = document.querySelector('.drift-alerts');
    alertsContainer.innerHTML = alerts.map(alert => 
        `<div class="alert alert-${alert.severity.toLowerCase()}">${alert.message}</div>`
    ).join('');
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

function updateTrendIndicator(metric, trend) {
    const trendIcon = document.querySelector(`#${metric} .trend-icon`);
    const trendValue = document.querySelector(`#${metric} .trend-value`);
    
    trendIcon.className = 'trend-icon ' + (trend > 0 ? 'trend-up' : 'trend-down');
    trendValue.textContent = `${Math.abs(trend).toFixed(2)}%`;
    trendValue.style.color = trend > 0 ? '#28a745' : '#dc3545';
}

function showError(message) {
    // Implementar sistema de notificaciones de error
    alert(message);
} 