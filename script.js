let chart = null;
let historyData = [];

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    loadHistory();
    loadStats();
    setupEventListeners();
    updateCharCounter();
    initializeChart();
    
    // Hide results section initially
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.classList.remove('show');
});

function setupEventListeners() {
    const analyzeBtn = document.getElementById('analyzeBtn');
    const commentInput = document.getElementById('commentInput');
    
    analyzeBtn.addEventListener('click', analyzeComment);
    commentInput.addEventListener('input', updateCharCounter);
    commentInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && e.ctrlKey) {
            analyzeComment();
        }
    });
}

function updateCharCounter() {
    const input = document.getElementById('commentInput');
    const counter = document.getElementById('charCount');
    const length = input.value.length;
    counter.textContent = length;
    
    // Add visual feedback for long comments
    if (length > 200) {
        counter.style.color = 'var(--warning)';
    } else if (length > 500) {
        counter.style.color = 'var(--danger)';
    } else {
        counter.style.color = 'var(--gray)';
    }
}

async function analyzeComment() {
    const commentInput = document.getElementById('commentInput');
    const comment = commentInput.value.trim();
    
    if (!comment) {
        showNotification('Please enter a comment to analyze', 'warning');
        return;
    }
    
    // Show loading state
    const analyzeBtn = document.getElementById('analyzeBtn');
    const originalHTML = analyzeBtn.innerHTML;
    analyzeBtn.innerHTML = '<span class="btn-text">Analyzing...</span><span class="btn-icon">⏳</span>';
    analyzeBtn.disabled = true;
    analyzeBtn.classList.add('loading');
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ comment: comment })
        });
        
        const result = await response.json();
        
        if (response.ok) {
            displayResults(result);
            playCelebration(result.label === 'Non-abusive');
            await loadHistory();
            await loadStats();
            updateChartFromStats();
        } else {
            showNotification(result.error || 'Analysis failed', 'error');
        }
    } catch (error) {
        console.error('Error:', error);
        showNotification('Network error. Please try again.', 'error');
    } finally {
        analyzeBtn.innerHTML = originalHTML;
        analyzeBtn.disabled = false;
        analyzeBtn.classList.remove('loading');
    }
}

function displayResults(result) {
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.style.display = 'block';
    setTimeout(() => {
        resultsSection.classList.add('show');
    }, 10);
    
    // Update scores
    const abusiveScore = (result.abusive_score * 100).toFixed(1);
    const nonAbusiveScore = (result.non_abusive_score * 100).toFixed(1);
    
    const abusiveScoreEl = document.getElementById('abusiveScore');
    const nonAbusiveScoreEl = document.getElementById('nonAbusiveScore');
    
    // Animate score numbers
    animateNumber(abusiveScoreEl, 0, parseFloat(abusiveScore), 500);
    animateNumber(nonAbusiveScoreEl, 0, parseFloat(nonAbusiveScore), 500);
    
    // Update bars
    const abusiveBar = document.getElementById('abusiveBar');
    const nonAbusiveBar = document.getElementById('nonAbusiveBar');
    
    // Reset animations
    abusiveBar.classList.remove('animate');
    nonAbusiveBar.classList.remove('animate');
    
    // Force reflow
    void abusiveBar.offsetWidth;
    void nonAbusiveBar.offsetWidth;
    
    // Set new widths
    abusiveBar.style.width = `${abusiveScore}%`;
    nonAbusiveBar.style.width = `${nonAbusiveScore}%`;
    
    // Add animation class
    setTimeout(() => {
        abusiveBar.classList.add('animate');
        nonAbusiveBar.classList.add('animate');
    }, 50);
    
    // Update badges
    const resultBadge = document.getElementById('resultBadge');
    const confidenceBadge = document.getElementById('confidenceBadge');
    const resultMessage = document.getElementById('resultMessage');
    
    resultBadge.textContent = `Result: ${result.label}`;
    resultBadge.className = `result-badge ${result.label === 'Abusive' ? 'abusive' : 'non-abusive'}`;
    
    const confidencePercent = (result.confidence * 100).toFixed(1);
    confidenceBadge.textContent = `Confidence: ${confidencePercent}%`;
    
    // Set result message
    resultMessage.className = `result-message ${result.label === 'Abusive' ? 'abusive-message' : 'safe-message'}`;
    if (result.label === 'Abusive') {
        resultMessage.innerHTML = `
            ⚠️ Warning: This comment contains abusive content! 
            Consider moderating or blocking this content.
        `;
        // Add shake animation
        resultsSection.classList.add('shake-animation');
        setTimeout(() => resultsSection.classList.remove('shake-animation'), 500);
    } else {
        resultMessage.innerHTML = `
            ✅ Safe Content: This comment is appropriate and non-abusive.
            Great community interaction!
        `;
    }
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function animateNumber(element, start, end, duration) {
    const startTime = performance.now();
    const updateNumber = (currentTime) => {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const current = start + (end - start) * progress;
        element.textContent = `${current.toFixed(1)}%`;
        
        if (progress < 1) {
            requestAnimationFrame(updateNumber);
        }
    };
    requestAnimationFrame(updateNumber);
}

async function loadHistory() {
    try {
        const response = await fetch('/history');
        const history = await response.json();
        historyData = history;
        displayHistory(history);
    } catch (error) {
        console.error('Error loading history:', error);
    }
}

function displayHistory(history) {
    const historyList = document.getElementById('historyList');
    
    if (!history || history.length === 0) {
        historyList.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">🔍</div>
                <p>No comments analyzed yet. Start by entering a comment above!</p>
            </div>
        `;
        return;
    }
    
    historyList.innerHTML = history.slice(0, 10).map(item => `
        <div class="history-item ${item.result.label.toLowerCase()}" onclick="window.reanalyzeComment('${escapeHtml(item.text)}')">
            <div class="history-text">${escapeHtml(item.text.substring(0, 100))}${item.text.length > 100 ? '...' : ''}</div>
            <div class="history-meta">
                <span class="history-label">${item.result.label}</span>
                <span class="history-time">${new Date(item.timestamp).toLocaleTimeString()}</span>
            </div>
        </div>
    `).join('');
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

async function loadStats() {
    try {
        const response = await fetch('/stats');
        const stats = await response.json();
        
        document.getElementById('totalCount').textContent = stats.total;
        document.getElementById('abusiveCount').textContent = stats.abusive_count;
        document.getElementById('nonAbusiveCount').textContent = stats.non_abusive_count;
        document.getElementById('avgConfidence').textContent = `${(stats.avg_confidence * 100).toFixed(1)}%`;
        
        return stats;
    } catch (error) {
        console.error('Error loading stats:', error);
        return null;
    }
}

function initializeChart() {
    const ctx = document.getElementById('distributionChart').getContext('2d');
    chart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Abusive', 'Non-Abusive'],
            datasets: [{
                data: [0, 0],
                backgroundColor: ['#ef4444', '#10b981'],
                borderWidth: 0,
                hoverOffset: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        font: {
                            size: 12,
                            weight: '500'
                        },
                        padding: 15,
                        usePointStyle: true,
                        pointStyle: 'circle'
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const label = context.label || '';
                            const value = context.parsed || 0;
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = total > 0 ? ((value / total) * 100).toFixed(1) : 0;
                            return `${label}: ${value} (${percentage}%)`;
                        }
                    }
                }
            },
            cutout: '60%'
        }
    });
}

async function updateChartFromStats() {
    const stats = await loadStats();
    if (chart && stats) {
        chart.data.datasets[0].data = [stats.abusive_count || 0, stats.non_abusive_count || 0];
        chart.update();
    }
}

function playCelebration(isSafe) {
    if (isSafe && typeof party !== 'undefined') {
        party.confetti(document.getElementById('analyzeBtn'), {
            count: party.variation.random(50, 100),
            size: party.variation.random(0.8, 1.2)
        });
    }
}

function showNotification(message, type) {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Global function for re-analysis
window.reanalyzeComment = function(text) {
    document.getElementById('commentInput').value = text;
    updateCharCounter();
    analyzeComment();
};