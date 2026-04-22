/**
 * analytics.js — Phase 6 Research Analytics Dashboard
 *
 * Provides Chart.js-powered visualizations for RL Playground metrics:
 *   - Per-NPC reward curves (individual, community, total)
 *   - Village social welfare index
 *   - Cooperation index progression (key hypothesis chart)
 *   - Action distribution shift (early vs late windows)
 *   - Shock timeline and response curves
 */

// ═══════════════════════════════════════════════════════════════
// Chart Color Palette
// ═══════════════════════════════════════════════════════════════

const CHART_COLORS = [
    'rgba(74, 144, 217, 1)',     // blue
    'rgba(46, 204, 113, 1)',     // green
    'rgba(243, 156, 18, 1)',     // amber
    'rgba(231, 76, 60, 1)',      // red
    'rgba(155, 89, 182, 1)',     // purple
    'rgba(26, 188, 156, 1)',     // teal
    'rgba(241, 196, 15, 1)',     // yellow
    'rgba(230, 126, 34, 1)',     // orange
];

const CHART_COLORS_ALPHA = CHART_COLORS.map(c => c.replace(', 1)', ', 0.15)'));

// Default Chart.js config for dark theme
const DARK_CHART_DEFAULTS = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 400 },
    plugins: {
        legend: {
            labels: {
                color: '#8b8b96',
                font: { family: "'Inter', system-ui, sans-serif", size: 11 },
                boxWidth: 12,
                padding: 12,
            },
        },
        tooltip: {
            backgroundColor: 'rgba(15, 15, 25, 0.95)',
            titleColor: '#e8e8f0',
            bodyColor: '#a0a0ac',
            borderColor: 'rgba(255, 255, 255, 0.08)',
            borderWidth: 1,
            cornerRadius: 8,
            titleFont: { family: "'Inter', system-ui, sans-serif", size: 12, weight: '600' },
            bodyFont: { family: "'JetBrains Mono', monospace", size: 11 },
            padding: 10,
        },
    },
    scales: {
        x: {
            ticks: { color: '#5a5a66', font: { size: 10 } },
            grid: { color: 'rgba(255, 255, 255, 0.04)' },
            border: { color: 'rgba(255, 255, 255, 0.06)' },
        },
        y: {
            ticks: { color: '#5a5a66', font: { size: 10 } },
            grid: { color: 'rgba(255, 255, 255, 0.04)' },
            border: { color: 'rgba(255, 255, 255, 0.06)' },
        },
    },
};

// ═══════════════════════════════════════════════════════════════
// State
// ═══════════════════════════════════════════════════════════════

let analyticsOpen = false;
let chartInstances = {};

// ═══════════════════════════════════════════════════════════════
// Panel Lifecycle
// ═══════════════════════════════════════════════════════════════

function toggleAnalyticsPanel() {
    analyticsOpen ? closeAnalyticsPanel() : openAnalyticsPanel();
}

function openAnalyticsPanel() {
    let panel = document.querySelector('#analytics-panel');
    if (!panel) {
        panel = createAnalyticsPanel();
        document.body.appendChild(panel);
    }
    panel.classList.add('active');
    analyticsOpen = true;
    refreshAnalytics();
}

function closeAnalyticsPanel() {
    const panel = document.querySelector('#analytics-panel');
    if (panel) panel.classList.remove('active');
    analyticsOpen = false;
    // Destroy chart instances to free memory
    Object.values(chartInstances).forEach(c => { if (c) c.destroy(); });
    chartInstances = {};
}

function createAnalyticsPanel() {
    const panel = document.createElement('div');
    panel.id = 'analytics-panel';
    panel.innerHTML = `
        <div class="analytics-header">
            <h2>Research Analytics</h2>
            <div class="analytics-header-actions">
                <button class="analytics-export-btn" id="analytics-export-btn" title="Download experiment bundle">
                    Export Bundle
                </button>
                <button class="analytics-close-btn" id="analytics-close-btn" title="Close (Esc)">×</button>
            </div>
        </div>
        <div class="analytics-content" id="analytics-content">
            <div class="analytics-loading">Loading analytics data...</div>
        </div>
    `;

    panel.querySelector('#analytics-close-btn').addEventListener('click', closeAnalyticsPanel);
    panel.querySelector('#analytics-export-btn').addEventListener('click', exportExperimentBundle);

    return panel;
}

// ═══════════════════════════════════════════════════════════════
// Data Fetching & Rendering
// ═══════════════════════════════════════════════════════════════

async function refreshAnalytics() {
    const content = document.querySelector('#analytics-content');
    if (!content) return;

    content.innerHTML = '<div class="analytics-loading">Loading analytics data...</div>';

    try {
        const data = await fetchAnalyticsData();
        renderAnalyticsDashboard(content, data);
    } catch (e) {
        content.innerHTML = `<div class="analytics-empty">Error loading analytics: ${e.message}</div>`;
    }
}

async function fetchAnalyticsData() {
    const API_BASE = `${window.location.origin}/api`;
    const resp = await fetch(`${API_BASE}/metrics/timeseries`);
    if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${resp.status}`);
    }
    return await resp.json();
}

async function exportExperimentBundle() {
    try {
        const API_BASE = `${window.location.origin}/api`;
        const resp = await fetch(`${API_BASE}/metrics/experiment`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();

        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `experiment_bundle_${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);

        // Toast via global function if available
        if (typeof showToast === 'function') {
            showToast('Experiment bundle exported!', 'success');
        }
    } catch (e) {
        if (typeof showToast === 'function') {
            showToast(`Export failed: ${e.message}`, 'error');
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Dashboard Layout
// ═══════════════════════════════════════════════════════════════

function renderAnalyticsDashboard(container, data) {
    // Destroy old charts
    Object.values(chartInstances).forEach(c => { if (c) c.destroy(); });
    chartInstances = {};

    // Build HTML
    let html = '';

    // Summary cards
    html += renderSummaryCards(data);

    // Charts grid
    html += '<div class="analytics-charts-grid">';
    html += renderChartCard('cooperation-chart', 'Cooperation Index Over Turns',
        'Key hypothesis: individualism → cooperation shift', false);
    html += renderChartCard('welfare-chart', 'Social Welfare Index',
        'Village-wide average cooperation tendency over time', false);
    html += renderChartCard('reward-chart', 'NPC Reward Curves',
        'Per-NPC total reward over turns (individual + community)', false);
    html += renderChartCard('community-reward-chart', 'Community Reward',
        'Village-level community reward averaged per turn', false);
    html += renderChartCard('action-dist-chart', 'Action Distribution Shift',
        'Early vs late window action frequencies', true);
    html += renderShockTimelineCard(data);
    html += '</div>';

    container.innerHTML = html;

    // Create charts (after DOM is ready)
    requestAnimationFrame(() => {
        createCooperationChart(data);
        createWelfareChart(data);
        createRewardChart(data);
        createCommunityRewardChart(data);
        createActionDistChart(data);
    });
}

function renderChartCard(canvasId, title, subtitle, wide) {
    return `
        <div class="analytics-chart-card${wide ? ' wide' : ''}">
            <h4 class="analytics-chart-title">${title}</h4>
            <p class="analytics-chart-subtitle">${subtitle}</p>
            <div class="analytics-chart-container">
                <canvas id="${canvasId}"></canvas>
            </div>
        </div>
    `;
}

// ═══════════════════════════════════════════════════════════════
// Summary Cards
// ═══════════════════════════════════════════════════════════════

function renderSummaryCards(data) {
    const coop = data.cooperation_index || {};
    const globalCoop = (coop.global ?? 0).toFixed(3);
    const npcCount = Object.keys(data.reward_series || {}).length;
    const shockCount = (data.shock_timeline || []).filter(s => s.status === 'active').length;

    // Average policy entropy
    const entropies = Object.values(data.policy_entropy || {});
    const avgEntropy = entropies.length
        ? (entropies.reduce((a, b) => a + b, 0) / entropies.length).toFixed(3)
        : '—';

    // Total turns from cooperation series
    const coopSeries = data.cooperation_series || {};
    const totalTurns = (coopSeries.turns || []).length;

    // Total shocks (active + expired)
    const totalShocks = (data.shock_timeline || []).length;

    return `
        <div class="analytics-summary-row">
            <div class="analytics-stat-card highlight">
                <div class="analytics-stat-value">${globalCoop}</div>
                <div class="analytics-stat-label">Cooperation Index</div>
            </div>
            <div class="analytics-stat-card">
                <div class="analytics-stat-value">${avgEntropy}</div>
                <div class="analytics-stat-label">Avg Policy Entropy</div>
            </div>
            <div class="analytics-stat-card">
                <div class="analytics-stat-value">${npcCount}</div>
                <div class="analytics-stat-label">NPCs Tracked</div>
            </div>
            <div class="analytics-stat-card">
                <div class="analytics-stat-value">${totalTurns}</div>
                <div class="analytics-stat-label">Data Points</div>
            </div>
            <div class="analytics-stat-card${shockCount > 0 ? ' warning' : ''}">
                <div class="analytics-stat-value">${shockCount} / ${totalShocks}</div>
                <div class="analytics-stat-label">Active / Total Shocks</div>
            </div>
        </div>
    `;
}

// ═══════════════════════════════════════════════════════════════
// Chart: Cooperation Index
// ═══════════════════════════════════════════════════════════════

function createCooperationChart(data) {
    const canvas = document.getElementById('cooperation-chart');
    if (!canvas) return;

    const series = data.cooperation_series || {};
    if (!series.turns || series.turns.length === 0) {
        canvas.parentElement.innerHTML = '<div class="analytics-empty">No cooperation data yet. Play some turns!</div>';
        return;
    }

    chartInstances.cooperation = new Chart(canvas, {
        type: 'line',
        data: {
            labels: series.turns,
            datasets: [{
                label: 'Global Cooperation',
                data: series.global_cooperation,
                borderColor: CHART_COLORS[0],
                backgroundColor: CHART_COLORS_ALPHA[0],
                fill: true,
                tension: 0.3,
                pointRadius: series.turns.length > 50 ? 0 : 3,
                pointHoverRadius: 5,
                borderWidth: 2,
            }],
        },
        options: {
            ...DARK_CHART_DEFAULTS,
            scales: {
                ...DARK_CHART_DEFAULTS.scales,
                x: { ...DARK_CHART_DEFAULTS.scales.x, title: { display: true, text: 'Turn', color: '#5a5a66' } },
                y: { ...DARK_CHART_DEFAULTS.scales.y, title: { display: true, text: 'Cooperation Index', color: '#5a5a66' }, min: 0, max: 1 },
            },
            plugins: {
                ...DARK_CHART_DEFAULTS.plugins,
                legend: { display: false },
            },
        },
    });
}

// ═══════════════════════════════════════════════════════════════
// Chart: Social Welfare
// ═══════════════════════════════════════════════════════════════

function createWelfareChart(data) {
    const canvas = document.getElementById('welfare-chart');
    if (!canvas) return;

    const series = data.social_welfare_series || {};
    if (!series.turns || series.turns.length === 0) {
        canvas.parentElement.innerHTML = '<div class="analytics-empty">No welfare data yet.</div>';
        return;
    }

    chartInstances.welfare = new Chart(canvas, {
        type: 'line',
        data: {
            labels: series.turns,
            datasets: [{
                label: 'Social Welfare Index',
                data: series.welfare_index,
                borderColor: CHART_COLORS[1],
                backgroundColor: CHART_COLORS_ALPHA[1],
                fill: true,
                tension: 0.3,
                pointRadius: series.turns.length > 50 ? 0 : 3,
                pointHoverRadius: 5,
                borderWidth: 2,
            }],
        },
        options: {
            ...DARK_CHART_DEFAULTS,
            scales: {
                ...DARK_CHART_DEFAULTS.scales,
                x: { ...DARK_CHART_DEFAULTS.scales.x, title: { display: true, text: 'Turn', color: '#5a5a66' } },
                y: { ...DARK_CHART_DEFAULTS.scales.y, title: { display: true, text: 'Welfare Index', color: '#5a5a66' }, min: 0, max: 1 },
            },
            plugins: {
                ...DARK_CHART_DEFAULTS.plugins,
                legend: { display: false },
            },
        },
    });
}

// ═══════════════════════════════════════════════════════════════
// Chart: NPC Reward Curves
// ═══════════════════════════════════════════════════════════════

function createRewardChart(data) {
    const canvas = document.getElementById('reward-chart');
    if (!canvas) return;

    const rewardSeries = data.reward_series || {};
    const npcIds = Object.keys(rewardSeries);

    if (npcIds.length === 0) {
        canvas.parentElement.innerHTML = '<div class="analytics-empty">No reward data yet.</div>';
        return;
    }

    const datasets = npcIds.map((uid, i) => {
        const npc = rewardSeries[uid];
        const colorIdx = i % CHART_COLORS.length;
        return {
            label: npc.npc_name || uid,
            data: npc.total,
            borderColor: CHART_COLORS[colorIdx],
            backgroundColor: 'transparent',
            tension: 0.3,
            pointRadius: (npc.turns || []).length > 50 ? 0 : 2,
            pointHoverRadius: 4,
            borderWidth: 1.5,
        };
    });

    // Use turns from first NPC
    const labels = rewardSeries[npcIds[0]]?.turns || [];

    chartInstances.reward = new Chart(canvas, {
        type: 'line',
        data: { labels, datasets },
        options: {
            ...DARK_CHART_DEFAULTS,
            scales: {
                ...DARK_CHART_DEFAULTS.scales,
                x: { ...DARK_CHART_DEFAULTS.scales.x, title: { display: true, text: 'Turn', color: '#5a5a66' } },
                y: { ...DARK_CHART_DEFAULTS.scales.y, title: { display: true, text: 'Total Reward', color: '#5a5a66' } },
            },
        },
    });
}

// ═══════════════════════════════════════════════════════════════
// Chart: Community Reward
// ═══════════════════════════════════════════════════════════════

function createCommunityRewardChart(data) {
    const canvas = document.getElementById('community-reward-chart');
    if (!canvas) return;

    const series = data.community_reward_series || {};
    if (!series.turns || series.turns.length === 0) {
        canvas.parentElement.innerHTML = '<div class="analytics-empty">No community reward data yet.</div>';
        return;
    }

    chartInstances.communityReward = new Chart(canvas, {
        type: 'line',
        data: {
            labels: series.turns,
            datasets: [
                {
                    label: 'Avg Community Reward',
                    data: series.avg_community,
                    borderColor: CHART_COLORS[4],
                    backgroundColor: CHART_COLORS_ALPHA[4],
                    fill: true,
                    tension: 0.3,
                    pointRadius: series.turns.length > 50 ? 0 : 3,
                    borderWidth: 2,
                },
                {
                    label: 'Avg Total Reward',
                    data: series.avg_total,
                    borderColor: CHART_COLORS[5],
                    backgroundColor: 'transparent',
                    tension: 0.3,
                    pointRadius: series.turns.length > 50 ? 0 : 3,
                    borderWidth: 1.5,
                    borderDash: [5, 3],
                },
            ],
        },
        options: {
            ...DARK_CHART_DEFAULTS,
            scales: {
                ...DARK_CHART_DEFAULTS.scales,
                x: { ...DARK_CHART_DEFAULTS.scales.x, title: { display: true, text: 'Turn', color: '#5a5a66' } },
                y: { ...DARK_CHART_DEFAULTS.scales.y, title: { display: true, text: 'Reward', color: '#5a5a66' } },
            },
        },
    });
}

// ═══════════════════════════════════════════════════════════════
// Chart: Action Distribution Shift
// ═══════════════════════════════════════════════════════════════

function createActionDistChart(data) {
    const canvas = document.getElementById('action-dist-chart');
    if (!canvas) return;

    const dist = data.action_distribution || {};
    const early = dist.early_window || {};
    const late = dist.late_window || {};

    const allActions = [...new Set([...Object.keys(early), ...Object.keys(late)])];
    if (allActions.length === 0) {
        canvas.parentElement.innerHTML = '<div class="analytics-empty">No action data yet.</div>';
        return;
    }

    // Sort by total frequency
    allActions.sort((a, b) => ((late[b] || 0) + (early[b] || 0)) - ((late[a] || 0) + (early[a] || 0)));
    const topActions = allActions.slice(0, 12); // Show top 12

    chartInstances.actionDist = new Chart(canvas, {
        type: 'bar',
        data: {
            labels: topActions,
            datasets: [
                {
                    label: 'Early Window',
                    data: topActions.map(a => early[a] || 0),
                    backgroundColor: CHART_COLORS_ALPHA[0].replace('0.15', '0.6'),
                    borderColor: CHART_COLORS[0],
                    borderWidth: 1,
                    borderRadius: 3,
                },
                {
                    label: 'Late Window',
                    data: topActions.map(a => late[a] || 0),
                    backgroundColor: CHART_COLORS_ALPHA[2].replace('0.15', '0.6'),
                    borderColor: CHART_COLORS[2],
                    borderWidth: 1,
                    borderRadius: 3,
                },
            ],
        },
        options: {
            ...DARK_CHART_DEFAULTS,
            scales: {
                ...DARK_CHART_DEFAULTS.scales,
                x: { ...DARK_CHART_DEFAULTS.scales.x, ticks: { ...DARK_CHART_DEFAULTS.scales.x.ticks, maxRotation: 45 } },
                y: { ...DARK_CHART_DEFAULTS.scales.y, title: { display: true, text: 'Count', color: '#5a5a66' }, beginAtZero: true },
            },
        },
    });
}

// ═══════════════════════════════════════════════════════════════
// Shock Timeline (non-chart, HTML-based)
// ═══════════════════════════════════════════════════════════════

function renderShockTimelineCard(data) {
    const timeline = data.shock_timeline || [];
    const responses = data.shock_responses || [];

    if (timeline.length === 0 && responses.length === 0) {
        return `
            <div class="analytics-chart-card">
                <h4 class="analytics-chart-title">Shock Timeline & Response</h4>
                <p class="analytics-chart-subtitle">Shock events and their impact on cooperation</p>
                <div class="analytics-empty">No shocks recorded yet.</div>
            </div>
        `;
    }

    let barsHtml = '';
    for (const shock of timeline) {
        const statusCls = shock.status === 'active' ? 'active' : 'expired';
        const typeCls = shock.shock_type || '';
        const intensity = shock.intensity != null ? ` (${(shock.intensity * 100).toFixed(0)}%)` : '';
        barsHtml += `
            <div class="shock-timeline-bar ${typeCls}">
                <span class="shock-type-label">${shock.shock_type}</span>
                <span class="shock-turns-label">T${shock.turn_started} → T${shock.turn_started + shock.duration}</span>
                <span class="shock-status-badge ${statusCls}">${shock.status}${intensity}</span>
            </div>
        `;
    }

    // Shock response summary
    let responseHtml = '';
    if (responses.length > 0) {
        responseHtml = '<table class="action-dist-table"><thead><tr>';
        responseHtml += '<th>Shock</th><th>Coop Before</th><th>Coop During</th><th>Coop After</th>';
        responseHtml += '<th>Reward Before</th><th>Reward During</th><th>Reward After</th>';
        responseHtml += '</tr></thead><tbody>';
        for (const r of responses) {
            const fmt = v => v != null ? v.toFixed(3) : '—';
            responseHtml += `<tr>
                <td>${r.shock_type}</td>
                <td>${fmt(r.avg_cooperation_before)}</td>
                <td>${fmt(r.avg_cooperation_during)}</td>
                <td>${fmt(r.avg_cooperation_after)}</td>
                <td>${fmt(r.avg_reward_before)}</td>
                <td>${fmt(r.avg_reward_during)}</td>
                <td>${fmt(r.avg_reward_after)}</td>
            </tr>`;
        }
        responseHtml += '</tbody></table>';
    }

    return `
        <div class="analytics-chart-card">
            <h4 class="analytics-chart-title">Shock Timeline & Response</h4>
            <p class="analytics-chart-subtitle">Shock events and their measured impact on cooperation and rewards</p>
            <div class="shock-timeline-container">${barsHtml}</div>
            ${responseHtml}
        </div>
    `;
}

// ═══════════════════════════════════════════════════════════════
// Expose globals for app.js integration
// ═══════════════════════════════════════════════════════════════

window.toggleAnalyticsPanel = toggleAnalyticsPanel;
window.openAnalyticsPanel = openAnalyticsPanel;
window.closeAnalyticsPanel = closeAnalyticsPanel;
