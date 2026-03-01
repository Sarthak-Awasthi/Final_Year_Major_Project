/**
 * MVP Research Game — Application Entry Point
 * 
 * ES6 module that bootstraps the game UI, connects to the backend API,
 * and wires up all interactive elements.
 */

// ═══════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════

const API_BASE = `${window.location.origin}/api`;
const WS_BASE = `${window.location.origin.replace('http', 'ws')}/ws`;

const ACTIONS = {
    navigation:  [{ id: 'move_to', name: 'Move To', ap: 3 }],
    exploration: [
        { id: 'look', name: 'Look', ap: 1 },
        { id: 'search', name: 'Search', ap: 5 },
        { id: 'examine', name: 'Examine', ap: 2 },
    ],
    social: [
        { id: 'talk', name: 'Talk', ap: 1 },
        { id: 'greet', name: 'Greet', ap: 1 },
        { id: 'ask_info', name: 'Ask Info', ap: 1 },
        { id: 'persuade', name: 'Persuade', ap: 2 },
        { id: 'trade', name: 'Trade', ap: 2 },
        { id: 'give_item', name: 'Give Item', ap: 1 },
        { id: 'deceive', name: 'Deceive', ap: 2 },
        { id: 'intimidate', name: 'Intimidate', ap: 2 },
    ],
    combat: [
        { id: 'attack', name: 'Attack', ap: 10 },
        { id: 'defend', name: 'Defend', ap: 5 },
        { id: 'flee', name: 'Flee', ap: 5 },
    ],
    stealth: [
        { id: 'sneak', name: 'Sneak', ap: 5 },
        { id: 'hide', name: 'Hide', ap: 3 },
        { id: 'steal', name: 'Steal', ap: 5 },
    ],
    utility: [
        { id: 'pick_up', name: 'Pick Up', ap: 1 },
        { id: 'use_item', name: 'Use Item', ap: 2 },
        { id: 'eat', name: 'Eat', ap: 1 },
        { id: 'rest', name: 'Rest', ap: 0 },
        { id: 'wait', name: 'Wait', ap: 0 },
        { id: 'drop_item', name: 'Drop Item', ap: 0 },
        { id: 'status', name: 'Status', ap: 0 },
        { id: 'equip', name: 'Equip', ap: 1 },
        { id: 'work', name: 'Work', ap: 5 },
    ],
};

const LOCATION_ADJACENCY = {
    village_center: ['elders_house', 'tavern', 'gate', 'fields'],
    elders_house:   ['village_center'],
    tavern:         ['village_center'],
    gate:           ['village_center'],
    fields:         ['village_center'],
};

const LOCATION_NAMES = {
    village_center: 'Village Center',
    elders_house:   "Elder's House",
    tavern:         'Tavern',
    gate:           'Gate',
    fields:         'Fields',
};

const REP_TIERS = [
    { min: 50,   cls: 'rep-trusted',    label: 'Trusted' },
    { min: 20,   cls: 'rep-friendly',   label: 'Friendly' },
    { min: -19,  cls: 'rep-neutral',    label: 'Neutral' },
    { min: -49,  cls: 'rep-suspicious', label: 'Suspicious' },
    { min: -100, cls: 'rep-hostile',    label: 'Hostile' },
];

// ═══════════════════════════════════════════════════════════════
// State
// ═══════════════════════════════════════════════════════════════

let gameState = null;
let ws = null;
let wsReconnectAttempts = 0;
let cy = null;             // Cytoscape instance
let pendingAction = null;  // Action waiting for target selection

// ═══════════════════════════════════════════════════════════════
// DOM References
// ═══════════════════════════════════════════════════════════════

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const dom = {
    // Modals
    newGameModal:     $('#new-game-modal'),
    gameOverOverlay:  $('#game-over-overlay'),
    saveLoadModal:    $('#save-load-modal'),
    moveModal:        $('#move-modal'),

    // New Game
    playerNameInput:  $('#player-name-input'),
    seedInput:        $('#seed-input'),
    difficultySelect: $('#difficulty-select'),
    startGameBtn:     $('#start-game-btn'),

    // Game Over
    gameOverIcon:     $('#game-over-icon'),
    gameOverTitle:    $('#game-over-title'),
    gameOverMessage:  $('#game-over-message'),

    // Container
    gameContainer:    $('#game-container'),

    // Header
    turnCounter:      $('#turn-counter'),
    timePeriod:       $('#time-period'),
    currentLocation:  $('#current-location'),
    statusText:       $('#status-text'),
    llmDot:           $('.llm-dot'),

    // Player
    playerName:       $('#player-name'),
    healthBar:        $('#health-bar'),
    healthText:       $('#health-text'),
    staminaBar:       $('#stamina-bar'),
    staminaText:      $('#stamina-text'),
    globalReputation: $('#global-reputation'),
    playerAttack:     $('#player-attack'),
    playerDefense:    $('#player-defense'),

    // Inventory & Equipment
    inventoryCount:   $('#inventory-count'),
    inventoryList:    $('#inventory-list'),
    equippedWeapon:   $('#equipped-weapon'),
    equippedArmor:    $('#equipped-armor'),

    // Quest
    questStage:       $('#quest-stage'),
    questCheckpoint:  $('#quest-checkpoint'),
    questProgressFill:$('#quest-progress-fill'),
    questDescription: $('#quest-description'),

    // Narrative
    narrativeContent: $('#narrative-content'),

    // Actions
    actionTabs:       $$('.action-tab'),
    actionGroups:     $$('.action-group'),
    actionBtns:       $$('.action-btn'),
    targetSelector:   $('#target-selector'),
    targetSelect:     $('#target-select'),
    confirmActionBtn: $('#confirm-action-btn'),
    cancelActionBtn:  $('#cancel-action-btn'),

    // Text input
    freeTextInput:    $('#free-text-input'),
    sendTextBtn:      $('#send-text-btn'),

    // MDP Graph
    mdpGraph:         $('#mdp-graph'),
    graphFullscreenBtn: $('#graph-fullscreen-btn'),

    // NPC
    npcList:          $('#npc-list'),

    // Events
    eventList:        $('#event-list'),

    // Footer
    footerDifficulty: $('#footer-difficulty'),
    saveBtn:          $('#save-btn'),
    loadBtn:          $('#load-btn'),
    quickSaveBtn:     $('#quick-save-btn'),

    // Toast
    toastContainer:   $('#toast-container'),
};

// ═══════════════════════════════════════════════════════════════
// Initialization
// ═══════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
    initActionPalette();
    initTextInput();
    initModals();
    initFooter();
    initMDPGraph();
});

// ═══════════════════════════════════════════════════════════════
// Action Palette
// ═══════════════════════════════════════════════════════════════

function initActionPalette() {
    // Tab switching
    dom.actionTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const category = tab.dataset.category;
            dom.actionTabs.forEach(t => t.classList.remove('active'));
            dom.actionGroups.forEach(g => g.classList.remove('active'));
            tab.classList.add('active');
            const group = $(`.action-group[data-category="${category}"]`);
            if (group) group.classList.add('active');
        });
    });

    // Action button clicks
    dom.actionBtns.forEach(btn => {
        btn.addEventListener('click', () => handleActionClick(btn));
    });

    // Target selector
    dom.confirmActionBtn.addEventListener('click', confirmTargetedAction);
    dom.cancelActionBtn.addEventListener('click', cancelTargetedAction);
}

function handleActionClick(btn) {
    const actionId = btn.dataset.action;

    // move_to opens location modal
    if (actionId === 'move_to') {
        openMoveModal();
        return;
    }

    // Actions needing a target NPC
    const needsNpcTarget = ['talk', 'greet', 'ask_info', 'persuade', 'trade',
        'give_item', 'deceive', 'intimidate', 'attack', 'steal'];

    if (needsNpcTarget.includes(actionId) && gameState?.npcs_here?.length) {
        showTargetSelector(actionId);
        return;
    }

    // Direct execution
    sendAction({ action_id: actionId, source: 'button' });
}

function showTargetSelector(actionId) {
    pendingAction = actionId;
    dom.targetSelector.classList.remove('hidden');

    // Populate target dropdown with NPCs at location
    dom.targetSelect.innerHTML = '<option value="">Select target...</option>';
    if (gameState?.npcs_here) {
        gameState.npcs_here.forEach(npc => {
            const opt = document.createElement('option');
            opt.value = npc.uid;
            opt.textContent = npc.name;
            dom.targetSelect.appendChild(opt);
        });
    }
}

function confirmTargetedAction() {
    const targetUid = dom.targetSelect.value;
    if (!targetUid || !pendingAction) return;

    sendAction({
        action_id: pendingAction,
        target_npc: targetUid,
        source: 'button',
    });
    cancelTargetedAction();
}

function cancelTargetedAction() {
    pendingAction = null;
    dom.targetSelector.classList.add('hidden');
}

// ═══════════════════════════════════════════════════════════════
// Text Input
// ═══════════════════════════════════════════════════════════════

function initTextInput() {
    dom.freeTextInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            submitFreeText();
        }
    });
    dom.sendTextBtn.addEventListener('click', submitFreeText);
}

function submitFreeText() {
    const text = dom.freeTextInput.value.trim();
    if (!text) return;

    sendAction({ raw_text: text, source: 'text' });
    dom.freeTextInput.value = '';
}

// ═══════════════════════════════════════════════════════════════
// API Communication
// ═══════════════════════════════════════════════════════════════

async function apiPost(endpoint, body = {}) {
    try {
        const resp = await fetch(`${API_BASE}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${resp.status}`);
        }
        return await resp.json();
    } catch (e) {
        showToast(`API Error: ${e.message}`, 'error');
        throw e;
    }
}

async function apiGet(endpoint) {
    try {
        const resp = await fetch(`${API_BASE}${endpoint}`);
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${resp.status}`);
        }
        return await resp.json();
    } catch (e) {
        showToast(`API Error: ${e.message}`, 'error');
        throw e;
    }
}

async function sendAction(payload) {
    try {
        const result = await apiPost('/action', payload);
        if (result) {
            handleActionResult(result);
        }
    } catch (e) {
        // Error already toasted by apiPost
    }
}

async function startGame(playerName, seed, difficulty) {
    try {
        const result = await apiPost('/game/new', {
            player_name: playerName,
            seed: seed || undefined,
            difficulty: difficulty,
        });
        if (result) {
            gameState = result;
            dom.newGameModal.classList.remove('active');
            dom.gameContainer.classList.remove('hidden');
            updateAllUI();
            connectWebSocket();
            showToast('Game started! Good luck.', 'success');
        }
    } catch (e) {
        // Error already toasted
    }
}

// ═══════════════════════════════════════════════════════════════
// WebSocket
// ═══════════════════════════════════════════════════════════════

function connectWebSocket() {
    if (ws && ws.readyState === WebSocket.OPEN) return;

    try {
        ws = new WebSocket(`${WS_BASE}/game`);
    } catch (e) {
        console.warn('WebSocket connection failed:', e);
        return;
    }

    ws.onopen = () => {
        wsReconnectAttempts = 0;
        console.log('WebSocket connected');
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleWSMessage(data);
        } catch (e) {
            console.warn('Invalid WS message:', e);
        }
    };

    ws.onclose = () => {
        console.log('WebSocket disconnected');
        scheduleReconnect();
    };

    ws.onerror = (err) => {
        console.warn('WebSocket error:', err);
    };
}

function scheduleReconnect() {
    if (wsReconnectAttempts >= 5) {
        showToast('Lost connection to server. Please refresh.', 'error');
        return;
    }
    const delay = Math.min(1000 * Math.pow(2, wsReconnectAttempts), 16000);
    wsReconnectAttempts++;
    setTimeout(() => connectWebSocket(), delay);
}

function handleWSMessage(data) {
    switch (data.type) {
        case 'state_update':
            gameState = data.payload;
            updateAllUI();
            break;
        case 'mdp_update':
            updateMDPGraph(data.payload);
            break;
        case 'narration':
            appendNarration(data.payload);
            break;
        case 'event':
            appendEvent(data.payload);
            break;
        case 'toast':
            showToast(data.payload.message, data.payload.level || 'info');
            break;
        case 'game_over':
            showGameOver(data.payload);
            break;
    }
}

// ═══════════════════════════════════════════════════════════════
// Action Result Handling
// ═══════════════════════════════════════════════════════════════

function handleActionResult(result) {
    if (result.state) {
        gameState = result.state;
        updateAllUI();
    }

    if (result.narration) {
        appendNarration(result.narration);
    }

    if (result.events) {
        result.events.forEach(evt => appendEvent(evt));
    }

    if (result.mdp_update) {
        updateMDPGraph(result.mdp_update);
    }

    if (result.game_over) {
        showGameOver(result.game_over);
    }
}

// ═══════════════════════════════════════════════════════════════
// UI Update Functions
// ═══════════════════════════════════════════════════════════════

function updateAllUI() {
    if (!gameState) return;
    updateHeader();
    updatePlayerPanel();
    updateInventory();
    updateEquipment();
    updateQuestPanel();
    updateNPCPanel();
    updateActionButtonStates();
}

function updateHeader() {
    const s = gameState;
    dom.turnCounter.textContent = s.turn ?? 0;
    dom.timePeriod.textContent = s.time_of_day ?? 'Morning';
    dom.currentLocation.textContent = LOCATION_NAMES[s.location] || s.location || '—';
}

function updatePlayerPanel() {
    const p = gameState.player || gameState;
    dom.playerName.textContent = p.name || '—';

    // Health
    const hp = p.health ?? 100;
    const maxHp = p.max_health ?? 100;
    const hpPct = Math.round((hp / maxHp) * 100);
    dom.healthBar.style.width = `${hpPct}%`;
    dom.healthText.textContent = `${hp} / ${maxHp}`;

    dom.healthBar.classList.remove('health-low', 'health-mid');
    if (hpPct <= 25) dom.healthBar.classList.add('health-low');
    else if (hpPct <= 50) dom.healthBar.classList.add('health-mid');

    // Stamina
    const sp = p.stamina ?? 50;
    const maxSp = p.max_stamina ?? 50;
    const spPct = Math.round((sp / maxSp) * 100);
    dom.staminaBar.style.width = `${spPct}%`;
    dom.staminaText.textContent = `${sp} / ${maxSp}`;

    dom.staminaBar.classList.toggle('stamina-low', spPct <= 20);

    // Reputation
    dom.globalReputation.textContent = p.global_reputation ?? 0;

    // Combat stats
    const combat = p.combat_stats || {};
    dom.playerAttack.textContent = (combat.base_attack ?? 8) + (combat.weapon_modifier ?? 0);
    dom.playerDefense.textContent = (combat.base_defense ?? 3) + (combat.armor_modifier ?? 0);
}

function updateInventory() {
    const items = gameState.player?.inventory || gameState.inventory || [];
    const maxInv = gameState.player?.max_inventory || 10;
    dom.inventoryCount.textContent = `${items.length}/${maxInv}`;

    if (items.length === 0) {
        dom.inventoryList.innerHTML = '<div class="empty-state">Empty</div>';
        return;
    }

    dom.inventoryList.innerHTML = items.map(item => `
        <div class="inventory-item" data-item-id="${item.id}" data-tooltip="${item.description || ''}">
            <span class="item-type-badge type-${item.type}">${item.type}</span>
            <span class="item-name">${item.name}</span>
            ${item.quest_relevant ? '<span class="item-quest-marker">★</span>' : ''}
        </div>
    `).join('');
}

function updateEquipment() {
    const equipped = gameState.player?.equipped || gameState.equipped || {};
    dom.equippedWeapon.textContent = equipped.weapon || 'None';
    dom.equippedWeapon.classList.toggle('equipped', !!equipped.weapon);
    dom.equippedArmor.textContent = equipped.armor || 'None';
    dom.equippedArmor.classList.toggle('equipped', !!equipped.armor);
}

function updateQuestPanel() {
    const quest = gameState.quest_state || gameState.quest || {};
    dom.questStage.textContent = quest.current_stage || '—';
    dom.questCheckpoint.textContent = quest.current_checkpoint || '—';

    const progress = quest.progress ?? 0;
    dom.questProgressFill.style.width = `${Math.round(progress * 100)}%`;

    dom.questDescription.textContent = quest.description || quest.stage_description || '';
}

function updateNPCPanel() {
    const npcs = gameState.npcs_here || [];
    if (npcs.length === 0) {
        dom.npcList.innerHTML = '<div class="empty-state">No NPCs at this location</div>';
        return;
    }

    dom.npcList.innerHTML = npcs.map(npc => {
        const rep = npc.reputation ?? 0;
        const tier = getRepTier(rep);
        const initial = (npc.name || '?')[0].toUpperCase();
        return `
            <div class="npc-card" data-npc-uid="${npc.uid}">
                <div class="npc-avatar">${initial}</div>
                <div class="npc-info">
                    <div class="npc-name">${npc.name}</div>
                    <div class="npc-archetype">${npc.archetype || ''}</div>
                </div>
                <span class="npc-rep ${tier.cls}" data-tooltip="${tier.label}">${rep >= 0 ? '+' : ''}${rep}</span>
            </div>
        `;
    }).join('');
}

function updateActionButtonStates() {
    const stamina = gameState.player?.stamina ?? gameState.stamina ?? 50;
    dom.actionBtns.forEach(btn => {
        const ap = parseInt(btn.dataset.ap, 10) || 0;
        btn.classList.toggle('insufficient-ap', ap > stamina && ap > 0);
    });
}

// ═══════════════════════════════════════════════════════════════
// Narrative Log
// ═══════════════════════════════════════════════════════════════

function appendNarration(entry) {
    if (typeof entry === 'string') {
        entry = { text: entry, type: 'system' };
    }

    const div = document.createElement('div');
    div.className = `narrative-entry narrative-${entry.type || 'system'}`;

    if (entry.importance) {
        div.dataset.importance = entry.importance;
    }

    let html = '';
    if (entry.turn !== undefined) {
        html += `<span class="narrative-time">T${entry.turn}</span>`;
    }
    if (entry.speaker) {
        const speakerClass = entry.speaker_type === 'player' ? 'speaker-player' : 'speaker-npc';
        html += `<span class="narrative-speaker ${speakerClass}">${entry.speaker}:</span> `;
    }
    html += `<span class="narrative-text">${escapeHtml(entry.text || entry.narration || '')}</span>`;

    div.innerHTML = html;
    dom.narrativeContent.appendChild(div);

    // Auto-scroll to bottom
    dom.narrativeContent.scrollTop = dom.narrativeContent.scrollHeight;
}

// ═══════════════════════════════════════════════════════════════
// Event Feed
// ═══════════════════════════════════════════════════════════════

function appendEvent(evt) {
    if (typeof evt === 'string') {
        evt = { text: evt, type: 'system' };
    }

    // Remove empty state
    const empty = dom.eventList.querySelector('.empty-state');
    if (empty) empty.remove();

    const div = document.createElement('div');
    div.className = `event-entry event-${evt.event_type || evt.type || 'system'}`;

    let html = '';
    if (evt.turn !== undefined) {
        html += `<span class="event-turn">T${evt.turn}</span>`;
    }
    html += escapeHtml(evt.text || evt.outcome || evt.narration || '');
    div.innerHTML = html;

    dom.eventList.prepend(div);

    // Keep max 50 events visible
    while (dom.eventList.children.length > 50) {
        dom.eventList.lastChild.remove();
    }
}

// ═══════════════════════════════════════════════════════════════
// MDP Graph (Cytoscape.js)
// ═══════════════════════════════════════════════════════════════

function initMDPGraph() {
    cy = cytoscape({
        container: dom.mdpGraph,
        style: getCytoscapeStyle(),
        layout: { name: 'preset' },
        elements: [],
        userZoomingEnabled: true,
        userPanningEnabled: true,
        boxSelectionEnabled: false,
        autoungrabify: true,
        minZoom: 0.3,
        maxZoom: 3,
    });

    // Fullscreen toggle
    dom.graphFullscreenBtn.addEventListener('click', () => {
        dom.mdpGraph.classList.toggle('fullscreen');
        setTimeout(() => cy.resize(), 100);
    });

    // Escape exits fullscreen
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            dom.mdpGraph.classList.remove('fullscreen');
            cy.resize();
            // Also close modals
            closeMoveModal();
            closeSaveLoadModal();
        }
    });
}

function getCytoscapeStyle() {
    return [
        // Default node
        {
            selector: 'node',
            style: {
                'label': 'data(label)',
                'text-valign': 'center',
                'text-halign': 'center',
                'font-size': '10px',
                'font-family': 'Inter, system-ui, sans-serif',
                'color': '#e8e8ed',
                'text-outline-width': 1.5,
                'text-outline-color': '#0a0a0f',
                'background-color': '#4A90D9',
                'width': 36,
                'height': 36,
                'shape': 'roundrectangle',
                'border-width': 1.5,
                'border-color': 'rgba(255,255,255,0.1)',
            },
        },
        // Static checkpoint
        {
            selector: 'node[type="static"]',
            style: {
                'background-color': '#4A90D9',
            },
        },
        // Dynamic checkpoint
        {
            selector: 'node[type="dynamic"]',
            style: {
                'background-color': '#E67E22',
                'shape': 'diamond',
                'width': 32,
                'height': 32,
            },
        },
        // Current node
        {
            selector: 'node[type="current"]',
            style: {
                'background-color': '#F39C12',
                'border-width': 2.5,
                'border-color': '#F39C12',
                'width': 42,
                'height': 42,
                'shadow-blur': 12,
                'shadow-color': 'rgba(243, 156, 18, 0.5)',
                'shadow-offset-x': 0,
                'shadow-offset-y': 0,
                'shadow-opacity': 1,
            },
        },
        // Completed
        {
            selector: 'node[type="completed"]',
            style: {
                'background-color': '#2ECC71',
                'opacity': 0.65,
            },
        },
        // Terminal success
        {
            selector: 'node[type="terminal_success"]',
            style: {
                'background-color': '#2ECC71',
                'shape': 'star',
                'width': 44,
                'height': 44,
            },
        },
        // Terminal fail
        {
            selector: 'node[type="terminal_fail"]',
            style: {
                'background-color': '#E74C3C',
                'shape': 'star',
                'width': 44,
                'height': 44,
            },
        },
        // Stage node (larger)
        {
            selector: 'node[type="stage"]',
            style: {
                'background-color': '#4A90D9',
                'width': 50,
                'height': 50,
                'font-size': '12px',
                'font-weight': 'bold',
            },
        },
        // Default edge
        {
            selector: 'edge',
            style: {
                'width': 1.5,
                'line-color': '#3a3a44',
                'target-arrow-color': '#3a3a44',
                'target-arrow-shape': 'triangle',
                'curve-style': 'bezier',
                'arrow-scale': 0.7,
            },
        },
        // Active path edge
        {
            selector: 'edge[type="active"]',
            style: {
                'line-color': '#F39C12',
                'target-arrow-color': '#F39C12',
                'width': 2.5,
            },
        },
        // Completed edge
        {
            selector: 'edge[type="completed"]',
            style: {
                'line-color': '#2ECC71',
                'target-arrow-color': '#2ECC71',
                'opacity': 0.5,
            },
        },
    ];
}

function updateMDPGraph(mdpData) {
    if (!cy || !mdpData) return;

    const elements = [];

    // Build nodes
    if (mdpData.nodes) {
        mdpData.nodes.forEach(node => {
            elements.push({
                group: 'nodes',
                data: {
                    id: node.id,
                    label: node.label || node.id,
                    type: node.status || node.type || 'static',
                },
                position: node.position || undefined,
            });
        });
    }

    // Build edges
    if (mdpData.edges) {
        mdpData.edges.forEach(edge => {
            elements.push({
                group: 'edges',
                data: {
                    id: `${edge.source}-${edge.target}`,
                    source: edge.source,
                    target: edge.target,
                    type: edge.status || edge.type || 'default',
                },
            });
        });
    }

    cy.json({ elements });

    // Relayout if no positions were provided
    if (mdpData.nodes && !mdpData.nodes[0]?.position) {
        cy.layout({
            name: 'breadthfirst',
            directed: true,
            spacingFactor: 1.2,
            animate: true,
            animationDuration: 300,
        }).run();
    }

    cy.fit(undefined, 20);
}

// ═══════════════════════════════════════════════════════════════
// Modals
// ═══════════════════════════════════════════════════════════════

function initModals() {
    // Start game
    dom.startGameBtn.addEventListener('click', () => {
        const name = dom.playerNameInput.value.trim() || 'Player';
        const seed = dom.seedInput.value ? parseInt(dom.seedInput.value, 10) : null;
        const difficulty = dom.difficultySelect.value;
        startGame(name, seed, difficulty);
    });

    // Enter on name field
    dom.playerNameInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') dom.startGameBtn.click();
    });

    // Game over buttons
    $('#new-game-after-btn')?.addEventListener('click', () => {
        dom.gameOverOverlay.classList.remove('active');
        dom.gameContainer.classList.add('hidden');
        dom.newGameModal.classList.add('active');
    });

    $('#export-log-btn')?.addEventListener('click', exportLog);

    // Move modal close
    $('#close-move-modal-btn')?.addEventListener('click', closeMoveModal);

    // Location buttons
    $$('.location-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const loc = btn.dataset.location;
            closeMoveModal();
            sendAction({ action_id: 'move_to', target_location: loc, source: 'button' });
        });
    });

    // Save/Load close
    $('#close-save-load-btn')?.addEventListener('click', closeSaveLoadModal);
}

function openMoveModal() {
    const currentLoc = gameState?.location || gameState?.player?.location;
    const adjacent = LOCATION_ADJACENCY[currentLoc] || [];

    $$('.location-btn').forEach(btn => {
        const loc = btn.dataset.location;
        btn.classList.remove('current-location', 'not-adjacent');
        btn.disabled = false;

        if (loc === currentLoc) {
            btn.classList.add('current-location');
            btn.disabled = true;
        } else if (!adjacent.includes(loc)) {
            btn.classList.add('not-adjacent');
            btn.disabled = true;
        }
    });

    dom.moveModal.classList.add('active');
}

function closeMoveModal() {
    dom.moveModal.classList.remove('active');
}

function closeSaveLoadModal() {
    dom.saveLoadModal.classList.remove('active');
}

function showGameOver(data) {
    const isVictory = data.result === 'success' || data.result === 'victory';
    dom.gameOverIcon.className = `game-over-icon ${isVictory ? 'victory' : 'defeat'}`;
    dom.gameOverTitle.textContent = isVictory ? 'Victory!' : 'Defeat';
    dom.gameOverMessage.textContent = data.message || '';

    $('#summary-turns').textContent = data.turns ?? gameState?.turn ?? '—';
    $('#summary-checkpoints').textContent = data.checkpoints_completed ?? '—';
    $('#summary-npcs').textContent = data.npcs_met ?? '—';
    $('#summary-combat').textContent = data.combat_encounters ?? '—';

    dom.gameOverOverlay.classList.add('active');
}

// ═══════════════════════════════════════════════════════════════
// Footer Controls
// ═══════════════════════════════════════════════════════════════

function initFooter() {
    dom.saveBtn.addEventListener('click', () => openSaveLoadModal('save'));
    dom.loadBtn.addEventListener('click', () => openSaveLoadModal('load'));

    dom.quickSaveBtn.addEventListener('click', async () => {
        try {
            await apiPost('/save', { slot: 0 });
            showToast('Quick saved!', 'success');
        } catch (e) { /* toasted */ }
    });

    dom.footerDifficulty.addEventListener('change', async () => {
        try {
            await apiPost('/difficulty', { difficulty: dom.footerDifficulty.value });
            showToast(`Difficulty set to ${dom.footerDifficulty.value}`, 'info');
        } catch (e) { /* toasted */ }
    });
}

async function openSaveLoadModal(mode) {
    const title = mode === 'save' ? 'Save Game' : 'Load Game';
    $('#save-load-title').textContent = title;

    // Fetch save info
    try {
        const saves = await apiGet('/saves');
        $$('.save-slot').forEach(slot => {
            const slotNum = parseInt(slot.dataset.slot, 10);
            const saveData = saves?.slots?.[slotNum];
            const infoEl = slot.querySelector('.slot-info');
            const actionBtn = slot.querySelector('.slot-action-btn');

            if (saveData) {
                infoEl.textContent = `Turn ${saveData.turn} — ${saveData.timestamp || ''}`;
            } else {
                infoEl.textContent = 'Empty';
            }

            actionBtn.textContent = mode === 'save' ? 'Save' : 'Load';
            actionBtn.onclick = async () => {
                try {
                    if (mode === 'save') {
                        await apiPost('/save', { slot: slotNum });
                        showToast(`Saved to Slot ${slotNum}`, 'success');
                    } else {
                        const result = await apiPost('/load', { slot: slotNum });
                        if (result) {
                            gameState = result;
                            updateAllUI();
                            showToast(`Loaded Slot ${slotNum}`, 'success');
                        }
                    }
                    closeSaveLoadModal();
                } catch (e) { /* toasted */ }
            };

            // Disable load on empty slots
            if (mode === 'load' && !saveData) {
                actionBtn.disabled = true;
            } else {
                actionBtn.disabled = false;
            }
        });
    } catch (e) {
        // Use fallback
    }

    dom.saveLoadModal.classList.add('active');
}

async function exportLog() {
    try {
        const data = await apiGet('/export');
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `game_log_${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
    } catch (e) {
        showToast('Export failed', 'error');
    }
}

// ═══════════════════════════════════════════════════════════════
// Toast Notifications
// ═══════════════════════════════════════════════════════════════

function showToast(message, level = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${level}`;

    const icons = { success: '✓', error: '✗', warning: '⚠', info: 'ℹ' };

    toast.innerHTML = `
        <span class="toast-icon">${icons[level] || icons.info}</span>
        <span class="toast-message">${escapeHtml(message)}</span>
        <button class="toast-close" onclick="this.parentElement.remove()">×</button>
    `;

    dom.toastContainer.appendChild(toast);

    // Auto-remove after 4s
    setTimeout(() => toast.remove(), 4000);
}

// ═══════════════════════════════════════════════════════════════
// Utilities
// ═══════════════════════════════════════════════════════════════

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function getRepTier(rep) {
    return REP_TIERS.find(t => rep >= t.min) || REP_TIERS[REP_TIERS.length - 1];
}

// ═══════════════════════════════════════════════════════════════
// Keyboard Shortcuts
// ═══════════════════════════════════════════════════════════════

document.addEventListener('keydown', (e) => {
    // Don't capture when typing in input fields
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;

    switch (e.key) {
        case '1': selectActionTab('navigation'); break;
        case '2': selectActionTab('exploration'); break;
        case '3': selectActionTab('social'); break;
        case '4': selectActionTab('combat'); break;
        case '5': selectActionTab('stealth'); break;
        case '6': selectActionTab('utility'); break;
        case '/':
        case 't':
            e.preventDefault();
            dom.freeTextInput.focus();
            break;
    }
});

function selectActionTab(category) {
    const tab = $(`.action-tab[data-category="${category}"]`);
    if (tab) tab.click();
}
