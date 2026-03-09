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
    talk: [
        { id: 'talk', name: 'Talk', ap: 1 },
        { id: 'greet', name: 'Greet', ap: 1 },
        { id: 'ask_info', name: 'Ask Info', ap: 1 },
        { id: 'persuade', name: 'Persuade', ap: 2 },
        { id: 'deceive', name: 'Deceive', ap: 2 },
        { id: 'intimidate', name: 'Intimidate', ap: 2 },
    ],
    social: [
        { id: 'trade', name: 'Trade', ap: 2 },
        { id: 'give_item', name: 'Give Item', ap: 1 },
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
    itemSelectRow:    $('#item-select-row'),
    itemSelect:       $('#item-select'),
    cancelActionBtn:  $('#cancel-action-btn'),

    // Text input
    freeTextInput:    $('#free-text-input'),
    sendTextBtn:      $('#send-text-btn'),

    // Talk Action Picker
    talkPicker:       $('#talk-action-picker'),
    talkPickerClose:  $('#talk-picker-close'),
    talkPickBtns:     $$('.talk-pick-btn'),
    talkPickerSendRaw:$('#talk-picker-send-raw'),

    // MDP Graph
    mdpGraph:         $('#mdp-graph'),
    graphFullscreenBtn: $('#graph-fullscreen-btn'),

    // NPC
    npcList:          $('#npc-list'),

    // Nearby Objects
    groundItemsList:  $('#ground-items-list'),
    groundItemCount:  $('#ground-item-count'),

    // Points of Interest
    poiList:          $('#poi-list'),
    poiCount:         $('#poi-count'),

    // Events
    eventList:        $('#event-list'),

    // RL Agent Activity
    rlAgentList:      $('#rl-agent-list'),

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
    applyURLParams();
    updateLLMStatus();
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

    // pick_up opens ground item selector when items are available
    if (actionId === 'pick_up') {
        const groundItems = gameState?.location?.items_on_ground || [];
        if (groundItems.length > 0) {
            showGroundItemPicker(groundItems);
            return;
        }
        // No items on ground — send directly, engine will respond "nothing to pick up"
        sendAction({ action_id: actionId, source: 'button' });
        return;
    }

    // Actions needing both NPC target AND item selection
    const needsNpcAndItem = ['give_item', 'present_item'];

    // Actions needing only NPC target
    const needsNpcTarget = ['talk', 'greet', 'ask_info', 'persuade', 'trade',
        'deceive', 'intimidate', 'attack', 'steal'];

    if (needsNpcAndItem.includes(actionId) && gameState?.npcs_here?.length) {
        showTargetSelector(actionId, true);
        return;
    }

    if (needsNpcTarget.includes(actionId) && gameState?.npcs_here?.length) {
        showTargetSelector(actionId, false);
        return;
    }

    // Direct execution
    sendAction({ action_id: actionId, source: 'button' });
}

function showTargetSelector(actionId, showItemSelect = false) {
    pendingAction = actionId;
    dom.targetSelector.classList.remove('hidden');

    // Populate NPC dropdown
    dom.targetSelect.innerHTML = '<option value="">Select NPC...</option>';
    if (gameState?.npcs_here) {
        gameState.npcs_here.forEach(npc => {
            const opt = document.createElement('option');
            opt.value = npc.npc_uid;
            opt.textContent = npc.name;
            dom.targetSelect.appendChild(opt);
        });
    }

    // Show/hide item selector
    if (showItemSelect) {
        dom.itemSelectRow.classList.remove('hidden');
        dom.itemSelect.innerHTML = '<option value="">Select item...</option>';
        const items = gameState.player?.inventory || gameState.inventory || [];
        items.forEach(item => {
            const opt = document.createElement('option');
            opt.value = item.id;
            opt.textContent = item.name + (item.quest_relevant ? ' ★' : '');
            dom.itemSelect.appendChild(opt);
        });
    } else {
        dom.itemSelectRow.classList.add('hidden');
    }
}

function confirmTargetedAction() {
    const targetUid = dom.targetSelect.value;
    if (!targetUid || !pendingAction) return;

    // pick_up: the dropdown contains ground item IDs, not NPC UIDs
    if (pendingAction === 'pick_up') {
        sendAction({
            action_id: 'pick_up',
            target_item: targetUid,
            source: 'button',
        });
        cancelTargetedAction();
        return;
    }

    // For item-requiring actions, also require item selection
    const needsItem = ['give_item', 'present_item'];
    const targetItem = needsItem.includes(pendingAction) ? dom.itemSelect?.value : null;
    if (needsItem.includes(pendingAction) && !targetItem) return;

    const payload = {
        action_id: pendingAction,
        target_npc: targetUid,
        source: 'button',
    };
    if (targetItem) payload.target_item = targetItem;

    sendAction(payload);
    cancelTargetedAction();
}

function cancelTargetedAction() {
    pendingAction = null;
    dom.targetSelector.classList.add('hidden');
    dom.itemSelectRow.classList.add('hidden');
}

/**
 * Show the target selector as a ground-item picker for pick_up.
 * Reuses the target-selector UI but populates the NPC dropdown with ground items.
 */
function showGroundItemPicker(groundItems) {
    pendingAction = 'pick_up';
    dom.targetSelector.classList.remove('hidden');
    dom.itemSelectRow.classList.add('hidden');

    // Repurpose the NPC dropdown for ground items
    dom.targetSelect.innerHTML = '<option value="">Select item to pick up...</option>';
    groundItems.forEach(item => {
        const opt = document.createElement('option');
        opt.value = item.id;
        const badge = item.quest_relevant ? ' ★' : '';
        opt.textContent = `${item.name}${badge} (${item.type || 'misc'})`;
        dom.targetSelect.appendChild(opt);
    });
}

// ═══════════════════════════════════════════════════════════════
// Text Input + Talk Action Picker
// ═══════════════════════════════════════════════════════════════

/**
 * Keywords that indicate the user is trying a talk-category action.
 * Maps keyword → best matching action_id from the talk category.
 */
const TALK_KEYWORDS = {
    // greet
    'greet':     'greet',
    'hello':     'greet',
    'hi ':       'greet',
    'hey ':      'greet',
    'wave':      'greet',
    'introduce': 'greet',
    'say hello': 'greet',
    'say hi':    'greet',
    // ask_info
    'ask':       'ask_info',
    'inquire':   'ask_info',
    'question':  'ask_info',
    'tell me':   'ask_info',
    'what do you know': 'ask_info',
    'any news':  'ask_info',
    'ask about': 'ask_info',
    // persuade
    'persuade':  'persuade',
    'convince':  'persuade',
    'plead':     'persuade',
    'reason with': 'persuade',
    'appeal':    'persuade',
    'coax':      'persuade',
    // deceive
    'deceive':   'deceive',
    'lie':       'deceive',
    'bluff':     'deceive',
    'trick':     'deceive',
    'mislead':   'deceive',
    // intimidate
    'intimidate':'intimidate',
    'threaten':  'intimidate',
    'scare':     'intimidate',
    'menace':    'intimidate',
    'demand':    'intimidate',
    'bully':     'intimidate',
    // talk (general)
    'talk':      'talk',
    'speak':     'talk',
    'chat':      'talk',
    'converse':  'talk',
    'discuss':   'talk',
    'compliment':'talk',
    'praise':    'talk',
};

/** Stored text when the talk picker is open, so we can send it later. */
let _pendingTalkText = null;
/** Best-guess action from keyword match. */
let _pendingTalkBestMatch = null;

function initTextInput() {
    dom.freeTextInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            submitFreeText();
        }
    });
    dom.sendTextBtn.addEventListener('click', submitFreeText);

    // Talk picker buttons
    dom.talkPickBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const actionId = btn.dataset.action;
            sendTalkPickerAction(actionId);
        });
    });
    dom.talkPickerClose.addEventListener('click', closeTalkPicker);
    dom.talkPickerSendRaw.addEventListener('click', sendTalkPickerRaw);
}

/**
 * Check if the user's text matches a talk-category keyword.
 * Returns { keyword, actionId } or null.
 */
function detectTalkKeyword(text) {
    const lower = text.toLowerCase();
    // Try longest keywords first for better matching
    const sorted = Object.keys(TALK_KEYWORDS).sort((a, b) => b.length - a.length);
    for (const kw of sorted) {
        if (lower.includes(kw.trim())) {
            return { keyword: kw, actionId: TALK_KEYWORDS[kw] };
        }
    }
    return null;
}

function submitFreeText() {
    const text = dom.freeTextInput.value.trim();
    if (!text) return;

    // Check if this looks like a talk-category action
    const talkMatch = detectTalkKeyword(text);
    if (talkMatch) {
        // Show the talk action picker instead of sending immediately
        _pendingTalkText = text;
        _pendingTalkBestMatch = talkMatch.actionId;
        showTalkPicker(talkMatch.actionId);
        return;
    }

    // Not a talk action — send directly to NLP parser
    sendAction({ text: text, source: 'text' });
    dom.freeTextInput.value = '';
}

function showTalkPicker(suggestedAction) {
    dom.talkPicker.classList.remove('hidden');
    // Highlight the suggested action
    dom.talkPickBtns.forEach(btn => {
        btn.classList.toggle('talk-pick-suggested', btn.dataset.action === suggestedAction);
    });
}

function closeTalkPicker() {
    dom.talkPicker.classList.add('hidden');
    _pendingTalkText = null;
    _pendingTalkBestMatch = null;
}

/**
 * User picked a specific action from the talk picker.
 * Send as a button action (precise) with the original text context.
 */
function sendTalkPickerAction(actionId) {
    if (!_pendingTalkText) return;

    // If there are NPCs here, prompt for target selection
    if (gameState?.npcs_here?.length) {
        closeTalkPicker();
        // Stash the text context and open the NPC target selector
        pendingAction = actionId;
        showTargetSelector(actionId, false);
    } else {
        // No NPCs — send directly, engine will handle "no target" gracefully
        sendAction({
            action_id: actionId,
            source: 'button',
        });
        dom.freeTextInput.value = '';
        closeTalkPicker();
    }
}

/**
 * User chose "Send as typed" — bypass picker, let NLP decide best action.
 */
function sendTalkPickerRaw() {
    if (!_pendingTalkText) return;
    sendAction({ text: _pendingTalkText, source: 'text' });
    dom.freeTextInput.value = '';
    closeTalkPicker();
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
        const result = await apiPost('/game/action', payload);
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
            // Clear the static welcome message before rendering game narration
            dom.narrativeContent.innerHTML = '';
            updateAllUI();
            // Display opening backstory narration
            if (result.opening_narration) {
                appendNarration({ text: result.opening_narration, type: 'backstory' });
            }
            // Render MDP graph from initial state
            if (result.graph) {
                updateMDPGraph(result.graph);
            }
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
        case 'state_sync':
        case 'state_update':
            gameState = data.data || data.payload;
            updateAllUI();
            if (gameState.graph) {
                updateMDPGraph(gameState.graph);
            }
            break;
        case 'graph_update':
        case 'mdp_update':
            updateMDPGraph(data.data || data.payload);
            break;
        case 'turn_result': {
            // The REST response already renders narration/events via
            // handleActionResult(), so we only sync state & graph here
            // to avoid duplicate narration entries.
            const tr = data.data || data.payload || {};
            if (tr.state) {
                gameState = tr.state;
                updateAllUI();
            }
            if (tr.state && tr.state.graph) {
                updateMDPGraph(tr.state.graph);
            }
            if (tr.game_over) {
                showGameOver(tr.game_over);
            }
            break;
        }
        case 'narration':
            appendNarration(data.data || data.payload);
            break;
        case 'event':
            appendEvent(data.data || data.payload);
            break;
        case 'npc_action':
            appendEvent(data.data || data.payload);
            break;
        case 'toast':
            showToast((data.data || data.payload).message, (data.data || data.payload).level || 'info');
            break;
        case 'game_over':
            showGameOver(data.data || data.payload);
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

    // Display NPC dialogue separately from narration
    if (result.dialogue) {
        appendNarration({
            text: result.dialogue,
            type: 'dialogue',
            speaker: result.dialogue_speaker || 'NPC',
            speaker_type: 'npc',
        });
    }

    // Handle events from both key names the engine may use
    const events = result.events || result.new_events || [];
    events.forEach(evt => appendEvent(evt));

    // NPC actions → RL Agent Activity panel (right sidebar)
    const npcActions = result.npc_actions || result.npc_narrations || [];
    npcActions.forEach(n => {
        const text = n.display_narration || n.narration || n.text || '';
        if (text) {
            appendRLUpdate({
                text: text,
                turn: result.turn,
                npc_name: n.name || n.npc_name || '',
            });
        }
    });

    // Update MDP graph from state or dedicated field
    if (result.mdp_update) {
        updateMDPGraph(result.mdp_update);
    } else if (result.state && result.state.graph) {
        updateMDPGraph(result.state.graph);
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
    updateNearbyObjects();
    updatePOIPanel();
    updateActionButtonStates();
}

function updateHeader() {
    const s = gameState;
    dom.turnCounter.textContent = s.turn ?? 0;
    dom.timePeriod.textContent = s.time_period ?? s.time_of_day ?? 'Morning';
    const locId = (typeof s.location === 'object' && s.location !== null) ? s.location.id : s.location;
    dom.currentLocation.textContent = LOCATION_NAMES[locId] || (s.location?.name) || locId || '—';
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
            <div class="npc-card" data-npc-uid="${npc.npc_uid}">
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

function updateNearbyObjects() {
    const loc = gameState.location || {};
    const items = loc.items_on_ground || [];
    dom.groundItemCount.textContent = items.length;

    if (items.length === 0) {
        dom.groundItemsList.innerHTML = '<div class="empty-state">Nothing on the ground</div>';
        return;
    }

    dom.groundItemsList.innerHTML = items.map(item => `
        <div class="ground-item" data-item-id="${item.id}">
            <span class="item-type-badge type-${item.type || 'misc'}">${item.type || 'misc'}</span>
            <span class="item-name">${item.name}</span>
            ${item.quest_relevant ? '<span class="item-quest-marker">★</span>' : ''}
            <button class="ground-item-pickup" data-item-id="${item.id}" title="Pick up ${item.name}">Pick Up</button>
        </div>
    `).join('');

    // Attach click handlers to pickup buttons
    dom.groundItemsList.querySelectorAll('.ground-item-pickup').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const itemId = btn.dataset.itemId;
            sendAction({ action_id: 'pick_up', target_item: itemId, source: 'button' });
        });
    });
}

function updatePOIPanel() {
    const loc = gameState.location || {};
    const pois = loc.discovered_pois || [];
    dom.poiCount.textContent = pois.length;

    if (pois.length === 0) {
        dom.poiList.innerHTML = '<div class="empty-state">No discoveries yet</div>';
        return;
    }

    dom.poiList.innerHTML = pois.map(poi => {
        const hint = poi.has_hidden_items ? '<div class="poi-hint">Something may be hidden here...</div>' : '';
        const searchBtn = poi.searchable
            ? `<button class="poi-btn poi-search" data-poi-id="${poi.poi_id}" title="Search near ${poi.name}">Search</button>`
            : '';
        return `
            <div class="poi-entry" data-poi-id="${poi.poi_id}">
                <div class="poi-header">
                    <span class="poi-name">${poi.name}</span>
                </div>
                <div class="poi-desc">${poi.description}</div>
                ${hint}
                <div class="poi-actions">
                    <button class="poi-btn poi-examine" data-poi-id="${poi.poi_id}" title="Examine ${poi.name}">Examine</button>
                    ${searchBtn}
                </div>
            </div>
        `;
    }).join('');

    // Attach examine handlers
    dom.poiList.querySelectorAll('.poi-examine').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            sendAction({ action_id: 'examine', target_item: btn.dataset.poiId, source: 'button' });
        });
    });

    // Attach search handlers
    dom.poiList.querySelectorAll('.poi-search').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            sendAction({ action_id: 'search', target_item: btn.dataset.poiId, source: 'button' });
        });
    });
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

function appendRLUpdate(entry) {
    if (!dom.rlAgentList) return;

    // Remove empty state
    const empty = dom.rlAgentList.querySelector('.empty-state');
    if (empty) empty.remove();

    const div = document.createElement('div');
    div.className = 'event-entry event-npc_action';

    let html = '';
    if (entry.turn !== undefined) {
        html += `<span class="event-turn">T${entry.turn}</span>`;
    }
    html += escapeHtml(entry.text || '');
    div.innerHTML = html;

    dom.rlAgentList.prepend(div);

    // Keep max 30 RL updates visible
    while (dom.rlAgentList.children.length > 30) {
        dom.rlAgentList.lastChild.remove();
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
        const isFullscreen = dom.mdpGraph.classList.toggle('fullscreen');
        setTimeout(() => {
            cy.resize();
            if (isFullscreen) {
                // In fullscreen: fit entire graph within viewport height
                cy.fit(undefined, 40);
            } else {
                // Normal: zoom to current node
                zoomToCurrentNode();
            }
        }, 150);
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
        /* ── Stage nodes (large orange circles) ───────────────── */
        {
            selector: 'node[kind="stage"]',
            style: {
                'label': 'data(label)',
                'text-valign': 'center',
                'text-halign': 'center',
                'font-size': '11px',
                'font-family': 'Inter, system-ui, sans-serif',
                'font-weight': 'bold',
                'color': '#1a1a1a',
                'text-outline-width': 0,
                'text-wrap': 'wrap',
                'text-max-width': '70px',
                'background-color': '#E67E22',
                'width': 80,
                'height': 80,
                'shape': 'ellipse',
                'border-width': 3,
                'border-color': '#1a1a1a',
            },
        },
        /* Stage completed → muted orange */
        {
            selector: 'node[type="stage_completed"]',
            style: {
                'background-color': '#E67E22',
                'opacity': 0.55,
                'border-color': '#2ECC71',
                'border-width': 3,
            },
        },
        /* Stage current → bright orange with glow */
        {
            selector: 'node[type="stage_current"]',
            style: {
                'background-color': '#F39C12',
                'border-color': '#F39C12',
                'border-width': 4,
                'shadow-blur': 15,
                'shadow-color': 'rgba(243, 156, 18, 0.6)',
                'shadow-offset-x': 0,
                'shadow-offset-y': 0,
                'shadow-opacity': 1,
            },
        },

        /* ── Checkpoint nodes (small circles) ─────────────────── */
        {
            selector: 'node[kind="checkpoint"]',
            style: {
                'label': 'data(label)',
                'text-valign': 'bottom',
                'text-halign': 'center',
                'text-margin-y': 6,
                'font-size': '9px',
                'font-family': "'JetBrains Mono', monospace",
                'color': '#8b8b96',
                'text-outline-width': 1,
                'text-outline-color': '#0a0a0f',
                'background-color': '#4A90D9',
                'width': 26,
                'height': 26,
                'shape': 'ellipse',
                'border-width': 2,
                'border-color': 'rgba(255,255,255,0.15)',
            },
        },
        /* Static (not yet reached) → blue */
        {
            selector: 'node[kind="checkpoint"][type="static"]',
            style: {
                'background-color': '#4A90D9',
            },
        },
        /* Completed → green */
        {
            selector: 'node[kind="checkpoint"][type="completed"]',
            style: {
                'background-color': '#2ECC71',
                'border-color': '#27ae60',
            },
        },
        /* Current → amber with pulse */
        {
            selector: 'node[kind="checkpoint"][type="current"]',
            style: {
                'background-color': '#F39C12',
                'border-color': '#F39C12',
                'border-width': 3,
                'width': 32,
                'height': 32,
                'shadow-blur': 12,
                'shadow-color': 'rgba(243, 156, 18, 0.6)',
                'shadow-offset-x': 0,
                'shadow-offset-y': 0,
                'shadow-opacity': 1,
            },
        },
        /* Dynamic → yellow */
        {
            selector: 'node[kind="checkpoint"][type="dynamic"]',
            style: {
                'background-color': '#F1C40F',
                'border-color': '#d4ac0d',
                'shape': 'diamond',
                'width': 28,
                'height': 28,
            },
        },

        /* ── Terminal nodes ────────────────────────────────────── */
        {
            selector: 'node[kind="terminal"]',
            style: {
                'label': 'data(label)',
                'text-valign': 'bottom',
                'text-halign': 'center',
                'text-margin-y': 6,
                'font-size': '9px',
                'font-family': 'Inter, system-ui, sans-serif',
                'color': '#8b8b96',
                'text-outline-width': 1,
                'text-outline-color': '#0a0a0f',
                'width': 32,
                'height': 32,
                'shape': 'star',
                'border-width': 2,
            },
        },
        {
            selector: 'node[type="terminal_success"]',
            style: {
                'background-color': '#2ECC71',
                'border-color': '#27ae60',
            },
        },
        {
            selector: 'node[type="terminal_fail"]',
            style: {
                'background-color': '#E74C3C',
                'border-color': '#c0392b',
            },
        },

        /* ── Edges ─────────────────────────────────────────────── */
        /* Stage-to-stage thick arrows */
        {
            selector: 'edge[type="stage_link"]',
            style: {
                'width': 4,
                'line-color': '#888',
                'target-arrow-color': '#888',
                'target-arrow-shape': 'triangle-backcurve',
                'curve-style': 'straight',
                'arrow-scale': 1.4,
            },
        },
        /* Stage-to-first-checkpoint */
        {
            selector: 'edge[type="stage_to_cp"]',
            style: {
                'width': 2,
                'line-color': '#4A72A8',
                'target-arrow-color': '#4A72A8',
                'target-arrow-shape': 'triangle',
                'curve-style': 'bezier',
                'arrow-scale': 0.8,
                'line-style': 'dashed',
                'line-dash-pattern': [6, 3],
            },
        },
        /* Default checkpoint edge */
        {
            selector: 'edge[type="default"]',
            style: {
                'width': 2,
                'line-color': '#4A72A8',
                'target-arrow-color': '#4A72A8',
                'target-arrow-shape': 'triangle',
                'curve-style': 'bezier',
                'arrow-scale': 0.8,
            },
        },
        /* Completed path */
        {
            selector: 'edge[type="completed"]',
            style: {
                'width': 2,
                'line-color': '#2ECC71',
                'target-arrow-color': '#2ECC71',
                'target-arrow-shape': 'triangle',
                'curve-style': 'bezier',
                'arrow-scale': 0.8,
                'opacity': 0.6,
            },
        },
        /* Terminal link */
        {
            selector: 'edge[type="terminal_link"]',
            style: {
                'width': 1.5,
                'line-color': '#5a5a66',
                'target-arrow-color': '#5a5a66',
                'target-arrow-shape': 'triangle',
                'curve-style': 'bezier',
                'arrow-scale': 0.7,
                'line-style': 'dotted',
            },
        },
    ];
}

function updateMDPGraph(mdpData) {
    if (!cy || !mdpData) return;

    const elements = [];

    // Build nodes with preset positions
    if (mdpData.nodes) {
        mdpData.nodes.forEach(node => {
            const ele = {
                group: 'nodes',
                data: {
                    id: node.id,
                    label: node.label || node.id,
                    type: node.type || 'static',
                    kind: node.kind || 'checkpoint',
                    stage_id: node.stage_id,
                    parent_stage: node.parent_stage || null,
                },
            };
            if (node.position) {
                ele.position = { x: node.position.x, y: node.position.y };
            }
            elements.push(ele);
        });
    }

    // Build edges
    if (mdpData.edges) {
        mdpData.edges.forEach(edge => {
            elements.push({
                group: 'edges',
                data: {
                    id: `${edge.source}->${edge.target}`,
                    source: edge.source,
                    target: edge.target,
                    type: edge.type || 'default',
                },
            });
        });
    }

    cy.json({ elements });
    cy.style(getCytoscapeStyle());

    // Fit the graph nicely
    setTimeout(() => {
        cy.fit(undefined, 30);
        // Then center on current stage area
        const current = cy.$('node[type="current"], node[type="stage_current"]');
        if (current.length) {
            cy.animate({
                center: { eles: current },
                zoom: Math.min(cy.zoom(), 1.5),
            }, { duration: 300 });
        }
    }, 100);
}

/**
 * Zoom/center the Cytoscape view on the current quest node.
 * Falls back to fitting the entire graph if no current node exists.
 */
function zoomToCurrentNode() {
    if (!cy) return;
    const current = cy.$('node[type="current"], node[type="stage_current"]');
    if (current.length) {
        cy.animate({
            center: { eles: current },
            zoom: Math.min(cy.zoom(), 1.5),
        }, { duration: 250 });
    } else {
        cy.fit(undefined, 20);
    }
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
    const rawLoc = gameState?.location;
    const currentLoc = (typeof rawLoc === 'object' && rawLoc !== null) ? rawLoc.id : (rawLoc || gameState?.player?.location);
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
        case 'm':
        case 'M':
            e.preventDefault();
            if (gameState && !dom.moveModal.classList.contains('active')) {
                openMoveModal();
            } else {
                closeMoveModal();
            }
            break;
        case 's':
            // Quick save (lowercase s)
            e.preventDefault();
            if (gameState && dom.quickSaveBtn) dom.quickSaveBtn.click();
            break;
        case 'S':
            // Open save modal (uppercase S)
            e.preventDefault();
            if (gameState) openSaveLoadModal('save');
            break;
        case 'h':
        case 'H':
            e.preventDefault();
            toggleHistoryPanel();
            break;
        case 'd':
        case 'D':
            e.preventDefault();
            toggleDebugOverlay();
            break;
        case 'n':
        case 'N':
            e.preventDefault();
            toggleNPCDetailPanel();
            break;
        case 'Tab':
            e.preventDefault();
            dom.freeTextInput.focus();
            break;
        case 'Escape':
            closeMoveModal();
            closeSaveLoadModal();
            dom.mdpGraph.classList.remove('fullscreen');
            if (cy) cy.resize();
            closeHistoryPanel();
            closeDebugOverlay();
            closeNPCDetailPanel();
            break;
    }
});

function selectActionTab(category) {
    const tab = $(`.action-tab[data-category="${category}"]`);
    if (tab) tab.click();
}

// ═══════════════════════════════════════════════════════════════
// History Panel (Turn-by-turn navigation)
// ═══════════════════════════════════════════════════════════════

let historyPanelOpen = false;

function toggleHistoryPanel() {
    historyPanelOpen ? closeHistoryPanel() : openHistoryPanel();
}

function openHistoryPanel() {
    let panel = $('#history-panel');
    if (!panel) {
        panel = document.createElement('div');
        panel.id = 'history-panel';
        panel.className = 'overlay-panel';
        panel.innerHTML = `
            <div class="overlay-panel-header">
                <h3>Turn History</h3>
                <button class="panel-close-btn" onclick="closeHistoryPanel()">×</button>
            </div>
            <div id="history-panel-content" class="overlay-panel-content"></div>
            <div class="history-nav">
                <button id="history-prev" class="btn-sm" onclick="navigateHistory(-1)">← Prev</button>
                <span id="history-indicator">—</span>
                <button id="history-next" class="btn-sm" onclick="navigateHistory(1)">Next →</button>
            </div>
        `;
        document.body.appendChild(panel);
    }
    panel.classList.add('active');
    historyPanelOpen = true;
    refreshHistoryPanel();
}

function closeHistoryPanel() {
    const panel = $('#history-panel');
    if (panel) panel.classList.remove('active');
    historyPanelOpen = false;
}

let historyViewTurn = null;

function refreshHistoryPanel() {
    const content = $('#history-panel-content');
    const indicator = $('#history-indicator');
    if (!content) return;

    const entries = dom.narrativeContent.querySelectorAll('.narrative-entry');
    if (entries.length === 0) {
        content.innerHTML = '<div class="empty-state">No history yet</div>';
        if (indicator) indicator.textContent = '—';
        return;
    }

    // Collect turns from narrative entries
    const turnMap = {};
    entries.forEach(entry => {
        const timeEl = entry.querySelector('.narrative-time');
        const turnText = timeEl ? timeEl.textContent : 'T?';
        const turn = turnText.replace('T', '') || '?';
        if (!turnMap[turn]) turnMap[turn] = [];
        turnMap[turn].push(entry.querySelector('.narrative-text')?.textContent || entry.textContent);
    });

    const turns = Object.keys(turnMap).sort((a, b) => parseInt(b) - parseInt(a));
    if (historyViewTurn === null) historyViewTurn = turns.length - 1;
    historyViewTurn = Math.max(0, Math.min(historyViewTurn, turns.length - 1));

    const currentKey = turns[historyViewTurn];
    const turnEntries = turnMap[currentKey] || [];

    content.innerHTML = turnEntries.map(t =>
        `<div class="history-entry">${escapeHtml(t)}</div>`
    ).join('');

    if (indicator) indicator.textContent = `Turn ${currentKey} (${historyViewTurn + 1}/${turns.length})`;
}

function navigateHistory(delta) {
    historyViewTurn = (historyViewTurn || 0) + delta;
    refreshHistoryPanel();
}

// ═══════════════════════════════════════════════════════════════
// NPC Detail Panel
// ═══════════════════════════════════════════════════════════════

let npcDetailPanelOpen = false;

function toggleNPCDetailPanel() {
    npcDetailPanelOpen ? closeNPCDetailPanel() : openNPCDetailPanel();
}

function openNPCDetailPanel() {
    let panel = $('#npc-detail-panel');
    if (!panel) {
        panel = document.createElement('div');
        panel.id = 'npc-detail-panel';
        panel.className = 'overlay-panel overlay-panel-wide';
        panel.innerHTML = `
            <div class="overlay-panel-header">
                <h3>NPC Details</h3>
                <button class="panel-close-btn" onclick="closeNPCDetailPanel()">×</button>
            </div>
            <div id="npc-detail-content" class="overlay-panel-content"></div>
        `;
        document.body.appendChild(panel);
    }
    panel.classList.add('active');
    npcDetailPanelOpen = true;
    refreshNPCDetailPanel();
}

function closeNPCDetailPanel() {
    const panel = $('#npc-detail-panel');
    if (panel) panel.classList.remove('active');
    npcDetailPanelOpen = false;
}

function refreshNPCDetailPanel() {
    const content = $('#npc-detail-content');
    if (!content || !gameState) return;

    const npcs = gameState.npcs_here || [];
    const allNpcs = gameState.all_npcs || npcs;

    if (allNpcs.length === 0) {
        content.innerHTML = '<div class="empty-state">No NPC data available</div>';
        return;
    }

    content.innerHTML = allNpcs.map(npc => {
        const rep = npc.reputation ?? 0;
        const tier = getRepTier(rep);
        const location = npc.location || '?';
        const mood = npc.mood || 'unknown';
        const hp = npc.health ?? npc.current_hp ?? '?';
        const status = npc.status || 'active';
        const conversations = npc.conversation_count ?? 0;

        return `
            <div class="npc-detail-card">
                <div class="npc-detail-header">
                    <span class="npc-detail-name">${npc.name || npc.npc_uid}</span>
                    <span class="npc-rep ${tier.cls}">${rep >= 0 ? '+' : ''}${rep} ${tier.label}</span>
                </div>
                <div class="npc-detail-body">
                    <div class="npc-detail-stat">Archetype: ${npc.archetype || '—'}</div>
                    <div class="npc-detail-stat">Location: ${LOCATION_NAMES[location] || location}</div>
                    <div class="npc-detail-stat">Status: ${status}</div>
                    <div class="npc-detail-stat">Mood: ${mood} | HP: ${hp}</div>
                    <div class="npc-detail-stat">Conversations: ${conversations}</div>
                </div>
            </div>
        `;
    }).join('');
}

// ═══════════════════════════════════════════════════════════════
// Debug Overlay (NPC thought bubbles, Q-values)
// ═══════════════════════════════════════════════════════════════

let debugOverlayOpen = false;

function toggleDebugOverlay() {
    debugOverlayOpen ? closeDebugOverlay() : openDebugOverlay();
}

function openDebugOverlay() {
    let panel = $('#debug-overlay');
    if (!panel) {
        panel = document.createElement('div');
        panel.id = 'debug-overlay';
        panel.className = 'overlay-panel overlay-panel-wide';
        panel.innerHTML = `
            <div class="overlay-panel-header">
                <h3>Debug / Research</h3>
                <button class="panel-close-btn" onclick="closeDebugOverlay()">×</button>
            </div>
            <div id="debug-content" class="overlay-panel-content"></div>
        `;
        document.body.appendChild(panel);
    }
    panel.classList.add('active');
    debugOverlayOpen = true;
    refreshDebugOverlay();
}

function closeDebugOverlay() {
    const panel = $('#debug-overlay');
    if (panel) panel.classList.remove('active');
    debugOverlayOpen = false;
}

async function refreshDebugOverlay() {
    const content = $('#debug-content');
    if (!content) return;
    content.innerHTML = '<div class="empty-state">Loading...</div>';

    try {
        const metrics = await apiGet('/metrics/summary');
        const llmStatus = await apiGet('/llm/status');

        let html = '<div class="debug-section"><h4>Game Metrics</h4>';
        html += `<div class="debug-stat">Turn: ${metrics.turn ?? '—'}</div>`;
        html += `<div class="debug-stat">Total Actions: ${metrics.total_actions ?? '—'}</div>`;
        html += `<div class="debug-stat">Difficulty: ${metrics.difficulty ?? '—'}</div>`;
        if (metrics.actions_by_type) {
            html += '<div class="debug-stat">Actions: ' +
                Object.entries(metrics.actions_by_type).map(([k, v]) => `${k}:${v}`).join(', ') +
                '</div>';
        }
        html += '</div>';

        html += '<div class="debug-section"><h4>LLM Status</h4>';
        html += `<div class="debug-stat">Available: ${llmStatus.available ? 'Yes' : 'No'}</div>`;
        html += `<div class="debug-stat">Model: ${llmStatus.model_path ?? '—'}</div>`;
        html += `<div class="debug-stat">Calls/min: ${llmStatus.calls_this_minute ?? 0}/${llmStatus.max_calls ?? 20}</div>`;
        html += '</div>';

        // NPC thought bubbles (Q-value insights)
        if (gameState?.npcs_here?.length) {
            html += '<div class="debug-section"><h4>NPC Thoughts (at location)</h4>';
            for (const npc of gameState.npcs_here) {
                const top_action = npc.debug_top_action || '?';
                const q_value = npc.debug_q_value != null ? npc.debug_q_value.toFixed(2) : '?';
                html += `<div class="debug-npc-thought">
                    <strong>${npc.name}</strong>: considering "${top_action}" (Q=${q_value})
                </div>`;
            }
            html += '</div>';
        }

        content.innerHTML = html;
    } catch (e) {
        content.innerHTML = `<div class="empty-state">Error loading debug data: ${escapeHtml(e.message)}</div>`;
    }
}

// ═══════════════════════════════════════════════════════════════
// Metrics Dashboard
// ═══════════════════════════════════════════════════════════════

async function showMetricsDashboard() {
    let panel = $('#metrics-panel');
    if (!panel) {
        panel = document.createElement('div');
        panel.id = 'metrics-panel';
        panel.className = 'overlay-panel overlay-panel-wide';
        panel.innerHTML = `
            <div class="overlay-panel-header">
                <h3>Session Metrics</h3>
                <button class="panel-close-btn" onclick="document.querySelector('#metrics-panel').classList.remove('active')">×</button>
            </div>
            <div id="metrics-content" class="overlay-panel-content"></div>
        `;
        document.body.appendChild(panel);
    }
    panel.classList.add('active');

    const content = $('#metrics-content');
    content.innerHTML = '<div class="empty-state">Loading metrics...</div>';

    try {
        const m = await apiGet('/metrics/summary');
        let html = '<div class="metrics-grid">';

        html += metricCard('Total Turns', m.turn ?? 0);
        html += metricCard('Actions Taken', m.total_actions ?? 0);
        html += metricCard('Difficulty', m.difficulty ?? 'normal');
        html += metricCard('Checkpoints', m.checkpoints_completed ?? 0);
        html += metricCard('Combat Encounters', m.combat_encounters ?? 0);
        html += metricCard('Rep Changes', m.reputation_changes ?? 0);
        html += metricCard('Dynamic CPs', m.dynamic_checkpoints ?? 0);
        html += metricCard('LLM Calls', m.llm_calls ?? 0);

        // Action breakdown
        if (m.actions_by_type && Object.keys(m.actions_by_type).length) {
            html += '</div><div class="debug-section"><h4>Action Breakdown</h4><div class="metrics-bar-chart">';
            const sorted = Object.entries(m.actions_by_type).sort((a, b) => b[1] - a[1]);
            const maxVal = sorted[0]?.[1] || 1;
            for (const [action, count] of sorted) {
                const pct = Math.round((count / maxVal) * 100);
                html += `<div class="metric-bar-row">
                    <span class="metric-bar-label">${action}</span>
                    <div class="metric-bar-track"><div class="metric-bar-fill" style="width:${pct}%"></div></div>
                    <span class="metric-bar-value">${count}</span>
                </div>`;
            }
            html += '</div></div>';
        } else {
            html += '</div>';
        }

        content.innerHTML = html;
    } catch (e) {
        content.innerHTML = `<div class="empty-state">Error: ${escapeHtml(e.message)}</div>`;
    }
}

function metricCard(label, value) {
    return `<div class="metric-card"><div class="metric-value">${value}</div><div class="metric-label">${label}</div></div>`;
}

// ═══════════════════════════════════════════════════════════════
// LLM Status Indicator
// ═══════════════════════════════════════════════════════════════

async function updateLLMStatus() {
    try {
        const status = await apiGet('/llm/status');
        if (dom.llmDot) {
            dom.llmDot.classList.toggle('llm-active', status.available === true);
            dom.llmDot.title = status.available ? 'LLM Active' : 'LLM Inactive';
        }
        const statusText = dom.statusText;
        if (statusText && status.available) {
            statusText.textContent = 'LLM Active';
        } else if (statusText) {
            statusText.textContent = 'LLM Offline';
        }
    } catch {
        if (dom.llmDot) {
            dom.llmDot.classList.remove('llm-active');
            dom.llmDot.title = 'LLM Status Unknown';
        }
    }
}

// Poll LLM status every 30 seconds when game is running
setInterval(() => {
    if (gameState) updateLLMStatus();
}, 30000);

// ═══════════════════════════════════════════════════════════════
// URL Parameter Overrides
// ═══════════════════════════════════════════════════════════════

function applyURLParams() {
    const params = new URLSearchParams(window.location.search);

    const autoStart = params.get('auto_start');
    const playerName = params.get('player_name');
    const seed = params.get('seed');
    const difficulty = params.get('difficulty');
    const profile = params.get('profile');

    if (playerName && dom.playerNameInput) dom.playerNameInput.value = playerName;
    if (seed && dom.seedInput) dom.seedInput.value = seed;
    if (difficulty && dom.difficultySelect) dom.difficultySelect.value = difficulty;

    if (autoStart === 'true' || autoStart === '1') {
        // Auto-start the game after a brief delay for DOM readiness
        setTimeout(() => {
            const name = dom.playerNameInput?.value?.trim() || playerName || 'Player';
            const s = dom.seedInput?.value ? parseInt(dom.seedInput.value, 10) : (seed ? parseInt(seed, 10) : null);
            const d = dom.difficultySelect?.value || difficulty || 'normal';
            startGame(name, s, d);
        }, 200);
    }
}
