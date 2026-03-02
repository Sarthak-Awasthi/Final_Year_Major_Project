"""
config.py — All configuration constants for the MVP game.

Centralizes tunable parameters so nothing is scattered in code.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SAVES_DIR = DATA_DIR / "saves"
METRICS_DIR = DATA_DIR / "metrics"
LOGS_DIR = DATA_DIR / "logs"
QUEST_DIR = DATA_DIR / "quests"
NPC_DIR = DATA_DIR / "npcs"
WORLD_DIR = DATA_DIR / "world"
CONFIG_DIR = DATA_DIR / "config"
MODELS_DIR = BASE_DIR.parent / "models"

# Ensure directories exist
for _d in (SAVES_DIR, METRICS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ─── Master Seed ──────────────────────────────────────────────────────────────
MASTER_SEED: int = 42

# ─── Session ──────────────────────────────────────────────────────────────────
MAX_TURNS: int = 200
GAME_VERSION: str = "mvp-1.0"
SAVE_VERSION: str = "1.0"

# ─── Time System ──────────────────────────────────────────────────────────────
TURNS_PER_PERIOD: int = 4
TIME_PERIODS: list[str] = ["morning", "midday", "afternoon", "evening", "night"]

# ─── Player Defaults ─────────────────────────────────────────────────────────
PLAYER_MAX_HEALTH: int = 100
PLAYER_MAX_STAMINA: int = 50
PLAYER_MAX_INVENTORY: int = 10
PLAYER_BASE_ATTACK: int = 8
PLAYER_BASE_DEFENSE: int = 3
STAMINA_REGEN_PER_TURN: int = 5

# ─── Universal Action Catalog ────────────────────────────────────────────────
UNIVERSAL_ACTIONS: dict[str, dict] = {
    # Navigation
    "move_to":    {"label": "Move to location",       "category": "navigation",  "base_ap": 3},
    # Exploration
    "look":       {"label": "Look around",             "category": "exploration", "base_ap": 1},
    "search":     {"label": "Search area thoroughly",  "category": "exploration", "base_ap": 5},
    "examine":    {"label": "Examine object / person", "category": "exploration", "base_ap": 2},
    # Talk (conversation / persuasion)
    "talk":       {"label": "Talk to NPC",             "category": "talk",        "base_ap": 1},
    "greet":      {"label": "Greet someone",           "category": "talk",        "base_ap": 1},
    "ask_info":   {"label": "Ask for information",     "category": "talk",        "base_ap": 1},
    "persuade":   {"label": "Persuade / convince",     "category": "talk",        "base_ap": 2},
    "deceive":    {"label": "Lie / bluff",             "category": "talk",        "base_ap": 2},
    "intimidate": {"label": "Threaten / intimidate",   "category": "talk",        "base_ap": 2},
    # Social (exchange / barter)
    "trade":      {"label": "Trade items",             "category": "social",      "base_ap": 2},
    "give_item":  {"label": "Give item to NPC",        "category": "social",      "base_ap": 1},
    "present_item": {"label": "Present / show item to NPC", "category": "social",  "base_ap": 1},
    # Combat
    "attack":     {"label": "Attack / fight",          "category": "combat",      "base_ap": 10},
    "defend":     {"label": "Defend / block",          "category": "combat",      "base_ap": 5},
    "flee":       {"label": "Flee / run away",         "category": "combat",      "base_ap": 5},
    # Stealth
    "sneak":      {"label": "Sneak / move stealthily", "category": "stealth",     "base_ap": 5},
    "hide":       {"label": "Hide from view",          "category": "stealth",     "base_ap": 3},
    "steal":      {"label": "Steal item",              "category": "stealth",     "base_ap": 5},
    # Utility
    "pick_up":    {"label": "Pick up item",            "category": "utility",     "base_ap": 1},
    "use_item":   {"label": "Use inventory item",      "category": "utility",     "base_ap": 2},
    "eat":        {"label": "Eat food",                "category": "utility",     "base_ap": 1},
    "rest":       {"label": "Rest / wait",             "category": "utility",     "base_ap": 0},
    "wait":       {"label": "Wait and observe",        "category": "utility",     "base_ap": 0},
    "drop_item":  {"label": "Drop an item",            "category": "utility",     "base_ap": 0},
    "status":     {"label": "Check quest journal",     "category": "utility",     "base_ap": 0},
    "equip":      {"label": "Equip weapon or armor",   "category": "utility",     "base_ap": 1},
    "work":       {"label": "Perform labor",           "category": "utility",     "base_ap": 5},
}

UNIVERSAL_ACTION_IDS: list[str] = list(UNIVERSAL_ACTIONS.keys())
ACTION_CATEGORIES: list[str] = ["navigation", "exploration", "talk", "social", "combat", "stealth", "utility"]

# Actions in the "talk" category — used by text-input interception
TALK_CATEGORY_ACTIONS: set[str] = {"talk", "greet", "ask_info", "persuade", "deceive", "intimidate"}

# ─── Reputation ──────────────────────────────────────────────────────────────
REPUTATION_MIN: int = -100
REPUTATION_MAX: int = 100
REPUTATION_DECAY_INTERVAL: int = 20
REPUTATION_DECAY_AMOUNT: int = 1
REPUTATION_THRESHOLDS: dict[str, tuple[int, int]] = {
    "trusted":    (50, 100),
    "friendly":   (20, 49),
    "neutral":    (-19, 19),
    "suspicious": (-49, -20),
    "hostile":    (-100, -50),
}

# ─── Gossip ──────────────────────────────────────────────────────────────────
GOSSIP_PROBABILITY: float = 0.4
GOSSIP_DECAY_FACTOR: float = 0.5
GOSSIP_MAX_HOPS: int = 3
GOSSIP_MIN_DELTA: int = 2

# ─── Combat ──────────────────────────────────────────────────────────────────
COMBAT_HIT_MIN: float = 0.1
COMBAT_HIT_MAX: float = 0.95
COMBAT_MIN_DAMAGE: int = 1
COMBAT_DAMAGE_VARIANCE: int = 3
COMBAT_DEFEND_REDUCTION: float = 0.5
COMBAT_FLEE_BASE_SUCCESS: float = 0.70
COMBAT_FLEE_BONUS_HIT: float = 0.20
COMBAT_FLEE_EXHAUSTED_SUCCESS: float = 0.40
INCAPACITATION_TURNS: int = 20
INCAPACITATION_REPUTATION: int = -80
INCAPACITATION_WITNESS_PENALTY: int = -15

# ─── Skill Checks ───────────────────────────────────────────────────────────
SOCIAL_MODIFIERS: dict[str, int] = {
    "polite": 5, "honest": 3, "cooperative": 4, "neutral": 0,
    "rude": -5, "deceptive": -3, "intimidating": 0,
}

# ─── NPC RL ──────────────────────────────────────────────────────────────────
NPC_Q_LEARNING_ALPHA: float = 0.1
NPC_Q_LEARNING_GAMMA: float = 0.9
NPC_EPSILON_PRETRAIN: float = 0.5
NPC_EPSILON_START: float = 0.15
NPC_EPSILON_MIN: float = 0.05
NPC_EPSILON_DECAY_RATE: float = 0.995
NPC_COLD_START_TURNS: int = 20
NPC_PRETRAIN_EPISODES: int = 100
NPC_PRETRAIN_TURNS: int = 50
NPC_INVALID_LOCATION_PENALTY: float = -5.0

# NPC state-space sizes for Q-table
NPC_NUM_LOCATIONS: int = 5
NPC_NUM_TIME_SLOTS: int = 5  # morning, midday, afternoon, evening, night
NPC_NUM_ENERGY_LEVELS: int = 3  # low, medium, high
NPC_NUM_MOOD_LEVELS: int = 3  # low, medium, high
NPC_STATE_SPACE_SIZE: int = NPC_NUM_LOCATIONS * NPC_NUM_TIME_SLOTS * NPC_NUM_ENERGY_LEVELS * NPC_NUM_MOOD_LEVELS
NPC_ACTION_SPACE_SIZE: int = len(UNIVERSAL_ACTIONS)

# ─── LLM ─────────────────────────────────────────────────────────────────────
LLM_ENABLED: bool = True
LLM_MODEL_PATH: str = str(MODELS_DIR / "Phi-3.5-mini-instruct-Q4_K_M.gguf")
LLM_CONTEXT_SIZE: int = 4096
LLM_GPU_LAYERS: int = -1  # -1 = offload all layers to GPU; 0 = CPU only
LLM_MAX_PROMPT_TOKENS: int = 2500
LLM_DEFAULT_TEMPERATURE: float = 0.7
LLM_TIMEOUT_SECONDS: int = 10
LLM_MAX_RETRIES: int = 2
LLM_MIN_INTERVAL_MS: int = 2000
LLM_MAX_CALLS_PER_MINUTE: int = 20

LLM_TEMPERATURES: dict[str, float] = {
    "input_parsing": 0.4,
    "npc_dialogue": 0.7,
    "narration": 0.7,
    "checkpoint_gen": 0.85,
}

# ─── Dynamic Checkpoints & Nudging ───────────────────────────────────────────
NUDGE_LAMBDA: float = 0.3
NUDGE_HINT_THRESHOLD: int = 3
NUDGE_FORCE_CONVERGENCE_THRESHOLD: int = 5
DYNAMIC_CP_LOOP_THRESHOLD: int = 3

# ─── Event Log ───────────────────────────────────────────────────────────────
MAX_EVENT_LOG_SIZE: int = 500
MAX_CONVERSATION_HISTORY: int = 10
LLM_CONVERSATION_CONTEXT: int = 5

# ─── Save/Load ───────────────────────────────────────────────────────────────
AUTO_SAVE_INTERVAL: int = 5
MAX_AUTO_SAVES: int = 3
MAX_MANUAL_SAVES: int = 3

# ─── Random Events ──────────────────────────────────────────────────────────
RANDOM_EVENT_FREQUENCY_MULTIPLIER: float = 1.0

# ─── Passive Perception ─────────────────────────────────────────────────────
PASSIVE_PERCEPTION_BASE: float = 0.4
PASSIVE_PERCEPTION_SOCIAL_BONUS: float = 0.2

# ─── Difficulty Presets ──────────────────────────────────────────────────────
DIFFICULTY_PRESETS: dict[str, dict] = {
    "easy": {
        "ap_cost_multiplier": 0.75,
        "combat_damage_to_player": 0.7,
        "combat_damage_from_player": 1.3,
        "npc_hostility_threshold": -60,
        "reputation_gain_multiplier": 1.5,
        "reputation_loss_multiplier": 0.7,
        "nudge_aggressiveness": "high",
        "max_deviations_before_convergence": 7,
        "stamina_regen_per_turn": 8,
        "gossip_propagation_rate": 0.2,
        "random_event_frequency": 0.5,
        "combat_flee_success_rate": 0.80,
    },
    "normal": {
        "ap_cost_multiplier": 1.0,
        "combat_damage_to_player": 1.0,
        "combat_damage_from_player": 1.0,
        "npc_hostility_threshold": -50,
        "reputation_gain_multiplier": 1.0,
        "reputation_loss_multiplier": 1.0,
        "nudge_aggressiveness": "medium",
        "max_deviations_before_convergence": 5,
        "stamina_regen_per_turn": 5,
        "gossip_propagation_rate": 0.4,
        "random_event_frequency": 1.0,
        "combat_flee_success_rate": 0.70,
    },
    "hard": {
        "ap_cost_multiplier": 1.5,
        "combat_damage_to_player": 1.4,
        "combat_damage_from_player": 0.8,
        "npc_hostility_threshold": -35,
        "reputation_gain_multiplier": 0.7,
        "reputation_loss_multiplier": 1.5,
        "nudge_aggressiveness": "low",
        "max_deviations_before_convergence": 3,
        "stamina_regen_per_turn": 3,
        "gossip_propagation_rate": 0.6,
        "random_event_frequency": 1.5,
        "combat_flee_success_rate": 0.55,
    },
}

# ─── NLP Keywords ────────────────────────────────────────────────────────────
EMOTION_KEYWORDS: dict[str, list[str]] = {
    "angry":       ["furious", "angry", "rage", "damn", "hate", "kill", "fuming", "livid", "outraged", "furiously"],
    "friendly":    ["please", "kindly", "hello", "hi", "friend", "thanks", "grateful", "appreciate",
                    "compliment", "praise", "flatter", "admire", "respect", "love", "nice", "good job",
                    "well done", "thank", "glad", "happy", "cheerful", "warm", "fondly", "gently"],
    "fearful":     ["scared", "afraid", "nervous", "run", "hide", "danger", "worried", "anxious",
                    "terrified", "frightened", "uneasy", "dread", "wary"],
    "curious":     ["wonder", "what", "why", "how", "interesting", "tell me", "know about",
                    "ask about", "curious", "explain", "learn", "understand", "question",
                    "any news", "what's going on", "what happened"],
    "threatening": ["or else", "better", "warn", "threat", "regret", "destroy", "suffer",
                    "punish", "consequences", "pay for"],
    "neutral":     [],
}

SOCIAL_KEYWORDS: dict[str, list[str]] = {
    "polite":       ["please", "excuse me", "thank you", "sir", "ma'am", "kindly", "respectfully",
                     "could you", "would you", "if you don't mind", "pardon", "with respect"],
    "rude":         ["idiot", "fool", "shut up", "get lost", "stupid", "out of my way",
                     "worthless", "scum", "pathetic", "useless"],
    "deceptive":    ["trick", "lie", "pretend", "disguise", "fool them", "bluff",
                     "mislead", "fabricate", "make up"],
    "honest":       ["truth", "honest", "truly", "really", "sincerely", "genuinely",
                     "frankly", "openly", "straightforward"],
    "intimidating": ["threaten", "scare", "force", "make them", "demand", "or else",
                     "obey", "comply", "submit"],
    "cooperative":  ["help", "together", "work with", "assist", "support", "cooperate",
                     "ally", "partner", "team up", "join"],
    "neutral":      [],
}

ACTION_SYNONYMS: dict[str, list[str]] = {
    "move_to":    ["go to", "move to", "walk to", "travel to", "head to", "go", "walk", "travel"],
    "look":       ["look around", "look", "observe", "see", "glance"],
    "search":     ["search", "investigate", "explore area", "rummage", "look for"],
    "examine":    ["examine", "inspect", "study", "check out"],
    "talk":       ["talk", "speak", "chat", "converse", "discuss", "say", "tell",
                   "compliment", "praise", "flatter", "respond", "reply", "address",
                   "call out", "comment", "remark", "mention", "bring up"],
    "greet":      ["greet", "hello", "say hello", "say hi", "hi", "wave", "approach",
                   "introduce myself", "introduce"],
    "ask_info":   ["ask", "ask about", "inquire", "question", "tell me about",
                   "what do you know", "know about", "ask for", "request info",
                   "what can you tell", "any information", "any news"],
    "persuade":   ["persuade", "convince", "plead", "reason with", "appeal", "coax", "urge"],
    "trade":      ["trade", "buy", "sell", "barter", "shop", "purchase", "deal"],
    "give_item":  ["give", "hand over", "offer", "donate"],
    "present_item": ["show to", "present to", "present item", "show item", "display to", "produce", "show", "present", "show papers", "present papers"],
    "deceive":    ["deceive", "lie", "bluff", "trick", "mislead", "fool"],
    "intimidate": ["intimidate", "threaten", "scare", "menace", "bully", "demand"],
    "attack":     ["attack", "fight", "hit", "strike", "punch", "slash", "combat"],
    "defend":     ["defend", "block", "shield", "parry", "guard"],
    "flee":       ["flee", "run", "escape", "run away", "retreat", "bolt"],
    "sneak":      ["sneak", "stealth", "creep", "slip past", "tiptoe", "sneak past"],
    "hide":       ["hide", "conceal", "duck", "cover", "lurk"],
    "steal":      ["steal", "pickpocket", "pilfer", "swipe", "take secretly", "nick"],
    "pick_up":    ["pick up", "grab", "take", "collect", "get"],
    "use_item":   ["use", "use item", "utilize", "activate", "apply"],
    "eat":        ["eat", "consume", "devour", "munch", "snack"],
    "rest":       ["rest", "sleep", "nap", "relax", "recover", "sit down", "take a break"],
    "wait":       ["wait", "hang around", "stay put", "bide time", "watch"],
    "drop_item":  ["drop", "discard", "throw away", "leave behind", "toss"],
    "status":     ["status", "journal", "quest log", "objectives", "check quest"],
    "equip":      ["equip", "wear", "wield", "put on", "arm"],
    "work":       ["work", "labor", "toil", "farm", "chop", "build"],
}

# ─── Location IDs ────────────────────────────────────────────────────────────
LOCATION_IDS: list[str] = ["gate", "village_center", "elders_house", "fields", "tavern"]

LOCATION_ADJACENCY: dict[str, list[str]] = {
    "gate":           ["village_center"],
    "village_center": ["gate", "elders_house", "fields", "tavern"],
    "elders_house":   ["village_center"],
    "fields":         ["village_center"],
    "tavern":         ["village_center"],
}

INDOOR_LOCATIONS: set[str] = {"elders_house", "tavern"}
OUTDOOR_LOCATIONS: set[str] = {"gate", "village_center", "fields"}
SOCIAL_LOCATIONS: set[str] = {"tavern", "village_center"}

NPC_INDOOR_REGEN: int = 2  # HP/turn for NPCs at indoor locations

# ─── Logging ─────────────────────────────────────────────────────────────────

class JSONFormatter(logging.Formatter):
    """Format log records as JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "module": record.module,
            "turn": getattr(record, "turn", None),
            "event": getattr(record, "event", None),
            "message": record.getMessage(),
        }
        return json.dumps(log_entry)


def setup_logging(session_id: str = "default") -> logging.Logger:
    """Configure structured JSON logging."""
    logger = logging.getLogger("mvp")
    logger.setLevel(logging.DEBUG)

    # File handler
    log_path = LOGS_DIR / f"session_{session_id}.jsonl"
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(JSONFormatter())
    logger.addHandler(fh)

    # Console handler (INFO only, JSON formatted)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(JSONFormatter())
    logger.addHandler(ch)

    return logger


# Create a default logger
logger = logging.getLogger("mvp")
