from typing import Dict, List, Any, Optional
import numpy as np
import time
from dataclasses import dataclass
from enum import Enum

class MemoryType(Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"

@dataclass
class Memory:
    type: MemoryType
    content: Any
    timestamp: float
    importance: float

@dataclass
class AttentionVector:
    focus: Dict[str, float]  # Maps features to attention weights
    decay_rate: float = 0.1

@dataclass
class AffectiveState:
    valence: float  # [-1, 1]
    arousal: float  # [0, 1]
    emotions: Dict[str, float]  # Maps emotion names to intensities

class CognitiveState:
    def __init__(self):
        # Initialize subspaces
        self.memory: Dict[MemoryType, List[Memory]] = {
            MemoryType.EPISODIC: [],
            MemoryType.SEMANTIC: [],
            MemoryType.PROCEDURAL: []
        }
        self.attention = AttentionVector({})
        self.traits: Dict[str, float] = {}
        self.intentions: List[Dict[str, Any]] = []
        self.affective = AffectiveState(0.0, 0.5, {})
        
        # State evolution parameters
        self.alpha = 0.7  # Trait update smoothing
        self.memory_capacity = 1000
        self.attention_decay = 0.1

    def update_memory(self, memory_type: MemoryType, content: Any, importance: float):
        """Update memory subspace according to equation (1)"""
        memory = Memory(memory_type, content, time.time(), importance)
        self.memory[memory_type].append(memory)
        
        # Maintain memory capacity
        if len(self.memory[memory_type]) > self.memory_capacity:
            self.memory[memory_type].sort(key=lambda x: x.importance)
            self.memory[memory_type] = self.memory[memory_type][-self.memory_capacity:]

    def update_attention(self, utility_gradient: Dict[str, float]):
        """Update attention vector field according to equation (2)"""
        # Decay existing attention
        for feature in self.attention.focus:
            self.attention.focus[feature] *= (1 - self.attention_decay)
        
        # Update based on utility gradient
        for feature, gradient in utility_gradient.items():
            self.attention.focus[feature] = self.attention.focus.get(feature, 0.0) + gradient

    def update_traits(self, new_traits: Dict[str, float]):
        """
        Update trait vector according to equation (3).
        Ensures trait values are properly scaled between 0 and 1.
        """
        for trait, value in new_traits.items():
            # Ensure value is a float and within [0,1]
            value = float(value)
            value = max(0.0, min(1.0, value))
            
            # Update with smoothing
            current = self.traits.get(trait, 0.0)
            self.traits[trait] = self.alpha * current + (1 - self.alpha) * value

    def update_affective(self, reward_prediction_error: float):
        """Update affective field according to equation (4)"""
        # Update valence based on prediction error
        self.affective.valence = np.clip(
            self.affective.valence + 0.1 * reward_prediction_error,
            -1.0, 1.0
        )
        
        # Update arousal based on prediction error magnitude
        self.affective.arousal = np.clip(
            self.affective.arousal + 0.1 * abs(reward_prediction_error),
            0.0, 1.0
        )

    def transition(self, external_input: Dict[str, Any]) -> 'CognitiveState':
        """State evolution according to Φt(st, et)"""
        new_state = CognitiveState()
        
        # Copy current state
        new_state.memory = self.memory.copy()
        new_state.attention = self.attention
        new_state.traits = self.traits.copy()
        new_state.intentions = self.intentions.copy()
        new_state.affective = self.affective
        
        # Process external input
        if 'memory' in external_input:
            for mem_type, content in external_input['memory'].items():
                new_state.update_memory(MemoryType(mem_type), content, 1.0)
        
        if 'attention' in external_input:
            new_state.update_attention(external_input['attention'])
        
        if 'traits' in external_input:
            new_state.update_traits(external_input['traits'])
        
        if 'reward_error' in external_input:
            new_state.update_affective(external_input['reward_error'])
        
        return new_state

    def get_utility(self, goal: Dict[str, Any]) -> float:
        """Compute utility function J(st) for a goal"""
        # Base utility from goal attributes
        utility = goal.get('base_utility', 0.0)
        
        # Adjust based on trait alignment
        for trait, value in self.traits.items():
            if trait in goal.get('trait_weights', {}):
                utility += value * goal['trait_weights'][trait]
        
        # Adjust based on affective state
        utility *= (1 + self.affective.valence * 0.5)  # Valence influence
        utility *= (1 + self.affective.arousal * 0.3)  # Arousal influence
        
        # Adjust based on memory relevance
        for mem_type, memories in self.memory.items():
            for memory in memories:
                if self._is_memory_relevant(memory, goal):
                    utility += memory.importance * 0.1
        
        return utility

    def _is_memory_relevant(self, memory: Memory, goal: Dict[str, Any]) -> bool:
        """Check if a memory is relevant to a goal"""
        # Simple relevance check - can be made more sophisticated
        if isinstance(memory.content, dict):
            return any(key in goal for key in memory.content)
        return False

    def get_state_vector(self) -> np.ndarray:
        """Convert cognitive state to a vector representation"""
        # Combine all subspaces into a single vector
        vectors = []
        
        # Memory vector (averaged importance per type)
        for mem_type in MemoryType:
            memories = self.memory[mem_type]
            if memories:
                vectors.append(np.mean([m.importance for m in memories]))
            else:
                vectors.append(0.0)
        
        # Attention vector
        vectors.extend(self.attention.focus.values())
        
        # Trait vector
        vectors.extend(self.traits.values())
        
        # Affective vector
        vectors.extend([self.affective.valence, self.affective.arousal])
        vectors.extend(self.affective.emotions.values())
        
        return np.array(vectors) 