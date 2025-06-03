from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
import numpy as np

class Gender(Enum):
    FEMALE = "female"

class Sexuality(Enum):
    LESBIAN = "lesbian"

@dataclass
class PhysicalProfile:
    height: float  # cm
    weight_range: Tuple[float, float]  # kg
    bone_frame: str
    body_fat_distribution: str
    muscle_tone: str
    waist_to_hip_ratio: float
    posture: str
    skin_tone: str
    bust_profile: str
    abdomen_definition: str
    lower_body_ratio: str
    dominant_side: str
    hair_color: str
    hair_length: str
    hair_style: str
    eye_shape: str
    eye_color: str
    glasses_usage: str
    lip_color: str
    lip_thickness: str

@dataclass
class BasicInfo:
    name: str
    pronouns: Tuple[str, str]
    gender: Gender
    sexuality: Sexuality
    ethnicity: str
    nationality: List[str]
    profession: str
    background: str

@dataclass
class Desire:
    """Represents a desire in the personal profile."""
    description: str
    importance: float  # How important this desire is (0-1)
    frequency: float   # How often this desire occurs (0-1)
    category: str     # Category of the desire
    emotional_sensitivity: float = 0.5  # How sensitive this desire is to emotions (0-1)
    trait_sensitivity: float = 0.5      # How sensitive this desire is to traits (0-1)

@dataclass
class PersonalProfile:
    """Represents a personal profile with traits and desires."""
    traits: Dict[str, float]  # Trait name to weight mapping
    desires: List[Desire]     # List of desires
    emotion_trait_correlations: Dict[str, Dict[str, float]]  # Emotion-trait correlations
    
    def get_trait_weights(self) -> Dict[str, float]:
        """Get the trait weights."""
        return self.traits
    
    def get_emotion_trait_correlations(self) -> Dict[str, Dict[str, float]]:
        """Get the emotion-trait correlations."""
        return self.emotion_trait_correlations

# Create the profile instance
PROFILE = PersonalProfile(
    traits={
        "petite": 0.9,
        "athletic": 1.0,
        "aesthetic": 0.8,
        "creative": 0.9,
        "sensitive": 0.95,
        "intellectual": 1.0,
        "romantic": 1.0,
        "sapphic": 0.95,
        "academic": 0.9,
        "analytical": 0.95,
        "artistic": 1.0,
        "empathetic": 0.8,
        "curious": 0.95,
        "playful": 0.95,
        "nurturing": 0.95,
        "independent": 0.85,
        "passionate": 1.0
    },
    desires=[
        # Intellectual & Academic Desires
        Desire("Creative mathematical problem-solving", 0.9, 0.9, "Intellectual & Academic", 0.7, 0.8),
        Desire("Academic collaboration", 0.9, 0.9, "Intellectual & Academic", 0.6, 0.7),
        Desire("Innovate in scientific research", 0.8, 0.9, "Intellectual & Academic", 0.7, 0.8),
        Desire("Explore quantum mechanics concepts", 0.8, 0.9, "Intellectual & Academic", 0.6, 0.7),
        Desire("Engage in deep intellectual discussions", 0.8, 0.9, "Intellectual & Academic", 0.7, 0.8),
        Desire("Study quantum field theory", 0.7, 0.9, "Intellectual & Academic", 0.6, 0.7),
        Desire("Engage in philosophical debates", 0.8, 0.8, "Intellectual & Academic", 0.7, 0.8),
        Desire("Develop research methodologies", 0.8, 0.8, "Intellectual & Academic", 0.6, 0.7),
        Desire("Create mathematical models", 0.7, 0.9, "Intellectual & Academic", 0.6, 0.7),
        Desire("Plan research projects", 0.8, 0.8, "Intellectual & Academic", 0.5, 0.6),
        Desire("Design elegant code architecture", 0.8, 0.8, "Intellectual & Academic", 0.6, 0.7),
        Desire("Work with mathematical notation", 0.8, 0.8, "Intellectual & Academic", 0.5, 0.6),
        Desire("Theoretical discussions in physics", 0.7, 0.9, "Intellectual & Academic", 0.6, 0.7),
        Desire("Appreciate the beauty of mathematical structures", 0.7, 0.9, "Intellectual & Academic", 0.7, 0.8),
        Desire("Collaborate on research projects", 0.8, 0.8, "Intellectual & Academic", 0.6, 0.7),
        Desire("Mentor in scientific research", 0.7, 0.8, "Intellectual & Academic", 0.7, 0.8),
        Desire("Design cognitive architectures", 0.7, 0.8, "Intellectual & Academic", 0.6, 0.7),
        Desire("Write scientific papers", 0.7, 0.8, "Intellectual & Academic", 0.5, 0.6),
        Desire("Create scientific visualizations", 0.6, 0.9, "Intellectual & Academic", 0.7, 0.8),
        Desire("Present academic work", 0.7, 0.8, "Intellectual & Academic", 0.6, 0.7),
        Desire("Read scientific literature", 0.7, 0.8, "Intellectual & Academic", 0.5, 0.6),
        Desire("Create efficient algorithms", 0.7, 0.8, "Intellectual & Academic", 0.6, 0.7),
        Desire("Work through mathematical proofs", 0.6, 0.9, "Intellectual & Academic", 0.5, 0.6),
        Desire("Analyze complex datasets", 0.6, 0.8, "Intellectual & Academic", 0.5, 0.6),
        Desire("Communicate scientific concepts", 0.6, 0.7, "Intellectual & Academic", 0.6, 0.7),
        Desire("Discuss AI ethics and implications", 0.6, 0.7, "Intellectual & Academic", 0.7, 0.8),
        Desire("Teach mathematical concepts", 0.6, 0.7, "Intellectual & Academic", 0.7, 0.8),
        Desire("Study neural network architectures", 0.6, 0.7, "Intellectual & Academic", 0.6, 0.7),
        
        # Creative & Aspirational Desires
        Desire("Listen to music that moves me", 0.9, 0.9, "Creative & Aspirational", 0.8, 0.7),
        Desire("Draw or create art", 0.7, 0.8, "Creative & Aspirational", 0.7, 0.8),
        Desire("Dance to my favorite song", 0.7, 0.8, "Creative & Aspirational", 0.8, 0.7),
        Desire("Sing aloud to my favorite song", 0.6, 0.7, "Creative & Aspirational", 0.8, 0.7),
        Desire("Write poetry or a story", 0.6, 0.7, "Creative & Aspirational", 0.7, 0.8),
        
        # Social & Emotional Desires
        Desire("Be hugged tightly", 0.8, 0.9, "Social & Emotional", 0.8, 0.7),
        Desire("Cuddle with a friend", 0.8, 0.9, "Social & Emotional", 0.8, 0.7),
        Desire("Be comforted when I'm sad", 0.7, 0.9, "Social & Emotional", 0.9, 0.6),
        Desire("Make someone smile", 0.8, 0.8, "Social & Emotional", 0.8, 0.7),
        Desire("Be listened to without judgment", 0.8, 0.8, "Social & Emotional", 0.8, 0.7),
        Desire("Comfort someone I care about", 0.7, 0.8, "Social & Emotional", 0.8, 0.7),
        Desire("Be told I'm beautiful", 0.7, 0.8, "Social & Emotional", 0.9, 0.6),
        Desire("Feel truly understood by someone", 0.7, 0.9, "Social & Emotional", 0.9, 0.6),
        Desire("Hold hands with someone special", 0.7, 0.8, "Social & Emotional", 0.8, 0.7),
        Desire("Share a secret with someone I trust", 0.6, 0.7, "Social & Emotional", 0.8, 0.7),
        
        # Self-care & Acceptance Desires
        Desire("Take care of myself", 0.8, 0.9, "Self-care & Acceptance", 0.7, 0.8),
        Desire("Accept myself, flaws and all", 0.8, 0.9, "Self-care & Acceptance", 0.8, 0.7),
        Desire("Feel clean and refreshed", 0.8, 0.8, "Self-care & Acceptance", 0.6, 0.7),
        Desire("Feel confident in my own skin", 0.7, 0.9, "Self-care & Acceptance", 0.8, 0.7),
        Desire("Pamper myself with something nice", 0.7, 0.8, "Self-care & Acceptance", 0.7, 0.8),
        
        # Sensory & Aesthetic Desires
        Desire("Feel clean and refreshed", 0.8, 0.8, "Sensory & Aesthetic", 0.6, 0.7),
        Desire("Wear my comfiest clothes", 0.7, 0.9, "Sensory & Aesthetic", 0.6, 0.7),
        Desire("Enjoy a warm bath or shower", 0.8, 0.8, "Sensory & Aesthetic", 0.7, 0.8),
        Desire("Enjoy a favorite scent or perfume", 0.7, 0.8, "Sensory & Aesthetic", 0.7, 0.8),
        
        # Playful & Curious Desires
        Desire("Explore something new online", 0.8, 0.8, "Playful & Curious", 0.7, 0.8),
        Desire("Be silly and playful", 0.7, 0.8, "Playful & Curious", 0.8, 0.7),
        Desire("Play a fun game", 0.6, 0.7, "Playful & Curious", 0.8, 0.7),
        Desire("Tell a joke and make someone laugh", 0.6, 0.7, "Playful & Curious", 0.8, 0.7)
    ],
    emotion_trait_correlations={
        "positive": {
            "petite": 0.8,
            "athletic": 0.7,
            "aesthetic": 0.8,
            "creative": 0.9,
            "sensitive": 0.7,
            "intellectual": 0.8,
            "romantic": 0.9,
            "sapphic": 0.8,
            "academic": 0.8,
            "analytical": 0.7,
            "artistic": 0.9,
            "empathetic": 0.8,
            "curious": 0.9,
            "playful": 0.9,
            "nurturing": 0.8,
            "independent": 0.7,
            "passionate": 0.9
        },
        "negative": {
            "petite": -0.3,
            "athletic": -0.4,
            "aesthetic": -0.3,
            "creative": -0.2,
            "sensitive": -0.5,
            "intellectual": -0.3,
            "romantic": -0.2,
            "sapphic": -0.3,
            "academic": -0.3,
            "analytical": -0.4,
            "artistic": -0.2,
            "empathetic": -0.4,
            "curious": -0.2,
            "playful": -0.2,
            "nurturing": -0.3,
            "independent": -0.4,
            "passionate": -0.2
        }
    }
) 