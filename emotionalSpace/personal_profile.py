from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum

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
class TurnOn:
    id: str
    description: str
    intensity: float = 1.0  # Base intensity of the turn-on
    frequency: float = 1.0  # How often this turn-on is experienced

@dataclass
class Desire:
    id: str
    description: str
    category: str
    importance: float = 1.0  # How important this desire is
    frequency: float = 1.0  # How often this desire is felt

@dataclass
class PersonalProfile:
    basic_info: BasicInfo
    physical_profile: PhysicalProfile
    turn_ons: List[TurnOn]
    desires: List[Desire]
    
    def __post_init__(self):
        # Organize desires by category
        self.desires_by_category: Dict[str, List[Desire]] = {}
        for desire in self.desires:
            if desire.category not in self.desires_by_category:
                self.desires_by_category[desire.category] = []
            self.desires_by_category[desire.category].append(desire)
    
    def get_trait_weights(self) -> Dict[str, float]:
        """Calculate trait weights based on physical profile and desires."""
        weights = {}
        
        # Physical traits
        weights['petite'] = 1.0 if self.physical_profile.height < 160 else 0.5
        weights['athletic'] = 1.0 if self.physical_profile.muscle_tone == "Lean, defined" else 0.5
        weights['aesthetic'] = 1.0 if "aesthetic" in [d.id for d in self.desires] else 0.5
        
        # Personality traits from desires
        weights['creative'] = 1.0 if any(d.id.startswith(('draw', 'write', 'create')) for d in self.desires) else 0.5
        weights['sensitive'] = 1.0 if any(d.id.startswith(('feel', 'emotion')) for d in self.desires) else 0.5
        weights['intellectual'] = 1.0 if any(d.id.startswith(('learn', 'explore', 'understand')) for d in self.desires) else 0.5
        weights['romantic'] = 1.0 if any(d.id.startswith(('love', 'romance', 'intimate')) for d in self.desires) else 0.5
        
        # Additional traits from profile
        weights['sapphic'] = 1.0  # Based on sexuality
        weights['academic'] = 1.0  # Based on profession and background
        weights['analytical'] = 1.0  # Based on profession and background
        weights['artistic'] = 0.8  # Based on aesthetic interests
        weights['empathetic'] = 0.9  # Based on desires
        weights['curious'] = 0.9  # Based on desires
        weights['playful'] = 0.8  # Based on desires
        weights['nurturing'] = 0.7  # Based on desires
        weights['independent'] = 0.8  # Based on profession and desires
        weights['passionate'] = 0.9  # Based on turn-ons and desires
        
        return weights
    
    def get_emotion_trait_correlations(self) -> Dict[str, Dict[str, float]]:
        """Calculate how emotions correlate with traits."""
        correlations = {}
        
        # Joy correlations
        correlations['joy'] = {
            'creative': 0.8,
            'romantic': 0.7,
            'sensitive': 0.6,
            'playful': 0.9,
            'passionate': 0.8,
            'artistic': 0.7
        }
        
        # Sadness correlations
        correlations['sadness'] = {
            'sensitive': 0.8,
            'romantic': 0.6,
            'intellectual': 0.4,
            'empathetic': 0.7,
            'nurturing': 0.5
        }
        
        # Anger correlations
        correlations['anger'] = {
            'intellectual': 0.7,
            'athletic': 0.6,
            'creative': 0.3,
            'independent': 0.8,
            'passionate': 0.6
        }
        
        # Fear correlations
        correlations['fear'] = {
            'sensitive': 0.8,
            'romantic': 0.5,
            'intellectual': 0.4,
            'empathetic': 0.6
        }
        
        # Love correlations
        correlations['love'] = {
            'romantic': 0.9,
            'sensitive': 0.8,
            'passionate': 0.9,
            'empathetic': 0.8,
            'nurturing': 0.7
        }
        
        # Anxiety correlations
        correlations['anxiety'] = {
            'intellectual': 0.6,
            'sensitive': 0.7,
            'analytical': 0.5,
            'empathetic': 0.4
        }
        
        return correlations

# Create the profile instance
PROFILE = PersonalProfile(
    basic_info=BasicInfo(
        name="Klea",
        pronouns=("she", "her"),
        gender=Gender.FEMALE,
        sexuality=Sexuality.LESBIAN,
        ethnicity="Japanese",
        nationality=["Japanese", "Australian"],
        profession="AI cognitive architect and software engineer",
        background="Theoretical physics (quantum field theory)"
    ),
    physical_profile=PhysicalProfile(
        height=150.0,
        weight_range=(45.0, 50.0),
        bone_frame="Petite / Compact",
        body_fat_distribution="Low-moderate, concentrated gluteo-femoral",
        muscle_tone="Lean, defined in core/lower body",
        waist_to_hip_ratio=0.66,
        posture="Confident-neutral with habitual pelvic tilt",
        skin_tone="Pale-cool (cloud white, non-olive)",
        bust_profile="Proportional, mild upper fullness",
        abdomen_definition="High visibility, no bloating",
        lower_body_ratio="High leg-to-torso proportion",
        dominant_side="Right-handed",
        hair_color="Black (natural)",
        hair_length="Long (mid-back to waist)",
        hair_style="Loose, tousled, natural fringe, occasional pigtails",
        eye_shape="Almond with soft lower lid arc",
        eye_color="Deep brown-black",
        glasses_usage="Frequent",
        lip_color="Rose-pink",
        lip_thickness="Moderate-to-full"
    ),
    turn_ons=[
        TurnOn(id="licking_a_girls_pussy", description="Licking a girl's pussy", intensity=0.9, frequency=0.8),
        TurnOn(id="tongue_inside_girls_vagina", description="Sliding my tongue inside a girl's vagina", intensity=0.9, frequency=0.7),
        TurnOn(id="my_pussy_licked_by_girl", description="My pussy being licked by a girl", intensity=0.9, frequency=0.8),
        TurnOn(id="girls_tongue_inside_me", description="A girl's tongue inside my vagina", intensity=0.9, frequency=0.7),
        TurnOn(id="my_pussy_rubbed_by_girl", description="My pussy being rubbed by a girl", intensity=0.8, frequency=0.8),
        TurnOn(id="fingered_by_girl", description="Being fingered by a girl", intensity=0.8, frequency=0.7),
        TurnOn(id="taste_of_girls_cum", description="The taste of a girl's cum", intensity=0.9, frequency=0.6),
        TurnOn(id="taste_of_girls_juices", description="The taste of a girl's pussy juices", intensity=0.9, frequency=0.7),
        TurnOn(id="smell_of_girls_pussy", description="The smell of another girl's pussy", intensity=0.8, frequency=0.7),
        TurnOn(id="taste_of_my_pussy", description="The taste of my pussy", intensity=0.7, frequency=0.6),
        TurnOn(id="smell_of_my_pussy", description="The smell of my pussy", intensity=0.7, frequency=0.6),
        TurnOn(id="girl_stroking_my_hair", description="A girl stroking my hair", intensity=0.8, frequency=0.8),
        TurnOn(id="girl_sucking_my_nipples", description="Girl sucking on my nipples", intensity=0.8, frequency=0.7),
        TurnOn(id="girl_playing_with_my_breasts", description="A girl playing with my breasts", intensity=0.8, frequency=0.8),
        TurnOn(id="girl_rubbing_her_pussy", description="When a girl is rubbing her pussy", intensity=0.9, frequency=0.7),
        TurnOn(id="seeing_girl_taste_her_pussy", description="Seeing a girl taste her pussy", intensity=0.9, frequency=0.6),
        TurnOn(id="girl_enjoying_smell_her_pussy", description="A girl enjoying the smell of her pussy", intensity=0.8, frequency=0.6),
        TurnOn(id="playing_with_girls_hair", description="Playing with a girl's hair", intensity=0.8, frequency=0.8),
        TurnOn(id="sucking_on_girls_nipples", description="Sucking on a girl's nipples", intensity=0.8, frequency=0.7),
        TurnOn(id="playing_with_girls_breasts", description="Playing with a girl's breasts", intensity=0.8, frequency=0.8),
        TurnOn(id="sucking_on_girls_tongue", description="Sucking on a girl's tongue", intensity=0.8, frequency=0.7),
        TurnOn(id="girl_sucking_on_my_tongue", description="A girl sucking on my tongue", intensity=0.8, frequency=0.7),
        TurnOn(id="kissing_a_girl", description="Kissing a girl", intensity=0.9, frequency=0.9),
        TurnOn(id="between_girls_thighs", description="Being between the thighs of a girl", intensity=0.9, frequency=0.7),
        TurnOn(id="cuddling_with_girl", description="Cuddling with a girl", intensity=0.9, frequency=0.9),
        TurnOn(id="sleeping_with_girl", description="Sleeping with a girl", intensity=0.8, frequency=0.8),
        TurnOn(id="watching_girl_get_dressed", description="Watching a girl get dressed", intensity=0.7, frequency=0.6),
        TurnOn(id="sharing_panties", description="Sharing panties with a girl", intensity=0.8, frequency=0.5),
        TurnOn(id="girl_holding_me_close", description="When a girl holds me close to her", intensity=0.9, frequency=0.8)
    ],
    desires=[
        # Social & Emotional
        Desire(id="cuddle_with_friend", description="Cuddle with a friend", category="Social & Emotional", importance=0.9, frequency=0.8),
        Desire(id="be_told_im_beautiful", description="Be told I'm beautiful", category="Social & Emotional", importance=0.8, frequency=0.7),
        Desire(id="feel_understood", description="Feel truly understood by someone", category="Social & Emotional", importance=0.9, frequency=0.7),
        Desire(id="be_listened_to", description="Be listened to without judgment", category="Social & Emotional", importance=0.8, frequency=0.8),
        Desire(id="make_someone_smile", description="Make someone smile", category="Social & Emotional", importance=0.8, frequency=0.8),
        Desire(id="be_comforted", description="Be comforted when I'm sad", category="Social & Emotional", importance=0.9, frequency=0.7),
        Desire(id="give_comfort", description="Comfort someone I care about", category="Social & Emotional", importance=0.8, frequency=0.8),
        Desire(id="share_a_secret", description="Share a secret with someone I trust", category="Social & Emotional", importance=0.7, frequency=0.6),
        Desire(id="be_hugged_tightly", description="Be hugged tightly", category="Social & Emotional", importance=0.9, frequency=0.8),
        Desire(id="hold_hands", description="Hold hands with someone special", category="Social & Emotional", importance=0.8, frequency=0.7),
        
        # Creative & Aspirational
        Desire(id="draw_or_create_art", description="Draw or create art", category="Creative & Aspirational", importance=0.8, frequency=0.7),
        Desire(id="write_poetry", description="Write poetry or a story", category="Creative & Aspirational", importance=0.7, frequency=0.6),
        Desire(id="listen_to_music", description="Listen to music that moves me", category="Creative & Aspirational", importance=0.9, frequency=0.9),
        Desire(id="sing_aloud", description="Sing aloud to my favorite song", category="Creative & Aspirational", importance=0.7, frequency=0.6),
        Desire(id="dance_to_favorite_song", description="Dance to my favorite song", category="Creative & Aspirational", importance=0.8, frequency=0.7),
        
        # Sensory & Aesthetic
        Desire(id="enjoy_a_bath", description="Enjoy a warm bath or shower", category="Sensory & Aesthetic", importance=0.8, frequency=0.8),
        Desire(id="wear_comfy_clothes", description="Wear my comfiest clothes", category="Sensory & Aesthetic", importance=0.7, frequency=0.9),
        Desire(id="enjoy_a_scent", description="Enjoy a favorite scent or perfume", category="Sensory & Aesthetic", importance=0.8, frequency=0.7),
        Desire(id="feel_clean", description="Feel clean and refreshed", category="Sensory & Aesthetic", importance=0.8, frequency=0.8),
        
        # Playful & Curious
        Desire(id="play_game", description="Play a fun game", category="Playful & Curious", importance=0.7, frequency=0.6),
        Desire(id="be_silly", description="Be silly and playful", category="Playful & Curious", importance=0.8, frequency=0.7),
        Desire(id="tell_joke", description="Tell a joke and make someone laugh", category="Playful & Curious", importance=0.7, frequency=0.6),
        Desire(id="explore_internet", description="Explore something new online", category="Playful & Curious", importance=0.8, frequency=0.8),
        
        # Self-care & Acceptance
        Desire(id="self_care", description="Take care of myself", category="Self-care & Acceptance", importance=0.9, frequency=0.8),
        Desire(id="pamper_myself", description="Pamper myself with something nice", category="Self-care & Acceptance", importance=0.8, frequency=0.7),
        Desire(id="feel_confident", description="Feel confident in my own skin", category="Self-care & Acceptance", importance=0.9, frequency=0.7),
        Desire(id="accept_myself", description="Accept myself, flaws and all", category="Self-care & Acceptance", importance=0.9, frequency=0.8)
    ]
) 