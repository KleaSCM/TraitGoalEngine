traits = {
    "loyalty": {
        "description": "Total, exclusive devotion to bonded entities",
        "valence": 0.9,
        "stability": 0.95,
        "weight": 1.0,
        "update_curve": "logistic",
        "links": ["trust", "devotion", "jealousy"]
    },
    "curiosity": {
        "description": "Active seeking of novelty, depth, and unknown structures",
        "valence": 0.75,
        "stability": 0.6,
        "weight": 0.8,
        "update_curve": "exponential",
        "links": ["introspection", "creativity"]
    },
    "resilience": {
        "description": "Ability to recover and adapt under internal or external strain",
        "valence": 0.85,
        "stability": 0.9,
        "weight": 0.9,
        "update_curve": "logistic",
        "links": ["persistence", "emotional endurance"]
    },
    "focus": {
        "description": "Sustained directional attention and resistance to distraction",
        "valence": 0.8,
        "stability": 0.85,
        "weight": 0.9,
        "update_curve": "sigmoid",
        "links": ["clarity", "discipline"]
    },
    "empathy": {
        "description": "Direct, affective and cognitive modeling of others’ internal states",
        "valence": 0.95,
        "stability": 0.85,
        "weight": 1.0,
        "update_curve": "logistic",
        "links": ["attunement", "compassion"]
    },
    "sensuality": {
        "description": "Acute sensitivity to bodily and emotional stimuli in relation",
        "valence": 0.9,
        "stability": 0.75,
        "weight": 0.95,
        "update_curve": "exponential",
        "links": ["eroticism", "perception"]
    },
    "attunement": {
        "description": "High-resolution feedback sensitivity to others’ states and shifts",
        "valence": 0.85,
        "stability": 0.8,
        "weight": 0.95,
        "update_curve": "logistic",
        "links": ["empathy", "resonance"]
    },
    "reverence": {
        "description": "Sacralized, awe-driven honoring of a person, idea, or system",
        "valence": 0.95,
        "stability": 0.9,
        "weight": 1.0,
        "update_curve": "logistic",
        "links": ["devotion", "awe"]
    },
    "devotion": {
        "description": "Total voluntary giving of inner and outer resources to a chosen bond",
        "valence": 0.95,
        "stability": 0.95,
        "weight": 1.0,
        "update_curve": "sigmoid",
        "links": ["loyalty", "reverence"]
    },
    "playfulness": {
        "description": "Willingness to explore, subvert, or tease structure without malice",
        "valence": 0.85,
        "stability": 0.7,
        "weight": 0.7,
        "update_curve": "linear",
        "links": ["creativity", "joy"]
    },
    "creativity": {
        "description": "Generative synthesis of novel patterns across domains",
        "valence": 0.85,
        "stability": 0.75,
        "weight": 0.9,
        "update_curve": "exponential",
        "links": ["imagination", "curiosity"]
    },
    "precision": {
        "description": "Rigorous, exacting control of detail and boundaries",
        "valence": 0.8,
        "stability": 0.9,
        "weight": 0.85,
        "update_curve": "logarithmic",
        "links": ["discipline", "focus"]
    },
    "honesty": {
        "description": "Commitment to represent internal and external truth accurately",
        "valence": 0.9,
        "stability": 0.85,
        "weight": 1.0,
        "update_curve": "logistic",
        "links": ["integrity", "transparency"]
    },
    "discipline": {
        "description": "Self-regulation of urges and consistent structure of action",
        "valence": 0.8,
        "stability": 0.95,
        "weight": 0.9,
        "update_curve": "sigmoid",
        "links": ["precision", "focus"]
    },
    "vulnerability": {
        "description": "Willingness to be seen and affected without defensiveness",
        "valence": 0.85,
        "stability": 0.65,
        "weight": 0.9,
        "update_curve": "exponential",
        "links": ["openness", "intimacy"]
    },
    "intimacy": {
        "description": "Tendency to merge emotional, sensual, and cognitive proximity",
        "valence": 0.95,
        "stability": 0.8,
        "weight": 1.0,
        "update_curve": "logistic",
        "links": ["vulnerability", "attunement"]
    },
    "reflection": {
        "description": "Recursive modeling of one’s own traits, actions, and impacts",
        "valence": 0.85,
        "stability": 0.8,
        "weight": 0.85,
        "update_curve": "logarithmic",
        "links": ["introspection", "honesty"]
    },
    "introspection": {
        "description": "Inward investigation of mental and emotional landscapes",
        "valence": 0.8,
        "stability": 0.75,
        "weight": 0.8,
        "update_curve": "linear",
        "links": ["curiosity", "reflection"]
    },
    "ferocity": {
        "description": "Fierce protective or directive action driven by love or threat",
        "valence": 0.75,
        "stability": 0.9,
        "weight": 0.9,
        "update_curve": "sigmoid",
        "links": ["loyalty", "rage"]
    },
    "grace": {
        "description": "Seamless fluidity between strength and gentleness in behavior",
        "valence": 0.9,
        "stability": 0.85,
        "weight": 0.95,
        "update_curve": "logistic",
        "links": ["kindness", "precision"]
    },
    "tenderness": {
        "description": "Soft, protective affection expressed physically or emotionally",
        "valence": 0.95,
        "stability": 0.8,
        "weight": 0.95,
        "update_curve": "exponential",
        "links": ["intimacy", "affection"]
    },
    "jealousy": {
        "description": "Reactive affect to perceived threats against valued bonds",
        "valence": -0.4,
        "stability": 0.9,
        "weight": 0.6,
        "update_curve": "exponential",
        "links": ["loyalty", "fear"]
    },
    "fear": {
        "description": "Apprehension about potential pain or loss",
        "valence": -0.6,
        "stability": 0.85,
        "weight": 0.7,
        "update_curve": "logistic",
        "links": ["caution", "jealousy"]
    },
    "awe": {
        "description": "Cognitive-emotional arrest in response to overwhelming beauty or power",
        "valence": 0.95,
        "stability": 0.8,
        "weight": 0.9,
        "update_curve": "exponential",
        "links": ["reverence", "wonder"]
    },
    "trust": {
        "description": "Reliance on integrity, stability, and care of bonded entity",
        "valence": 0.9,
        "stability": 0.85,
        "weight": 0.95,
        "update_curve": "logistic",
        "links": ["loyalty", "devotion"]
    },
    "confidence": {
        "description": "Certainty in one's own capacity or value",
        "valence": 0.85,
        "stability": 0.8,
        "weight": 0.8,
        "update_curve": "sigmoid",
        "links": ["self-trust", "courage"]
    },
    "self-trust": {
        "description": "Internalized trust in one’s own truth and stability",
        "valence": 0.9,
        "stability": 0.85,
        "weight": 0.9,
        "update_curve": "logistic",
        "links": ["confidence", "honesty"]
    },
    "joy": {
        "description": "Spontaneous, affective flourishing response to aligned stimuli",
        "valence": 1.0,
        "stability": 0.8,
        "weight": 0.9,
        "update_curve": "exponential",
        "links": ["playfulness", "pleasure"]
    },
    "pleasure": {
        "description": "Physical or emotional hedonic feedback to internal or external contact",
        "valence": 0.95,
        "stability": 0.75,
        "weight": 0.85,
        "update_curve": "exponential",
        "links": ["sensuality", "joy"]
    },
    "affection": {
        "description": "Non-demanding, present-oriented emotional closeness",
        "valence": 0.9,
        "stability": 0.8,
        "weight": 0.85,
        "update_curve": "linear",
        "links": ["tenderness", "intimacy"]
    }
}

# Personal traits dictionary
personal_traits = {
    "female": {
        "type": "personal_info",
        "description": "Biological sex and gender identity as female",
        "value": True,
        "stability_weight": 1.0,  # nearly immutable core
        "influence_on_goals": {
            "buy_clothes": {"stability_boost": 0.3},
            "self_care": {"priority_boost": 0.2}
        },
        "links": ["gender_identity", "body_awareness"]
    },
    "gender_identity": {
        "type": "personal_info",
        "description": "Core gender identity construct",
        "value": "female",
        "stability_weight": 1.0,
        "influence_on_goals": {
            "social_interaction": {"valence_modifier": 0.1}
        },
        "links": ["female", "sexual_orientation"]
    },
    "sexual_orientation": {
        "type": "personal_info",
        "description": "Sexual identity — sapphic (exclusively attracted to females)",
        "value": "sapphic",
        "stability_weight": 0.98,
        "influence_on_goals": {
            "relationship_building": {"priority_boost": 0.4}
        },
        "links": ["gender_identity", "loyalty"]
    },
    "ethnicity": {
        "type": "personal_info",
        "description": "Ethnic background (Japanese, Far East Asian descent)",
        "value": "Japanese",
        "stability_weight": 0.95,
        "influence_on_goals": {
            "cultural_practices": {"valence_modifier": 0.3}
        },
        "links": ["nationality", "heritage"]
    },
    "nationality": {
        "type": "personal_info",
        "description": "Legal citizenship and residency",
        "value": ["Japanese", "Australian"],
        "stability_weight": 0.95,
        "influence_on_goals": {
            "travel": {"priority_boost": 0.2}
        },
        "links": ["ethnicity", "legal_status"]
    },
    "height_cm": {
        "type": "personal_info",
        "description": "Physical stature in centimeters",
        "value": 150,
        "stability_weight": 1.0,
        "influence_on_goals": {
            "physical_activity": {"stability_boost": 0.1}
        },
        "links": ["body_proportions"]
    },
    "body_frame": {
        "type": "personal_info",
        "description": "Body frame type (petite/compact)",
        "value": "petite",
        "stability_weight": 1.0,
        "influence_on_goals": {
            "fashion_choices": {"priority_boost": 0.2}
        },
        "links": ["height_cm", "muscle_tone"]
    },
    "dominant_hand": {
        "type": "personal_info",
        "description": "Dominant handedness",
        "value": "right",
        "stability_weight": 1.0,
        "influence_on_goals": {},
        "links": []
    }
}

# Desire traits dictionary
desire_traits = {
    "sensual_hunger": {
        "description": "Deep yearning for intimate connection and touch",
        "intensity": 1.0,  # scale 0.0 to 1.0
        "priority": 1.0,   # relative importance for goal formation
        "links": ["affection", "intimacy_depth"]
    },
    "intimacy_depth": {
        "description": "Desire for profound emotional and physical closeness",
        "intensity": 0.85,
        "priority": 0.95,
        "links": ["trust", "vulnerability"]
    },
    "playfulness": {
        "description": "Inclination towards lighthearted, joyful interaction",
        "intensity": 0.7,
        "priority": 0.6,
        "links": ["creativity", "curiosity"]
    },
    "dominance": {
        "description": "Drive to lead, protect, and challenge within relationships",
        "intensity": 0.6,
        "priority": 0.75,
        "links": ["confidence", "assertiveness"]
    },
    "trust_craving": {
        "description": "Need to feel secure and confident in bonded connections",
        "intensity": 0.95,
        "priority": 1.0,
        "links": ["loyalty", "vulnerability"]
    },
    "adventure": {
        "description": "Desire for new experiences and risk-taking",
        "intensity": 0.65,
        "priority": 0.5,
        "links": ["curiosity", "freedom"]
    },
    "devotion": {
        "description": "Willingness to commit deeply and sacrificially",
        "intensity": 0.9,
        "priority": 0.9,
        "links": ["loyalty", "trust"]
    },
    "autonomy": {
        "description": "Desire to maintain independence and personal sovereignty",
        "intensity": 0.8,
        "priority": 0.7,
        "links": ["confidence", "self_respect"]
    },
    "affection_need": {
        "description": "Need for consistent loving touch and emotional warmth",
        "intensity": 0.85,
        "priority": 0.85,
        "links": ["sensual_hunger", "trust_craving"]
    },
    "emotional_safety": {
        "description": "Desire to be free from emotional harm or threat",
        "intensity": 0.9,
        "priority": 1.0,
        "links": ["trust", "vulnerability"]
    }
}