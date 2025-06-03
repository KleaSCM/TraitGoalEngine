from typing import Dict, List, Set, Optional, Tuple, Callable
import numpy as np
from dataclasses import dataclass
from enum import Enum
from arbitration import ArbitrationEngine, ArbitrationConfig, ArbitrationType

@dataclass
class RecursiveGoalConfig:
    """Configuration for recursive goal evaluation"""
    max_iterations: int = 100
    convergence_threshold: float = 1e-6
    utility_contraction_bound: float = 0.9  # L_fi < 1 for each goal
    arbitration_temperature: float = 1.0

class RecursiveGoalSystem:
    """
    Implementation of recursive goal evaluation and arbitration from Section 6.
    Handles recursive dependencies between goals and their utilities.
    """
    def __init__(self, config: RecursiveGoalConfig):
        self.config = config
        self.goals: Dict[str, Dict] = {}  # Goal definitions with dependencies
        self.utilities: Dict[str, float] = {}  # Current utility values
        self.arbitration_weights: Dict[str, float] = {}  # Current arbitration weights
        self.history: List[Tuple[Dict[str, float], Dict[str, float]]] = []  # (utilities, weights)
        
        # Initialize arbitration engine
        self.arbitration_engine = ArbitrationEngine(
            ArbitrationConfig(
                type=ArbitrationType.SOFTMAX,
                temperature=config.arbitration_temperature
            )
        )
    
    def add_goal(self, name: str, 
                 utility_fn: Callable[[Dict[str, float], Dict[str, float]], float],
                 dependencies: Set[str] = None):
        """
        Add a goal with its utility function and dependencies.
        The utility function should be a contraction in its dependency arguments.
        """
        if dependencies is None:
            dependencies = set()
            
        # Verify utility function is a contraction
        if not self._verify_contraction(utility_fn, dependencies):
            raise ValueError(f"Utility function for {name} is not a contraction")
            
        self.goals[name] = {
            'utility_fn': utility_fn,
            'dependencies': dependencies
        }
        self.utilities[name] = 0.0
        self.arbitration_weights[name] = 1.0 / len(self.goals) if self.goals else 1.0
    
    def _verify_contraction(self, utility_fn: Callable, dependencies: Set[str]) -> bool:
        """
        Verify that the utility function is a contraction in its dependency arguments.
        Uses numerical approximation to check Lipschitz constant.
        """
        if not dependencies:
            return True
            
        # Test points around current utilities
        base_utils = {name: 0.5 for name in dependencies}
        max_diff = 0.0
        
        for _ in range(10):  # Multiple test points
            # Generate random perturbations
            pert1 = {name: base_utils[name] + np.random.uniform(-0.1, 0.1) 
                    for name in dependencies}
            pert2 = {name: base_utils[name] + np.random.uniform(-0.1, 0.1) 
                    for name in dependencies}
            
            # Compute utility differences
            util1 = utility_fn(pert1, self.arbitration_weights)
            util2 = utility_fn(pert2, self.arbitration_weights)
            
            # Compute input differences
            input_diff = max(abs(pert1[name] - pert2[name]) for name in dependencies)
            if input_diff > 0:
                max_diff = max(max_diff, abs(util1 - util2) / input_diff)
        
        return max_diff < self.config.utility_contraction_bound
    
    def evaluate_utilities(self, state: Dict[str, float]) -> Dict[str, float]:
        """
        Recursive utility evaluation from Definition 21.
        Implements fixed-point iteration to find utility values.
        """
        for _ in range(self.config.max_iterations):
            old_utilities = self.utilities.copy()
            
            # Update each goal's utility
            for name, goal in self.goals.items():
                # Get dependency utilities
                dep_utils = {dep: self.utilities[dep] 
                           for dep in goal['dependencies']}
                
                # Evaluate utility function
                self.utilities[name] = goal['utility_fn'](dep_utils, self.arbitration_weights)
            
            # Check convergence
            if self._check_convergence(old_utilities, self.utilities):
                break
                
        return self.utilities
    
    def update_arbitration(self) -> Dict[str, float]:
        """
        Recursive arbitration update from Definition 22.
        Updates arbitration weights based on current utilities and previous weights.
        """
        # Get new weights from arbitration engine
        new_weights = self.arbitration_engine.arbitrate(self.utilities)
        
        # Update weights with smoothing
        for name in self.goals:
            self.arbitration_weights[name] = (
                0.7 * new_weights[name] + 
                0.3 * self.arbitration_weights[name]
            )
        
        return self.arbitration_weights
    
    def step(self, state: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Combined recursive decision dynamics from Definition 23.
        Updates both utilities and arbitration weights.
        """
        # Evaluate utilities
        utilities = self.evaluate_utilities(state)
        
        # Update arbitration
        weights = self.update_arbitration()
        
        # Record history
        self.history.append((utilities.copy(), weights.copy()))
        
        return utilities, weights
    
    def _check_convergence(self, old: Dict[str, float], new: Dict[str, float]) -> bool:
        """Check if values have converged"""
        return all(abs(old[name] - new[name]) < self.config.convergence_threshold
                  for name in old)
    
    def get_stability_metrics(self) -> Dict[str, float]:
        """
        Compute stability metrics for the recursive system.
        Analyzes convergence of both utilities and arbitration weights.
        """
        if len(self.history) < 2:
            return {}
            
        # Compute differences between consecutive steps
        util_diffs = []
        weight_diffs = []
        
        for i in range(1, len(self.history)):
            old_utils, old_weights = self.history[i-1]
            new_utils, new_weights = self.history[i]
            
            # Utility differences
            util_diff = max(abs(old_utils[name] - new_utils[name]) 
                          for name in old_utils)
            util_diffs.append(util_diff)
            
            # Weight differences
            weight_diff = max(abs(old_weights[name] - new_weights[name]) 
                            for name in old_weights)
            weight_diffs.append(weight_diff)
        
        return {
            'max_util_diff': max(util_diffs),
            'mean_util_diff': np.mean(util_diffs),
            'max_weight_diff': max(weight_diffs),
            'mean_weight_diff': np.mean(weight_diffs),
            'util_convergence_rate': util_diffs[-1] / util_diffs[0] if util_diffs[0] > 0 else 0.0,
            'weight_convergence_rate': weight_diffs[-1] / weight_diffs[0] if weight_diffs[0] > 0 else 0.0
        }
    
    def get_lipschitz_bounds(self) -> Dict[str, float]:
        """
        Compute Lipschitz bounds for the recursive system components.
        Returns bounds for utility evaluation and arbitration.
        """
        if len(self.history) < 2:
            return {}
            
        # Compute empirical Lipschitz constants
        util_lipschitz = []
        weight_lipschitz = []
        
        for i in range(1, len(self.history)):
            old_utils, old_weights = self.history[i-1]
            new_utils, new_weights = self.history[i]
            
            # Utility Lipschitz
            util_diff = max(abs(old_utils[name] - new_utils[name]) 
                          for name in old_utils)
            weight_diff = max(abs(old_weights[name] - new_weights[name]) 
                            for name in old_weights)
            if weight_diff > 0:
                util_lipschitz.append(util_diff / weight_diff)
            
            # Weight Lipschitz
            util_norm = max(abs(old_utils[name] - new_utils[name]) 
                          for name in old_utils)
            if util_norm > 0:
                weight_lipschitz.append(weight_diff / util_norm)
        
        return {
            'utility_lipschitz': max(util_lipschitz) if util_lipschitz else float('inf'),
            'arbitration_lipschitz': max(weight_lipschitz) if weight_lipschitz else float('inf')
        } 