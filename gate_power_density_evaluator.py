#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gate Power Density Prediction Task Evaluator 

Implements power density regression-based Gate power prediction evaluation:
- Unified interface: metrics = evaluator.eval(y_pred, y_true)
- Five core metrics: MAE, RMSE, R², power distribution uniformity, hotspot prediction accuracy
- Supports multiple power density prediction formats

Author: EDA for AI Team
"""

import torch
from torch_geometric.data import Data
import numpy as np
from typing import Dict, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Core Evaluator Module
# ============================================================================

class GatePowerDensityEvaluator:
    """
    Gate Power Density Prediction Evaluator (Streamlined Version, OGB-inspired Design)
    
    Focuses on core evaluation metrics with unified eval(y_pred, y_true) interface:
    - Five core metrics: MAE, RMSE, R², power distribution uniformity, hotspot prediction accuracy
    - Supports multiple power density prediction formats
    - Automatic input format conversion and normalization
    """
    
    def __init__(self, hotspot_threshold: float = 0.8):
        """
        Initialize evaluator
        
        Args:
            hotspot_threshold: Hotspot determination threshold
        """
        self.hotspot_threshold = hotspot_threshold
        
    def eval(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, float]:
        """
        Main evaluation function (OGB-inspired interface design)
        
        Focuses on Gate power density prediction evaluation with 5 core metrics:
        1. Mean Absolute Error (mae) - regression error metric
        2. Root Mean Square Error (rmse) - regression error metric  
        3. Coefficient of Determination (r2) - regression fit metric
        4. Power Distribution Uniformity (power_uniformity) - EDA professional metric
        5. Hotspot Prediction Accuracy (hotspot_accuracy) - EDA professional metric
        
        Args:
            y_pred: Predicted power density [num_gates] or [num_gates, 1]
            y_true: True power density [num_gates] or [num_gates, 1]
            
        Returns:
            Dictionary containing 5 core evaluation metrics
        """
        # Input format processing and validation
        pred_power, true_power = self._process_inputs(y_pred, y_true)
        
        metrics = {}
        
        # 1-3. Power regression evaluation metrics (MAE, RMSE, R²)
        metrics.update(self._compute_regression_metrics(pred_power, true_power))
        
        # 4-5. EDA power domain professional metrics
        metrics.update(self._compute_eda_metrics(pred_power, true_power))
        
        return metrics
    
    def _process_inputs(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process and validate input formats
        
        Args:
            y_pred: Predicted power density
            y_true: True power density
            
        Returns:
            (pred_power, true_power): Processed power density tensors [num_gates]
        """
        # Process prediction results
        if y_pred.dim() == 2 and y_pred.size(1) == 1:
            pred_power = y_pred.squeeze(1)
        elif y_pred.dim() == 1:
            pred_power = y_pred
        else:
            raise ValueError(f"Unsupported prediction dimension: {y_pred.shape}, expected [num_gates] or [num_gates, 1]")
            
        # Process true labels
        if y_true.dim() == 2 and y_true.size(1) == 1:
            true_power = y_true.squeeze(1)
        elif y_true.dim() == 1:
            true_power = y_true
        else:
            raise ValueError(f"Unsupported true label dimension: {y_true.shape}, expected [num_gates] or [num_gates, 1]")
            
        # Validate shape matching
        if pred_power.shape != true_power.shape:
            raise ValueError(f"Prediction and true label shapes mismatch: {pred_power.shape} vs {true_power.shape}")
            
        # Validate power range
        if torch.any(pred_power < 0) or torch.any(true_power < 0):
            warnings.warn("Detected negative power values, which are physically unreasonable")
            
        return pred_power.float(), true_power.float()
    
    def _compute_regression_metrics(self, pred_power: torch.Tensor, true_power: torch.Tensor) -> Dict[str, float]:
        """
        Compute three power regression evaluation metrics: MAE, RMSE, R²
        """
        try:
            # 1. MAE (Mean Absolute Error)
            mae = torch.mean(torch.abs(pred_power - true_power)).item()
            
            # 2. RMSE (Root Mean Square Error)
            mse = torch.mean((pred_power - true_power) ** 2)
            rmse = torch.sqrt(mse).item()
            
            # 3. R² (Coefficient of Determination)
            ss_res = torch.sum((true_power - pred_power) ** 2)
            ss_tot = torch.sum((true_power - torch.mean(true_power)) ** 2)
            r2 = (1 - ss_res / ss_tot).item() if ss_tot > 0 else 0.0
            
            return {
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            }
            
        except Exception as e:
            warnings.warn(f"Error computing regression metrics: {e}")
            return {'mae': float('inf'), 'rmse': float('inf'), 'r2': -float('inf')}
    
    def _compute_eda_metrics(self, pred_power: torch.Tensor, true_power: torch.Tensor) -> Dict[str, float]:
        """
        Compute EDA power domain professional metrics
        
        Returns:
            power_uniformity: Power distribution uniformity [0, 1]
            hotspot_accuracy: Hotspot prediction accuracy [0, 1]
        """
        try:
            # 1. Power distribution uniformity
            power_uniformity = self._compute_power_uniformity(pred_power, true_power)
            
            # 2. Hotspot prediction accuracy
            hotspot_accuracy = self._compute_hotspot_accuracy(pred_power, true_power)
            
            return {
                'power_uniformity': power_uniformity,
                'hotspot_accuracy': hotspot_accuracy
            }
            
        except Exception as e:
            warnings.warn(f"Error computing EDA metrics: {e}")
            return {'power_uniformity': 0.0, 'hotspot_accuracy': 0.0}
    
    def _compute_power_uniformity(self, pred_power: torch.Tensor, true_power: torch.Tensor) -> float:
        """
        Compute power distribution uniformity metric
        
        Uses Jensen-Shannon divergence of histogram distributions to quantify distribution differences
        
        Returns:
            uniformity: Power distribution uniformity [0, 1], higher values indicate more similar distributions
        """
        try:
            bins = 20
            
            # Compute histograms
            pred_hist = torch.histc(pred_power, bins=bins, min=0, max=1)
            true_hist = torch.histc(true_power, bins=bins, min=0, max=1)
            
            # Normalize to probability distributions
            pred_hist = pred_hist / (pred_hist.sum() + 1e-8)
            true_hist = true_hist / (true_hist.sum() + 1e-8)
            
            # Compute Jensen-Shannon divergence
            js_divergence = self._jensen_shannon_divergence(pred_hist, true_hist)
            
            # Convert to similarity metric (higher is better)
            uniformity = 1.0 - js_divergence
            
            return max(0.0, min(1.0, uniformity))
            
        except Exception as e:
            warnings.warn(f"Error computing power distribution uniformity: {e}")
            return 0.0
    
    def _compute_hotspot_accuracy(self, pred_power: torch.Tensor, true_power: torch.Tensor) -> float:
        """
        Compute hotspot prediction accuracy metric
        
        Uses binary classification accuracy to evaluate hotspot identification performance
        
        Returns:
            accuracy: Hotspot prediction accuracy [0, 1]
        """
        try:
            # Determine hotspots based on threshold
            pred_hotspots = pred_power >= self.hotspot_threshold
            true_hotspots = true_power >= self.hotspot_threshold
            
            # Calculate accuracy
            correct_predictions = torch.sum(pred_hotspots == true_hotspots).item()
            total_predictions = len(pred_power)
            
            accuracy = correct_predictions / total_predictions
            
            return float(accuracy)
            
        except Exception as e:
            warnings.warn(f"Error computing hotspot prediction accuracy: {e}")
            return 0.0
    
    def _jensen_shannon_divergence(self, p: torch.Tensor, q: torch.Tensor) -> float:
        """
        Compute Jensen-Shannon divergence
        
        Args:
            p: Probability distribution 1
            q: Probability distribution 2
        
        Returns:
            js_div: Jensen-Shannon divergence [0, 1]
        """
        # Avoid log(0)
        p = p + 1e-8
        q = q + 1e-8
        
        # Compute average distribution
        m = 0.5 * (p + q)
        
        # Compute KL divergence
        kl_pm = torch.sum(p * torch.log(p / m))
        kl_qm = torch.sum(q * torch.log(q / m))
        
        # Jensen-Shannon divergence
        js_div = 0.5 * kl_pm + 0.5 * kl_qm
        
        # Normalize to [0,1]
        js_div = js_div / np.log(2)
        
        return float(js_div)

# ============================================================================
# Utility Functions
# ============================================================================

def create_evaluator(hotspot_threshold: float = 0.8) -> GatePowerDensityEvaluator:
    """
    Utility function to create Gate power density prediction evaluator
    
    Args:
        hotspot_threshold: Hotspot threshold
        
    Returns:
        Evaluator instance
    """
    return GatePowerDensityEvaluator(hotspot_threshold)

def evaluate_predictions_res(y_pred: torch.Tensor, y_true: torch.Tensor, 
                        hotspot_threshold: float = 0.8) -> Dict[str, float]:
    """
    Utility function to directly evaluate prediction results
    
    Args:
        y_pred: Predicted power density
        y_true: True power density labels
        hotspot_threshold: Hotspot threshold
        
    Returns:
        Evaluation metrics dictionary
    """
    evaluator = GatePowerDensityEvaluator(hotspot_threshold)
    return evaluator.eval(y_pred, y_true)