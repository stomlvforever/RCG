#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gate Position Prediction Task Evaluator 

Implements grid-based region classification evaluation for Gate position prediction:
- Unified interface: metrics = evaluator.eval(y_pred, y_true)
- Five core metrics: MAE, RMSE, R², congestion hotspot distribution, region distribution uniformity
- Supports both classification logits and region ID input formats

Author: EDA for AI Team
"""

import torch
import numpy as np
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Core Evaluator Module
# ============================================================================
        
class GatePositionEvaluator:
    """
    Gate Position Prediction Evaluator (Streamlined Version, OGB-inspired Design)
    
    Focuses on core evaluation metrics with unified eval(y_pred, y_true) interface:
    - Five core metrics: MAE, RMSE, R², congestion hotspot distribution, region distribution uniformity
    - Supports both classification logits and region ID input formats
    - Automatic input format conversion
    """
    
    def __init__(self, grid_size: Tuple[int, int] = (16, 16)):
        """
        Initialize evaluator
        
        Args:
            grid_size: Grid dimensions (rows, cols)
        """
        self.grid_size = grid_size
        self.num_regions = grid_size[0] * grid_size[1]
    
    def eval(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, float]:
        """
        Main evaluation function (OGB-inspired interface design)
        
        Focuses on Gate position prediction evaluation with 5 core metrics:
        1. Mean Absolute Error (mae) - regression error metric
        2. Root Mean Square Error (rmse) - regression error metric  
        3. Coefficient of Determination (r2) - regression fit metric
        4. Congestion Hotspot Distribution (congestion_hotspot_score) - EDA professional metric
        5. Region Distribution Uniformity (distribution_uniformity) - layout uniformity metric
        
        Args:
            y_pred: Prediction results [num_gates, num_regions] or [num_gates] 
                   - logits format: [num_gates, num_regions] classification probabilities
                   - region ID format: [num_gates] direct region indices
            y_true: True region labels [num_gates]
            
        Returns:
            Dictionary containing 5 core evaluation metrics
        """
        # Input format processing and validation
        pred_regions, true_regions = self._process_inputs(y_pred, y_true)
        
        metrics = {}
        
        # 1-3. GNN node regression evaluation metrics (MAE, RMSE, R²)
        metrics.update(self._compute_regression_metrics(pred_regions, true_regions))
        
        # 4. Congestion hotspot distribution - EDA professional metric
        metrics.update(self._compute_congestion_hotspot_metrics(pred_regions, true_regions))
        
        # 5. Region distribution uniformity
        metrics.update(self._compute_distribution_metrics(pred_regions, true_regions))
        
        return metrics
    
    def _process_inputs(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process and validate input formats
        
        Args:
            y_pred: Prediction results (logits or region IDs)
            y_true: True region labels
            
        Returns:
            (pred_regions, true_regions): Processed region ID tensors
        """
        # Process prediction results
        if y_pred.dim() == 2:  # logits format [num_gates, num_regions]
            pred_regions = torch.argmax(y_pred, dim=1)
        elif y_pred.dim() == 1:  # region ID format [num_gates]
            pred_regions = y_pred.long()
        else:
            raise ValueError(f"Unsupported prediction dimension: {y_pred.dim()}, expected 1D or 2D")
            
        # Process true labels
        true_regions = y_true.long()
        
        # Validate shape matching
        if pred_regions.shape != true_regions.shape:
            raise ValueError(f"Prediction and true label shapes mismatch: {pred_regions.shape} vs {true_regions.shape}")
            
        # Validate region ID range
        max_region = self.num_regions - 1
        if torch.max(pred_regions) > max_region or torch.max(true_regions) > max_region:
            warnings.warn(f"Detected out-of-range region IDs, maximum allowed: {max_region}")
            
        return pred_regions, true_regions
    

    
    def _compute_regression_metrics(self, pred_regions: torch.Tensor, 
                                  true_regions: torch.Tensor) -> Dict[str, float]:
        """
        Compute three GNN node regression evaluation metrics: MAE, RMSE, R²
        
        Convert region classification problem to coordinate regression for evaluation
        """
        # Convert to normalized grid coordinates
        pred_coords = self._region_to_coords(pred_regions)
        true_coords = self._region_to_coords(true_regions)
        
        # 1. MAE (Mean Absolute Error)
        mae = torch.mean(torch.abs(pred_coords - true_coords)).item()
        
        # 2. RMSE (Root Mean Square Error)
        mse = torch.mean((pred_coords - true_coords) ** 2)
        rmse = torch.sqrt(mse).item()
        
        # 3. R² (Coefficient of Determination)
        ss_res = torch.sum((true_coords - pred_coords) ** 2)
        ss_tot = torch.sum((true_coords - torch.mean(true_coords, dim=0)) ** 2)
        r2 = (1 - ss_res / ss_tot).item() if ss_tot > 0 else 0.0
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
    
    def _compute_congestion_hotspot_metrics(self, pred_regions: torch.Tensor, 
                                          true_regions: torch.Tensor) -> Dict[str, float]:
        """
        Compute congestion hotspot distribution metrics - EDA professional evaluation metric
        
        Evaluate the rationality of region density distribution, avoiding overly congested hotspot areas
        
        Returns:
            congestion_hotspot_score: Congestion hotspot distribution rationality [0, 1], higher is better
        """
        try:
            # Calculate predicted and true region density distributions
            pred_density = torch.histc(
                pred_regions.float(), 
                bins=self.num_regions, 
                min=0, 
                max=self.num_regions-1
            )
            true_density = torch.histc(
                true_regions.float(), 
                bins=self.num_regions, 
                min=0, 
                max=self.num_regions-1
            )
            
            # Normalize density distributions
            pred_density = pred_density / pred_density.sum() if pred_density.sum() > 0 else pred_density
            true_density = true_density / true_density.sum() if true_density.sum() > 0 else true_density
            
            # Calculate congestion hotspot metrics
            # 1. Maximum density region congestion level (lower is better)
            max_pred_density = torch.max(pred_density).item()
            max_true_density = torch.max(true_density).item()
            
            # 2. Number of high-density regions (regions exceeding 2x average density)
            avg_density = 1.0 / self.num_regions  # Ideal uniform distribution average density
            hotspot_threshold = avg_density * 2.0
            
            pred_hotspots = torch.sum(pred_density > hotspot_threshold).item()
            true_hotspots = torch.sum(true_density > hotspot_threshold).item()
            
            # 3. Density variance (smaller is better, indicates more uniform distribution)
            pred_variance = torch.var(pred_density).item()
            true_variance = torch.var(true_density).item()
            
            # Comprehensive congestion hotspot scoring
            # Comprehensive evaluation based on maximum density, hotspot count, and variance
            max_density_score = 1.0 - min(max_pred_density / (max_true_density + 1e-8), 1.0)
            hotspot_count_score = 1.0 - min(pred_hotspots / max(true_hotspots, 1), 1.0)
            variance_score = 1.0 - min(pred_variance / (true_variance + 1e-8), 1.0)
            
            # Weighted average (maximum density has highest weight)
            congestion_score = (
                0.5 * max_density_score + 
                0.3 * hotspot_count_score + 
                0.2 * variance_score
            )
            
            return {
                'congestion_hotspot_score': max(0.0, min(1.0, congestion_score))
            }
            
        except Exception as e:
            print(f"Warning: Congestion hotspot distribution calculation failed: {e}")
            return {'congestion_hotspot_score': 0.0}
    
    def _compute_distribution_metrics(self, pred_regions: torch.Tensor, 
                                    true_regions: torch.Tensor) -> Dict[str, float]:
        """
        Compute region distribution related metrics
        
        Returns:
            Distribution uniformity improvement [-1, 1], positive values indicate more uniform predicted distribution
        """
        try:
            # Calculate region distribution histograms
            pred_hist = torch.histc(
                pred_regions.float(), 
                bins=self.num_regions, 
                min=0, 
                max=self.num_regions-1
            )
            true_hist = torch.histc(
                true_regions.float(), 
                bins=self.num_regions, 
                min=0, 
                max=self.num_regions-1
            )
            
            # Calculate distribution standard deviation (smaller is more uniform)
            pred_std = torch.std(pred_hist).item()
            true_std = torch.std(true_hist).item()
            
            # Calculate uniformity improvement
            if true_std > 0:
                uniformity = (true_std - pred_std) / true_std
                uniformity = max(-1.0, min(1.0, uniformity))
            else:
                uniformity = 0.0
            
            return {'distribution_uniformity': uniformity}
            
        except Exception as e:
            warnings.warn(f"Distribution uniformity calculation failed: {e}")
            return {'distribution_uniformity': 0.0}
    
    def _region_to_coords(self, regions: torch.Tensor) -> torch.Tensor:
        """
        Convert region IDs to normalized grid coordinates [0, 1]
        
        Args:
            regions: Region IDs [num_samples]
            
        Returns:
            coords: Normalized coordinates [num_samples, 2]
        """
        rows = regions // self.grid_size[1]
        cols = regions % self.grid_size[1]
        
        # Normalize to [0,1] range
        norm_cols = cols.float() / max(1, self.grid_size[1] - 1)
        norm_rows = rows.float() / max(1, self.grid_size[0] - 1)
        
        return torch.stack([norm_cols, norm_rows], dim=1)
    
# ============================================================================
# Utility Functions
# ============================================================================

def create_evaluator(grid_size: Tuple[int, int] = (16, 16)) -> GatePositionEvaluator:
    """
    Utility function to create Gate position prediction evaluator
    
    Args:
        grid_size: Grid dimensions (rows, cols)
        
    Returns:
        Evaluator instance
    """
    return GatePositionEvaluator(grid_size)

def evaluate_predictions(y_pred: torch.Tensor, y_true: torch.Tensor, 
                        grid_size: Tuple[int, int] = (16, 16)) -> Dict[str, float]:
    """
    Utility function to directly evaluate prediction results
    
    Args:
        y_pred: Prediction results (logits or region IDs)
        y_true: True region labels
        grid_size: Grid dimensions
        
    Returns:
        Evaluation metrics dictionary
    """
    evaluator = GatePositionEvaluator(grid_size)
    return evaluator.eval(y_pred, y_true)