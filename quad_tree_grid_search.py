import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import math


class QuadNode:
  def __init__(self, bounds, level=0):
    self.bounds = bounds  # (C_min, C_max, gamma_min, gamma_max)
    self.children = []
    self.level = level
    self.is_leaf = False
    self.score = None
    self.n_support = None


class QuadTreeGridSearch:
  def __init__(
    self,
    C_range=(-8, 8),
    gamma_range=(-8, 8),
    cv=5,
    min_cv_score=0.7,
    max_sv_ratio=0.5,
    min_grid_size=0.5,
  ):
    self.C_range = C_range
    self.gamma_range = gamma_range
    self.cv = cv
    self.min_cv_score = min_cv_score
    self.max_sv_ratio = max_sv_ratio
    self.min_grid_size = min_grid_size
    self.best_params = None
    self.best_score = 0

  def _evaluate_node(self, node, X, y):
    """Evaluate SVM performance at node vertices"""
    C_min, C_max, g_min, g_max = node.bounds
    vertices = [(C_min, g_min), (C_min, g_max), (C_max, g_min), (C_max, g_max)]

    scores = []
    sv_ratios = []

    for C, gamma in vertices:
      # Convert from log2 space
      C_actual = 2**C
      gamma_actual = 2**gamma

      # Train SVM
      svm = SVC(C=C_actual, gamma=gamma_actual, kernel='rbf')
      cv_scores = cross_val_score(svm, X, y, cv=self.cv)
      mean_score = np.mean(cv_scores)

      # Fit to get number of support vectors
      svm.fit(X, y)
      sv_ratio = len(svm.support_) / len(X)

      scores.append(mean_score)
      sv_ratios.append(sv_ratio)

      # Track best parameters
      if mean_score > self.best_score and sv_ratio <= self.max_sv_ratio:
        self.best_score = mean_score
        self.best_params = {'C': C_actual, 'gamma': gamma_actual}

    return scores, sv_ratios

  def _should_split(self, scores, sv_ratios):
    """Determine if node should be split based on evaluation criteria"""
    all_good = all(
      score >= self.min_cv_score and sv <= self.max_sv_ratio
      for score, sv in zip(scores, sv_ratios)
    )
    all_bad = all(
      score < self.min_cv_score or sv > self.max_sv_ratio
      for score, sv in zip(scores, sv_ratios)
    )
    return not (all_good or all_bad)

  def _split_node(self, node):
    """Split node into four quadrants"""
    C_min, C_max, g_min, g_max = node.bounds
    C_mid = (C_min + C_max) / 2
    g_mid = (g_min + g_max) / 2

    # Create four children nodes
    quads = [
      (C_min, C_mid, g_min, g_mid),  # SW
      (C_mid, C_max, g_min, g_mid),  # SE
      (C_min, C_mid, g_mid, g_max),  # NW
      (C_mid, C_max, g_mid, g_max),  # NE
    ]

    for bounds in quads:
      child = QuadNode(bounds, node.level + 1)
      node.children.append(child)

  def fit(self, X, y):
    """Main fitting method"""
    # Create root node
    root = QuadNode(
      (self.C_range[0], self.C_range[1], self.gamma_range[0], self.gamma_range[1])
    )

    nodes_to_process = [root]

    while nodes_to_process:
      print(f'Level {root.level}: {len(nodes_to_process)} nodes to process')
      node = nodes_to_process.pop(0)

      # Check if quadrant is too small
      C_size = node.bounds[1] - node.bounds[0]
      g_size = node.bounds[3] - node.bounds[2]
      if C_size <= self.min_grid_size or g_size <= self.min_grid_size:
        node.is_leaf = True
        continue

      # Evaluate node
      scores, sv_ratios = self._evaluate_node(node, X, y)

      # Determine if node should be split
      if self._should_split(scores, sv_ratios):
        self._split_node(node)
        nodes_to_process.extend(node.children)
      else:
        node.is_leaf = True
        node.score = np.mean(scores)
        node.n_support = np.mean(sv_ratios)

    return self

  def get_best_params(self):
    """Return the best parameters found"""
    return self.best_params
