import logging
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


LOGGER = logging.getLogger(__name__)


def load_prior_statistics(statistics_path: Optional[str]):
	if not statistics_path:
		return None
	if not os.path.exists(statistics_path):
		LOGGER.warning('Prior statistics file not found: %s', statistics_path)
		return None
	try:
		return np.load(statistics_path, allow_pickle=True)
	except Exception as exc:
		LOGGER.warning('Failed to load prior statistics from %s: %s', statistics_path, exc)
		return None


def _sample_from_histogram(grid: np.ndarray, pdf: np.ndarray, num_samples: int, fallback_min=0.0, fallback_max=1.0):
	if grid is None or pdf is None or len(grid) == 0 or len(pdf) == 0 or num_samples <= 0:
		return np.random.uniform(fallback_min, fallback_max, size=(num_samples,)).astype(np.float32)

	grid = np.asarray(grid, dtype=np.float32)
	pdf = np.asarray(pdf, dtype=np.float32)
	if pdf.ndim != 1:
		pdf = pdf.reshape(-1)
	if grid.ndim != 1:
		grid = grid.reshape(-1)
	if pdf.sum() <= 0:
		return np.random.uniform(fallback_min, fallback_max, size=(num_samples,)).astype(np.float32)

	cdf = np.cumsum(pdf)
	cdf /= cdf[-1]
	base = np.linspace(0.0, 1.0, num_samples, endpoint=False, dtype=np.float32)
	jitter = np.random.uniform(0.0, 1.0 / max(num_samples, 1), size=(num_samples,)).astype(np.float32)
	quantiles = np.clip(base + jitter, 0.0, 1.0)
	sample_positions = np.interp(quantiles, cdf, np.linspace(0.0, 1.0, len(cdf), dtype=np.float32))
	index_positions = sample_positions * max(len(grid) - 1, 1)
	return np.interp(index_positions, np.arange(len(grid), dtype=np.float32), grid).astype(np.float32)


def _init_uniform_priors(prior_embeddings: nn.Embedding, num_priors: int):
	bottom_priors_nums = num_priors * 3 // 4
	left_priors_nums, _ = num_priors // 8, num_priors // 8
	strip_size = 0.5 / (left_priors_nums // 2 - 1)
	bottom_strip_size = 1 / (bottom_priors_nums // 4 + 1)

	for i in range(left_priors_nums):
		nn.init.constant_(prior_embeddings.weight[i, 0], (i // 2) * strip_size)
		nn.init.constant_(prior_embeddings.weight[i, 1], 0.0)
		nn.init.constant_(prior_embeddings.weight[i, 2], 0.16 if i % 2 == 0 else 0.32)

	for i in range(left_priors_nums, left_priors_nums + bottom_priors_nums):
		nn.init.constant_(prior_embeddings.weight[i, 0], 0.0)
		nn.init.constant_(prior_embeddings.weight[i, 1], ((i - left_priors_nums) // 4 + 1) * bottom_strip_size)
		nn.init.constant_(prior_embeddings.weight[i, 2], 0.2 * (i % 4 + 1))

	for i in range(left_priors_nums + bottom_priors_nums, num_priors):
		nn.init.constant_(prior_embeddings.weight[i, 0], ((i - left_priors_nums - bottom_priors_nums) // 2) * strip_size)
		nn.init.constant_(prior_embeddings.weight[i, 1], 1.0)
		nn.init.constant_(prior_embeddings.weight[i, 2], 0.68 if i % 2 == 0 else 0.84)


def init_priors(prior_embeddings: nn.Embedding, cfg, img_w: int, img_h: int, num_priors: int):
	use_data_driven = bool(getattr(cfg, 'use_data_driven_priors', False))
	stats = load_prior_statistics(getattr(cfg, 'dataset_statistics', None))

	with torch.no_grad():
		if use_data_driven and stats is not None and 'cluster_centers' in stats:
			centers = np.asarray(stats['cluster_centers'], dtype=np.float32)
			if 'cluster_sizes' in stats and len(stats['cluster_sizes']) == len(centers):
				order = np.argsort(np.asarray(stats['cluster_sizes']))[::-1]
				centers = centers[order]

			n_centers = min(len(centers), num_priors)
			if centers.shape[1] >= 3:
				start_y = centers[:n_centers, 0]
				start_x = centers[:n_centers, 1]
				theta = centers[:n_centers, 2]
				if np.nanmax(np.abs(start_y)) > 2.0:
					start_y = start_y / max(img_h - 1, 1)
				if np.nanmax(np.abs(start_x)) > 2.0:
					start_x = start_x / max(img_w - 1, 1)
				if np.nanmax(np.abs(theta)) > 1.0:
					theta = theta / np.pi
				prior_embeddings.weight[:n_centers, 0] = torch.from_numpy(start_y.astype(np.float32))
				prior_embeddings.weight[:n_centers, 1] = torch.from_numpy(start_x.astype(np.float32))
				prior_embeddings.weight[:n_centers, 2] = torch.from_numpy(theta.astype(np.float32))

			remaining = num_priors - n_centers
			if remaining > 0:
				start_x = _sample_from_histogram(
					stats['start_x_grid'] if 'start_x_grid' in stats else None,
					stats['start_x_pdf'] if 'start_x_pdf' in stats else None,
					remaining,
					fallback_min=0.0,
					fallback_max=1.0,
				)
				start_y = _sample_from_histogram(
					stats['start_y_grid'] if 'start_y_grid' in stats else None,
					stats['start_y_pdf'] if 'start_y_pdf' in stats else None,
					remaining,
					fallback_min=0.2,
					fallback_max=1.0,
				)
				if 'thetas' in stats and len(stats['thetas']) > 0:
					theta_pool = np.asarray(stats['thetas'], dtype=np.float32)
					if np.nanmax(np.abs(theta_pool)) > 1.0:
						theta_pool = theta_pool / np.pi
					theta_index = np.linspace(0, len(theta_pool) - 1, remaining, dtype=np.int64)
					theta = theta_pool[theta_index]
				else:
					theta = np.random.uniform(0.15, 0.85, size=(remaining,)).astype(np.float32)

				prior_embeddings.weight[n_centers:, 0] = torch.from_numpy(start_y)
				prior_embeddings.weight[n_centers:, 1] = torch.from_numpy(start_x)
				prior_embeddings.weight[n_centers:, 2] = torch.from_numpy(theta)
			return stats

		_init_uniform_priors(prior_embeddings, num_priors)
		return stats
