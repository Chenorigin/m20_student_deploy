"""Bounded NumPy elevation estimator; no Torch, CUDA, PCL, LIO or global map."""
from collections import deque
from dataclasses import dataclass, fields
import math
import numpy as np
from .contract import GRID, ROWS, COLS, N, RESOLUTION, HEIGHT_OFFSET
from .geometry import yaw_rotation


def row_nanmedian(values):
    """Median of each finite-support row, without NumPy's masked-array path."""
    present = ~np.isnan(values)
    counts = present.sum(axis=1)
    ordered = np.sort(np.where(present, values, np.inf), axis=1)
    rows = np.arange(len(values))
    lower = ordered[rows, np.maximum(0, (counts-1)//2)]
    upper = ordered[rows, counts//2]
    return np.where(counts > 0, (lower+upper)*.5, np.nan)


@dataclass
class EstimatorConfig:
    # Engineering starting values; sensor/pose variances require physical calibration.
    range_std_m: float = .01
    range_quadratic: float = .001
    angular_std_rad: float = .002
    common_height_std_m: float = .015
    correlated_height_std_m: float = .01
    min_points: int = 5
    max_points_per_cell: int = 128
    max_effective_samples: int = 8
    min_xy_spread_m: float = .008
    max_extrapolation_m: float = .025
    max_slope: float = 2.
    max_residual_m: float = .035
    layer_gap_m: float = .08
    minority_fraction: float = .20
    outlier_floor_m: float = .025
    max_map_age_s: float = .30
    confidence_reference_std: float = .05
    confidence_age_tau_s: float = .50
    confidence_step_ambiguity_weight: float = 1.
    variance_growth_per_s: float = .001
    max_packets: int = 32
    min_range_m: float = .05
    max_range_m: float = 5.
    min_relative_z_m: float = -1.5
    max_relative_z_m: float = .5

    def validate(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError("Nonfinite estimator option: " + f.name)
            if f.name == "confidence_step_ambiguity_weight":
                if not 0 <= value <= 1:
                    raise ValueError("Confidence step ambiguity weight must be in [0,1]")
                continue
            if f.name != "min_relative_z_m" and value <= 0:
                raise ValueError("Estimator options must be positive: " + f.name)
        for name in ("min_points", "max_points_per_cell", "max_effective_samples", "max_packets"):
            if type(getattr(self, name)) is not int:
                raise ValueError(name + " must be an integer")
        if self.min_points < 3 or self.max_points_per_cell < self.min_points or self.max_packets > 128:
            raise ValueError("Invalid cell or packet capacity")
        if not 0 < self.minority_fraction < .5 or not self.min_range_m < self.max_range_m or not self.min_relative_z_m < self.max_relative_z_m:
            raise ValueError("Invalid estimator bounds")


@dataclass
class Patch:
    stamp: float
    anchor_stamp: float
    anchor: np.ndarray
    z: np.ndarray
    valid: np.ndarray
    variance: np.ndarray
    blocked: np.ndarray
    pose_variance: np.ndarray
    cell_stamps: np.ndarray = None
    block_stamps: np.ndarray = None
    cell_ends: np.ndarray = None
    block_ends: np.ndarray = None
    covariance_stamp: float = None


def confidence(valid, variance, age, cfg):
    usable = valid & np.isfinite(variance) & (variance >= 0) & np.isfinite(age) & (age >= -1e-8) & (age <= cfg.max_map_age_s)
    out = np.zeros_like(variance)
    out[usable] = np.exp(-np.maximum(0, age[usable])/cfg.confidence_age_tau_s)/(1+variance[usable]/cfg.confidence_reference_std**2)
    return out


def age_confidence(value, source_age, elapsed, valid, cfg):
    """Age the sender's quality variance, independent of full geometric variance."""
    out = np.zeros_like(value, dtype=np.float64)
    source_decay = np.exp(-np.maximum(0., source_age[valid])/cfg.confidence_age_tau_s)
    base = np.minimum(value[valid], source_decay)
    quality_weight = np.divide(base, source_decay, out=np.zeros_like(base), where=source_decay > 0)
    growth = elapsed*cfg.variance_growth_per_s/cfg.confidence_reference_std**2
    out[valid] = base*np.exp(-elapsed/cfg.confidence_age_tau_s)/(1+growth*quality_weight)
    return out


def edge_variance_components(z, valid, xy_variance, yaw_variance):
    """Separate continuous surface propagation from discrete step-location ambiguity."""
    z, m = z.reshape(ROWS, COLS), valid.reshape(ROWS, COLS)
    dx = np.diff(z, axis=1)*(m[:, 1:] & m[:, :-1])
    dy = np.diff(z, axis=0)*(m[1:] & m[:-1])
    left, right = np.pad(dx, ((0, 0), (1, 0))), np.pad(dx, ((0, 0), (0, 1)))
    down, up = np.pad(dy, ((1, 0), (0, 0))), np.pad(dy, ((0, 1), (0, 0)))
    gx = np.where(np.abs(left) > np.abs(right), left, right).ravel()/RESOLUTION
    gy = np.where(np.abs(down) > np.abs(up), down, up).ravel()/RESOLUTION
    edge = np.maximum.reduce([np.abs(left), np.abs(right), np.abs(down), np.abs(up)]).ravel()
    jy = -GRID[:, 1]*gx+GRID[:, 0]*gy
    displacement = np.sqrt(xy_variance)+np.linalg.norm(GRID, axis=1)*np.sqrt(yaw_variance)
    propagation = (gx*gx+gy*gy)*xy_variance + jy*jy*yaw_variance
    ambiguity = edge*edge*np.clip(2*displacement/RESOLUTION, 0, 1)
    return propagation, ambiguity


def edge_variance(z, valid, xy_variance, yaw_variance):
    propagation, ambiguity = edge_variance_components(z, valid, xy_variance, yaw_variance)
    return propagation+ambiguity


class ElevationMapper:
    def __init__(self, profile="v1", cfg=None, retain=True):
        if profile not in ("v0", "v1"):
            raise ValueError("Unknown profile")
        self.profile, self.cfg, self.retain = profile, cfg or EstimatorConfig(), retain
        self.cfg.validate()
        self.patches = deque(maxlen=self.cfg.max_packets)
        self.last_stamp = -math.inf
        self.stats = {}

    def reset(self):
        self.patches.clear()
        self.last_stamp = -math.inf

    def ingest(self, world_points, ranges, pose, motion_std_m=0., capture_stamp=None, sample_stamps=None,
               scan_stamp=None):
        cfg = self.cfg
        points = np.asarray(world_points, dtype=np.float64)
        ranges = np.asarray(ranges, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or ranges.shape != (len(points),):
            raise ValueError("Expected XYZ points and one range per point")
        input_stamp = pose.stamp if scan_stamp is None else float(scan_stamp)
        if not np.isfinite(input_stamp) or input_stamp <= self.last_stamp or not np.isfinite(motion_std_m) or motion_std_m < 0:
            raise ValueError("Repeated/reversed scan time or invalid motion uncertainty")
        capture_stamp = pose.stamp if capture_stamp is None else float(capture_stamp)
        if not np.isfinite(capture_stamp):
            raise ValueError("Nonfinite acquisition time")
        times = np.full(len(points),capture_stamp) if sample_stamps is None else np.asarray(sample_stamps,dtype=np.float64)
        if times.shape != (len(points),) or not np.isfinite(times).all():
            raise ValueError('Expected one finite acquisition time per point')
        self.last_stamp = input_stamp
        local = (points-pose.position) @ yaw_rotation(pose.yaw)
        keep = (np.isfinite(points).all(1) & np.isfinite(ranges) &
                (ranges >= cfg.min_range_m) & (ranges <= cfg.max_range_m) &
                (np.abs(local[:, 0]) <= .85) & (np.abs(local[:, 1]) <= .55) &
                (local[:, 2] >= cfg.min_relative_z_m) & (local[:, 2] <= cfg.max_relative_z_m))
        local, points, ranges, times = local[keep], points[keep], ranges[keep], times[keep]
        ix = np.floor((local[:, 0]+.85)/RESOLUTION).astype(int)
        iy = np.floor((local[:, 1]+.55)/RESOLUTION).astype(int)
        ids = iy*COLS+ix
        keep = (ix >= 0) & (ix < COLS) & (iy >= 0) & (iy < ROWS)
        local, points, ranges, ids, times = local[keep], points[keep], ranges[keep], ids[keep], times[keep]
        order = np.argsort(ids, kind="stable")
        ids, local, points, ranges, times = ids[order], local[order], points[order], ranges[order], times[order]
        boundaries = np.searchsorted(ids, np.arange(N+1))
        z, var = np.zeros(N), np.zeros(N)
        valid, blocked = np.zeros(N, dtype=bool), np.zeros(N, dtype=bool)
        cell_stamps,block_stamps = np.full(N,capture_stamp),np.full(N,capture_stamp)
        cell_ends,block_ends = np.full(N,capture_stamp),np.full(N,capture_stamp)
        counts = np.diff(boundaries)
        # Diagnostics count overlapping predicates; they do not change support,
        # fit decisions or unknown-only tombstones.
        fit_rejections = {"insufficient_points": int((counts < cfg.min_points).sum()),
                          "multiple_layers": 0, "narrow_xy": 0, "extrapolated": 0,
                          "insufficient_inliers": 0, "slope_or_residual": 0}
        cells = np.flatnonzero(counts >= cfg.min_points)
        if len(cells):
            # Bound every cell to the same sampled support as the scalar implementation.
            # Batch the robust fits to avoid hundreds of Python/LAPACK calls on ARM.
            sizes = np.minimum(counts[cells], cfg.max_points_per_cell)
            columns = np.arange(cfg.max_points_per_cell)[None, :]
            sample_mask = columns < sizes[:, None]
            steps = (counts[cells]-1)/np.maximum(sizes-1, 1)
            indices = boundaries[cells, None] + (columns*steps[:, None]).astype(int)
            indices = np.minimum(indices, boundaries[cells+1, None]-1)
            indices = np.where(columns == sizes[:, None]-1, boundaries[cells+1, None]-1, indices)
            p, r = local[indices], ranges[indices]
            support_times = times[indices]
            block_stamps[cells] = np.min(np.where(sample_mask,support_times,np.inf),axis=1)
            block_ends[cells] = np.max(np.where(sample_mask,support_times,-np.inf),axis=1)
            heights = np.sort(np.where(sample_mask, p[:, :, 2], np.inf), axis=1)
            support = np.maximum(3, np.ceil(sizes*cfg.minority_fraction))
            split = np.arange(1, cfg.max_points_per_cell)[None, :]
            gaps = np.zeros((len(cells), cfg.max_points_per_cell-1))
            np.subtract(heights[:, 1:], heights[:, :-1], out=gaps, where=split < sizes[:, None])
            multiple = np.any((gaps > cfg.layer_gap_m) & (split >= support[:, None]) &
                              (sizes[:, None]-split >= support[:, None]), axis=1)
            blocked[cells[multiple]] = True
            center = row_nanmedian(np.where(sample_mask, p[:, :, 2], np.nan))
            mad = 1.4826*row_nanmedian(np.where(sample_mask, np.abs(p[:, :, 2]-center[:, None]), np.nan))
            inliers = sample_mask & (np.abs(p[:, :, 2]-center[:, None]) <=
                                    np.maximum(cfg.outlier_floor_m, 3*mad)[:, None])
            n = inliers.sum(1)
            xy = p[:, :, :2]-GRID[cells, None, :]
            mean = np.sum(np.where(inliers[:, :, None], xy, 0.), axis=1)/np.maximum(n, 1)[:, None]
            d = xy-mean[:, None, :]
            cov = np.einsum('bni,bnj,bn->bij', d, d, inliers.astype(float))/np.maximum(n, 1)[:, None, None]
            eigen_min = .5*(cov[:, 0, 0]+cov[:, 1, 1]-np.hypot(cov[:, 0, 0]-cov[:, 1, 1], 2*cov[:, 0, 1]))
            # Preserve the scalar decision at the exact support threshold (ulp boundary).
            threshold = cfg.min_xy_spread_m**2
            boundary = np.abs(eigen_min-threshold) <= 64*np.finfo(float).eps*np.maximum(threshold, np.trace(cov, axis1=1, axis2=2))
            for row in np.flatnonzero(boundary & (n >= cfg.min_points)):
                eigen_min[row] = np.linalg.eigvalsh(np.cov(xy[row, inliers[row]].T, bias=True))[0]
            wall = (n >= cfg.min_points) & (eigen_min < cfg.min_xy_spread_m**2)
            blocked[cells[wall & ~multiple]] = True
            lo = np.min(np.where(inliers[:, :, None], xy, np.inf), axis=1)
            hi = np.max(np.where(inliers[:, :, None], xy, -np.inf), axis=1)
            extrapolated = np.any(lo > cfg.max_extrapolation_m, axis=1) | np.any(hi < -cfg.max_extrapolation_m, axis=1)
            eligible = ~multiple & ~wall & ~extrapolated & (n >= cfg.min_points)
            fit_rejections.update(multiple_layers=int(multiple.sum()), narrow_xy=int(wall.sum()),
                                  extrapolated=int(extrapolated.sum()),
                                  insufficient_inliers=int((n < cfg.min_points).sum()))
            fit_rows = np.flatnonzero(eligible)
            if len(fit_rows):
                selected = cells[fit_rows]
                p, r, inliers = p[fit_rows], r[fit_rows], inliers[fit_rows]
                xy = xy[fit_rows]
                # Scaling XY keeps the 3x3 systems well conditioned at centimeter support.
                A = np.concatenate((xy/RESOLUTION, np.ones((*xy.shape[:2], 1))), axis=2)
                weights = inliers.astype(float)
                coeff = np.zeros((len(fit_rows), 3))
                for _ in range(3):
                    lhs = np.einsum('bni,bnj,bn->bij', A, A, weights)
                    rhs = np.einsum('bni,bn,bn->bi', A, p[:, :, 2], weights)
                    try:
                        coeff = np.linalg.solve(lhs, rhs)
                    except np.linalg.LinAlgError as exc:
                        raise ValueError("Numerically singular supported plane fit") from exc
                    residual = p[:, :, 2]-np.einsum('bni,bi->bn', A, coeff)
                    med = row_nanmedian(np.where(inliers, residual, np.nan))
                    scale = np.maximum(cfg.range_std_m, 1.4826*row_nanmedian(
                        np.where(inliers, np.abs(residual-med[:, None]), np.nan)))
                    weights = inliers*np.minimum(1., 1.5*scale[:, None]/np.maximum(np.abs(residual), 1e-12))
                slopes = coeff[:, :2]/RESOLUTION
                rms = np.sqrt(np.sum(weights*residual**2, axis=1)/weights.sum(1))
                slope_norm = np.linalg.norm(slopes, axis=1)
                fit_boundary = ((np.abs(slope_norm-cfg.max_slope) <= 64*np.finfo(float).eps*cfg.max_slope) |
                                (np.abs(rms-cfg.max_residual_m) <= 64*np.finfo(float).eps*cfg.max_residual_m))
                for row in np.flatnonzero(fit_boundary):
                    mask = inliers[row]
                    original_A = np.column_stack((xy[row, mask], np.ones(mask.sum())))
                    original_z = p[row, mask, 2]
                    original_w = np.ones(mask.sum())
                    for _ in range(3):
                        original_c = np.linalg.lstsq(original_A*np.sqrt(original_w[:, None]), original_z*np.sqrt(original_w), rcond=None)[0]
                        original_r = original_z-original_A@original_c
                        original_scale = max(cfg.range_std_m, 1.4826*np.median(np.abs(original_r-np.median(original_r))))
                        original_w = np.minimum(1., 1.5*original_scale/np.maximum(np.abs(original_r), 1e-12))
                    slopes[row], coeff[row, 2] = original_c[:2], original_c[2]
                    residual[row, mask], weights[row, mask] = original_r, original_w
                    rms[row] = np.sqrt(np.average(original_r**2, weights=original_w))
                rejected = (np.linalg.norm(slopes, axis=1) > cfg.max_slope) | (rms > cfg.max_residual_m)
                fit_rejections["slope_or_residual"] = int(rejected.sum())
                blocked[selected[rejected]] = True
                sensor = (cfg.range_std_m+cfg.range_quadratic*r*r)**2+(r*cfg.angular_std_rad)**2
                sensor_var = np.sum(np.where(inliers, sensor, 0.), axis=1)/inliers.sum(1)
                effective_n = np.minimum(cfg.max_effective_samples, weights.sum(1)**2/np.sum(weights**2, axis=1))
                body = (points[indices[fit_rows]]-pose.position) @ pose.R
                e = pose.R[2]
                jr = e[2]*body[:, :, 1]-e[1]*body[:, :, 2]
                jp = e[0]*body[:, :, 2]-e[2]*body[:, :, 0]
                tilt = (np.sum(np.where(inliers, jr**2, 0.), axis=1)*pose.variance[3] +
                        np.sum(np.where(inliers, jp**2, 0.), axis=1)*pose.variance[4])/inliers.sum(1)
                xy_uncertainty = np.max(pose.variance[:2])*np.sum(slopes**2, axis=1)
                variance = (sensor_var/effective_n + rms*rms + cfg.common_height_std_m**2 +
                            cfg.correlated_height_std_m**2 + pose.variance[2] + tilt + xy_uncertainty + motion_std_m**2)
                accepted = ~rejected
                var[selected[accepted]] = variance[accepted]
                z[selected[accepted]] = pose.position[2]+coeff[accepted, 2]
                valid[selected[accepted]] = True
                actual_support = inliers & (weights > 0)
                oldest_support = np.min(np.where(actual_support,support_times[fit_rows],np.inf),axis=1)
                cell_stamps[selected[accepted]] = oldest_support[accepted]
                newest_support = np.max(np.where(actual_support,support_times[fit_rows],-np.inf),axis=1)
                cell_ends[selected[accepted]] = newest_support[accepted]
        if not self.retain or self.profile == "v0":
            self.patches.clear()
        self.patches.append(Patch(capture_stamp, pose.stamp, np.array([*pose.position[:2], pose.yaw]), z, valid, var, blocked, pose.variance.copy(),cell_stamps,block_stamps,cell_ends,block_ends,input_stamp))
        self.stats = {"input_points": len(world_points), "roi_points": len(local), "valid_cells": int(valid.sum()),
                      "ambiguous_cells": int(blocked.sum()), "packets": len(self.patches),
                      "fit_rejections": fit_rejections,
                      "fit_rejection_counts_overlap": True,
                      "fit_rejection_scope": "187 fitting cells before query visibility mask"}

    def snapshot(self, pose):
        cfg = self.cfg
        tracing = getattr(self, 'trace_sources', False)
        if tracing:
            selected_scan = np.full(N, np.nan)
            selected_cell = np.full(N, -1, dtype=int)
            selected_end = np.full(N, np.nan)
        # Individual source timestamps survive querying and packet loss.
        def cell_times(p):
            return p.cell_stamps if p.cell_stamps is not None else np.full(N,p.stamp)
        def block_times(p):
            return p.block_stamps if p.block_stamps is not None else np.full(N,p.stamp)
        def cell_ends(p):
            return p.cell_ends if p.cell_ends is not None else cell_times(p)
        def block_ends(p):
            return p.block_ends if p.block_ends is not None else block_times(p)
        self.patches = deque((p for p in self.patches if
            np.any(p.valid & (pose.stamp-cell_times(p) <= cfg.max_map_age_s)) or
            np.any(p.blocked & (pose.stamp-block_ends(p) <= cfg.max_map_age_s))),maxlen=cfg.max_packets)
        z, var, capture = np.zeros(N), np.zeros(N), np.full(N, -math.inf)
        xy_var, yaw_var = np.zeros(N), np.zeros(N)
        barrier = np.full(N,-math.inf)
        query_world = GRID @ yaw_rotation(pose.yaw)[:2, :2].T + pose.position[:2]
        for patch in self.patches:
            local = (query_world-patch.anchor[:2]) @ yaw_rotation(patch.anchor[2])[:2, :2]
            inside = (np.abs(local[:, 0]) <= .8+1e-7) & (np.abs(local[:, 1]) <= .5+1e-7)
            ix = np.rint((local[:, 0]+.8)/RESOLUTION).astype(int).clip(0, COLS-1)
            iy = np.rint((local[:, 1]+.5)/RESOLUTION).astype(int).clip(0, ROWS-1)
            source = iy*COLS+ix
            measured,blocked_at = cell_times(patch)[source],block_ends(patch)[source]
            # A contradictory interval invalidates older surfaces through its
            # newest evidence. Retain this unknown-only tombstone until every
            # affected older observation must itself have expired.
            active_block = inside & patch.blocked[source] & (blocked_at <= pose.stamp+1e-8) & (pose.stamp-blocked_at <= cfg.max_map_age_s)
            barrier[active_block] = np.maximum(barrier[active_block],blocked_at[active_block])
            invalidated = active_block & (capture <= barrier)
            capture[invalidated] = -math.inf
            chosen = (inside & patch.valid[source] & (measured <= pose.stamp+1e-8) &
                      (cell_ends(patch)[source] <= pose.stamp+1e-8) &
                      (pose.stamp-measured <= cfg.max_map_age_s) & (measured >= capture) & (measured > barrier))
            z[chosen], var[chosen], capture[chosen] = patch.z[source[chosen]], patch.variance[source[chosen]], measured[chosen]
            if tracing:
                selected_scan[chosen] = patch.covariance_stamp if patch.covariance_stamp is not None else patch.stamp
                selected_cell[chosen] = source[chosen]
                selected_end[chosen] = cell_ends(patch)[source[chosen]]
            reference_stamp = patch.anchor_stamp if patch.covariance_stamp is None else patch.covariance_stamp
            if pose.stamp > reference_stamp+1e-8:
                # Relative vertical uncertainty with unknown temporal correlation:
                # Var(z_query-z_capture) <= (sigma_query+sigma_capture)^2.
                # Capture variance is already in the packet; only add the remainder.
                var[chosen] += pose.variance[2] + 2*np.sqrt(pose.variance[2]*patch.pose_variance[2])
            # Capture covariance was already propagated through the local fitted surface.
            # These relative map/query terms are used only for reprojection and step ambiguity.
            xy_var[chosen] = max(patch.pose_variance[0], patch.pose_variance[1], pose.variance[0], pose.variance[1])
            yaw_var[chosen] = max(patch.pose_variance[5], pose.variance[5])
        age = pose.stamp-capture
        valid = np.isfinite(capture) & (age >= -1e-8) & (age <= cfg.max_map_age_s)
        if tracing:
            self.query_sources = {
                'scan_stamp_s': np.where(valid, selected_scan, np.nan),
                'source_cell': np.where(valid, selected_cell, -1),
                'oldest_support_s': np.where(valid, capture, np.nan),
                'newest_support_s': np.where(valid, selected_end, np.nan)}
        quality_var = var.copy()
        if self.profile == "v1":
            propagation, ambiguity = edge_variance_components(z, valid, xy_var, yaw_var)
            var += propagation+ambiguity
            # A supported tread's height quality need not include the full
            # uncertainty about which side of a discontinuity the query lies on.
            # Keep that uncertainty in the published geometric variance.
            quality_var += propagation+cfg.confidence_step_ambiguity_weight*ambiguity
        var[valid] += np.maximum(0, age[valid])*cfg.variance_growth_per_s
        quality_var[valid] += np.maximum(0, age[valid])*cfg.variance_growth_per_s
        c = confidence(valid, quality_var, age, cfg)
        h = np.where(valid, np.clip(pose.position[2]-z-HEIGHT_OFFSET, -1, 1), 0.)
        channels = np.stack((h, valid.astype(float), c)).astype(np.float32)
        return channels, np.where(valid, var, 0.).astype(np.float32), np.where(valid, np.maximum(0, age), -1.).astype(np.float32)
