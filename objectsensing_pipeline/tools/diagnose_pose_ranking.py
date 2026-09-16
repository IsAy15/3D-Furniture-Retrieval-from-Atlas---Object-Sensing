"""Compare pose search and ranking on trusted, locally generated run caches.

No retrieval settings or previous run artifacts are modified. Pickle inputs must
only come from a trusted pipeline run; loading arbitrary pickle executes code.
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def centered_seed(points, baseline, degrees, scale_factor=1.0):
    """Change yaw/scale while preserving the model's bottom-center in world space."""
    from geometry import rotation_about_axis
    from transforms import GroundTransform
    pivot = (points.min(0) + points.max(0)) / 2
    pivot += baseline.up * ((points @ baseline.up).min() - pivot @ baseline.up)
    theta = np.radians(degrees)
    scale = baseline.scale * scale_factor
    translation = baseline.apply(pivot[None])[0] - scale * (
        rotation_about_axis(baseline.up, theta) @ pivot
    )
    return GroundTransform(theta, scale, translation, baseline.up.copy())


def render_report(directory):
    """Render saved numeric results and NPZ overlays, without loading pickles."""
    from PIL import Image, ImageDraw, ImageFont
    directory = Path(directory)
    report = json.loads((directory / 'comparison.json').read_text(encoding='utf-8'))
    models = report['models']
    font = ImageFont.load_default(size=18)
    chart = Image.new('RGB', (1100, 420 * len(models)), 'white')
    draw = ImageDraw.Draw(chart)
    for j, (name, data) in enumerate(models.items()):
        top = j * 420
        draw.text((60, top + 15), name, font=font, fill='black')
        draw.text((60, top + 42), 'Rotation only; same pivot and scale, no ICP', font=font, fill='#555555')
        for tick in range(6):
            y = top + 350 - tick * 52
            draw.line((60, y, 1040, y), fill='#dddddd')
            draw.text((10, y - 10), f'{tick/5:.1f}', font=font, fill='black')
        for tick in range(0, 361, 60):
            x = 60 + tick / 360 * 980
            draw.text((x - 15, top + 355), str(tick), font=font, fill='black')
        rows = [row for row in data['poses'] if row['mode'] == 'yaw_only']
        for field, color in (('current_score', '#157d91'), ('geometric_score', '#b44a54')):
            draw.line([(60 + r['yaw_deg'] / 360 * 980, top + 350 - r[field] * 260)
                       for r in rows], fill=color, width=3)
        base = data['best']['baseline']
        x, y = 60 + base['yaw_deg'] / 360 * 980, top + 350 - base['current_score'] * 260
        draw.ellipse((x-5, y-5, x+5, y+5), fill='black')
        draw.text((60, top+390), 'Teal: current score | Red: geometric score | Black: original pose | X: yaw (deg)', font=font, fill='black')
    chart.save(directory / 'yaw_scores.png')
    labels = ('baseline', 'yaw_only_current_score', 'yaw_icp_geometric_score')
    image = Image.new('RGB', (1500, 560 * len(models) + 45), 'white')
    draw = ImageDraw.Draw(image)
    draw.text((20, 10), 'Gray: scan. Orange: model. Same projection per row; display cropped, scores use full scan.', font=font, fill='black')
    # Fixed orthographic camera, world Y is up. No per-pose recentering.
    az, el = np.radians(-55), np.radians(22)
    horizontal = np.array([np.cos(az), 0, -np.sin(az)])
    vertical = np.array([-np.sin(az)*np.sin(el), np.cos(el), -np.cos(az)*np.sin(el)])
    for row_index, (name, data) in enumerate(models.items()):
        overlays = []
        for label in labels:
            with np.load(directory / f'{name}_{label}.npz', allow_pickle=False) as cloud:
                pts = cloud['points']
                colors = cloud['colors']
            mask = colors[:, 0] > 200
            overlays.append((pts, mask))
        model_points = np.vstack([points[mask] for points, mask in overlays])
        low, high = model_points.min(0) - 0.12, model_points.max(0) + 0.12
        center = (low + high) / 2
        factor = 350 / np.linalg.norm(high - low)
        for col, (label, (points, mask)) in enumerate(zip(labels, overlays)):
            inside = np.all((points >= low) & (points <= high), axis=1)
            for selected, color in ((inside & ~mask, '#b5b5b5'), (mask, '#da8b22')):
                p = points[selected] - center
                coords = np.column_stack((col*500 + 250 + p @ horizontal * factor,
                                          row_index*560 + 350 - p @ vertical * factor)).astype(int)
                for x, y in coords:
                    draw.point((int(x), int(y)), fill=color)
            r = data['best'][label]
            title = ('Original', 'Best yaw / current score', 'Best ICP / geometric score')[col]
            draw.text((col*500+20, row_index*560+70), f'{name[6:12]} | {title}\nYaw {r["yaw_deg"]:.1f}, scale {r["scale"]:.3f}\n'
                      f'Current {r["current_score"]:.3f} / geometric {r["geometric_score"]:.3f}', font=font, fill='black')
    image.save(directory / 'pose_comparison.png')


def main():
    import db_store
    from config import PipelineConfig
    from parallelism import configure_process_worker
    from progressive import _final_revalidate
    from verify import icp_refine, symmetric_coverage_score
    from viz import write_points_ply

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--yaw-step', type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.yaw_step <= 90:
        parser.error('--yaw-step must be between 1 and 90')
    if args.out.exists():
        parser.error('Output already exists; use a new diagnostic directory')
    configure_process_worker()
    config = json.loads((args.run / 'run_config.json').read_text(encoding='utf-8'))
    if config.get('query_mode') != 'single':
        parser.error('This diagnostic currently requires a single-query run')
    scene_caches = list((args.run / 'stage_cache').glob('scene_*.pkl'))
    if len(scene_caches) != 1:
        parser.error('Expected exactly one scene cache for an unambiguous comparison')
    with scene_caches[0].open('rb') as stream:
        scene = pickle.load(stream)['scene']
    with (args.run / 'resume/single_query_full.pkl').open('rb') as stream:
        results = pickle.load(stream)['results']
    registrations = {
        reg.model_name: reg for result in results.values()
        for reg in result['registrations']
    }
    cfg = PipelineConfig()
    for section in (cfg.keypoint, cfg.matching):
        for key, value in config.items():
            if hasattr(section, key) and value is not None:
                setattr(section, key, value)
    db = config['db_dir']
    names = db_store.load_index(db)['names']
    files = db_store.model_files(db)
    args.out.mkdir(parents=True)
    report = {'source_run': str(args.run.resolve()), 'yaw_step_deg': args.yaw_step,
              'icp_iterations': 10, 'scale_factors': [0.9, 1.0, 1.1],
              'ranking': 'harmonic(forward, visibility support if sufficiently known)',
              'warning': 'Best sampled score is not a ground-truth pose.',
              'geometric_best_scope': 'Only poses passing current final validation and score gate.',
              'models': {}}
    start = time.monotonic()
    for name in args.models:
        model = db_store.load_model(files[names.index(name)])
        baseline = registrations[name]
        poses = [copy.deepcopy(baseline)]
        tags = [('baseline', float(np.degrees(baseline.transform.theta)), 1.0)]
        failures = []
        points = model.cloud.points
        sub = points[np.linspace(0, len(points) - 1, min(2500, len(points))).astype(int)]
        for angle in range(0, 360, args.yaw_step):
            fixed = copy.deepcopy(baseline)
            fixed.transform = centered_seed(points, baseline.transform, angle)
            poses.append(fixed)
            tags.append(('yaw_only', angle, 1.0))
            for factor in report['scale_factors']:
                seed = centered_seed(points, baseline.transform, angle, factor)
                transform = icp_refine(sub, scene.cloud, seed, scene.up)
                if transform is None:
                    failures.append({'angle': angle, 'scale_factor': factor})
                    continue
                reg = copy.deepcopy(baseline)
                reg.transform = transform
                poses.append(reg)
                tags.append(('yaw_icp', angle, factor))
            if angle % 30 == 0:
                print(f'{name}: yaw {angle}/360, elapsed {time.monotonic()-start:.1f}s', flush=True)
        _, diagnostics = _final_revalidate(
            poses, db, scene, cfg, config['coverage'], config['final_reverse_gate'],
        )
        rows = []
        for reg, tag, diagnostic in zip(poses, tags, diagnostics):
            score = symmetric_coverage_score(reg.coverage, reg.cov_reverse)
            rows.append({
                'mode': tag[0], 'seed_yaw_deg': tag[1], 'seed_scale_factor': tag[2],
                'yaw_deg': float(np.degrees(reg.transform.theta) % 360),
                'scale': reg.transform.scale, 'translation': reg.transform.t.tolist(),
                'forward': reg.coverage, 'geometric_reverse': reg._geometric_cov_reverse,
                'visibility_support': getattr(reg, '_visibility_support', None),
                'known_fraction': getattr(reg, '_visibility_known_fraction', None),
                'current_score': score,
                'geometric_score': symmetric_coverage_score(reg.coverage, reg._geometric_cov_reverse),
                'surface_distance': reg.mean_surface_dist,
                'eligible': diagnostic['status'] == 'kept' and score >= config['final_min_symmetric_score'],
                'validation': diagnostic,
            })
        selected = {'baseline': 0}
        for mode in ('yaw_only', 'yaw_icp'):
            eligible = [i for i, row in enumerate(rows) if row['mode'] == mode and row['eligible']]
            if eligible:
                for score in ('current_score', 'geometric_score'):
                    selected[f'{mode}_{score}'] = max(eligible, key=lambda i: rows[i][score])
        for label, index in selected.items():
            # One overlay per alternative avoids confusing superposed hypotheses.
            combined = np.vstack((scene.cloud.points, poses[index].transform.apply(points)))
            colors = np.vstack((np.tile([150, 150, 150], (scene.cloud.size, 1)),
                                np.tile([235, 170, 65], (len(points), 1))))
            write_points_ply(args.out / f'{name}_{label}.ply', combined, colors)
            np.savez_compressed(args.out / f'{name}_{label}.npz', points=combined, colors=colors)
        report['models'][name] = {'poses': rows, 'best': {k: rows[i] for k, i in selected.items()},
                                  'invalid_icp_seeds': failures}
        (args.out / 'comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps({name: {k: {field: rows[i][field] for field in
              ('yaw_deg', 'scale', 'current_score', 'geometric_score', 'eligible')}
              for k, i in selected.items()}}, indent=2), flush=True)
    report['elapsed_seconds'] = time.monotonic() - start
    (args.out / 'comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
