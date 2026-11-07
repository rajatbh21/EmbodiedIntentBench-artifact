"""Freeze and acquire broad curved mission layouts without changing old studies.

The shared terrain is still the known Lake map. Each new scenario has a separate
authored route, scenery layout and distributed target population. Geometry is
declared before RGB or model predictions exist. This module acquires sensor data;
it does not silently reuse the old 120-second mission or claim router results.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from experiments.lake_missions.assets import verify as verify_assets
from experiments.lake_missions.placement import collision_bounds, place, terrain_height
from experiments.wide_area_missions.routes import build_route

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name('config.json')
DEFAULT_ASSETS = ROOT/'results/lake_assets_collision_aligned_v2'
TEMPLATE = ROOT/'results/routing_development_matrix_v1_20261017/east_shore_day/rapid_survey/scenario.json'
DAY = dict(ambient=.5, sunlight=.9, fog_density=0., rgb_noise_stddev=0., width_px=960, height_px=720)


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clearance_surface(assets, bounds):
    """Inflate known physical scenery in XY before the route altitude envelope."""
    def height(x,y):
        floor=terrain_height(assets,x,y)
        for _,low,high in bounds:
            if low[0]-5 <= x <= high[0]+5 and low[1]-5 <= y <= high[1]+5:
                floor=max(floor,float(high[2]))
        return floor
    return height


def distributed_targets(assets, route, bounds, *, sectors, per_sector, name):
    """Find grounded clusters beside the route using geometry, never predictions."""
    template=read(TEMPLATE)['world']['objects'][0]
    points=route['points']
    arc=np.asarray([p['distance_along_route_m'] for p in points])
    occupied,objects,groups=[],[],[]
    for sector,fraction in enumerate(np.linspace(.07,.92,sectors),1):
        center=int(np.argmin(np.abs(arc-fraction*arc[-1])))
        candidates=sorted(range(max(1,center-80),min(len(points)-1,center+81)),
                          key=lambda i:abs(arc[i]-arc[center]))
        chosen=None
        for index in candidates:
            point=np.asarray(points[index]['position_m'])
            tangent=np.asarray(points[index+1]['position_m'][:2])-points[index-1]['position_m'][:2]
            norm=np.linalg.norm(tangent)
            if norm<1e-6:continue
            tangent=tangent/norm
            normal=np.array([-tangent[1],tangent[0]])
            trial=[]
            try:
                for number in range(per_sector):
                    desired=point[:2]+normal*((number-(per_sector-1)/2)*5)+tangent*4
                    pos=place(assets,desired,bounds,occupied+trial)
                    if np.linalg.norm(np.asarray(pos[:2])-point[:2])>18:
                        raise ValueError('Cluster moved outside the declared near-route corridor')
                    trial.append(pos)
            except ValueError:
                continue
            if any(math.dist(point[:2],g['center_xy_m'])<45 for g in groups):continue
            chosen=(index,point,trial)
            break
        if chosen is None:raise ValueError(f'No grounded target cluster in sector {sector}: {name}')
        index,point,trial=chosen
        ids=[]
        for number,pos in enumerate(trial,1):
            label=len(objects)+1
            obj=copy.deepcopy(template)
            obj.update(object_id=f'{name}_sector_{sector}_person_{number}',label=label,position_m=pos)
            obj['mesh']['pose'][-1]=(label%8)*math.pi/4
            objects.append(obj);occupied.append(pos);ids.append(obj['object_id'])
        groups.append(dict(sector_id=f'sector_{sector}',center_xy_m=point[:2].tolist(),
                           nominal_pass_time_s=points[index]['time_s'],evaluator_target_ids=ids))
    return objects,groups


def _new_bounds(layout):
    result=[]
    for entry in layout.get('collision_bounds',[]):
        if isinstance(entry,dict):
            result.append((entry.get('name',entry.get('id','new_prop')),np.asarray(entry['low']),np.asarray(entry['high'])))
        else:
            name,low,high=entry;result.append((name,np.asarray(low),np.asarray(high)))
    return result


def prepare(output, *, config=DEFAULT_CONFIG, assets=DEFAULT_ASSETS):
    from experiments.wide_area_missions.scenery import build_layout
    output=Path(output).resolve();assets=Path(assets).resolve()
    if output.exists():raise FileExistsError('Preserve previous acquisitions; choose a new output directory')
    verify_assets(assets)
    settings=read(config)
    original_bounds=collision_bounds(assets)
    output.mkdir(parents=True)
    cases={}
    for number,name in enumerate(settings['routes']):
        folder=output/name;folder.mkdir()
        route=build_route(name,clearance_surface(assets,original_bounds),
                          **{key:settings[key] for key in ('sample_spacing_m','cruise_speed_mps',
                             'max_horizontal_acceleration_mps2','max_vertical_speed_mps','clearance_m')})
        layout=build_layout(assets,route['points'],folder/'scenery',seed=settings['seed']+number,clearance_m=12.,count=settings.get('scenery_prop_count',72))
        objects,sectors=distributed_targets(assets,route,original_bounds+_new_bounds(layout),
                    sectors=settings['sector_count'],per_sector=settings['targets_per_sector'],name=name)
        duration=math.ceil(route['duration_s']/settings['capture_interval_s'])*settings['capture_interval_s']+settings['final_hover_s']
        points=copy.deepcopy(route['points'])
        points[-1]['velocity_mps']=[0.,0.,0.]
        end=copy.deepcopy(points[-1]);end.update(time_s=duration,speed_mps=0.)
        points.append(end)
        scenario=dict(schema_version='wide_area_physical_scenario_v1',scenario_id=name,
            asset_directory=str(assets),environment=dict(DAY,seed=settings['seed']+number),
            warmup_s=settings['warmup_s'],duration_s=duration,capture_interval_s=settings['capture_interval_s'],
            world=dict(objects=objects,initial_position_m=points[0]['position_m'],trajectory=points,
                       max_speed_mps=settings['cruise_speed_mps'],camera_pitch_rad=math.pi/2),
            scenery=layout,
            controller=dict(position_gain=.8,max_acceleration_mps2=1.5,
                            command_speed_limit_mps=settings['cruise_speed_mps']+1,
                            tracking_tolerance_m=5.,minimum_clearance_m=10.),
            scope='New authored known-map development acquisition; not independent map or field data')
        save(folder/'route.json',route)
        save(folder/'scenario.json',scenario)
        save(folder/'mission_design.json',dict(name=route['name'],sectors=sectors,target_count=len(objects),
            mission_style=route['design'],
            timing_reference='Nominal planned route passes; no mission scoring thresholds have been inferred from model outputs.',
            routing_evaluation_status='Acquisition only. Requires separately frozen intent, model cache, calibrated long-flight costs and paired router protocol.',
            flight_energy_note='Do not reuse the historical 120 s / 30 Wh battery contract for these longer flights.'))
        cases[name]=dict(scenario=str((folder/'scenario.json').relative_to(output)),
            route_name=route['name'],horizontal_distance_m=route['horizontal_distance_m'],
            duration_s=duration,bounds=route['bounds'],target_count=len(objects),
            sectors=len(sectors),new_scenery_count=len(layout.get('props',[])))
        print(f'Prepared {name}: {route["distance_m"]:.0f} m, {duration:.0f} s, {len(objects)} targets',flush=True)
    external_mesh_manifests={obj['mesh']['manifest'] for name in cases
                            for obj in read(output/name/'scenario.json')['world']['objects'] if 'mesh' in obj}
    sources=[*Path(__file__).parent.glob('*.py'),
             ROOT/'experiments/lake_missions/world.py',ROOT/'experiments/lake_missions/placement.py',
             ROOT/'experiments/v4_gazebo_pilot/world.py',ROOT/'experiments/v4_gazebo_pilot/transport.py',TEMPLATE]
    frozen={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()}
    protocol=dict(schema_version='wide_area_acquisition_cohort_v1',scope='authored_known_map_development',
        terrain_count=1,authored_layout_count=len(cases),cases=cases,settings=settings,
        asset_directory=str(assets),asset_manifest_sha256=sha(assets/'manifest.json'),
        settings_source_sha256=sha(config),
        configuration=dict(path=str(Path(config).resolve()),sha256=sha(config)),
        external_mesh_manifest_sha256={str(Path(p).resolve()):sha(p) for p in external_mesh_manifests},
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},frozen_files=frozen,
        predictions_used_for_design=False,
        references=[{'title':'QGroundControl Survey and terrain following',
                     'url':'https://docs.qgroundcontrol.com/master/en/qgc-user-guide/plan_view/pattern_survey.html'},
                    {'title':'ArduPilot spline waypoints',
                     'url':'https://ardupilot.org/copter/docs/mission-command-list.html'}],
        limitations=['Authored spline paths, not recorded real-world UAV missions.',
          'Route extent is not complete area coverage; camera footprint must be measured separately.',
          'New layouts share the existing 1.2 km Lake terrain and asset family.',
          'Ideal-odometry controller and simplified quadrotor; no claim of PX4/ArduPilot integration.',
          'Historical 576-router replay results remain tied to their original narrow recordings.'])
    save(output/'protocol.json',protocol)
    return protocol


def verify(output):
    from experiments.v4_gazebo_pilot.assets import load_asset
    output=Path(output).resolve();p=read(output/'protocol.json')
    for name,digest in p['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError(f'Changed acquisition source: {name}')
    for name,digest in p['frozen_files'].items():
        if sha(output/name)!=digest:raise ValueError(f'Changed frozen layout: {name}')
    if sha(Path(p['asset_directory'])/'manifest.json')!=p['asset_manifest_sha256']:
        raise ValueError('Changed terrain asset manifest')
    verify_assets(Path(p['asset_directory']))
    if sha(p['configuration']['path'])!=p['configuration']['sha256']:
        raise ValueError('Changed acquisition configuration')
    for path,digest in p['external_mesh_manifest_sha256'].items():
        if sha(path)!=digest:raise ValueError(f'Changed target mesh manifest: {path}')
        load_asset(Path(path))
    for name,case in p['cases'].items():
        expected=f'{name}/scenario.json'
        candidate=Path(case['scenario'])
        if candidate.is_absolute() or '..' in candidate.parts or str(candidate)!=expected or expected not in p['frozen_files']:
            raise ValueError('Case scenario must reference its own frozen file')
        scenario=read(output/candidate)
        if Path(scenario['asset_directory']).resolve()!=Path(p['asset_directory']).resolve():
            raise ValueError('Case scenario must use the pinned terrain assets')
    return p


def audit(output):
    output=Path(output).resolve();protocol=verify(output);cases={}
    for name in protocol['cases']:
        recording=output/name/'recording'
        if not (recording/'recording.json').exists():
            cases[name]={'recorded':False,'full_mission':False};continue
        m=read(recording/'recording.json');v=read(recording/'physical_route_validation.json')
        scenario=read(output/name/'scenario.json');labels={o['label']:o['object_id'] for o in scenario['world']['objects']}
        counts={obj:0 for obj in labels.values()};peak={obj:0 for obj in labels.values()}
        for frame in m['frames']:
            if sha(recording/frame['file'])!=frame['sha256']:raise ValueError('Changed frame bytes')
            with np.load(recording/frame['file'],allow_pickle=False) as a:
                unique,n=np.unique(a['object_ids'],return_counts=True)
                for label,pixels in zip(unique,n):
                    if int(label) in labels:
                        obj=labels[int(label)];peak[obj]=max(peak[obj],int(pixels))
                        counts[obj]+=int(pixels>=4)
        cases[name]=dict(recorded=True,full_mission=m['provenance']['full_declared_mission'],
            physical_route_passed=v['passed'],frames=len(m['frames']),tracking_error_max_m=v['tracking_error_max_m'],
            minimum_terrain_clearance_m=v['minimum_observed_terrain_clearance_m'],
            targets_with_at_least_4_pixels=sum(n>0 for n in counts.values()),target_count=len(labels),
            visible_capture_counts=counts,peak_target_pixels=peak,
            visibility_scope='Evaluator-only geometry/sensor diagnostic; not perception-model recall or mission success.')
    result=dict(schema_version='wide_area_cohort_audit_v1',cases=cases,
                all_recorded_and_physically_valid=all(c.get('full_mission') and c.get('physical_route_passed') for c in cases.values()))
    save(output/'audit.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','record','audit'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--assets',type=Path,default=DEFAULT_ASSETS)
    parser.add_argument('--flight')
    args=parser.parse_args()
    if args.mode=='prepare':prepare(args.output,config=args.config,assets=args.assets)
    elif args.mode=='audit':print(json.dumps(audit(args.output),indent=2))
    else:
        from experiments.wide_area_missions.runtime import capture
        protocol=verify(args.output)
        names=[args.flight] if args.flight else list(protocol['cases'])
        for name in names:
            if name not in protocol['cases']:raise ValueError(f'Unknown flight: {name}')
            capture(args.output/protocol['cases'][name]['scenario'],args.output/name/'recording')


if __name__=='__main__':main()
