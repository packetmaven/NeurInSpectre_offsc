"""Generate nuScenes label map JSON for NeurInSpectre evaluation."""
import json
from nuscenes.nuscenes import NuScenes

# Use v1.0-mini (not trainval)
nusc = NuScenes(version='v1.0-mini', 
                dataroot='/Volumes/OWC Envoy/nuscenes_mini', 
                verbose=True)

# Map categories to indices
CATEGORY_MAP = {
    'vehicle.car': 0,
    'vehicle.truck': 1,
    'vehicle.bus.bendy': 2,
    'vehicle.bus.rigid': 2,
    'vehicle.construction': 3,
    'vehicle.motorcycle': 4,
    'vehicle.bicycle': 5,
    'human.pedestrian.adult': 6,
    'human.pedestrian.child': 6,
    'human.pedestrian.construction_worker': 6,
    'movable_object.barrier': 7,
    'movable_object.trafficcone': 8,
    'static_object.bicycle_rack': 9,
}

label_map = {}
for sample in nusc.sample:
    token = sample['token']
    anns = [nusc.get('sample_annotation', t) for t in sample['anns']]
    
    if not anns:
        label_map[token] = 0
        continue
    
    # Majority vote on category
    cats = [CATEGORY_MAP.get(a['category_name'], 0) for a in anns]
    label_map[token] = max(set(cats), key=cats.count)

# Save to external drive
output_path = '/Volumes/OWC Envoy/nuscenes_mini/label_map.json'
with open(output_path, 'w') as f:
    json.dump(label_map, f, indent=2)

print(f"✓ Generated labels for {len(label_map)} samples")
print(f"✓ Saved to: {output_path}")
