#!/usr/bin/env python3
"""Convert OBJ to COLLADA DAE with embedded colors - no external deps needed."""
import os
import numpy as np

OBJ_DIR = "/home/praveensingh/VisionAidedScript/src/vehicle_simulator/world/Toolkit/OBJ"

# Tool colors as RGBA floats (0-1) matching the reference image
TOOL_COLORS = {
    "Hammer.obj":       (0.05, 0.15, 0.85, 1.0),   # Royal blue
    "Pliers.obj":       (0.05, 0.15, 0.85, 1.0),   # Deep blue
    "Saw.obj":          (0.95, 0.68, 0.05, 1.0),    # Orange/gold
    "Screw driver.obj": (1.0,  0.55, 0.0,  1.0),    # Bright orange
    "Spanner.obj":      (0.85, 0.55, 0.35, 1.0),    # Copper
    "monkey ranch.obj": (0.05, 0.15, 0.85, 1.0),    # Royal blue
    "screw.obj":        (0.9,  0.72, 0.1,  1.0),    # Gold
}

def parse_obj(filepath):
    """Parse OBJ file, return vertices and face indices."""
    vertices = []
    normals = []
    faces = []
    
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v':
                vertices.append([float(x) for x in parts[1:4]])
            elif parts[0] == 'vn':
                normals.append([float(x) for x in parts[1:4]])
            elif parts[0] == 'f':
                face_verts = []
                for p in parts[1:]:
                    # Handle f v, f v/vt, f v/vt/vn, f v//vn
                    idx = int(p.split('/')[0]) - 1  # OBJ is 1-indexed
                    face_verts.append(idx)
                # Triangulate if quad or polygon
                for i in range(1, len(face_verts) - 1):
                    faces.append([face_verts[0], face_verts[i], face_verts[i+1]])
    
    return np.array(vertices), np.array(faces)


def write_dae(filepath, vertices, faces, color):
    """Write a COLLADA DAE file with embedded color material."""
    r, g, b, a = color
    nv = len(vertices)
    nf = len(faces)
    
    # Build vertex position string
    vert_str = ' '.join(f'{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}' for v in vertices)
    
    # Build face index string (triangle indices)
    face_str = ''
    for f in faces:
        face_str += f'{f[0]} {f[1]} {f[2]} '
    
    dae_content = f'''<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset>
    <created>2026-02-27</created>
    <modified>2026-02-27</modified>
    <unit name="meter" meter="1"/>
    <up_axis>Z_UP</up_axis>
  </asset>
  <library_effects>
    <effect id="material0-effect">
      <profile_COMMON>
        <technique sid="common">
          <phong>
            <ambient><color>{r*0.3:.4f} {g*0.3:.4f} {b*0.3:.4f} {a:.4f}</color></ambient>
            <diffuse><color>{r:.4f} {g:.4f} {b:.4f} {a:.4f}</color></diffuse>
            <specular><color>0.5 0.5 0.5 1.0</color></specular>
            <shininess><float>50.0</float></shininess>
          </phong>
        </technique>
      </profile_COMMON>
    </effect>
  </library_effects>
  <library_materials>
    <material id="material0" name="material0">
      <instance_effect url="#material0-effect"/>
    </material>
  </library_materials>
  <library_geometries>
    <geometry id="mesh0" name="mesh0">
      <mesh>
        <source id="mesh0-positions">
          <float_array id="mesh0-positions-array" count="{nv*3}">{vert_str}</float_array>
          <technique_common>
            <accessor source="#mesh0-positions-array" count="{nv}" stride="3">
              <param name="X" type="float"/>
              <param name="Y" type="float"/>
              <param name="Z" type="float"/>
            </accessor>
          </technique_common>
        </source>
        <vertices id="mesh0-vertices">
          <input semantic="POSITION" source="#mesh0-positions"/>
        </vertices>
        <triangles material="material0" count="{nf}">
          <input semantic="VERTEX" source="#mesh0-vertices" offset="0"/>
          <p>{face_str}</p>
        </triangles>
      </mesh>
    </geometry>
  </library_geometries>
  <library_visual_scenes>
    <visual_scene id="Scene" name="Scene">
      <node id="Node" name="Node" type="NODE">
        <instance_geometry url="#mesh0">
          <bind_material>
            <technique_common>
              <instance_material symbol="material0" target="#material0"/>
            </technique_common>
          </bind_material>
        </instance_geometry>
      </node>
    </visual_scene>
  </library_visual_scenes>
  <scene>
    <instance_visual_scene url="#Scene"/>
  </scene>
</COLLADA>'''
    
    with open(filepath, 'w') as f:
        f.write(dae_content)


# Convert each tool
for obj_name, color in TOOL_COLORS.items():
    obj_path = os.path.join(OBJ_DIR, obj_name)
    dae_path = os.path.join(OBJ_DIR, obj_name.replace('.obj', '.dae'))
    
    if not os.path.exists(obj_path):
        print(f"SKIP: {obj_name} not found")
        continue
    
    print(f"Converting {obj_name} -> {os.path.basename(dae_path)}...")
    try:
        verts, faces = parse_obj(obj_path)
        write_dae(dae_path, verts, faces, color)
        print(f"  OK: {len(verts)} vertices, {len(faces)} triangles")
    except Exception as e:
        print(f"  FAILED: {e}")

print("\nAll done!")
