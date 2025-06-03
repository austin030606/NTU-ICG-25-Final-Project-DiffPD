# generate_intermediate_meshes.py
import open3d as o3d

def count_faces(path):
    return sum(1 for line in open(path) if line.startswith("f "))

def subdivide_to_target(mesh, target_faces, method="midpoint"):
    """
    Subdivide mesh (in­place) until mesh.triangles >= target_faces.
    Uses either 'midpoint' or 'loop' subdivision.
    """
    subdiv_fn = {
        "midpoint": o3d.geometry.TriangleMesh.subdivide_midpoint,
        "loop":     o3d.geometry.TriangleMesh.subdivide_loop
    }[method]

    iters = 0
    while len(mesh.triangles) < target_faces:
        mesh = subdiv_fn(mesh, number_of_iterations=1)
        iters += 1
        # Safety: avoid infinite loops
        if iters > 10:
            print(f"  ⚠️  reached 10 subdivisions, still only {len(mesh.triangles)} faces")
            break
    # If we overshot (got more faces than target), we can optionally decimate a bit:
    if len(mesh.triangles) > target_faces:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
    return mesh

if __name__ == "__main__":
    low = "../../asset/mesh/armadillo_low_res.obj"
    high = "../../asset/mesh/armadillo_high_res.obj"

    n_low  = count_faces(low)
    n_high = count_faces(high)
    print(f"Low­res faces:  {n_low}")
    print(f"High­res faces: {n_high}")

    mid1 = int((2*n_low + n_high) / 3)
    mid2 = int((n_low + 2*n_high) / 3)
    print(f"Intermediate targets: {mid1}, {mid2}\n")

    # Read only the low­res mesh
    mesh_low = o3d.io.read_triangle_mesh(low)

    # Make mid1
    print(f"Generating mid1 ({mid1} faces) via subdivision…")
    m1 = subdivide_to_target(mesh_low, mid1, method="loop")
    o3d.io.write_triangle_mesh("../../asset/mesh/armadillo_mid1.obj", m1)
    print(f"  → got {len(m1.triangles)} faces\n")

    # For mid2, just start again from low
    print(f"Generating mid2 ({mid2} faces) via subdivision…")
    mesh_low = o3d.io.read_triangle_mesh(low)
    m2 = subdivide_to_target(mesh_low, mid2, method="loop")
    o3d.io.write_triangle_mesh("../../asset/mesh/armadillo_mid2.obj", m2)
    print(f"  → got {len(m2.triangles)} faces\n")

    print("Done generating mid1 and mid2")
