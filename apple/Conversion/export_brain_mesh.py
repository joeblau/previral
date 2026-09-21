"""Export the fsaverage5 cortical surface + Yeo-7 ROI labels to Models/BrainMesh.bin.

Vertex ordering (verified against the TRIBE v2 reference):

- ``tribev2/utils_fmri.py`` ``TribeSurfaceProjector.apply`` (volumetric path) loops
  ``for hemi in ("left", "right")`` and ``np.vstack``s ``nilearn.surface.vol_to_surf``
  results sampled on ``fetch_surf_fsaverage("fsaverage5")`` pial meshes -> output
  vertices are [left(10242) | right(10242)] in nilearn's fsaverage5 vertex order.
- The surface path of the same method takes ``data[:10242]`` as left and
  ``data[n:n+10242]`` as right, and ``tribev2/plotting/base.py`` splits predictions
  the same way (``left = data[:len//2]``) and maps them onto the left/right
  fsaverage5 meshes. So using nilearn's fsaverage5 surfaces in native order matches
  TRIBE's 20,484-vertex output order by construction.
- The Yeo annots are authored by the Yeo lab for the canonical FreeSurfer
  fsaverage5 mesh; nilearn's fsaverage5 is a byte-identical redistribution of that
  mesh. Registration between the annot labels and nilearn's geometry is verified
  numerically below (medial-wall vertices must sit near the midline, and must
  overlap nilearn's own Destrieux medial wall).

ROI source: Yeo 2011 7-network parcellation on fsaverage5, authoritative .annot
files from the Yeo lab's CBIG repo
(``stable_projects/brain_parcellation/Yeo2011_fcMRI_clustering/1000subjects_reference/
Yeo_JNeurophysiol11_SplitLabels/fsaverage5/label/{lh,rh}.Yeo2011_7Networks_N1000.annot``),
cached in Conversion/cache/yeo2011/. 0 = unassigned (medial wall / corpus
callosum), 1..7 = networks in canonical order:
1 Visual, 2 Somatomotor, 3 DorsalAttention, 4 VentralAttention, 5 Limbic,
6 Frontoparietal, 7 Default.

Binary layout (little-endian):
    char magic[4] = "BMSH"
    uint32 version = 2
    uint32 nVertices (= 20484)
    uint32 nFaces
    uint32 nROIs
    float32 positionsPial[nVertices*3]
    float32 positionsInflated[nVertices*3]
    uint32 faces[nFaces*3]
    uint8 roiLabel[nVertices]
    float32 sulcalDepth[nVertices]
    nROIs x (uint16 byteLength + UTF-8 name bytes)
"""

import struct
from pathlib import Path

import nibabel
import numpy as np

CONVERSION_DIR = Path(__file__).resolve().parent
OUT_PATH = CONVERSION_DIR.parent / "Models" / "BrainMesh.bin"
YEO_CACHE = CONVERSION_DIR / "cache" / "yeo2011"
YEO_BASE_URL = (
    "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/"
    "brain_parcellation/Yeo2011_fcMRI_clustering/1000subjects_reference/"
    "Yeo_JNeurophysiol11_SplitLabels/fsaverage5/label"
)
YEO_NAMES = [
    "Visual",
    "Somatomotor",
    "DorsalAttention",
    "VentralAttention",
    "Limbic",
    "Frontoparietal",
    "Default",
]
VERTICES_PER_HEMI = 10242

MAGIC = b"BMSH"
VERSION = 2


def _mesh_coords_faces(mesh) -> tuple[np.ndarray, np.ndarray]:
    """Handle both nilearn InMemoryMesh objects and file paths."""
    if hasattr(mesh, "coordinates"):
        return np.asarray(mesh.coordinates), np.asarray(mesh.faces)
    darrays = nibabel.load(mesh).darrays
    return darrays[0].data, darrays[1].data


def fetch_surfaces():
    from nilearn import datasets

    fs = datasets.fetch_surf_fsaverage("fsaverage5")
    surfaces = {}
    for kind in ("pial", "infl"):
        for hemi in ("left", "right"):
            coords, faces = _mesh_coords_faces(fs[f"{kind}_{hemi}"])
            surfaces[(kind, hemi)] = (
                coords.astype(np.float32),
                faces.astype(np.uint32),
            )
    return surfaces


def fetch_yeo_annot(hemi: str) -> Path:
    """Download the Yeo-7 fsaverage5 .annot for one hemisphere (cached)."""
    YEO_CACHE.mkdir(parents=True, exist_ok=True)
    path = YEO_CACHE / f"{hemi}.Yeo2011_7Networks_N1000.annot"
    if not path.exists():
        import urllib.request

        urllib.request.urlretrieve(f"{YEO_BASE_URL}/{path.name}", path)
    return path


def yeo_labels(hemi: str) -> np.ndarray:
    """Yeo-7 network index per vertex (uint8, 0 = unassigned) for one hemisphere."""
    labels, _ctab, names = nibabel.freesurfer.read_annot(fetch_yeo_annot(hemi))
    assert labels.shape[0] == VERTICES_PER_HEMI, (
        f"{hemi} annot has {labels.shape[0]} vertices, expected {VERTICES_PER_HEMI}"
    )
    out = np.zeros(labels.shape[0], dtype=np.uint8)
    for i, name in enumerate(names):
        name = name.decode() if isinstance(name, bytes) else name
        if name.startswith("7Networks_"):
            network = int(name.removeprefix("7Networks_"))
            assert 1 <= network <= 7, f"unexpected Yeo label {name!r}"
            out[labels == i] = network
        # "unknown" / "corpuscallosum" stay 0 (unassigned / medial wall)
    return out


def verify_label_registration(coords: np.ndarray, labels: np.ndarray, hemi: str) -> None:
    """Numeric check that annot labels are registered to the nilearn mesh vertices.

    Medial-wall (unassigned) vertices must hug the midline (|x| near 0), while
    labeled cortical vertices sit laterally. If the annot vertex order did not
    match the mesh, |x| distributions would be indistinguishable.
    """
    abs_x = np.abs(coords[:, 0])
    medial = abs_x[labels == 0]
    cortical = abs_x[labels > 0]
    print(
        f"  {hemi}: median |x| medial-wall {np.median(medial):.2f} mm vs "
        f"cortical {np.median(cortical):.2f} mm"
    )
    assert np.median(medial) < 15.0, "medial-wall labels not near midline"
    assert np.median(cortical) > 25.0, "cortical labels not lateral"
    # Independent cross-check: Yeo unassigned should overlap nilearn's Destrieux
    # medial wall (destrieux map uses label index of 'Unknown'/'Medial_wall').
    from nilearn import datasets

    dest = datasets.fetch_atlas_surf_destrieux()
    dest_map = dest[f"map_{hemi}"]
    dest_names = [n.decode() if isinstance(n, bytes) else n for n in dest.labels]
    wall_ids = [
        i for i, n in enumerate(dest_names) if n in ("Unknown", "Medial_wall")
    ]
    dest_wall = np.isin(dest_map, wall_ids)
    overlap = (labels == 0) & dest_wall
    frac_yeo = overlap.sum() / max((labels == 0).sum(), 1)
    print(f"  {hemi}: {frac_yeo:.1%} of Yeo unassigned vertices are Destrieux medial wall")
    assert frac_yeo > 0.9, "Yeo/Destrieux medial-wall mismatch: ordering broken?"


def write_brain_mesh(path: Path, pial, inflated, faces, roi, sulcal_depth, names) -> None:
    n_vertices, n_faces = pial.shape[0], faces.shape[0]
    with open(path, "wb") as f:
        f.write(struct.pack("<4s4I", MAGIC, VERSION, n_vertices, n_faces, len(names)))
        f.write(pial.astype("<f4").tobytes())
        f.write(inflated.astype("<f4").tobytes())
        f.write(faces.astype("<u4").tobytes())
        f.write(roi.astype(np.uint8).tobytes())
        f.write(sulcal_depth.astype("<f4").tobytes())
        for name in names:
            encoded = name.encode("utf-8")
            f.write(struct.pack("<H", len(encoded)))
            f.write(encoded)


def load_brain_mesh(path: str | Path) -> dict:
    """Round-trip reader for the BrainMesh.bin format (mirror of the Swift loader)."""
    blob = Path(path).read_bytes()
    magic, version, n_vertices, n_faces, n_rois = struct.unpack_from("<4s4I", blob, 0)
    assert magic == MAGIC, f"bad magic {magic!r}"
    assert version == VERSION, f"unsupported version {version}"
    offset = struct.calcsize("<4s4I")

    def take(dtype, count):
        nonlocal offset
        arr = np.frombuffer(blob, dtype=np.dtype(dtype).newbyteorder("<"),
                            count=count, offset=offset)
        offset += arr.nbytes
        return arr

    pial = take("f4", n_vertices * 3).reshape(n_vertices, 3)
    inflated = take("f4", n_vertices * 3).reshape(n_vertices, 3)
    faces = take("u4", n_faces * 3).reshape(n_faces, 3)
    roi = take("u1", n_vertices)
    sulcal_depth = take("f4", n_vertices)
    names = []
    for _ in range(n_rois):
        (length,) = struct.unpack_from("<H", blob, offset)
        offset += 2
        names.append(blob[offset : offset + length].decode("utf-8"))
        offset += length
    assert offset == len(blob), f"trailing bytes: {len(blob) - offset}"
    return {
        "positionsPial": pial,
        "positionsInflated": inflated,
        "faces": faces,
        "roiLabel": roi,
        "roiNames": names,
        "sulcalDepth": sulcal_depth,
    }


def main() -> None:
    print("fetching fsaverage5 surfaces via nilearn ...")
    surfaces = fetch_surfaces()
    pial_l, faces_l = surfaces[("pial", "left")]
    pial_r, faces_r = surfaces[("pial", "right")]
    infl_l, _ = surfaces[("infl", "left")]
    infl_r, _ = surfaces[("infl", "right")]

    for kind in ("pial", "infl"):
        for hemi in ("left", "right"):
            coords, faces = surfaces[(kind, hemi)]
            assert coords.shape == (VERTICES_PER_HEMI, 3), (kind, hemi, coords.shape)
    print(f"vertices per hemisphere: {pial_l.shape[0]} (expected {VERTICES_PER_HEMI})")

    print("verifying Yeo-7 label registration against the nilearn meshes ...")
    labels_l = yeo_labels("lh")
    labels_r = yeo_labels("rh")
    verify_label_registration(pial_l, labels_l, "left")
    verify_label_registration(pial_r, labels_r, "right")

    # Concatenate [left | right] — matches TribeSurfaceProjector.apply ordering.
    pial = np.concatenate([pial_l, pial_r], axis=0)
    inflated = np.concatenate([infl_l, infl_r], axis=0)
    faces = np.concatenate([faces_l, faces_r + VERTICES_PER_HEMI], axis=0)
    roi = np.concatenate([labels_l, labels_r], axis=0)
    from nilearn import datasets, surface

    fs = datasets.fetch_surf_fsaverage("fsaverage5")
    sulcal_depth = np.concatenate([
        surface.load_surf_data(fs[f"sulc_{hemi}"]) for hemi in ("left", "right")
    ]).astype(np.float32)
    assert sulcal_depth.shape == (2 * VERTICES_PER_HEMI,)
    assert np.isfinite(sulcal_depth).all()
    n_vertices, n_faces = pial.shape[0], faces.shape[0]
    assert n_vertices == 2 * VERTICES_PER_HEMI
    assert faces.min() >= 0 and faces.max() < n_vertices, "faces out of range"
    assert roi.max() <= len(YEO_NAMES), "roi label exceeds nROIs"

    print("\nROI vertex counts:")
    print(f"  {'0 Unassigned':<20} {(roi == 0).sum():>6}")
    for i, name in enumerate(YEO_NAMES, start=1):
        print(f"  {i} {name:<19} {(roi == i).sum():>6}")

    for label, arr in (("pial", pial), ("inflated", inflated)):
        lo, hi = arr.min(axis=0), arr.max(axis=0)
        centroid = arr.mean(axis=0)
        print(
            f"\n{label}: min {np.round(lo, 1)} max {np.round(hi, 1)} "
            f"centroid {np.round(centroid, 1)} (mm, fsaverage RAS: +x right, "
            f"+y anterior, +z superior)"
        )
        for hemi, sl in (("left", slice(0, VERTICES_PER_HEMI)),
                         ("right", slice(VERTICES_PER_HEMI, None))):
            c = arr[sl].mean(axis=0)
            print(f"  {hemi} centroid {np.round(c, 1)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_brain_mesh(OUT_PATH, pial, inflated, faces, roi, sulcal_depth, YEO_NAMES)
    size = OUT_PATH.stat().st_size
    print(f"\nwrote {OUT_PATH} ({size / 1e6:.2f} MB)")

    mesh = load_brain_mesh(OUT_PATH)
    assert np.array_equal(mesh["positionsPial"], pial)
    assert np.array_equal(mesh["positionsInflated"], inflated)
    assert np.array_equal(mesh["faces"], faces)
    assert np.array_equal(mesh["roiLabel"], roi)
    assert mesh["roiNames"] == YEO_NAMES
    assert np.array_equal(mesh["sulcalDepth"], sulcal_depth)
    print("round-trip read-back: all arrays identical")


if __name__ == "__main__":
    main()
