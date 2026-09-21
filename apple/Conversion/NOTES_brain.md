# Brain visualization

The native renderer keeps the TRIBE v2 prediction layout (left 10,242 vertices,
then right 10,242 vertices). Run `Conversion/.venv/bin/python
Conversion/export_brain_mesh.py` to export the fsaverage5 pial / inflated
surfaces, Yeo-7 network labels, and FreeSurfer sulcal depth. `make build` bundles
`Models/BrainMesh.bin`; `make verify-brain` checks the display mapping and writes
review images under `build/brain-review/` using a labeled synthetic test field.
Models and generated previews are gitignored; the exporter is the source of truth.

## What Meta does

Inspected September 10, 2026:

- [TRIBE v2 live demo](https://aidemos.atmeta.com/tribev2/) loads separate folded
  and inflated GLB hemispheres, including high-resolution versions. Its
  `BrainViewer-15466291.js` module uses a Three.js standard material with
  roughness around 0.75–0.9 and zero metalness. A shader reads per-face color
  atlases, maps each high-resolution face to three weighted low-resolution
  faces, and blends neighboring time frames. The geometry and activity data
  are separate. This is observed implementation, not a claim that the site
  publishes its complete preprocessing pipeline.
- [Published Python renderer](https://github.com/facebookresearch/tribev2/blob/main/tribev2/plotting/cortical_pv.py)
  blends activity RGBA over grayscale sulcal maps, uses smooth shading, and
  supports thresholds and robust percentile normalization. The shared mesh
  loader supports folded, half-inflated, and inflated surfaces.
- [Model card](https://huggingface.co/facebook/tribev2) specifies fsaverage5
  cortical predictions (~20k vertices), so higher display resolution does not
  imply additional predicted neural detail.
- [Research publication](https://ai.meta.com/research/publications/a-foundation-model-of-vision-audition-and-language-for-in-silico-neuroscience/)
  describes the multimodal brain-response model; it is not rendering documentation.

## Native implementation

The default is folded anatomy with its own area-weighted smooth normals.
Inflated anatomy uses independently computed normals. Previously inflated
positions were lit with pial normals, and an erroneous division by 360 in the
HSV conversion turned virtually every brain intensity red.

A sulcal grayscale underlay preserves the folds. One level of render-only
subdivision softens silhouettes without reordering prediction vertices.
Explicit key, fill, and ambient lighting plus ambient occlusion reveal depth.
The shared sRGB palette is converted to linear RGB for SceneKit vertex attributes
so the surface retains the intended saturation.
This is subdivision of our existing anatomy, not Meta's high-resolution GLB.
The SceneKit scene and camera persist while only vertex colors change. No
sampled color hash can miss activity confined to a small region.

Positive activity uses an ember → orange → gold → pale-yellow ramp. The display
scale is the 98th percentile of positive predictions across the clip (a bounded,
deterministic sample for long clips). Responses below 25% of that scale remain
gray; the overlay fades in through 50%. Negative predictions and the medial wall
remain anatomical gray. These are visual settings, not statistical significance
thresholds. Raw model values, ROI means, RMS summaries, and cached results are
unchanged. The legend explicitly identifies this as positive response.

Timeline rows retain their existing within-row normalization and use a shared
OKLab-interpolated charcoal → plum → ember → amber → cream ramp with monotonically
increasing luminance. The label makes the relative scaling explicit: colors do
not compare absolute values across rows or clips. Pixel-level scalar
interpolation removes hard TR boundaries; the brain likewise interpolates
adjacent predictions using the fractional, lag-adjusted playhead time. No-data
tracks are neutral instead of simulated activity. Seeking is measured within
the band itself, excluding the row labels and padding.

`BrainMesh.bin` version 2 adds one Float32 sulcal-depth value per vertex after
ROI labels. The Swift reader also accepts version 1, using a uniform gray
underlay. Re-export to get the full anatomical shading.
