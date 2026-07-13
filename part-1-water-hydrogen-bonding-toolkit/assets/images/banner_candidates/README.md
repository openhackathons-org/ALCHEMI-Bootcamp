# Water IR banner candidates

Four art-only 2:1 candidates for the Part 1 notebook hero. Each final PNG is
2880×1440. The notebook places its real HTML title and badge over the image, so
the artwork contains no generated words or logos.

Shared direction:

- cinematic black abstract compute field, matching the existing ALCHEMI palette
  without depicting a material surface;
- subject weighted to the right, with a dark left text-safe area;
- red oxygen, white hydrogen, and restrained NVIDIA-green accents;
- chemically plausible bent water molecules and hydrogen bonds;
- generated with the built-in image generator, then resized once with Lanczos
  filtering from the generated 1774×887 raster to the repository's 2880×1440
  banner convention.

Candidates:

1. `water-ir-v2-01-hydrogen-bond-landscape.png` — one water dimer and a clear
   donor-H···acceptor-O hydrogen bond.
2. `water-ir-v2-02-batched-water-systems.png` — four independent water systems
   in parallel green compute lanes.
3. `water-ir-v2-03-composed-potential.png` — a water dimer surrounded by
   restrained field and dispersion motifs.
4. `water-ir-v2-04-trajectory-to-spectrum.png` — a water dimer, motion trails,
   and a vibrational-signal waveform motif. This is the notebook default.

Prompt concepts used with the existing Part 1 and Part 2 banners as visual
references:

- **Hydrogen-bond landscape:** one scientifically plausible suspended water
  dimer; the dashed green bond must run from donor hydrogen to acceptor oxygen;
  leave the left 58% quiet and dark.
- **Batched systems:** exactly four separate bent water molecules, one per
  parallel green compute lane; no bonds between lanes; preserve the left
  text-safe field.
- **Composed potential:** one plausible water dimer with a restrained cyan
  electrostatic field and green dispersion ripples; use abstract effects, not
  extra atoms or false bonds.
- **Trajectory to spectrum:** one plausible water dimer with subtle motion
  trails beside a green vibrational-signal motif; retain two complete bent
  waters, a valid donor-H···acceptor-O bond, and an uncluttered left text area.

Every prompt also required a 2:1 cinematic scientific render, red oxygen and
white hydrogen, no labels, no typography, no logo, no watermark, and no extra
atoms. A final image-edit pass explicitly removed every atom-resolved substrate,
lattice, and adsorption cue. The dark contours are a computational-space
metaphor; this notebook models isolated, finite, nonperiodic systems.

The built-in generator did not expose a public model identifier or seed. The
output hashes below are therefore the definitive asset identities.

SHA-256:

```text
b00294544bf0ac8ee11a5d58af765ae8daf3d7b6c7b3aaa80752cb7f63169041  water-ir-v2-01-hydrogen-bond-landscape.png
2adfbfa4b1ea0805ff3ad233fae3a6f6c1aa9eaa257681de33c4ceb38a7451ef  water-ir-v2-02-batched-water-systems.png
b430f4141f9c6d80b7ea88f96b568253a57d076bbdbc5cba36fc549aa2ec752f  water-ir-v2-03-composed-potential.png
d80b61940272bf9f170c3b7229023e31c40ac89975a55c810a5ec9c648aece0f  water-ir-v2-04-trajectory-to-spectrum.png
```

All four require a final human visual review in the target notebook theme before
publication. The generated assets are original tutorial artwork; no third-party
image is redistributed.
