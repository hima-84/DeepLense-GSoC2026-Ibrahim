# Specific Test II — Agentic AI for Gravitational Lensing Simulation

**GSoC 2026 · ML4SCI · DeepLense (DEEPLENSE1)**
**Applicant:** Ibrahim Nagy Abd Elrazek Elsaid Emara · [github.com/hima-84](https://github.com/hima-84)

---

## Task

Build an agentic workflow using **Pydantic AI** that wraps the DeepLenseSim simulation
pipeline, accepting natural-language prompts and generating strong gravitational lensing
images with structured metadata. The agent must:

- Accept user prompts describing desired simulations in plain English
- Orchestrate calls to DeepLenseSim / lenstronomy
- Return generated images with structured, validated metadata
- Include a **human-in-the-loop (HITL)** component for parameter clarification
- Support at least two model configurations (Model_I and Model_II)
- Define Pydantic models for all simulation parameters

---

## Architecture

```
User Natural Language Prompt
           │
           ▼
┌─────────────────────────────────────────────┐
│         DeepLenseAgent  (Pydantic AI)        │
│         Backend: Google Gemini 2.0 Flash     │
│                                              │
│  1. parse_intent()      — NL → params        │
│  2. identify_missing()  — find ambiguities   │
│  3. HITL questions      — ask user           │
│  4. build_request()     — construct model    │
│  5. Pydantic validation — enforce physics    │
└──────────────┬──────────────────────────────┘
               │  6 registered tool functions
               ▼
       ┌───────────────┐
       │ Backend Router │
       └───────┬───────┘
        ┌──────┴──────┐
        ▼             ▼
  lenstronomy    Mock SIE Engine
  + pyHalo       (numpy + scipy)
  (full physics)
        │             │
        └──────┬──────┘
               ▼
   SimulationResult (Pydantic model)
   + 13-field ImageMetadata JSON per image
   + .npy arrays saved to disk
```

---

## Supported Configurations

### Model Configurations

| Config | Instrument | PSF FWHM | Pixel Scale | Exposure |
|--------|-----------|----------|-------------|---------|
| Model_I | Generic / SNR~25 | 0.10" | 0.10"/px | 1000s |
| Model_II | Euclid VIS | 0.18" | 0.10"/px | 565s |
| Model_III | HST ACS | 0.085" | 0.05"/px | 2028s |
| Model_IV | Euclid (alt.) | 0.18" | 0.10"/px | 565s |

### Dark Matter Substructure Classes

| Class | Physics | Lensing Signature | Model Config |
|-------|---------|-------------------|-------------|
| `no_substructure` | Smooth SIE+NFW | Symmetric ring | Any |
| `cdm_subhalo` | CDM point-mass subhalos | Flux-ratio anomalies | Model_I / III / IV |
| `axion_vortex` | FDM quantum vortex filaments | Astrometric perturbations | Model_II |

---

## Key Features

### 1. Natural Language Interface
```python
agent = DeepLenseConversation()
result = agent.chat("Generate 6 axion dark matter lensing images with Euclid settings")
```
The agent parses instrument shorthand ("Euclid", "HST"), substructure keywords ("CDM",
"axion", "vortex"), redshifts, image counts, and resolution from plain English.

### 2. Human-in-the-Loop (HITL) Clarification
When parameters are ambiguous, the agent asks targeted questions before running:
```
AGENT: I need a few clarifications before running the simulation:

  Q1. What dark matter substructure type would you like?
      (a) CDM subhalos — cold dark matter clumps  [Model_I/III/IV]
      (b) Axion/vortex — fuzzy dark matter patterns  [Model_II]
      (c) No substructure — smooth Einstein ring baseline

  Q2. How many images? (default: 5)

  Q3. Redshifts: z_lens=0.5, z_source=1.5 — keep defaults?
```
Fully-specified prompts skip this step entirely. Pass `answers=` to the second call:
```python
result = agent.chat("Generate lensing images", answers="(a) CDM, 8 images, defaults")
```

### 3. Physics Enforcement via Pydantic Validators
```python
@model_validator(mode='after')
def check_redshifts(self):
    # General Relativity: source must be behind the lens
    if self.source.z_source <= self.lens.z_lens:
        raise ValueError(
            f"z_source ({self.source.z_source}) must exceed "
            f"z_lens ({self.lens.z_lens}). Violation of GR."
        )
    return self
```
This fires **before any simulation backend is invoked**, preventing unphysical
ray-tracing regardless of what the LLM produces.

### 4. Dual-Backend Router
```python
def run_simulation(request: SimulationRequest):
    if LENSTRONOMY_AVAILABLE and DEEPLENSESIM_AVAILABLE:
        return _simulate_full(request), 'full'   # lenstronomy + pyHalo
    else:
        return _simulate_mock(request), 'mock'   # numpy + scipy SIE
```
The mock engine generates physically correct SIE Einstein ring morphologies, PSF
convolution, and CDM/axion substructure overlays using only standard libraries.
All agent logic, HITL flow, and metadata are **identical in both modes**.

### 5. 13-Field Metadata Schema
Every `.npy` image is bound to a structured JSON record:
```python
class ImageMetadata(BaseModel):
    image_id:          str    # e.g. "img_42_0"
    seed:              int    # RNG seed for exact reproducibility
    model_config_name: str    # Model_I / Model_II / ...
    substructure_type: str    # cdm_subhalo / axion_vortex / no_substructure
    z_lens:            float  # Lens redshift
    z_source:          float  # Source redshift
    einstein_radius:   float  # θ_E in arcsec
    pixel_scale:       float  # arcsec/pixel
    num_pixels:        int    # Image resolution
    peak_flux:         float  # Max pixel value
    mean_flux:         float  # Mean pixel value
    snr_estimate:      float  # Signal-to-noise ratio
    timestamp:         str    # ISO 8601 generation time
```
Any image in the dataset can be reproduced from its metadata entry alone.

---

## Six Registered Agent Tools

| Tool | Purpose |
|------|---------|
| `parse_simulation_request` | Extract params from NL prompt |
| `identify_missing_params` | Find ambiguities → ClarificationRequest |
| `validate_params` | Domain-level physics validation |
| `run_simulation` | Execute backend and return results |
| `describe_model_configs` | Return human-readable Model_I–IV descriptions |
| `list_substructure_types` | Return DM substructure options |

---

## Pydantic Model Hierarchy

```
SimulationRequest
├── model_config_name : DeepLenseModel     (MODEL_I – MODEL_IV)
├── substructure_type : SubstructureType   (no_substructure / cdm_subhalo / axion_vortex)
├── num_images        : int                (1 – 100)
├── lens              : LensParams         (z_lens, einstein_radius, axis_ratio, ...)
├── source            : SourceParams       (z_source, source_size, source_x, source_y)
├── instrument        : InstrumentParams   (pixel_scale, num_pixels, psf_fwhm, noise_level)
├── subhalo_params    : Optional[SubhaloParams]  (sigma_sub, m_low, m_high, ...)
├── axion_params      : Optional[AxionParams]    (log10_m_axion, vortex_density)
└── random_seed, output_dir, user_description

SimulationResult
├── request         : SimulationRequest
├── images          : list[np.ndarray]
├── metadata        : list[ImageMetadata]   ← 13 fields each
├── output_paths    : list[str]
├── simulation_mode : Literal['full', 'mock']
├── total_time_sec  : float
└── summary         : str
```

---

## How to Run

### 1. Install dependencies
```bash
pip install pydantic-ai[gemini] pydantic>=2.5 numpy matplotlib scipy

# Optional — for full lenstronomy physics backend:
pip install lenstronomy pyHalo astropy
```

### 2. Set your Gemini API key
```bash
export GEMINI_API_KEY=your_key_here
```
Or inside the notebook:
```python
import os
os.environ["GEMINI_API_KEY"] = "your_key_here"
```

### 3. Run the notebook

The notebook works in **two modes**:
- **Live mode** (with `GEMINI_API_KEY`): Full LLM-powered intent parsing and tool calling
- **Demo mode** (no key): Rule-based regex parser with identical agent logic and HITL flow

### 4. Example interactions

**Ambiguous prompt — agent asks questions:**
```python
agent = DeepLenseConversation()
result = agent.chat("Generate some strong lensing images for training a classifier")
# → Agent prints 3 clarifying questions and waits
```

**Second call with answers:**
```python
result_cdm = agent.chat(
    "Generate some strong lensing images for training a classifier",
    answers="(a) CDM subhalos, 8 images, use the defaults for redshift"
)
```

**Fully specified prompt — no questions asked:**
```python
result_axion = agent.chat(
    "Simulate 6 axion dark matter lensing images with z_lens=0.4, "
    "z_source=2.0, 150x150 pixels, include PSF with fwhm=0.1 arcsec",
    answers="6 images"
)
```

---

## Technical Challenges Solved

### 1. The Redshift Consistency Paradox
Early iterations allowed `z_source < z_lens` (physically impossible — source must be
*behind* the lens for lensing to occur). Fixed by `@model_validator` enforcing
$z_{\rm source} > z_{\rm lens}$ at the Pydantic schema level.

### 2. The 12-Field ValidationError Bug
Three co-occurring bugs caused `ImageMetadata(**m)` to raise
`ValidationError: 12 validation errors ... Field required`:
1. `image_id` declared `int` but built as `"img_42_0"` (string)
2. A stub cell silently redefined `tool_run_simulation` returning only 2 of 13 fields
   (Python last-definition-wins scoping in Jupyter)
3. A parallel stub stripped `SimulationResult` of 6 required fields

**Fix:** Corrected `image_id: str`, merged all stubs into one authoritative definition,
restored full `SimulationResult`. **Lesson:** always kernel-restart → run-all before
committing Jupyter notebooks.

### 3. Dependency Hell
`lenstronomy 1.13.6` + `pyHalo 1.4.3` = ~3.5 GB, fails on lightweight environments.
Fixed by the dual-backend router — the mock engine runs on any machine with only
`numpy` + `scipy`.

### 4. Axion-Only ModelConfig Constraint
`axion_vortex` substructure is physically only valid for Model_II (which has the
PSF and noise characteristics matching the vortex generation process).
Enforced by a second `@model_validator` that raises `ValueError` if
`AXION_VORTEX + MODEL_I` is requested.

---

## Generated Output Example

The agent generated a 4×3 grid of SNR≈1000 smooth lensing images from:
```
"Generate 12 smooth lensing images for curriculum learning"
```
Each image has a unique seed, correct SIE ellipticity variation, and arc-break
modulation — all at SNR≈1000 ("clean-room" data for curriculum learning baselines).

---

## Strategy Discussion

**Why Pydantic AI over raw function calling?**
Pydantic AI gives structured output by construction — every tool's input and return
type is a validated Pydantic model. For scientific simulation, silent type errors
(wrong redshift, wrong mass units) are as dangerous as code errors.

**Why tool decomposition?**
Separating parse / clarify / validate / simulate into 6 individual tools makes
each step independently testable and retriable. If `run_simulation` fails,
the agent can retry without re-running the LLM parsing step.

**Why HITL before execution?**
Lensing simulations with `lenstronomy` can take minutes per image. Spending that
compute on misspecified parameters wastes GPU credits and produces garbage training data.
The HITL step costs milliseconds and prevents that entirely.

---

## File Structure

```
Specific_Test_II/
├── DeepLense_GSoC2026_TestII_AgenticAI_FIXED.ipynb
├── requirements.txt
└── README.md  ← this file
```
