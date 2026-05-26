# Audio Concatenation Projects

Project configuration files for audio sample concatenation. Each project references external sample locations (samples are not stored in this repo due to size/copyright).

## Project Status

| Project | Location | Status | Notes |
|---------|----------|--------|-------|
| **sarabande** | `/Users/soney/Music/sarabande/` | Ready | Satie Sarabande, 98 samples |
| **wildrose** | `/Users/soney/Music/sample-laboratory/wildrose/` | Ready | MacDowell To a Wild Rose |
| **mayer** | `/Users/soney/Music/sample-laboratory/mayer/` | Ready | John Mayer samples |
| **ilyaf** | `/Users/soney/Music/sample-laboratory/ilyaf/` | Ready | I Love You Always Forever |
| **rhcp** | `/Users/soney/Music/sample-laboratory/rhcp/` | Ready | Red Hot Chili Peppers |
| **mamamia** | `/Users/soney/Music/sample-laboratory/mamamia/` | Ready | ABBA Mamma Mia |
| **whiteroom** | `/Users/soney/Music/sample-laboratory/whiteroom/` | Ready | Cream White Room |
| **takemitsu** | `/Users/soney/Music/sample-laboratory/takemitsu/` | Ready | Takemitsu percussion |
| **feldman_piano** | `/Users/soney/Music/sample-laboratory/feldman_piano/` | Ready | Morton Feldman piano |

## Usage

```bash
# List available projects
python scripts/concatenate_audio.py list-projects

# Run a project
python scripts/concatenate_audio.py run sarabande -n 50

# Run with custom output
python scripts/concatenate_audio.py run wildrose -n 30 -o output/wildrose_test.wav
```

## Config Format

Each `.yaml` file defines:
- `name`: Project identifier
- `path`: Absolute path to project directory (with scales_dir, samples, etc.)
- `sample_prefix`: Naming pattern for samples (e.g., "wildrose", "satie_sarabande")
- `sample_pattern`: How sample numbers are formatted (e.g., "{prefix}_{num}_" or "{prefix}_{num:04d}")
- `num_samples`: Total number of samples/measures
- `transpositions`: Range of transpositions used
