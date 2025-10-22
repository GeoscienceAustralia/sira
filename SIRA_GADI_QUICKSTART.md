# SIRA on Gadi - Complete Guide

## Quick Start

1. **Edit the PBS script** `sira-on-gadi.pbs`:
   ```bash
   # Modify these lines for your setup:
   ASSET="SS_Cunderdin"                                    # Your model name
   CODE_DIR="/scratch/y57/mr3457/code/sira"                # SIRA code location
   MODELS_DIR="/scratch/y57/mr3457/code/_SYSTEM_MODELS/epn" # Models directory
   ```

2. **Submit the job**:
   ```bash
   qsub sira-on-gadi.pbs
   ```

3. **Monitor progress**:
   ```bash
   qstat -u $USER                              # Check job status
   tail -f sira-ramsey.o*                     # Watch live output (use actual job ID)
   tail -f <model_dir>/output/log.txt         # Watch SIRA execution log
   tail -f logs/monitoring.log                # Watch resource usage
   ```

## Pre-Job Testing (Important!)

Before submitting your PBS job, test your environment setup on a login node:

```bash
# 1. Load modules and activate venv
module load python3/3.11.7
module load openmpi/4.1.4
source "$HOME/venv/sira-env-mpi/bin/activate"

# 2. Test mpi4py installation
python -c "
import mpi4py.MPI as MPI
print(f'[OK] mpi4py: {mpi4py.__file__}')
print(f'[OK] MPI version: {MPI.Get_version()}')
print(f'[OK] MPI library: {MPI.Get_library_version()}')
"

# 3. Test SIRA imports
python -c "
import sira
from sira.simulation import calculate_response
from sira.__main__ import safe_mpi_import
print('[OK] SIRA imports working')

MPI, comm, rank, size = safe_mpi_import()
print(f'[OK] SIRA MPI detection: {MPI is not None}')
"

# 4. Test small MPI run (optional)
echo "Testing MPI execution..."
mpirun -np 2 python -c "
from mpi4py import MPI
comm = MPI.COMM_WORLD
print(f'Rank {comm.Get_rank()} of {comm.Get_size()}')
"
```

**Expected output:**
- [OK] All imports should work without errors
- [OK] MPI version should show OpenMPI details
- [OK] Small MPI test should show "Rank 0 of 2" and "Rank 1 of 2"

If any test fails, fix the environment before submitting jobs.

## Key Features

- [OK] **Automatic OUTPUT_DIR detection** - Reads from your config file  
- [OK] **MPI with multiprocessing fallback** - Maximum reliability  
- [OK] **JobFS streaming** - Fast I/O and memory efficiency  
- [OK] **Resource monitoring** - Memory and disk usage tracking  
- [OK] **Comprehensive verification** - Checks outputs and streaming files  
- [OK] **Error resilience** - Continues processing despite individual failures  
- [OK] **Organized logging** - All logs stored in `logs/` directory  

## Resources Configured

- **CPUs**: 96 cores (adjust `-l ncpus=` as needed)
- **Memory**: 2.81TB (adjust `-l mem=` as needed)
- **JobFS**: 400GB for streaming
- **Walltime**: 12 hours
- **Queue**: hugemem

## Expected Performance

- **Target**: ~0.4 seconds per hazard (achieved in recent runs)
- **Memory usage**: ~250GB actual (well under 2.81TB limit)
- **I/O**: All streaming to fast JobFS storage
- **MPI vs Multiprocessing**: ~6x faster with MPI (10min vs 1h+ for large models)

## Performance Optimisations Included

### Environment Variables (All Set Automatically)
```bash
# Thread limiting (prevents CPU thrashing)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Logging optimisation
export SIRA_LOG_LEVEL=WARNING    # Reduce verbosity
export SIRA_QUIET_MODE=1         # Suppress verbose progress
export PYTHONUNBUFFERED=0        # Enable output buffering

# Performance tuning
export SIRA_CHUNKS_PER_SLOT=1            # Coarser chunks for MPI
export SIRA_STREAM_COMPRESSION=lz4       # Fast compression
export SIRA_STREAM_ROW_GROUP=262144      # Larger row groups

# JobFS streaming (critical for performance)
export SIRA_STREAM_DIR="$PBS_JOBFS/sira_stream"
```

### Why These Optimisations Matter
- **Thread limiting**: Prevents CPU oversubscription and thrashing
- **JobFS streaming**: 10-50x faster I/O than network storage
- **LZ4 compression**: 2-3x faster than snappy compression
- **Coarse chunking**: Reduces MPI communication overhead
- **Memory bounds**: Streaming keeps memory usage constant regardless of dataset size

### Automatic Data Management

SIRA generates output files automatically from streaming data with user control:

- [OK] **JobFS Streaming**: Fast results written to `/jobfs/$PBS_JOBID/sira_stream`  
- [OK] **Complete Reconstruction**: All response data reconstructed from streaming artifacts  
- [OK] **Selective Generation**: Control large file generation via environment variables  
- [OK] **Performance Optimised**: Reconstruction happens after simulation completes  

**Core output files (always generated):**
1. `system_response.csv` (with recovery analysis)
2. `system_output_vs_hazard_intensity.csv` 
3. `risk_summary_statistics.csv`
4. `risk_summary_statistics.json`

**Optional output files (controlled by environment variables):**
5. `component_response.csv` - controlled by `SIRA_SAVE_COMPONENT_RESPONSE` (default: disabled for large datasets)
6. `comptype_response.csv` - controlled by `SIRA_SAVE_COMPTYPE_RESPONSE` (default: enabled)

**Environment Variable Control:**
- `SIRA_SAVE_COMPONENT_RESPONSE=1` - Enable component_response.csv generation
- `SIRA_SAVE_COMPTYPE_RESPONSE=0` - Disable comptype_response.csv generation

## Troubleshooting

### If the job fails:

1. **Check the PBS log**: `cat sira-ramsey.o[JOBID]`
2. **Check SIRA execution log**: `cat <model_dir>/output/log.txt`
3. **Check for errors**: `grep -i error <model_dir>/output/log.txt`
4. **Monitor resource usage**: `cat logs/monitoring.log`
5. **Verify environment**: The script tests MPI availability before running
6. **Check paths**: Ensure `ASSET`, `CODE_DIR`, and `MODELS_DIR` are correct

### Common Issues and Solutions

#### Issue: "Streaming manifest not found" Error
**Symptoms:**
```
WARNING [sira.infrastructure_response:475] Streaming manifest not found at /path/to/output_dir/stream/manifest.jsonl. Skipping consolidation.
```

**Solution:** This is fixed in the latest code. The PBS script now dynamically reads OUTPUT_DIR from your config file instead of assuming hardcoded paths.

**Manual Recovery (if needed):**
```bash
# Determine your actual OUTPUT_DIR from config
CONFIG_FILE=$(find /path/to/your/model -name "config*.json" | head -1)
CONFIG_OUTPUT_DIR=$(python -c "
import json
with open('$CONFIG_FILE') as f: 
    config = json.load(f)
    output_dir = config.get('OUTPUT_DIR', './output')
    if output_dir.startswith('./'): output_dir = output_dir[2:]
    print(output_dir)
")

# Check JobFS location
ls -la $PBS_JOBFS/sira_stream/

# Copy results from JobFS before job ends
cp -r $PBS_JOBFS/sira_stream /path/to/model/$CONFIG_OUTPUT_DIR/stream_from_jobfs
```

#### Issue: Partial Results Only
**Symptoms:** Job completes successfully but manifest missing

**Recovery:**
```bash
cd $PBS_JOBFS/sira_stream
python -c "
import json
from pathlib import Path

# Consolidate orphaned rank manifests
rank_manifests = list(Path('.').glob('manifest_rank_*.jsonl'))
if rank_manifests:
    with open('manifest.jsonl', 'w') as main_mf:
        for rank_manifest in sorted(rank_manifests):
            with open(rank_manifest, 'r') as rank_mf:
                for line in rank_mf:
                    main_mf.write(line)
            rank_manifest.unlink()
    print(f'Consolidated {len(rank_manifests)} rank manifests')
"
```

### Model Diagnostics

If performance is still slow, check your model characteristics:
```bash
python -c "
import json
with open('input/config*.json') as f: config = json.load(f)
with open('input/model*.json') as f: model = json.load(f)
print(f'Samples: {config.get(\"SIMULATION_PARAMETERS\", {}).get(\"NUM_SAMPLES\", \"N/A\")}')
print(f'Components: {len(model.get(\"components\", {}))}')
print(f'Hazard events: {len(config.get(\"HAZARD_SCENARIOS\", []))}')
"
```

## What the Script Does

1. **Environment setup**: Loads modules, activates venv
2. **Logging organization**: Creates `logs/` directory for all log files
3. **Performance tuning**: Sets optimal environment variables automatically
4. **MPI execution**: Runs with MPI backend, falls back to multiprocessing if needed
5. **Output verification**: Checks both traditional and streaming outputs using dynamic OUTPUT_DIR
6. **Resource reporting**: Shows memory and disk usage throughout execution
7. **Error handling**: Continues processing despite individual hazard failures

## Resource Scaling Guidelines

### For Different Model Sizes:

**Small models (< 100 components, < 10k hazards):**
```bash
#PBS -l ncpus=24
#PBS -l mem=500GB
```

**Medium models (100-1000 components, 10k-100k hazards):**
```bash
#PBS -l ncpus=48
#PBS -l mem=1.5TB
```

**Large models (> 1000 components, > 100k hazards):**
```bash
#PBS -l ncpus=96
#PBS -l mem=2.81TB
```

### Memory vs Performance Trade-offs:
- **More cores**: Faster execution, but diminishing returns above 96 cores
- **More memory per core**: Better for complex models with large intermediate results
- **Streaming**: Essential for memory bounds, ~10-20% performance overhead but prevents OOM

## Architecture Overview

The PBS script implements a sophisticated parallel processing pipeline:

1. **MPI Priority**: Uses MPI for maximum performance on HPC systems
2. **Streaming Architecture**: Results written incrementally to JobFS (fast local storage)
3. **Manifest System**: Tracks all output files with rank-specific manifests to prevent race conditions
4. **Consolidation**: Combines individual rank outputs into final results
5. **Error Resilience**: Individual hazard failures don't crash entire job

This approach has proven to achieve ~0.4 seconds per hazard on large models, representing a significant improvement over previous Dask-based approaches that suffered from serialisation overhead and worker management issues.

The script handles all the complexity automatically - just edit the three path variables and submit!