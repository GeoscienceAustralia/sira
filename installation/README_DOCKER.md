# SIRA Docker Setup

This directory contains Docker configuration for running SIRA in containers.

## Quick Start

### 1. Prepare Directory Structure

Create data directories **outside** the SIRA code repository. The default configuration assumes they are at the same level as the `sira` directory:

```bash
# From the parent directory of your sira code (e.g., c:\code\)
cd ..
mkdir -p sira_inputs sira_outputs
```

Your directory structure should look like:

```
c:\code\                   # or your workspace root
├── sira/                  # SIRA code repository
│   └── installation/
│       └── docker-compose.yml
├── sira_inputs/        # Your scenario data (outside repo)
│   └── my_project/
│       ├── input/
│       │   ├── config_simulation.json
│       │   └── model_infrastructure.json
│       └── output/        # SIRA creates this
└── sira_outputs/          # Additional outputs (outside repo)
```

### 2. Add Your Scenario

Place your scenario directory in `sira_inputs/`:

```bash
# Example structure
sira_inputs/
└── my_project/
    ├── input/
    │   ├── config_simulation.json
    │   └── model_infrastructure.json
    └── output/
```

### 3. Update docker-compose.yml

Edit the `command` section to point to your scenario:

```yaml
command: >
  python -m sira 
  -d /scenarios/my_project 
  -sfl
```

### 4. Build and Run

```bash
# Build the image
docker compose build

# Run the simulation
docker compose up sira

# Or run in detached mode
docker compose up -d sira
```

## Usage Modes

### Run a Simulation (default)

```bash
docker compose up sira
```

### Interactive Mode

```bash
docker compose run --rm sira-interactive
```

Inside the container:
```bash
python -m sira -d /scenarios/my_project -s
```

### Run Tests

```bash
docker compose run --rm sira-test
```

## Volume Bindings

The compose file binds these directories:

- `../../sira_inputs` → `/scenarios` (read-write: scenario inputs and outputs)
- `../../sira_outputs` → `/outputs` (read-write: additional outputs)
- `../hazard` → `/hazard` (read-only: shared hazard data from repo)
- `../tests` → `/tests` (read-only: test data from repo)

**Important:** The default paths assume your data directories are outside the code repository. Adjust the paths in `docker-compose.yml` if your setup differs:

```yaml
volumes:
  - /path/to/your/scenarios:/scenarios
  - /path/to/your/outputs:/outputs
```

## Environment Variables

Customize behavior via environment variables in `docker-compose.yml`:

```yaml
environment:
  - SIRA_LOG_LEVEL=INFO
  - SIRA_QUIET_MODE=0
  - SIRA_HPC_MODE=0
```

See main documentation for all available environment flags.

## Building Only

```bash
docker compose build
```

## Cleanup

```bash
# Stop and remove containers
docker compose down

# Remove volumes and images
docker compose down -v --rmi all
```

## Direct Docker Commands (without Compose)

Build:
```bash
docker build -f installation/Dockerfile -t sira:latest .
```

Run simulation:
```bash
# Linux/Mac - from repository root
docker run -v /path/to/sira_inputs:/scenarios -v /path/to/sira_outputs:/outputs \
  sira:latest python -m sira -d /scenarios/my_project -sfl

# Windows PowerShell - from repository root
docker run -v C:\path\to\sira_inputs:/scenarios -v C:\path\to\sira_outputs:/outputs `
  sira:latest python -m sira -d /scenarios/my_project -sfl
```

Interactive shell:
```bash
# Linux/Mac
docker run -it -v /path/to/sira_inputs:/scenarios -v /path/to/sira_outputs:/outputs sira:latest /bin/bash

# Windows PowerShell
docker run -it -v C:\path\to\sira_inputs:/scenarios -v C:\path\to\sira_outputs:/outputs sira:latest /bin/bash
```

## Troubleshooting

**Permission Issues:**
If you encounter permission errors with output files, ensure the container user has write access to mounted volumes. On Linux, you may need to adjust ownership:

```bash
sudo chown -R $USER:$USER /path/to/sira_inputs /path/to/sira_outputs
```

**Path Issues:**
If volumes are not mounting correctly, verify:
1. The paths in `docker-compose.yml` match your actual directory structure
2. On Windows, ensure paths use forward slashes in Docker commands or are properly escaped
3. The directories exist before running `docker compose up`

**Build Failures:**
Ensure you're building from the repository root context via the compose file, or use:

```bash
docker build -f installation/Dockerfile -t sira:latest .
```

from the repository root.
