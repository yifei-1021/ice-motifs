#!/usr/bin/env bash
# Build a clean, scrubbed tarball for the Zenodo deposit.
#
#   ./tools/make-deposit.sh [OUTDIR]
#
# It cleans build artefacts, scans for anything that identifies the machine the
# code was developed on, and REFUSES to build the archive if it finds any.
#
# Why this exists rather than a one-off manual pass: compiled bytecode embeds
# the absolute path of the source it was compiled from, so a single stray
# __pycache__ puts a full home-directory path -- username included -- inside a
# public archive. It reappears every time anyone imports the package, so the
# check has to run at packaging time, not once.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/dist}"
NAME="icemotifs-$(sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT/pyproject.toml" | head -1)"

echo "=== 1. clean build artefacts"
find "$ROOT" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$ROOT" -name "*.py[cod]" -delete 2>/dev/null || true
rm -rf "$ROOT/build" "$ROOT"/*.egg-info "$ROOT/.pytest_cache"
# .ipynb_checkpoints holds full copies of whatever was open -- a checkpoint of
# a 1.7 MB movie is 1.7 MB of duplicate in the tarball. .gitignore covers git;
# this covers the archive.
find "$ROOT" -name ".ipynb_checkpoints" -type d -prune -exec rm -rf {} + 2>/dev/null || true
echo "    ok"

echo "=== 2. scan for machine-identifying content"
# Extend this list rather than trimming it. A false positive costs a second;
# a miss is public.
PATTERNS='/scratch/|/mnt/|/home/[a-z]|-scratch-[a-z]|-home-[a-z]'
PATTERNS="$PATTERNS"'|[a-z]+[0-9]{6,}|anaconda3/envs|conda activate'
PATTERNS="$PATTERNS"'|#PBS|qsub|sbatch|SLURM|PBS_O_|walltime'
PATTERNS="$PATTERNS"'|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
# Job-scratch paths that the rules above do not catch on their own. A session
# directory like /var/tmp/pbs.NNNNN.<cluster>/claude-NNNNN/-<fs>-scratch-<user>-...
# has had its slashes replaced by hyphens, which is why `-scratch-[a-z]` and
# `-home-[a-z]` sit beside the anchored forms above.
# NO MACHINE, FILESYSTEM OR USER NAME IS LISTED HERE, deliberately: this file is
# itself part of the archive and is exempt from its own scan, so a list of names
# to scrub for would publish exactly the names it is meant to keep out. Every
# rule here is generic, and several catch a flattened session path on their own.
PATTERNS="$PATTERNS"'|/var/tmp|pbs\.[0-9]{4,}\.|claude-[0-9]{4,}'

# Allow-list: scientific provenance that names published artefacts but no
# machine -- the deposited matrix filename, its row count, the checkpoint
# directory. A reproducer needs these.
# The seed/run names of the published trajectories (8154387, s242774, ...) are
# scientific provenance: examples/README.md says which trajectory each
# panel was drawn from, and a reader needs that. They match the generic
# `[a-z]+[0-9]{6,}` rule, so they are named here rather than the rule loosened.
ALLOW='TRAJ18058040|18058040|TRAIN\.T8\.10|/path/to/|github.com|usr/bin/env'
ALLOW="$ALLOW"'|8154387|8157657|822188|8255250|s242774|s252646|s259836'

# A notebook stores its figures as base64 INSIDE the .ipynb, one payload per
# line. Random base64 matches the generic `[a-z]+[0-9]{6,}` rule about once per
# 100 kB, so a notebook with stored figures refuses the build every time -- on
# an image, not on a path. The payload line is dropped whole: it is one quoted
# blob, so nothing else on it could be a leak.
IMAGE_LINE='"image/(png|jpeg|gif)": "'

# .git is excluded for the same reason as in step 3 and in the archive: nothing
# in it reaches the deposit. It is also a source of false positives -- a packed
# repository writes `objects/info/packs`, a text file naming a pack whose hex
# digest contains a run of digits, which the generic rule matches.
hits=$(grep -rnIE --exclude-dir=.git "$PATTERNS" "$ROOT" 2>/dev/null \
        | grep -vE "$ALLOW" \
        | grep -vE "$IMAGE_LINE" \
        | grep -v "tools/make-deposit.sh" || true)
if [ -n "$hits" ]; then
    echo "    REFUSING TO BUILD -- machine-identifying content found:"
    echo "$hits" | sed "s|$ROOT/|      |"
    exit 1
fi
echo "    clean"

echo "=== 3. binary files (bytecode embeds build paths; there should be none)"
# .git is excluded here for the same reason --exclude-vcs excludes it from the
# archive below: its objects are zlib blobs that never reach the deposit. Keep
# the two exclusions in step, or this check fails on any repo.
bins=$(find "$ROOT" -path "$ROOT/.git" -prune -o -type f -print0 2>/dev/null \
        | xargs -0 file 2>/dev/null \
        | grep -iE "ELF|compiled|data$" \
        | grep -vE "\.(png|gif|jpg|jpeg|ipynb|json|csv)" || true)
if [ -n "$bins" ]; then
    echo "    REFUSING TO BUILD -- unexpected binary files:"
    echo "$bins" | sed "s|$ROOT/|      |"
    exit 1
fi
echo "    none"

echo "=== 4. archive"
mkdir -p "$OUT"
tar --exclude-vcs --exclude="dist" --exclude="runs" \
    -czf "$OUT/$NAME.tar.gz" -C "$(dirname "$ROOT")" "$(basename "$ROOT")"
echo "    $OUT/$NAME.tar.gz  ($(du -h "$OUT/$NAME.tar.gz" | cut -f1))"

echo
echo "Before uploading, settle the placeholders:"
grep -rn "FILL IN\|TODO(author)" "$ROOT" --include="README.md" \
     --include="*.toml" 2>/dev/null | sed "s|$ROOT/|  |" || echo "  none left"
