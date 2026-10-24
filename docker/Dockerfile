# =============================================================================
# OPERA audit -- fully pinned reproducible environments
#
# Two INDEPENDENT build targets (they do not share a base, because IBM CPLEX
# Studio 22.1.1 ships Python bindings only up to Python 3.10, while the
# environment of record of the paper is Python 3.11.15):
#
#   highs  -- environment of record. Open target: anyone can build it with no
#             licence. It retrains the pools, rebuilds the out-of-fold
#             registers and reproduces 100% of the numbers published in
#             Section 6 (HiGHS 1.12.0, bundled in scipy 1.17.1, is the solver
#             of record).
#
#   cplex  -- cross-check target. Re-solves the EXACT registers exported from
#             the environment of record (results/registers_seed20260811.npz)
#             with IBM CPLEX 22.1.1 and asserts that every published optimum
#             is certified. Requires a user-supplied IBM CPLEX Studio 22.1.1
#             (or later, at your own compatibility risk) Linux x86-64
#             installer placed at docker/cplex_installer.bin. The resulting
#             image contains proprietary IBM software: BUILD IT LOCALLY AND
#             NEVER PUSH IT TO ANY REGISTRY. Only this Dockerfile is
#             redistributable.
#
# Build (from the repository root):
#   docker build -f docker/Dockerfile --target highs -t opera-audit:highs .
#
#   cp /path/to/your/cplex_studio2211.linux_x86_64.bin docker/cplex_installer.bin
#   docker build -f docker/Dockerfile --target cplex -t opera-audit:cplex .
#
# Run: see README.md (step-by-step) or ./reproduce_highs.sh / ./reproduce_cplex.sh
#
# Keep the cplex image locally / move it between machines (never publish it):
#   docker save opera-audit:cplex | gzip > opera-audit-cplex.tar.gz
#   docker load < opera-audit-cplex.tar.gz
# =============================================================================

# -----------------------------------------------------------------------------
# Target 1: environment of record (HiGHS). Python 3.11.15, digest-pinned.
# python:3.11.15-slim-bookworm, linux/amd64
# -----------------------------------------------------------------------------
FROM python@sha256:77923445c077d8eb971b14b2b114a1d9cd4a87edb4c75654820ca4832ee8cb15 AS highs

WORKDIR /opera

# Exact transitive closure of the environment of record.
COPY requirements-highs.txt .
RUN pip install --no-cache-dir -r requirements-highs.txt \
 && pip freeze > /opera/environment-lock.txt \
 && python -c "import sys, scipy.optimize._highspy._core as h; \
print('python', sys.version.split()[0], '| HiGHS', \
'.'.join(map(str,(h.HIGHS_VERSION_MAJOR,h.HIGHS_VERSION_MINOR,h.HIGHS_VERSION_PATCH))))"

COPY opera/ opera/
COPY experiments/ experiments/
COPY validation/ validation/
COPY tests/ tests/
COPY data/ data/
COPY results/registers_seed20260811.npz results/

# The MIP is validated against exhaustive enumeration at build time:
# an image that builds is an image whose solver layer is correct.
RUN python -m pytest tests -q

CMD ["python", "validation/compare_solvers.py", "highs"]

# -----------------------------------------------------------------------------
# Target 2: CPLEX cross-check. Python 3.10.19, digest-pinned.
# python:3.10.19-slim-bookworm, linux/amd64
# (3.10 is the newest interpreter supported by the CPLEX 22.1.1 bindings.)
# -----------------------------------------------------------------------------
FROM python@sha256:7de0b3589a2b35452fab55d753aa773d59afcbb88220ab118d6604a398cd65e1 AS cplex

WORKDIR /opera

COPY requirements-cplex.txt .
RUN pip install --no-cache-dir -r requirements-cplex.txt

# The installer is an InstallAnywhere self-extractor with a bundled JRE;
# default-jre-headless is a fallback for hosts where the bundled one fails.
COPY docker/cplex_installer.bin /tmp/cplex_installer.bin
COPY docker/response.properties /tmp/response.properties
RUN apt-get update \
 && apt-get install -y --no-install-recommends default-jre-headless \
 && chmod +x /tmp/cplex_installer.bin \
 && /tmp/cplex_installer.bin -f /tmp/response.properties \
 && pip install --no-cache-dir \
      /opt/ibm/ILOG/CPLEX_Studio2211/cplex/python/3.10/x86-64_linux/ \
 && rm -f /tmp/cplex_installer.bin /tmp/response.properties \
 && rm -rf /opt/ibm/ILOG/CPLEX_Studio2211/doc \
           /opt/ibm/ILOG/CPLEX_Studio2211/examples \
           /opt/ibm/ILOG/CPLEX_Studio2211/opl \
           /opt/ibm/ILOG/CPLEX_Studio2211/cpoptimizer \
           /opt/ibm/ILOG/CPLEX_Studio2211/python \
 && apt-get purge -y default-jre-headless \
 && apt-get autoremove -y && rm -rf /var/lib/apt/lists/* \
 && python -c "import cplex, docplex; print('cplex', cplex.__version__)"

COPY opera/ opera/
COPY validation/ validation/
COPY tests/ tests/
COPY results/registers_seed20260811.npz results/

RUN pip freeze > /opera/environment-lock.txt \
 && python -m pytest tests -q

CMD ["python", "validation/compare_solvers.py"]
