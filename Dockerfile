# Use an official Python runtime as a parent image
FROM python:3.12-slim-bookworm

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Set the working directory
WORKDIR /code

# Copy the pyproject.toml and uv.lock files
COPY pyproject.toml uv.lock /code/

# Install dependencies using uv (this also installs the spaCy en_core_web_lg model,
# which is listed in pyproject.toml). --no-dev skips pytest/httpx.
RUN uv sync --frozen --no-dev --no-install-project

# Copy the application code
COPY ./app /code/app

EXPOSE 80

# Command to run the application
CMD ["uv", "run", "--no-dev", "fastapi", "run", "app/main.py", "--port", "80"]
