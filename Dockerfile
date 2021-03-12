#FROM continuumio/miniconda3
FROM zhuwq0/phasenet

WORKDIR /app

# Create the environment:
# COPY env.yml /app
# RUN conda env create --name cs329s --file=env.yml
# Make RUN commands use the new environment:
# SHELL ["conda", "run", "-n", "cs329s", "/bin/bash", "-c"]
RUN pip install uvicorn fastapi kafka-python

# Copy files
COPY . /app

# Expose API port
EXPOSE 8000

ENV PYTHONUNBUFFERED=1

# Start API server
#ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "cs329s", "uvicorn", "--app-dir", "phasenet", "app:app", "--reload", "--port", "8000", "--host", "0.0.0.0"]
ENTRYPOINT ["uvicorn", "--app-dir", "phasenet", "app:app", "--reload", "--port", "8000", "--host", "0.0.0.0"]
