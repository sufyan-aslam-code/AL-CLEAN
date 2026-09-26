from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from main import run_full_pipeline

app = FastAPI(title="AL-Clean API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PipelineConfig(BaseModel):
    noise_level: float
    iterations: int
    sampling_metric: str

@app.post("/run-pipeline")
def run_pipeline(config: PipelineConfig):
    import mlflow
    mlflow.set_tracking_uri("http://localhost:5000")
    run = mlflow.start_run(run_name="AL-Clean Execution")
    run_id = run.info.run_id
    mlflow.end_run() # End it so it can be safely resumed
    
    # Execute synchronously (blocks until done)
    run_full_pipeline(
        noise_level=config.noise_level,
        iterations=config.iterations,
        metric=config.sampling_metric,
        run_id=run_id
    )
    return {"status": "success", "message": "Pipeline completed successfully", "run_id": run_id}
