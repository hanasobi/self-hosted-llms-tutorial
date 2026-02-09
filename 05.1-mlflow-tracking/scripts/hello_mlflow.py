"""
Hello World MLflow Example
Demonstrates basic MLflow tracking with central server setup
"""
import mlflow

# Tracking URI setzen - verbindet zu zentralem Server
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("hello-mlflow")

with mlflow.start_run():
    # Parameter loggen
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("batch_size", 32)
    
    # Metriken loggen (simuliertes Training)
    for step in range(10):
        loss = 1.0 / (step + 1)
        mlflow.log_metric("loss", loss, step=step)
    
    print("✓ Run completed! Check http://localhost:5000")
    print(f"  Experiment: hello-mlflow")
    print(f"  Run ID: {mlflow.active_run().info.run_id}")