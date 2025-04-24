from metaflow import FlowSpec, step, Parameter, kubernetes, timeout, retry, catch, conda_base

import pandas as pd
import mlflow.pyfunc
from updated_preprocessing import load_and_preprocess

@conda_base(
    libraries={
        "mlflow": "2.3.0",
        "pandas": "1.5.3",
        "databricks-cli": "0.17.7",   # Add this line
    },
    python="3.9.16"
)
class HappinessScoringFlow(FlowSpec):

    input_csv = Parameter("input_csv", default="CSV file with new data to predict")
    output_csv = Parameter("output_csv", default="../data/lab06/predictions.csv")

    @timeout(minutes=5)
    @retry(times=2)
    @catch(var="start_error")
    @step
    def start(self):
        self.data = load_and_preprocess(
            file_path=self.input_csv,
            save_path="../data/lab06/updated_scoring.csv"
        )
        self.next(self.predict)

    @kubernetes(cpu=2, memory=4000)
    @timeout(minutes=10)
    @retry(times=2)
    @catch(var="predict_error")
    @step
    def predict(self):
        import mlflow
        # mlflow.set_tracking_uri("sqlite:///mlflow.db")  # for local
        mlflow.set_tracking_uri("https://mlops-service-616366923242.us-west2.run.app")  # using GCP
        model_uri = "models:/happiness-model/Production"  # Use MLflow model registry if available
        self.model = mlflow.pyfunc.load_model(model_uri)
        X_new = self.data.drop(columns=["Happiness_Score", "Country", "Year"], errors="ignore")
        self.predictions = self.model.predict(X_new)
        self.data["Predicted_Happiness_Score"] = self.predictions
        self.data.to_csv(self.output_csv, index=False)
        self.next(self.end)

    @step
    def end(self):
        print(f"Predictions saved to {self.output_csv}")
        print(self.data[["Country", "Year", "Predicted_Happiness_Score"]].head())

if __name__ == "__main__":
    HappinessScoringFlow()
