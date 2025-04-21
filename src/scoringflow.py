from metaflow import FlowSpec, step, Parameter
import pandas as pd
import mlflow.pyfunc
from updated_preprocessing import load_and_preprocess

class HappinessScoringFlow(FlowSpec):
    input_csv = Parameter("input_csv", default="CSV file with new data to predict")
    output_csv = Parameter("output_csv", default="../data/lab06/predictions.csv")

    @step 
    def start(self):
        self.data = load_and_preprocess(  #same preprocessing step as the training flow
            file_path=self.input_csv,
            save_path="../data/lab06/updated_scoring.csv"
        )
        self.next(self.predict)

    @step
    def predict(self):
        import mlflow
        mlflow.set_tracking_uri("sqlite:///mlflow.db") #connect to local db
        model_uri = "runs:/e18d35dd4b1b463eada1089c034fc24a/best_model" #might need to change this
        self.model = mlflow.pyfunc.load_model(model_uri)
        X_new = self.data.drop(columns=["Happiness_Score", "Country", "Year"], errors="ignore")
        self.predictions = self.model.predict(X_new)
        self.data["Predicted_Happiness_Score"] = self.predictions
        self.data.to_csv(self.output_csv, index=False) #save predictions in csv file 
        self.next(self.end)


    @step
    def end(self):
        print(f"Predictions saved to {self.output_csv}")
        print(self.data[["Country", "Year", "Predicted_Happiness_Score"]].head())

if __name__ == "__main__":
    HappinessScoringFlow()


# example of running file below 
# python scoringflow.py run --input_csv ../data/lab06/test_happy.csv
