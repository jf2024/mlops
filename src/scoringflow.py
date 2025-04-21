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

# Example Output below and saved to csv file on my laptop

# 2025-04-21 01:14:02.411 Workflow starting (run-id 1745223242410795):
# 2025-04-21 01:14:02.426 [1745223242410795/start/1 (pid 64115)] Task is starting.
# 2025-04-21 01:14:03.929 [1745223242410795/start/1 (pid 64115)] Task finished successfully.
# 2025-04-21 01:14:03.939 [1745223242410795/predict/2 (pid 64117)] Task is starting.
# 2025-04-21 01:14:06.742 [1745223242410795/predict/2 (pid 64117)] Task finished successfully.
# 2025-04-21 01:14:06.752 [1745223242410795/end/3 (pid 64120)] Task is starting.
# 2025-04-21 01:14:08.132 [1745223242410795/end/3 (pid 64120)] Predictions saved to ../data/lab06/predictions.csv
# 2025-04-21 01:14:08.138 [1745223242410795/end/3 (pid 64120)] Country  Year  Predicted_Happiness_Score
# 2025-04-21 01:14:08.332 [1745223242410795/end/3 (pid 64120)] 0      China  2009                   4.541433
# 2025-04-21 01:14:08.332 [1745223242410795/end/3 (pid 64120)] 1     Brazil  2019                   5.049367
# 2025-04-21 01:14:08.333 [1745223242410795/end/3 (pid 64120)] 2  Australia  2020                   5.116909
# 2025-04-21 01:14:08.333 [1745223242410795/end/3 (pid 64120)] 3      India  2005                   4.623467
# 2025-04-21 01:14:08.333 [1745223242410795/end/3 (pid 64120)] 4     France  2020                   5.064047
# 2025-04-21 01:14:08.334 [1745223242410795/end/3 (pid 64120)] Task finished successfully.
# 2025-04-21 01:14:08.334 Done!

