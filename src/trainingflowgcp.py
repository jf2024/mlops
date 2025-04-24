# from metaflow import FlowSpec, step, Parameter, resources, kubernetes, timeout, retry, catch, conda_base

# import mlflow
# import mlflow.sklearn
# from updated_preprocessing import load_and_preprocess

# @conda_base(
#     libraries={
#         "scikit-learn": "1.2.2",
#         "mlflow": "2.3.0",
#         "pandas": "1.5.3",
#         "databricks-cli": "0.17.7",   # Add this line
#     },
#     python="3.9.16"
# )
# class HappinessTrainFlow(FlowSpec):

#     seed = Parameter("seed", default=42)
#     test_size = Parameter("test_size", default=0.2)

#     @timeout(minutes=10)
#     @retry(times=2)
#     @catch(var="start_error")
#     @step
#     def start(self):
#         self.data = load_and_preprocess(
#             file_path="../data/lab06/train_happy.csv",
#             save_path="../data/lab06/preprocess_train_happy.csv"
#         )
#         self.X = self.data.drop(columns=["Happiness_Score", "Country", "Year"])
#         self.y = self.data["Happiness_Score"]
#         from sklearn.model_selection import train_test_split
#         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
#             self.X, self.y, test_size=self.test_size, random_state=self.seed
#         )
#         self.next(self.train_rf, self.train_gb, self.train_lr)

#     @kubernetes(cpu=2, memory=8000)
#     @timeout(minutes=15)
#     @retry(times=2)
#     @catch(var="rf_error")
#     @step
#     def train_rf(self):
#         from sklearn.ensemble import RandomForestRegressor
#         self.model_name = "RandomForestRegressor"
#         self.model = RandomForestRegressor(random_state=self.seed)
#         self.model.fit(self.X_train, self.y_train)
#         self.score = self.model.score(self.X_test, self.y_test)
#         self.next(self.choose_model)

#     @kubernetes(cpu=2, memory=8000)
#     @timeout(minutes=15)
#     @retry(times=2)
#     @catch(var="gb_error")
#     @step
#     def train_gb(self):
#         from sklearn.ensemble import GradientBoostingRegressor
#         self.model_name = "GradientBoostingRegressor"
#         self.model = GradientBoostingRegressor(random_state=self.seed)
#         self.model.fit(self.X_train, self.y_train)
#         self.score = self.model.score(self.X_test, self.y_test)
#         self.next(self.choose_model)

#     @kubernetes(cpu=1, memory=4000)
#     @timeout(minutes=10)
#     @retry(times=2)
#     @catch(var="lr_error")
#     @step
#     def train_lr(self):
#         from sklearn.linear_model import LinearRegression
#         self.model_name = "LinearRegression"
#         self.model = LinearRegression()
#         self.model.fit(self.X_train, self.y_train)
#         self.score = self.model.score(self.X_test, self.y_test)
#         self.next(self.choose_model)

#     @timeout(minutes=5)
#     @retry(times=2)
#     @catch(var="choose_model_error")
#     @step
#     def choose_model(self, inputs):
#         import mlflow
#         # mlflow.set_tracking_uri("sqlite:///mlflow.db") # for local
#         mlflow.set_tracking_uri("https://mlops-service-616366923242.us-west2.run.app")  # using GCP

#         mlflow.set_experiment("happiness-experiment")
#         self.results = [(inp.model_name, inp.model, inp.score) for inp in inputs]
#         self.results.sort(key=lambda x: -x[2])
#         self.best_model_name, self.best_model, self.best_score = self.results[0]

#         with mlflow.start_run():
#             for name, model, score in self.results:
#                 mlflow.sklearn.log_model(sk_model=model, artifact_path=f"{name}_model")
#                 mlflow.log_metric(f"{name}_score", score)

#             mlflow.sklearn.log_model(
#                 sk_model=self.best_model,
#                 artifact_path="best_model",
#                 registered_model_name="happiness-model"
#             )
#             mlflow.log_param("best_model_name", self.best_model_name)
#             mlflow.log_metric("best_test_score", self.best_score)

#         self.next(self.end)

#     @step
#     def end(self):
#         print("Model scores:")
#         for name, model, score in self.results:
#             print(f"{name}: {score:.4f}")
#         print(f"Best model: {self.best_model_name} (score: {self.best_score:.4f})")


# if __name__ == "__main__":
#     HappinessTrainFlow()
from metaflow import FlowSpec, step, batch, Parameter
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib

class TrainingFlow(FlowSpec):

    # Parameters
    input_path = Parameter('input_path',
                           help="Path to the raw happiness CSV file",
                           default="data/world-happiness-report.csv")

    @step
    def start(self):
        import boto3
        self.s3 = boto3.client('s3')
        self.bucket = "storage-mlops_lab_seven-metaflow-default"
        self.next(self.preprocess)

    @step
    def preprocess(self):
        # Load the raw data
        df = pd.read_csv(self.input_path)

        # Drop duplicates based on Country and Year
        df = df.drop_duplicates(subset=["Country name", "year"])

        # Assign unique Country_ID
        df["Country_ID"] = df["Country name"].astype("category").cat.codes

        # Scale selected numerical features
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        num_cols = [col for col in num_cols if col not in ["Country_ID", "year"]]
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])

        # Save preprocessed data
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/preprocessed_happy.csv", index=False)

        self.df = df
        self.next(self.train)

    @batch(cpu=2, memory=4000)
    @step
    def train(self):
        df = self.df

        # Example: Linear Regression to predict Life Ladder score
        features = [col for col in df.columns if col not in ["Life Ladder", "Country name", "year"]]
        X = df[features]
        y = df["Life Ladder"]

        model = LinearRegression()
        model.fit(X, y)

        # Save model locally
        os.makedirs("model", exist_ok=True)
        joblib.dump(model, "model/life_ladder_model.pkl")

        self.model_path = "model/life_ladder_model.pkl"
        self.next(self.end)

    @step
    def end(self):
        print("Model training complete.")
        print(f"Saved model at: {self.model_path}")

if __name__ == '__main__':
    TrainingFlow()
