from metaflow import FlowSpec, step, Parameter
import mlflow
import mlflow.sklearn
from updated_preprocessing import load_and_preprocess

class HappinessTrainFlow(FlowSpec):

    seed = Parameter("seed", default=42)
    test_size = Parameter("test_size", default=0.2)

    @step
    def start(self):
        self.data = load_and_preprocess(    #import our data preprocessing from previous labs
            file_path="../data/lab06/train_happy.csv",
            save_path="../data/lab06/preprocess_train_happy.csv"
        )
        # Prepare features and target
        self.X = self.data.drop(columns=["Happiness_Score", "Country", "Year"])
        self.y = self.data["Happiness_Score"]
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.seed
        )
        self.next(self.train_rf, self.train_gb, self.train_lr)

    """
    will train a random forest, gradient boost, and linear regression model
    """
    @step
    def train_rf(self):
        from sklearn.ensemble import RandomForestRegressor
        self.model_name = "RandomForestRegressor"
        self.model = RandomForestRegressor(random_state=self.seed)
        self.model.fit(self.X_train, self.y_train)
        self.score = self.model.score(self.X_test, self.y_test)
        self.next(self.choose_model)

    @step
    def train_gb(self):
        from sklearn.ensemble import GradientBoostingRegressor
        self.model_name = "GradientBoostingRegressor"
        self.model = GradientBoostingRegressor(random_state=self.seed)
        self.model.fit(self.X_train, self.y_train)
        self.score = self.model.score(self.X_test, self.y_test)
        self.next(self.choose_model)

    @step
    def train_lr(self):
        from sklearn.linear_model import LinearRegression
        self.model_name = "LinearRegression"
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        self.score = self.model.score(self.X_test, self.y_test)
        self.next(self.choose_model)

    #choosing best model and registering that to mlflow locally
    @step
    def choose_model(self, inputs):
        import mlflow
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("happiness-experiment")

        self.results = [(inp.model_name, inp.model, inp.score) for inp in inputs]
        self.results.sort(key=lambda x: -x[2])
        self.best_model_name, self.best_model, self.best_score = self.results[0]

        with mlflow.start_run():
            for name, model, score in self.results:
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=f"{name}_model"
                )
                mlflow.log_metric(f"{name}_score", score)
            mlflow.sklearn.log_model(
                sk_model=self.best_model,
                artifact_path="best_model",
                registered_model_name="happiness-model"
            )
            mlflow.log_param("best_model_name", self.best_model_name)
            mlflow.log_metric("best_test_score", self.best_score)
        self.next(self.end)


    @step
    def end(self):
        print("Model scores:")
        for name, model, score in self.results:
            print(f"{name}: {score:.4f}")
        print(f"Best model: {self.best_model_name} (score: {self.best_score:.4f})")

if __name__ == "__main__":
    HappinessTrainFlow()

# example of running file
# python trainingflow.py run --seed 123 --test_size 0.25