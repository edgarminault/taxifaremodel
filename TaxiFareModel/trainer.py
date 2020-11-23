import time
import warnings

from TaxiFareModel.data import get_data, clean_df
from TaxiFareModel.utils import compute_rmse, simple_time_tracker
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer, DIST_ARGS
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)


class Trainer(object):
    ESTIMATOR = "Linear"

    def __init__(self, X, y, **kwargs):
        """
        FYI:
        __init__ is called every time you instatiate Trainer
        Consider kwargs as a dict containing all possible parameters given to your constructor
        Example:
            TT = Trainer(nrows=1000, estimator="Linear")
               ==> kwargs = {"nrows": 1000,
                            "estimator": "Linear"}
        :param X: pandas DataFrame
        :param y: pandas DataFrame
        :param kwargs:
        """
        self.pipeline = None
        self.kwargs = kwargs
        self.split = self.kwargs.get("split", True)
        self.estimator = self.kwargs.get("estimator", "Linear")
        if self.split == True:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.33, random_state=42
            )
        else:
            self.X_train = X
            self.y_train = y
        self.distance_transformer_input = self.kwargs.get("distance_transformer_input")

    def get_estimator(self):
        """
        INPUT: self
        -----------------------
        Extracts the estimator kwargs
        -----------------------
        OUTPUT : appropriate model with regard to the key word argument.
        """
        self.estimator = self.kwargs.get("estimator")

        if self.estimator == "Linear":
            model = LinearRegression()

        elif self.estimator == "RandomForest":
            model = RandomForestRegressor()

        elif self.estimator == "Ridge":
            model = Ridge()

        elif self.estimator == "Lasso":
            model = Lasso()

        return model

    def set_pipeline(self):
        """
        Will setup a pipeline with the following steps:
        1. Encoding the pickup_datetime and 1HOT encoding HOURS and DAY features.
        2. Encode the the lats and longs to a HAVERSINE DISTANCE scaled.
        3. Add a MODEL with get_estimator()
        """

        pipe_time_features = make_pipeline(
            TimeFeaturesEncoder(time_column="pickup_datetime"),
            OneHotEncoder(handle_unknown="ignore"),
        )

        pipe_distance = make_pipeline(
            DistanceTransformer(self.distance_type), StandardScaler()
        )

        features_encoder = ColumnTransformer(
            [
                (
                    "distance",
                    pipe_distance,
                    list(DIST_ARGS.values()),
                ),
                ("time_features", pipe_time_features, ["pickup_datetime"]),
            ],
            n_jobs=None,
            remainder="drop",
        )

        pipe = Pipeline(
            steps=[
                ("features", features_encoder),
                ("model", self.get_estimator()),
            ]
        )
        self.pipeline = pipe

        return self

    @simple_time_tracker
    def train(self):
        tic = time.time()
        self.set_pipeline()
        self.pipeline.fit(self.X_train, self.y_train)

    @simple_time_tracker
    def evaluate(self):
        if self.pipeline is None:
            raise ("Cannot evaluate an empty pipeline")
        y_pred = self.pipeline.predict(self.X_test)
        rmse = compute_rmse(y_pred, self.y_test)
        return round(rmse, 3)


if __name__ == "__main__":
    # Get and clean data
    N = 10000
    df = get_data(nrows=N)
    df = clean_df(df)
    y_train = df["fare_amount"]
    X_train = df.drop("fare_amount", axis=1)

    # Train and save model, locally and
    t = Trainer(
        X=X_train,
        y=y_train,
        estimator="xgboostt",
        distance_transformer_input=DIST_ARGS,
    )
    t.train()
    print(t.evaluate())
