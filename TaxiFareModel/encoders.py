import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from TaxiFareModel.utils import haversine_vectorized


DIST_ARGS = dict(
    start_lat="pickup_latitude",
    start_lon="pickup_longitude",
    end_lat="dropoff_latitude",
    end_lon="dropoff_longitude",
)

# Implement DistanceTransformer and TimeFeaturesEncoder
class DistanceTransformer(BaseEstimator, TransformerMixin):
    """
    INPUT: DF and distance_type
    OUTPUT:
    --------------------
    Column distance added to the df which is the haversine distance between dropoff and pickup!
    """

    def __init__(self, distance_type="haversine"):
        self.distance_type = distance_type

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X["distance"] = haversine_vectorized(X, **DIST_ARGS)

    def fit(self, X, y=None):
        return self


class TimeFeaturesEncoder(BaseEstimator, TransformerMixin):
    """
    OUTPUT:
    Additional columns thanks to the time_column (in this case "pickup_datetime"):
    - dow;
    - hour;
    - month;
    - year.
    """

    def __init__(self, time_column, time_zone_name="America/New_York"):
        self.time_column = time_column
        self.time_zone_name = time_zone_name

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X.index = pd.to_datetime(X[self.time_column])
        X.index = X.index.tz_convert(self.time_zone_name)
        X["dow"] = X.index.weekday
        X["hour"] = X.index.hour
        X["month"] = X.index.month
        X["year"] = X.index.year
        return X[["dow", "hour", "month", "year"]].reset_index(drop=True)

    def fit(self, X, y=None):
        return self
