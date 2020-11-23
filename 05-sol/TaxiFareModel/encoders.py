import pandas as pd
import pygeohash as gh
from TaxiFareModel.data import get_data, clean_df, DIST_ARGS
from TaxiFareModel.utils import (
    haversine_vectorized,
    minkowski_distance,
    calculate_direction,
)
from sklearn.base import BaseEstimator, TransformerMixin


class TimeFeaturesEncoder(BaseEstimator, TransformerMixin):
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


class AddGeohash(BaseEstimator, TransformerMixin):
    def __init__(self, precision=6):
        self.precision = precision

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X["geohash_pickup"] = X.apply(
            lambda x: gh.encode(
                x.pickup_latitude, x.pickup_longitude, precision=self.precision
            ),
            axis=1,
        )
        X["geohash_dropoff"] = X.apply(
            lambda x: gh.encode(
                x.dropoff_latitude, x.dropoff_longitude, precision=self.precision
            ),
            axis=1,
        )
        return X[["geohash_pickup", "geohash_dropoff"]]


class DistanceTransformer(BaseEstimator, TransformerMixin):
    """
    INPUT: DF

    OUTPUT:
    --------------------
    Column Distance added to the df which is  the direction to follow to go from
    pickup to dropoff!
    """

    def __init__(self, distance_type="haversine"):
        self.distance_type = distance_type

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        if self.distance_type == "haversine":
            X["distance"] = haversine_vectorized(X, **DIST_ARGS)

        #  Euclidian distance is distance of squared to root
        if self.distance_type == "euclidian":
            X["distance"] = minkowski_distance(X, 2, **DIST_ARGS)

        #  Manhattan distance is distance of 1D to 1D
        if self.distance_type == "manhattan":
            X["distance"] = minkowski_distance(X, 1, **DIST_ARGS)

        return X[["distance"]]

    def fit(self, X, y=None):
        return self


class DirectionTransformer(BaseEstimator, TransformerMixin):
    """
    INPUT: DF

    OUTPUT:
    --------------------
    Column direction added to the df which is  the direction to follow to go from
    pickup to dropoff!
    """

    def __init__(
        self,
        start_lat="pickup_latitude",
        start_lon="pickup_longitude",
        end_lat="dropoff_latitude",
        end_lon="dropoff_longitude",
    ):
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.end_lat = end_lat
        self.end_lon = end_lon

    def transform(self, X, y=None):
        X["delta_lon"] = X[self.start_lon] - X[self.end_lon]
        X["delta_lat"] = X[self.start_lat] - X[self.end_lat]
        X["direction"] = calculate_direction(X.delta_lon, X.delta_lat)
        return X[["delta_lon", "delta_lat", "direction"]]

    def fit(self, X, y=None):
        return self


class DistanceToCenter(BaseEstimator, TransformerMixin):
    def __init__(self, distance_type="haversine"):
        self.distance_type = distance_type

    def transform(self, X, y=None):
        #  NYC center coordinates found on Wikipedia
        nyc_center = {"lat": 40.7141667, "lon": -74.0063889}
        X["nyc_lat"], X["nyc_lng"] = nyc_center["lat"], nyc_center["lon"]

        # Splitting the distance to center with a center-pickup distance
        pickup_args = dict(
            start_lat="nyc_lat",
            start_lon="nyc_lng",
            end_lat="pickup_latitude",
            end_lon="pickup_longitude",
        )

        # Second hald of the split which is a center-dropoff
        dropoff_args = dict(
            start_lat="nyc_lat",
            start_lon="nyc_lng",
            end_lat="dropoff_latitude",
            end_lon="dropoff_longitude",
        )

        # Use the minkowski or haversine vectorized formulas to compute the distance
        # in accordance with the distance parameter set previously

        # HAVERSINE
        if self.distance_type == "haversine":
            X["pickup_distance_to_center"] = haversine_vectorized(X, **pickup_args)
            X["dropoff_distance_to_center"] = haversine_vectorized(X, **dropoff_args)

        #  EUCLIDIAN
        if self.distance_type == "euclidian":
            X["pickup_distance_to_center"] = minkowski_distance(X, 2, **pickup_args)
            X["dropoff_distance_to_center"] = minkowski_distance(X, 2, **dropoff_args)

        # MANHATTAN
        if self.distance_type == "manhattan":
            X["pickup_distance_to_center"] = minkowski_distance(X, 1, **pickup_args)
            X["dropoff_distance_to_center"] = minkowski_distance(X, 1, **dropoff_args)

        return X[["pickup_distance_to_center", "dropoff_distance_to_center"]]

    def fit(self, X, y=None):
        return self


if __name__ == "__main__":
    params = dict(
        nrows=1000,
        upload=False,
        local=False,  # set to False to get data from GCP (Storage or BigQuery)
        optimize=False,
    )
    df = get_data(**params)
    df = clean_df(df)
    dist = DistanceTransformer()
    X = dist.transform(df)
