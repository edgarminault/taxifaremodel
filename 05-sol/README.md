# Iterate with MlFlow

Now that we have already built a simple, we want to make it better! The ultimate goal is having a model that makes more accurate predictions on the test set, hence getting a RMSE as low as possible.

**So what can we do?**

There are many different things that make models better:
- build and try to use different or more features
- test with different estimators (linear, non linear, etc..)
- tune hyperparameters

In this series of exercise, you will get hands on using the [MLFlow Tracking Api](https://www.mlflow.org/docs/latest/tracking.html) in order to experiment with different features, models and parameters.

Since that now we have a good workflow to make model improvements, it is very important to track all our different experiments. We want to be able to save all our differents training runs and compare their performance.

This is what MLFlow tracking is about.

## Summary
1. [Try different models](#part2)
2. [Features engineering](#part3)

## Prerequisites
Before carrying on make sure you understood:
- the structure of the `Trainer()` class from ex1 is clear, if not please reach out to one of your TA's
- the way to log a metric and a parameter on [wagon_hosted_mlflow server](https://mlflow.lewagon.co/#/experiments/0)
- how to pass random number of parameters to a class or functions (`**kwargs`), and how to access them
- you might need to `pip install category_encoders memoized_property psutil xgboost pygeohash`

Please also inspect new `trainer.py` from today, the structure is slightly different from yesterday.
👉 Amongst other we added `save_model()` method, keep it in mind for later
👉 Once you'll be happy about your final model, you will submit your your best model's predictions to kaggle

Last thing, we suggested here a package structure to organize your code and your run, feel free to tweak is as you like if need be.
## 0. Define your experiment name
You will all log you run results and parameters on the same MLFLOW instance hosted on le wagon server.
From now on, you will have have your own `experiment` defined inside `trainer.py` as follow:
```python
experiment = "taxifare_YOURNAME"
```
Please update your name for all following exercices

## 1. Try different models
- Think about the different estimators that you know that can be used to solve prediction problems
- Implement a short script that will loop through all estimators, train the model and evaluate it on a validation set.
👉 Here you might need to tweek `TaxiFareModel` package
- Be careful: make sure you always use the same validation set accross all your trainings. **Tip** you can set the random seed for `train_test_split` to make sure the split is always the same.
- View results with on [wagon_hosted_mlflow server](https://mlflow.lewagon.co/#/experiments/0)

And last advice, while building your pipeline, run it on small datasample, and preferably locally

## 2. Features engineering and selection
**Now it is time to be creative!**

You just tried different models, and you now see that some estimators may be more powerful than others. Another area where you can experiment is about `features engineering`.

- Try different combinations of features (by removing or adding some) and track the runs.
💡`direction`, `manhattan distance`, `euclidian distance`, `geohash`
- Use some "context knowledge" to generate new features that you think might be explaining the fares.
 👉 For example: we know that taxi apply a fixed amount for airport transfers
- Try different methods for outliers removals
- Look at how the size of training set helps the reduce the RMSE on the validation set

### 💡 Suggested method to track influence of Feature Engineering:
You will use the benefits of Pipelines integrated into our custom class.
We will start with one additional feature: `distance_to_center`:
- Get back to day2 first notebook with complete feature engineering
- Implement Custom Transformer inside `encoders.py` called `DistanceToCenter`, that adds `distance_to_center`
💡 Use notions from day2 on Pipeline's custom transformers here
- adapt `set_pipeline()` method inside our main `Trainer()` class so that it integrates your new bloc
- modify params to feed to our `Trainer()` class in order to easily run 2 runs to compare influence of added feature
- launch 2 new runs and check influence of `distance_to_center` new feature

Once you've added the new feature, add as many features as you want and analyse impact on performances
Bonus - Use [PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html) transfomer to generate new features from distance.

