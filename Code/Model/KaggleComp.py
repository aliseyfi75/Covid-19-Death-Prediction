from AutoRegressionModel import AutoRegressionModel as arm
import numpy as np
import pandas as pd


def main():
    training_df = pd.read_csv('../data/phase1_training_data.csv')
    canada_data = training_df[training_df['country_id'] == 'CA']

    model = arm(14, bias=False, method="normal")

    train_series = []

    train_series.append(canada_data["deaths"][150:].values)
    # train_series.append(canada_data["cases"][140:-10].values)

    model.fit(train_series)

    predict_series = []

    predict_series.append(canada_data["deaths"][-14:].values)
    # predict_series.append(canada_data["cases"][-34:-10].values)

    num = 11

    #final_answer = predict_1_future(model, predict_series)

    final_answer = predict_n_future(num, model, predict_series)

    print(final_answer)

    save_results(final_answer, "../Results/test.csv")


def predict_1_future(model, predictors):
    temp_predictor = []
    for predictor in predictors:
        temp_predictor.append(predictor[-14:])
    predictor = np.hstack(temp_predictor)

    final_answer = np.round(model.predict(predictor, num=len(predictors))).astype(int)
    return final_answer


def predict_n_future(n, model, predictors):
    predictor = predictors[0]
    for i in range(n):
        predictor = np.append(predictor, np.round(model.predict(np.hstack(predictor[-14:], predictors[1][i:i+14]), num=len(predictors))).astype(int))

    final_answer = predictors[-n:]
    return final_answer


def save_results(final_answer, file_name):
    y = range(0, final_answer.shape[0])
    out = np.append(np.transpose([final_answer]), np.transpose([y]), axis=1)
    np.savetxt(file_name, out, delimiter=',', fmt=['%d', '%d'], header='deaths,Id', comments='')


if __name__ == "__main__":
    main()
