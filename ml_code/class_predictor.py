import sys
import pandas as pd
sys.path.insert(0, '/home/sid/github/thought-classifier/')
from sklearn.externals import joblib
from ml_code.pre_processing import clean_text
from ml_code.pre_processing import text_count
loaded_model = joblib.load('ml_code/output/best_logreg_final_model.pkl')


def read_model_and_predict(text):
    """
    This method reads the already defined model from the output/ directory using joblib.
    Uses it to predict the text polarity.
    :return: None
    """
    try:
        global loaded_model
        new_positive_tweets = pd.Series(text)
        tc = text_count.TextCount()
        ct = clean_text.CleanText()
        df_counts_pos = tc.transform(new_positive_tweets)
        df_clean_pos = ct.transform(new_positive_tweets)
        df_model_pos = df_counts_pos
        df_model_pos['clean_text'] = df_clean_pos
        print("Predicting from the loaded pickled model...")
        return loaded_model.predict(df_model_pos).tolist()
    except RuntimeError as e:
        pass
    finally:
        #################################################################
        # This statement required to try using less memory while running#
        #################################################################
        del loaded_model


if __name__ == '__main__':
    text = ["TThank you @VirginAmerica for you amazing customer support team on Tuesday 11/28 at @EWRairport and returning my lost bag in less than 24h! #efficiencyiskey #virginamerica",
            "Love flying with you guys ask these years. Sad that this will be the last trip 😂 @VirginAmerica #LuxuryTravel",
            "Wow @VirginAmerica main cabin select is the way to fly!! This plane is nice and clean & I have tons of legroom! Wahoo! NYC bound! ✈️",
            "This service is shit. I hate it."]

    read_model_and_predict(text=text)