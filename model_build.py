import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import pickle
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool
from sklearn.neural_network import MLPClassifier
import torch


class ModelBuilder:
    def load_data(self, path):
        with open(path, 'rb') as f:
            embeddings = pickle.load(f)

        embeddings = np.array(embeddings)
        np.random.shuffle(embeddings)

        labels = embeddings[:, -1]
        embeddings = embeddings[:, :-1]

        X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test, path.replace("embeddings-", "").replace(".pkl", "")

    def create_model(self, transformer_name, model_name, X_train, y_train):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_name == "xgboost":
            model = XGBClassifier(n_estimators=10000, learning_rate=0.1, max_depth=2, early_stopping_rounds=100, device=device)
            print("XGBoost model is training...")
            model.fit(X_train, y_train, verbose=True, eval_set=[(X_train, y_train)])
            # pickle.dump(model, open("model/" + transformer_name + "_" + model_name + "_model.pkl", "wb"))
            model.save_model("model/" + transformer_name + "_xgboost_model.json")
        elif model_name == "catboost":
            model = CatBoostClassifier(task_type="GPU", devices='0:1', iterations=10000, learning_rate=1, depth=2)
            print("CatBoost model is training...")
            pool = Pool(X_train, y_train)
            model.fit(X_train, y_train, verbose=True, eval_set=(X_train, y_train), early_stopping_rounds=100)
            # pickle.dump(model, open("model/sbert_catboost_model.pkl", "wb"))
            model.save_model("model/" + transformer_name + "_catboost_model.cbm", format="cbm")
        elif model_name == "ann":
            model = MLPClassifier(hidden_layer_sizes=(500, 200, 10), max_iter=200, activation='relu', solver='adam',
                                  random_state=1, verbose=True, n_iter_no_change=30)
            print("ANN model is training...")
            model.fit(X_train, y_train)
            pickle.dump(model, open("model/" + transformer_name + "_ann_model.pkl", "wb"))
        else:
            print("Model name is not valid")
            exit(1)

    def load_model(self, transformer_name, model_name):
        if model_name == "xgboost":
            model = XGBClassifier()
            model.load_model("model/" + transformer_name + "_xgboost_model.json")
        elif model_name == "catboost":
            model = CatBoostClassifier()
            model.load_model("model/" + transformer_name + "_catboost_model.cbm", format="cbm")
        elif model_name == "ann":
            model = pickle.load(open("model/" + transformer_name + "_" + model_name + "_model.pkl", "rb"))
        else:
            print("Model name is not valid")
            exit(1)
        return model

    def test_model(self, model, X_test, y_test):
        print("Testing the model... " + model.__class__.__name__)

        y_pred = model.predict(X_test)

        print(y_pred)
        print(y_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Precision:", precision_score(y_test, y_pred))
        print("Recall:", recall_score(y_test, y_pred))
        print("F1 Score:", f1_score(y_test, y_pred))

    def test_specific_model(self, model_name, path, X_test, y_test):
        print("Testing the model... " + model_name)
        if model_name == "xgboost":
            model = XGBClassifier()
            model.load_model(path)
        elif model_name == "catboost":
            model = CatBoostClassifier()
            model.load_model(path, format="cbm")
        elif model_name == "ann":
            model = pickle.load(open(path, "rb"))

        y_pred = model.predict(X_test)

        print(y_pred)
        print(y_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Precision:", precision_score(y_test, y_pred))
        print("Recall:", recall_score(y_test, y_pred))
        print("F1 Score:", f1_score(y_test, y_pred))

    def save_copy_model(self, path, transformer_name, model_name):
        if model_name == "xgboost":
            model = XGBClassifier()
            model1 = pickle.load(open(path, "rb"))
            model = model1.get_booster().copy()
            model.save_model("model/" + transformer_name + "_xgboost_model.json")
        elif model_name == "catboost":
            model = CatBoostClassifier()
            model1 = pickle.load(open(path, "rb"))
            model = model1.copy()
            model.save_model("model/" + transformer_name + "_catboost_model.cbm", format="cbm")


if __name__ == '__main__':
    os.makedirs('model', exist_ok=True)
    model_name = sys.argv[1]  # xgboost, catboost, ann
    model_builder = ModelBuilder()
    X_train, X_test, y_train, y_test, transformer_name = model_builder.load_data(sys.argv[2])  # embeddings file path
    model_builder.create_model(transformer_name.replace("embeddings/", ""), model_name, X_train, y_train)
    loaded_model = model_builder.load_model(transformer_name.replace("embeddings/", ""), model_name)
    model_builder.test_model(loaded_model, X_test, y_test)
    # model_builder.save_copy_model(sys.argv[3], transformer_name.replace("embeddings/", ""), model_name)
    # model_builder.test_specific_model("ann", "model/xlm-roberta_ann_model.pkl", X_test, y_test)
