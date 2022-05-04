import os
from pathlib import Path
import pickle
import numpy as np
from deepgesture.config import Config
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score

def main():
    suturing_processed_dir = Path(os.getenv("SUTURING_PROCESSED_DIR"))
    trained_models_dir = suturing_processed_dir / "models"
    pickle_path = os.path.join(trained_models_dir, "embedding_dict.pkl")
    fileo = open(pickle_path, 'rb')
    embed_dict = pickle.load(fileo)
    embeddings, gestures = np.array(embed_dict['embeddings']), np.array(embed_dict['gesture'])
    x_train, x_test, y_train, y_test = train_test_split(embeddings, gestures, test_size=0.2, random_state=12499)
    x_train = np.squeeze(x_train)
    x_test = np.squeeze(x_test)
    # parameters = {'hidden_layer_sizes': [(100,), (50,50), (25, 25, 25, 25)], 'solver': ['adam'], 'activation': ['identity', 'logistic', 'tanh', 'relu'], 'beta_1': [0.85, 0.9, 0.95], 'beta_2': [0.9, 0.99, 0.999], 'max_iter': [1000,1200,1400,1600,1800,2000]}
    #parameters = {'hidden_layer_sizes': [(50,50)], 'solver': ['adam'], 'activation': ['tanh'], 'beta_1': [0.85, 0.9], 'max_iter': [1200, 1500, 1800]}
    parameters = {'hidden_layer_sizes': [(75,75, 75)], 'solver': ['adam'], 'activation': ['tanh'], 'beta_1': [0.85], 'max_iter': [1500]}
#accuracy: 0.572327
#Best parameters found: {'activation': 'tanh', 'beta_1': 0.85, 'hidden_layer_sizes': (75, 75, 75), 'max_iter': 1500, 'solver': 'adam'}
    clf = GridSearchCV(MLPClassifier(), parameters, cv=3) # 0.559748
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    accuracy = float(np.sum(preds == y_test)) / y_test.shape[0]
    print("MLP Accuracy: %f" % (accuracy))
    #print('Best parameters found:', clf.best_params_)

    model_to_set = OneVsRestClassifier(SVC(kernel="rbf"))

    #parameters = { # 0.654088
    #    "estimator__C": [1,2,4,8],
    #    "estimator__kernel": ["poly","rbf"],
    #    "estimator__degree":[1, 2, 3, 4],
    #}

    parameters = { # 0.654088
        "estimator__C": [8],
        "estimator__kernel": ["rbf"],
        "estimator__degree":[1],
    }


    model_tuning = GridSearchCV(model_to_set, param_grid=parameters, scoring='f1_weighted', cv=3)

    model_tuning.fit(x_train, y_train)
    preds = model_tuning.predict(x_test)
    accuracy = float(np.sum(preds == y_test)) / y_test.shape[0]
    print("OneVsRest Accuracy: %f" % (accuracy))
    #print('Best parameters found:\n', model_tuning.best_params_)

    df = pd.DataFrame(embed_dict)

if __name__ == "__main__":
    main()