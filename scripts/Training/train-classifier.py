import os
from pathlib import Path
import pickle
import numpy as np
from deepgesture.config import Config
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV

def main():
    suturing_processed_dir = Path(os.getenv("SUTURING_PROCESSED_DIR"))
    trained_models_dir = suturing_processed_dir / "models"
    pickle_path = os.path.join(trained_models_dir, "embedding_dict.pkl")
    fileo = open(pickle_path, 'rb')
    embed_dict = pickle.load(fileo)
    embeddings, gestures = np.array(embed_dict['embeddings']), np.array(embed_dict['gesture'])
    #print(embed_dict.keys())
    #print(len(embeddings))
    #print(len(gestures))
    x_train, x_test, y_train, y_test = train_test_split(embeddings, gestures, test_size=0.2, random_state=12499)
    x_train = np.squeeze(x_train)
    x_test = np.squeeze(x_test)
    #print(len(x_train))
    #print(len(x_test))
    #print(len(y_train))
    #print(len(y_test))
    #print(x_train.shape)
    #print(y_train.shape)
    xg_cl = xgb.XGBClassifier()
    #xg_cl.fit(x_train, y_train)
    #preds = xg_cl.predict(x_test)
    # TODO: look at code in TestEmbeddings.py to get XGBoost to work for comparison
    parameters = {'solver': ['adam'], 'activation': ['identity', 'logistic', 'tanh', 'relu'], 'max_iter': [1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000], 'random_state':[0,1,2,3,4,5,6,7,8,9]}
    clf = GridSearchCV(MLPClassifier(), parameters, cv=3) # 0.559748
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    accuracy = float(np.sum(preds == y_test)) / y_test.shape[0]
    print("accuracy: %f" % (accuracy))

if __name__ == "__main__":
    main()