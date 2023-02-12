[[index]](https://infull.github.io/knowledge-base/index.md)

### pipe_implementation

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline

most_frequent_imputer = SimpleImputer(strategy='most_frequent')
onehot_encoder = OneHotEncoder(handle_unknown='ignore')
vectorizer = TfidfVectorizer()
svd = TruncatedSVD(n_components=2, random_state=42)

pipe1 = make_pipeline(most_frequent_imputer, onehot_encoder, svd)
pipe2 = make_pipeline(vectorizer, svd)

### pipeline_and_ColumnTransformer

most_frequent_imputer = SimpleImputer(strategy='most_frequent')
onehot_encoder = OneHotEncoder(handle_unknown='ignore')
vectorizer = CountVectorizer()
mean_imputer = SimpleImputer(strategy='mean')
svd = TruncatedSVD(n_components=2, random_state=42)

pipe1 = make_pipeline(most_frequent_imputer, onehot_encoder, svd)
pipe2 = make_pipeline(vectorizer, svd)

column_trans = make_column_transformer(
    (pipe1, ['Sex', 'Embarked']),
    (pipe2, 'Name'),
    (mean_imputer, ['Age']),
    ('passthrough', ['Fare', 'SibSp', 'Parch', 'Pclass']))

    ### modeling_pipeline

    pipe1 = make_pipeline(most_frequent_imputer, onehot_encoder, svd)
pipe2 = make_pipeline(vectorizer, svd)

column_trans = make_column_transformer(
    (pipe1, ['Sex', 'Embarked']),
    (pipe2, 'Name'),
    (mean_imputer, ['Age']),
    ('passthrough', ['Fare', 'SibSp', 'Parch', 'Pclass']))

scaler = StandardScaler()
classifier = LogisticRegression(random_state=42)

#### Final Pipeline
pipeline = make_pipeline(column_trans, scaler, classifier)

[[index]](https://infull.github.io/knowledge-base/index.md)