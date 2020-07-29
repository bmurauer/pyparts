# PyParts: Parts for your SciKit pipeline

## How?


```python
pipeline, parameters = sequential('root', [
    union('features', [
        Part('word', TfidfVectorizer(analyzer='word'), {
            'ngram_range': [(1, 1), (2, 2)],
        })
        Part('char', TfidfVectorizer(analyzer='char'), {
            'ngram_range': [(2, 2), (3, 3)],
        })
    ]),
    pick_one('classifier', [
        Part('linearsvm', LinearSVC(), {
            'C': [0.1, 1],
        }),
        Part('mnb', MultinominalNB()), # parameters are optional
    ])
])
grid = GridSearchCV(pipeline, parameters)
grid.fit(X, y)
...
 ```

## But why?
This package aims to improve the coupling between transformers in a scikit
pipeline and their parameters.

Nesting Pipelines can be useful for re-using parts of your code where
you have custom transformers or estimators, but as soon as you start nesting
Pipelines, things become confusing quickly:

```python
pipeline = Pipeline([
    ('features', FeatureUnion([
        ('word_features', Pipeline([
            ('count', CountVectorizer(analyzer='word')),
            ('tfidf', TfidfTransformer()),
        ])),
        ('char_features', Pipeline([
            ('count', CountVectorizer(analyzer='char')),
            ('tfidf', TfidfTransformer()),
        ])),
    ])),
    ('svm', LinearSVC()),
])

params = {
    'features__word_features__count__ngram_range': [(1,1), (2,2), (3,3)],
    'features__char_features__count__ngram_range': [(1,1), (2,2), (3,3)],
}
```


* The keys of your parameters become very long. Not really a problem, but
  annoying.
* re-using the same parts of your pipeline along with their parameters is
  tedious. This is easy with pyparts:

```python
def word_features():
  return sequential('word', [
     Part('count', CountVectorizer(analyzer='word'), {
         'ngram_range': [(1, 1), (2, 2)],
     }),
     Part('tfidf', TfidfTransformer(), {
        'use_idf': [True, False],
     })
  ])

# Later, in the model:
model, parameters = sequential('root', [
    union('features', [
        word_features(),
        char_features(),
    ]),
    Part('svm', LinearSVC()), # parameters are optional
])
```

This way,
 * you define the parameters where they are "used"
 * if you add a parameter later, you can't forget to update any models
defined earlier
