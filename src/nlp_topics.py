"""
Helpers for topic modeling on news headlines.

We use:
- CountVectorizer  -> convert text to numbers (word counts)
- LatentDirichletAllocation (LDA) -> find topics
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def build_vectorizer(
    max_df: float = 0.95,
    min_df: int = 20,
    stop_words: str | None = "english",
) -> CountVectorizer:
    """
    Create a CountVectorizer with common defaults for news headlines.

    Parameters
    ----------
    max_df : float
        Ignore words that appear in more than this fraction of documents.
    min_df : int
        Ignore words that appear in fewer than this many documents.
    stop_words : str or None
        Stop word setting passed to CountVectorizer. "english" is usually fine.

    Returns
    -------
    CountVectorizer
    """
    return CountVectorizer(
        max_df=max_df,
        min_df=min_df,
        stop_words=stop_words,
    )


def fit_lda(
    X,
    n_topics: int = 5,
    random_state: int = 42,
) -> LatentDirichletAllocation:
    """
    Fit an LDA topic model on the document-term matrix X.

    Parameters
    ----------
    X : sparse matrix
        Output of CountVectorizer.fit_transform.
    n_topics : int
        Number of topics to learn.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    LatentDirichletAllocation
    """
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=random_state,
        learning_method="batch",
    )
    lda.fit(X)
    return lda


def extract_topic_words(
    lda_model: LatentDirichletAllocation,
    feature_names: List[str],
    n_top_words: int = 10,
) -> List[List[str]]:
    """
    Extract top words for each topic from a fitted LDA model.

    Parameters
    ----------
    lda_model : LatentDirichletAllocation
        Fitted LDA model.
    feature_names : list of str
        Vocabulary from the vectorizer (vectorizer.get_feature_names_out()).
    n_top_words : int
        Number of top words to return per topic.

    Returns
    -------
    List[List[str]]
        Outer list: one item per topic.
        Inner list: top words for that topic.
    """
    topics: List[List[str]] = []

    for topic_idx, topic in enumerate(lda_model.components_):
        top_indices = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_indices]
        topics.append(top_words)

    return topics


def topic_model_from_texts(
    texts: Iterable[str],
    n_topics: int = 5,
    n_top_words: int = 10,
    max_df: float = 0.95,
    min_df: int = 20,
    stop_words: str | None = "english",
) -> Tuple[LatentDirichletAllocation, CountVectorizer, List[List[str]]]:
    """
    Convenience function:
    - builds vectorizer
    - vectorizes texts
    - fits LDA
    - extracts top words

    This is useful both in notebooks and scripts.

    Parameters
    ----------
    texts : iterable of str
        Collection of documents (e.g. headlines).
    n_topics : int
        Number of topics.
    n_top_words : int
        Number of top words per topic.
    max_df, min_df, stop_words :
        Passed to build_vectorizer().

    Returns
    -------
    (lda_model, vectorizer, topics)
    - lda_model : fitted LDA instance
    - vectorizer : CountVectorizer used
    - topics : list of list of top words
    """
    vectorizer = build_vectorizer(
        max_df=max_df,
        min_df=min_df,
        stop_words=stop_words,
    )
    X = vectorizer.fit_transform(texts)

    lda_model = fit_lda(X, n_topics=n_topics)
    feature_names = list(vectorizer.get_feature_names_out())
    topics = extract_topic_words(lda_model, feature_names, n_top_words=n_top_words)

    return lda_model, vectorizer, topics
