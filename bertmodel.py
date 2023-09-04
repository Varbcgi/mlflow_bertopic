from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired,PartOfSpeech,MaximalMarginalRelevance
from bertopic import BERTopic

def create_topic_model(n_neighbors, n_components, min_dist, umap_metric, min_cluster_size, cluster_selection_method, prediction_data, hdbscan_metric,reduce_frequent_words,diversity):
    # Step 1 - Extract embeddings
    embedding_model = SentenceTransformer("distiluse-base-multilingual-cased-v1")

    # Step 2 - Reduce dimensionality
    umap_model = UMAP(n_neighbors=n_neighbors, n_components=n_components, min_dist=min_dist, metric=umap_metric)

    # Step 3 - Cluster reduced embeddings
    hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, metric=hdbscan_metric, cluster_selection_method=cluster_selection_method, prediction_data=prediction_data)

    # Step 4 - Tokenize topics
    vectorizer_model = CountVectorizer(ngram_range=(1,4))

    # Step 5 - Create topic representation
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=reduce_frequent_words)
    mmr = MaximalMarginalRelevance(diversity=diversity)
   # pos = PartOfSpeech("xx_ent_wiki_sm")
    keyBert = KeyBERTInspired()
    representation_models = [mmr,keyBert]


    # All steps together
    topic_model = BERTopic(
      embedding_model=embedding_model,    # Step 1 - Extract embeddings
      umap_model=umap_model,              # Step 2 - Reduce dimensionality
      hdbscan_model=hdbscan_model,        # Step 3 - Cluster reduced embeddings
      vectorizer_model=vectorizer_model,  # Step 4 - Tokenize topics
      ctfidf_model=ctfidf_model,          # Step 5 - Extract topic words
      # representation_model=representation_models,
      nr_topics='auto'
    )
    return topic_model

def fitmodel(clean_df ,topic_model ):
    docs = clean_df['Comments_no_stop'].values
    topics, probs = topic_model.fit_transform(docs)
    return topic_model