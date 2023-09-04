import mlflow
from bertopic import BERTopic
from bertmodel import create_topic_model, fitmodel
from data_in import ingest_data, retrain_ingest_data
from preprocessor import clean_data, remove_stopwords_from_comments
from datetime import datetime
import plotly.io as py
import argparse

# Retraining pipeline for dataset

# Training pipeline


def init_pipeline(file_path, n_neighbors, n_components, min_dist, umap_metric, min_cluster_size,
                  cluster_selection_method, prediction_data, hdbscan_metric, reduce_frequent_words, diversity):

    # Data ingestion step
    data = ingest_data(file_path)

    # Data cleaning
    cleaned_data = clean_data(data)

    # Remove stop words
    stopwords_rem = remove_stopwords_from_comments(cleaned_data)

    # Model Creation Step
    model = create_topic_model(n_neighbors, n_components, min_dist, umap_metric, min_cluster_size,
                               cluster_selection_method, prediction_data, hdbscan_metric, reduce_frequent_words,
                               diversity)

    # Model training Step
    topic_model = fitmodel(stopwords_rem, model)

    return topic_model


def init_retrain_pipeline(file_path, n_neighbors, n_components, min_dist, umap_metric, min_cluster_size,
                          cluster_selection_method, prediction_data, hdbscan_metric, reduce_frequent_words, diversity):

    # Data ingestion step
    data = retrain_ingest_data(file_path)

    # Data cleaning
    cleaned_data = clean_data(data)

    # Remove stop words
    stopwords_rem = remove_stopwords_from_comments(cleaned_data)

    # Model Creation Step
    model = create_topic_model(n_neighbors, n_components, min_dist, umap_metric, min_cluster_size,
                               cluster_selection_method, prediction_data, hdbscan_metric, reduce_frequent_words,
                               diversity)

    # Model training Step
    topic_model = fitmodel(stopwords_rem, model)

    return topic_model


class Bert_model(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        # print(context)

        self.model = BERTopic.load(args.model_path)

    def predict(self, context, model_input):
        return self.model.transform(model_input)

    def visualize(self):
        fig = self.model.visualize_topics()
        py.write_image(fig, 'filename.png')
        return fig

    def vis_doc(self, model_input):
        docs = model_input['Comments_no_stop'].values
        labels = self.model.generate_topic_labels(nr_words=2, separator=', ')
        self.model.set_topic_labels(labels)
        fig = self.model.visualize_documents(docs, custom_labels=True)
        py.write_image(fig, 'filename_vis_docs.png')


if __name__ == '__main__':

    print("")

    class FileArgumentParser(argparse.ArgumentParser):
        def convert_arg_line_to_args(self, arg_line):
            return arg_line.split()


    # Create the argument parser
    parser = FileArgumentParser(fromfile_prefix_chars='@')

    # Add your arguments
    parser.add_argument('--data_file_path')
    parser.add_argument('--data_dir_path')
    parser.add_argument('--model_path')
    parser.add_argument('--retrain_new_data', type=bool)
    parser.add_argument('--retrain_old_model', type=bool)
    parser.add_argument('--n_neighbors', type=int)
    parser.add_argument('--n_components', type=int)
    parser.add_argument('--min_dist', type=float)
    parser.add_argument('--umap_metric', type=str)
    parser.add_argument('--min_cluster_size', type=int)
    parser.add_argument('--cluster_selection_method', type=str)
    parser.add_argument('--prediction_data', type=bool)
    parser.add_argument('--hdbscan_metric', type=str)
    # parser.add_argument('--ngram_range', type=tuple)
    parser.add_argument('--reduce_frequent_words', type=bool)
    parser.add_argument('--diversity', type=float)

    # Parse the arguments
    args = parser.parse_args(['@args.txt'])

    # Pipeline initiation
    if args.retrain_new_data is True:
        print("New Data condition")
        if args.data_dir_path is not None:
            print("Data path found")
            topic_model = init_retrain_pipeline(args.data_dir_path, args.n_neighbors, args.n_components, args.min_dist,
                                                args.umap_metric, args.min_cluster_size, args.cluster_selection_method,
                                                args.prediction_data, args.hdbscan_metric, args.reduce_frequent_words,
                                                args.diversity)
        else:
            raise ValueError("data_dir_path cannot be None")
    else:
        print("Conventional")

        topic_model = init_pipeline(args.data_file_path, args.n_neighbors, args.n_components, args.min_dist,
                                    args.umap_metric, args.min_cluster_size, args.cluster_selection_method,
                                    args.prediction_data, args.hdbscan_metric, args.reduce_frequent_words,
                                    args.diversity)

    print("-------Creating Topic Visualization-------")
    fig1 = topic_model.visualize_topics()
    py.write_image(fig1, 'filename.png')

    print("-------Catalouging Topic Frequency-------")
    table1 = topic_model.get_topic_freq()
    table1.to_csv("topic_frequencies.csv", index=False)

    print("-------Creating Topic Visualization-------")
    fig2 = topic_model.visualize_heatmap(n_clusters=True)
    py.write_image(fig2, 'heatmap.png')

    print("-------Creating Document Visualization-------")
    data = retrain_ingest_data(args.data_dir_path)
    cleaned_data = clean_data(data)
    stopwords_rem = remove_stopwords_from_comments(cleaned_data)
    docs = stopwords_rem['Comments_no_stop'].values
    labels = topic_model.generate_topic_labels(nr_words=2, separator=', ')
    topic_model.set_topic_labels(labels)
    fig3 = topic_model.visualize_documents(docs, custom_labels=True)
    py.write_image(fig3, 'Document.png')

    # (topic_distr, topic_token_distr) = topic_model.approximate_distribution(docs, calculate_tokens=True)
    # df = topic_model.visualize_approximate_distribution(docs[1001], topic_token_distr[1001])
    # df

    print("-------Saving Model-------")

    topic_model.save(args.model_path)

    # MLFlow Pipeline setup
    # Replace with your MLflow server URI make sure the server has started
    print("-------Setting up mlflow-------")
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = "BERTopic_model_" + current_datetime
    mlflow.set_experiment(experiment_name)

    # Start an MLflow experiment
    print("-------Staring mlflow experiment-------")
    run_name = "run_" + current_datetime
    with mlflow.start_run(run_name=run_name) as run:

        print("-------Logging Dataset-------")
        # Log dataset
        mlflow.log_artifact(args.data_file_path)

        print("-------Logging Metrics-------")
        # Log dataset
        mlflow.log_artifact('filename.png')
        mlflow.log_artifact("topic_frequencies.csv")
        mlflow.log_artifact("heatmap.png")
        mlflow.log_artifact("Document.png")

        print("-------Logging parameters-------")
        # Log hyperparameters
        mlflow.log_param("umap_n_neighbors", args.n_neighbors)
        mlflow.log_param("umap_n_components", args.n_components)
        mlflow.log_param("umap_min_dist", args.min_dist)
        mlflow.log_param("umap_metric", args.umap_metric)
        mlflow.log_param("hdbscan_min_cluster_size", args.min_cluster_size)
        mlflow.log_param("hdbscan_cluster_selection_method", args.cluster_selection_method)
        mlflow.log_param("hdbscan_prediction_data", args.prediction_data)
        mlflow.log_param("hdbscan_metric", args.hdbscan_metric)
        # mlflow.log_param("vectorizer_ngram_range", args.ngram_range)
        mlflow.log_param("ctfidf_reduce_frequent_words", args.reduce_frequent_words)
        mlflow.log_param("mmr_diversity", args.diversity)

        # Log  Model
        print("-------Logging model-------")
        mlflow.pyfunc.log_model("model",
                                python_model=Bert_model()
                                )

    # Get Run ID
    run_id = run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    print(model_uri)

    print("-------Loading model-------")
    model_test = mlflow.pyfunc.load_model(model_uri)

    print("-------Unwrapping model-------")
    unwrapped = model_test.unwrap_python_model()

    context = None
'''
    print("-------Visualizing topics-------")
    unwrapped.visualize()

    print("-------Logging topics artifacts-------")
    mlflow.log_artifact('filename.png', 'topics')
'''
