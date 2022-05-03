import time
import optuna

from optuna.integration import AllenNLPExecutor


def objective(trial):
    trial.suggest_categorical('elmo_embedder_do_layer_norm', [0, 1])
    trial.suggest_uniform('elmo_embedder_dropout', 0.0, 0.5)
    elmo_embedder_projection_dim = trial.suggest_categorical(
        'elmo_embedder_projection_dim', [16, 32, 64, 128, 256, 512])

    trial.suggest_categorical('encoder_type', ['gru', 'lstm', 'rnn'])

    trial.suggest_categorical('gru_input_size', [elmo_embedder_projection_dim])
    trial.suggest_categorical('gru_hidden_size', [16, 32, 64, 128, 256, 512])
    trial.suggest_int('gru_num_layers', 1, 6)
    trial.suggest_categorical('gru_bias', [1, 0])
    trial.suggest_uniform('gru_dropout', 0.0, 0.5)
    trial.suggest_categorical('gru_bidirectional', [1, 0])

    trial.suggest_categorical('lstm_input_size', [elmo_embedder_projection_dim])
    trial.suggest_categorical('lstm_hidden_size', [16, 32, 64, 128, 256, 512])
    trial.suggest_int('lstm_num_layers', 1, 6)
    trial.suggest_categorical('lstm_bias', [1, 0])
    trial.suggest_uniform('lstm_dropout', 0.0, 0.5)
    trial.suggest_categorical('lstm_bidirectional', [1, 0])

    trial.suggest_categorical('rnn_input_size', [elmo_embedder_projection_dim])
    trial.suggest_categorical('rnn_hidden_size', [16, 32, 64, 128, 256, 512])
    trial.suggest_int('rnn_num_layers', 1, 6)
    trial.suggest_categorical('rnn_nonlinearity', ['tanh'])
    trial.suggest_categorical('rnn_bias', [1, 0])
    trial.suggest_uniform('rnn_dropout', 0.0, 0.5)
    trial.suggest_categorical('rnn_bidirectional', [1, 0])

    trial.suggest_uniform('learning_rate', 1e-5, 1e-4)
    trial.suggest_categorical('batch_size', [16, 32])

    trial.suggest_categorical('max_length', [64, 128, 256])

    serialization_dir = f"outputs/elmo/{int(time.time())}"

    executor = AllenNLPExecutor(
        trial,
        config_file="configs/elmo.jsonnet",
        serialization_dir=serialization_dir,
        file_friendly_logging=True,
        include_package='classifier'
    )

    return executor.run()


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
