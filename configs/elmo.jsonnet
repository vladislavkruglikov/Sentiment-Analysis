local learning_rate = std.parseJson(std.extVar('learning_rate'));
local batch_size = std.parseInt(std.extVar('batch_size'));
local max_length = std.parseInt(std.extVar('max_length'));
local encoder_type = std.extVar('encoder_type');
local cuda_device = 0;
local num_epochs = 10;


local TRAIN_PATH = 'dataset/train.txt';
local TEST_PATH = 'dataset/test.txt';


local gru_input_size = std.parseInt(std.extVar('gru_input_size'));
local gru_hidden_size = std.parseInt(std.extVar('gru_hidden_size'));
local gru_num_layers = std.parseInt(std.extVar('gru_num_layers'));
local gru_bias = std.parseInt(std.extVar('gru_bias'));
local gru_dropout = std.parseJson(std.extVar('gru_dropout'));
local gru_bidirectional = std.parseInt(std.extVar('gru_bidirectional'));


local gru_encoder = {
  type: 'gru',
  input_size: gru_input_size,
  hidden_size: gru_hidden_size,
  num_layers: gru_num_layers,
  bias: gru_bias,
  dropout: gru_dropout,
  bidirectional: gru_bidirectional,
};


local lstm_input_size = std.parseInt(std.extVar('lstm_input_size'));
local lstm_hidden_size = std.parseInt(std.extVar('lstm_hidden_size'));
local lstm_num_layers = std.parseInt(std.extVar('lstm_num_layers'));
local lstm_bias = std.parseInt(std.extVar('lstm_bias'));
local lstm_dropout = std.parseJson(std.extVar('lstm_dropout'));
local lstm_bidirectional = std.parseInt(std.extVar('lstm_bidirectional'));


local lstm_encoder = {
  type: 'lstm',
  input_size: lstm_input_size,
  hidden_size: lstm_hidden_size,
  num_layers: lstm_num_layers,
  bias: lstm_bias,
  dropout: lstm_dropout,
  bidirectional: lstm_bidirectional,
};


local rnn_input_size = std.parseInt(std.extVar('rnn_input_size'));
local rnn_hidden_size = std.parseInt(std.extVar('rnn_hidden_size'));
local rnn_num_layers = std.parseInt(std.extVar('rnn_num_layers'));
local rnn_nonlinearity = std.extVar('rnn_nonlinearity');
local rnn_bias = std.parseInt(std.extVar('rnn_bias'));
local rnn_dropout = std.parseJson(std.extVar('rnn_dropout'));
local rnn_bidirectional = std.parseInt(std.extVar('rnn_bidirectional'));


local rnn_encoder = {
  type: 'rnn',
  input_size: rnn_input_size,
  hidden_size: rnn_hidden_size,
  num_layers: rnn_num_layers,
  nonlinearity: rnn_nonlinearity,
  bias: rnn_bias,
  dropout: rnn_dropout,
  bidirectional: rnn_bidirectional,
};


local bert_pooler_dropout = std.parseJson(std.extVar('bert_pooler_dropout'));


local get_encoder(type) =
  if type == 'gru' then
    gru_encoder
  else if type == 'lstm' then
    lstm_encoder
  else if type == 'rnn' then
    rnn_encoder;


local elmo_embedder_do_layer_norm = std.parseInt(std.extVar('elmo_embedder_do_layer_norm'));
local elmo_embedder_dropout = std.parseJson(std.extVar('elmo_embedder_dropout'));
local elmo_embedder_projection_dim = std.parseInt(std.extVar('elmo_embedder_projection_dim'));

{
  model: {
    type: 'classifier',
    embedder: {
      token_embedders: {
        tokens: {
          type: 'elmo_token_embedder',
          do_layer_norm: elmo_embedder_do_layer_norm,
          dropout: elmo_embedder_dropout,
          projection_dim: elmo_embedder_projection_dim,
        },
      },
    },
    encoder: get_encoder(encoder_type),
  },
  dataset_reader: {
    type: 'dataset',
    tokenizer: {
      type: 'whitespace',
    },
    token_indexers: {
      tokens: {
        type: 'elmo_characters',
      },
    },
  },
  trainer: {
    num_epochs: num_epochs,
    cuda_device: cuda_device,
    optimizer: {
      type: 'adam',
      lr: learning_rate,
    },
    callbacks: [
      {
        type: 'wandb',
        project: 'Sentiment Analysis',
        entity: 'lildatascientist',
        group: 'ELMo',
        tags: ['ELMo', encoder_type],
      },
    ],
    validation_metric: '+accuracy',
  },
  train_data_path: TRAIN_PATH,
  validation_data_path: TEST_PATH,
  data_loader: {
    batch_size: batch_size,
    shuffle: true,
  },
}
