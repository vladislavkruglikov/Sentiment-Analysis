local learning_rate = std.parseJson(std.extVar('learning_rate'));
local batch_size = std.parseInt(std.extVar('batch_size'));
local max_length = std.parseInt(std.extVar('max_length'));
local pretrained_model = std.extVar('pretrained_model');
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


local bert_encoder = {
  type: 'bert_pooler',
  dropout: bert_pooler_dropout,
  pretrained_model: pretrained_model,
};


local get_encoder(type) =
  if type == 'gru' then
    gru_encoder
  else if type == 'lstm' then
    lstm_encoder
  else if type == 'rnn' then
    rnn_encoder
  else if type == 'bert_pooler' then
    bert_encoder;


{
  model: {
    type: 'classifier',
    embedder: {
      token_embedders: {
        tokens: {
          type: 'pretrained_transformer',
          model_name: pretrained_model,
          max_length: max_length,
        },
      },
    },
    encoder: get_encoder(encoder_type),
  },
  dataset_reader: {
    type: 'dataset',
    tokenizer: {
      type: 'pretrained_transformer',
      model_name: pretrained_model,
      max_length: max_length,
    },
    token_indexers: {
      tokens: {
        type: 'pretrained_transformer',
        model_name: pretrained_model,
        max_length: max_length,
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
        group: 'BERT',
        tags: [pretrained_model, encoder_type],
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
