train_params = {
    'max_epoch': 1000,
    'batch_size': 64,
    'atk': 'nrnd',
    'isd': 1,
    'label': 'all',
    'cnn_model': 'cnn_rp',
    # 'cnn_model': 'cnn_spp',
    # 'cnn_model': 'cnn_lfr'
}

test_params = {
    'atk': 'nrnd',
    'isd': 1,
    'label': 'all',
    # 'cnn_model': 'cnn_rp',
    'cnn_model': 'cnn_spp',
    # 'cnn_model': 'cnn_lfr'
}
