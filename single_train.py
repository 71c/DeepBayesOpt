from utils_train.train_acquisition_function_net import single_train


SAVE_INCREMENTAL_BEST_MODELS = False
USE_CPROFILE = False
MEASURE_TIME = True
VERBOSE = True


if __name__ == "__main__":
    single_train(
        save_incremental_best_models=SAVE_INCREMENTAL_BEST_MODELS,
        use_cprofile=USE_CPROFILE,
        measure_time=MEASURE_TIME,
        verbose=VERBOSE,
    )
