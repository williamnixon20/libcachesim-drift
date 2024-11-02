#include <math.h>
#include <xgboost/c_api.h>

#include "GLCacheInternal.h"
#include "obj.h"
#include "utils.h"


#define MAX_LEARNERS 100

typedef struct {
    int index;
    float validation_loss;
} ranking_t;

typedef struct {
    BoosterHandle models[MAX_LEARNERS]; // Array of model handles loaded from saved files
    ranking_t rankings[MAX_LEARNERS]; // Rankings based on validation loss
    int model_count; // Number of models in the ensemble
} ENSEMBLE;

// Global ENSEMBLE instance
ENSEMBLE global_ensemble;

// Initialize the ENSEMBLE
void ENSEMBLE_init(ENSEMBLE *ensemble) {
    ensemble->model_count = 0;
}

// Add a model to the ensemble from a saved file
void ENSEMBLE_add_model_from_file(ENSEMBLE *ensemble, const char *file_path) {
    if (ensemble->model_count < MAX_LEARNERS) {
        BoosterHandle model;
        safe_call(XGBoosterCreate(NULL, 0, &model));
        if (XGBoosterLoadModel(model, file_path) == 0) {
            ensemble->models[ensemble->model_count++] = model;
        } else {
            fprintf(stderr, "Error: Could not load model from %s\n", file_path);
            safe_call(XGBoosterFree(model));
        }
    } else {
        fprintf(stderr, "Error: Maximum number of models reached.\n");
    }
}

// Helper function to evaluate a single model
float evaluate_model(BoosterHandle model, DMatrixHandle eval_dmats[2], const char **eval_names) {
    const char *eval_result;
    safe_call(XGBoosterEvalOneIter(model, 0, eval_dmats, eval_names, 2, &eval_result));
    const char *valid_pos = strstr(eval_result, "valid-rmse:") + 11;
    return strtof(valid_pos, NULL);
}

// Evaluate and rank models in the ensemble
void ENSEMBLE_evaluate_ranking(ENSEMBLE *ensemble, DMatrixHandle eval_dmats[2], const char **eval_names) {
    for (int i = 0; i < ensemble->model_count; ++i) {
        float valid_loss = evaluate_model(ensemble->models[i], eval_dmats, eval_names);
        ensemble->rankings[i].index = i;
        ensemble->rankings[i].validation_loss = valid_loss;
    }

    // Sort the rankings based on validation loss
    for (int i = 0; i < ensemble->model_count - 1; ++i) {
        for (int j = i + 1; j < ensemble->model_count; ++j) {
            if (ensemble->rankings[i].validation_loss > ensemble->rankings[j].validation_loss) {
                ranking_t temp = ensemble->rankings[i];
                ensemble->rankings[i] = ensemble->rankings[j];
                ensemble->rankings[j] = temp;
            }
        }
    }
    // print all model ranking
    for (int i = 0; i < ensemble->model_count; ++i) {
        printf("Model %d: %.4f\n", ensemble->rankings[i].index, ensemble->rankings[i].validation_loss);
    }
}

// Get the best-performing model's BoosterHandle
BoosterHandle ENSEMBLE_get_best_model(ENSEMBLE *ensemble) {
    if (ensemble->model_count == 0) {
        return NULL; // No models available
    }
    int best_model_index = ensemble->rankings[0].index;
    printf("Best model: %d\n", best_model_index);
    return ensemble->models[best_model_index];
}


static void debug_print_feature_matrix(const DMatrixHandle handle, int print_n_row) {
  unsigned long out_len;
  const float *out_data;
  XGDMatrixGetFloatInfo(handle, "label", &out_len, &out_data);

  printf("out_len: %lu\n", out_len);
  for (int i = 0; i < print_n_row; i++) {
    printf("%.4f, ", out_data[i]);
  }
  printf("\n");
}

static void train_xgboost(cache_t *cache) {
  GLCache_params_t *params = (GLCache_params_t *) cache->eviction_params;
  learner_t *learner = &params->learner;

  // if (learner->n_train != 0) {
  //   safe_call(XGBoosterFree(learner->booster));
  //   safe_call(XGDMatrixFree(learner->train_dm));
  //   safe_call(XGDMatrixFree(learner->valid_dm));
  // }

  prepare_training_data(cache);

  DMatrixHandle eval_dmats[2] = {learner->train_dm, learner->valid_dm};
  static const char *eval_names[2] = {"train", "valid"};
  const char *eval_result;
  double train_loss, valid_loss, last_valid_loss = 0;
  int n_stable_iter = 0;

  safe_call(XGBoosterCreate(eval_dmats, 1, &learner->booster));
  safe_call(XGBoosterSetParam(learner->booster, "booster", "gbtree"));
  safe_call(XGBoosterSetParam(learner->booster, "verbosity", "1"));
  safe_call(XGBoosterSetParam(learner->booster, "nthread", "1"));
#if OBJECTIVE == REG
  safe_call(XGBoosterSetParam(learner->booster, "objective", "reg:squarederror"));
#elif OBJECTIVE == LTR
  safe_call(XGBoosterSetParam(learner->booster, "objective", "rank:pairwise"));
#endif

  for (int i = 0; i < N_TRAIN_ITER; ++i) {
    // Update the model performance for each iteration
    safe_call(XGBoosterUpdateOneIter(learner->booster, i, learner->train_dm));
    if (learner->n_valid_samples < 10) continue;
    safe_call(XGBoosterEvalOneIter(learner->booster, i, eval_dmats, eval_names, 2, &eval_result));
#if OBJECTIVE == REG
    const char *train_pos = strstr(eval_result, "train-rmse:") + 11;
    const char *valid_pos = strstr(eval_result, "valid-rmse:") + 11;

    train_loss = strtof(train_pos, NULL);
    valid_loss = strtof(valid_pos, NULL);

    // DEBUG("%.2lf hour, cache size %.2lf MB, iter %d, train loss %.4lf, valid
    // loss %.4lf\n",
    //     (double) params->curr_rtime / 3600.0,
    //     (double) cache->cache_size / 1024.0 / 1024.0,
    //     i, train_loss, valid_loss);

    if (fabs(last_valid_loss - valid_loss) / valid_loss < 0.01) {
      n_stable_iter += 1;
      if (n_stable_iter > 2) {
        break;
      }
    } else {
      n_stable_iter = 0;
    }
    last_valid_loss = valid_loss;
#elif OBJECTIVE == LTR
    char *train_pos = strstr(eval_result, "train-map") + 10;
    char *valid_pos = strstr(eval_result, "valid-map") + 10;
    // DEBUG("%s\n", eval_result);
#else
#error
#endif
  }
#ifndef __APPLE__
  safe_call(XGBoosterBoostedRounds(learner->booster, &learner->n_trees));
#endif
  WARN("EVAL RESULT: %s\n", eval_result);
  WARN(
      "%.2lf hour, cache size %.2lf MB, vtime %ld, train/valid %d/%d samples, "
      "%d trees, "
      "rank intvl %.4lf\n",
      (double)params->curr_rtime / 3600.0, (double)cache->cache_size / 1024.0 / 1024.0, (long)params->curr_vtime,
      (int)learner->n_train_samples, (int)learner->n_valid_samples, learner->n_trees, params->rank_intvl);

  if (cache->should_dump || cache->is_matchmaker) {
    {
      static __thread char s[128];
      snprintf(s, 128, "dump/model_%d.bin", learner->n_train);
      safe_call(XGBoosterSaveModel(learner->booster, s));
      INFO("dump model %s\n", s);
      ENSEMBLE_add_model_from_file(&global_ensemble, s);
      ENSEMBLE_evaluate_ranking(&global_ensemble, eval_dmats, eval_names);
    }
  }
}

bool have_loaded = false;
bool is_first_time = true;

void train(cache_t *cache) {
  if (is_first_time) {
    ENSEMBLE_init(&global_ensemble);
    is_first_time = false;
  }
  GLCache_params_t *params = (GLCache_params_t *)cache->eviction_params;
  learner_t *learner = &params->learner;

  uint64_t start_time = gettime_usec();

  if (cache->should_load_initial_model) {
    if (have_loaded) {
      // printf("Already loaded\n");
      return;
    }
    printf("Loading model\n");
    have_loaded = true;
    static __thread char s[128];
    snprintf(s, 128, cache->initial_model_file, 1);

    safe_call(XGBoosterCreate(NULL, 0, &learner->booster));

    safe_call(XGBoosterLoadModel(learner->booster, s));
    INFO("Load model %s\n", s);
    bst_ulong num_of_features = 0;

    safe_call(XGBoosterGetNumFeature(learner->booster, &num_of_features));
    printf("num_feature: %lu\n", (unsigned long)(num_of_features));

  } else {
    printf("Training XGBOOST\n");
    train_xgboost(cache);
  }

  uint64_t end_time = gettime_usec();
  
  // INFO("training time %.4lf sec\n", (end_time - start_time) / 1000000.0);
  BoosterHandle best_model = ENSEMBLE_get_best_model(&global_ensemble);
  if (best_model != NULL && (cache->should_dump || cache->is_matchmaker)) {
      printf("Setting best model\n");
      learner->booster = best_model;
  }
  params->learner.n_train += 1;
  params->learner.last_train_rtime = params->curr_rtime;
  params->learner.n_train_samples = 0;
  params->learner.n_valid_samples = 0;
}