#include <math.h>
#include <xgboost/c_api.h>

#include "GLCacheInternal.h"
#include "obj.h"
#include "utils.h"

#define MAX_LEARNERS 5
#define NUM_TREES 20
#define MAX_LEAFS 100

typedef struct {
  int index;
  float validation_loss;
} ranking_t;

typedef struct {
  BoosterHandle models[MAX_LEARNERS];          // Array of model handles
  feature_t *train_x[MAX_LEARNERS];            // Training data features for each model
  pred_t *train_y[MAX_LEARNERS];               // Training data labels for each model
  int sample_counts[MAX_LEARNERS];             // Number of samples for each model's training data
  ranking_t concept_rankings[MAX_LEARNERS];    // Rankings based on validation loss (concept)
  ranking_t covariate_rankings[MAX_LEARNERS];  // Rankings based on covariate similarity
  BoosterHandle global_tree;                   // Global model trained on combined data
  DMatrixHandle global_data;                   // Combined data matrix from all models
  int model_count;                             // Number of models in the ensemble
  int num_trees;
  int leaf_distribution[MAX_LEARNERS][NUM_TREES][MAX_LEAFS];
} ENSEMBLE;

// Global ENSEMBLE instance
ENSEMBLE global_ensemble;

// Initialize the ENSEMBLE
void ENSEMBLE_init(ENSEMBLE *ensemble) { ensemble->model_count = 0; }

// Helper function to evaluate a single model (concept ranking)
float evaluate_model(BoosterHandle model, DMatrixHandle eval_dmats[2], const char **eval_names) {
  const char *eval_result;
  safe_call(XGBoosterEvalOneIter(model, 0, eval_dmats, eval_names, 2, &eval_result));
  const char *valid_pos = strstr(eval_result, "valid-rmse:") + 11;
  return strtof(valid_pos, NULL);
}
void ENSEMBLE_remove_lowest_ranked_model(ENSEMBLE *ensemble) {
  if (ensemble->model_count == 0) return;

  int lowest_ranked_index = 0;
  for (int i = 1; i < ensemble->model_count; ++i) {
    if (ensemble->concept_rankings[i].validation_loss >
        ensemble->concept_rankings[lowest_ranked_index].validation_loss) {
      lowest_ranked_index = i;
    }
  }

  safe_call(XGBoosterFree(ensemble->models[lowest_ranked_index]));
  my_free(sizeof(feature_t) * ensemble->sample_counts[lowest_ranked_index], ensemble->train_x[lowest_ranked_index]);
  my_free(sizeof(pred_t) * ensemble->sample_counts[lowest_ranked_index], ensemble->train_y[lowest_ranked_index]);

  for (int i = lowest_ranked_index; i < ensemble->model_count - 1; ++i) {
    ensemble->models[i] = ensemble->models[i + 1];
    ensemble->train_x[i] = ensemble->train_x[i + 1];
    ensemble->train_y[i] = ensemble->train_y[i + 1];
    ensemble->sample_counts[i] = ensemble->sample_counts[i + 1];
    ensemble->concept_rankings[i] = ensemble->concept_rankings[i + 1];
    ensemble->covariate_rankings[i] = ensemble->covariate_rankings[i + 1];
  }

  ensemble->model_count--;
}

// Evaluate and rank models in the ensemble based on concept ranking (validation loss)
void ENSEMBLE_evaluate_concept_ranking(ENSEMBLE *ensemble, DMatrixHandle eval_dmats[2], const char **eval_names) {
  for (int i = 0; i < ensemble->model_count; ++i) {
    float valid_loss = evaluate_model(ensemble->models[i], eval_dmats, eval_names);
    ensemble->concept_rankings[i].index = i;
    ensemble->concept_rankings[i].validation_loss = valid_loss;
  }

  // Sort the concept rankings based on validation loss
  for (int i = 0; i < ensemble->model_count - 1; ++i) {
    for (int j = i + 1; j < ensemble->model_count; ++j) {
      if (ensemble->concept_rankings[i].validation_loss > ensemble->concept_rankings[j].validation_loss) {
        ranking_t temp = ensemble->concept_rankings[i];
        ensemble->concept_rankings[i] = ensemble->concept_rankings[j];
        ensemble->concept_rankings[j] = temp;
      }
    }
  }
}

void clear_global_leaf_distribution(ENSEMBLE *ensemble) {
  for (int i = 0; i < ensemble->model_count; ++i) {
    for (int j = 0; j < NUM_TREES; ++j) {
      for (int k = 0; k < MAX_LEAFS; ++k) {
        ensemble->leaf_distribution[i][j][k] = 0;
      }
    }
  }
}

void ENSEMBLE_train_global_tree(ENSEMBLE *ensemble, int n_features) {
  if (ensemble->model_count == 0) {
    // fprintf(stderr, "Error: No models in the ensemble to train a global tree.\n");
    return;
  }

  if (ensemble->global_tree) {
    safe_call(XGBoosterFree(ensemble->global_tree));
  }

  int total_samples = 0;
  for (int i = 0; i < ensemble->model_count; ++i) {
    total_samples += ensemble->sample_counts[i];
  }

  feature_t *combined_x = my_malloc_n(feature_t, total_samples * n_features);
  pred_t *combined_y = my_malloc_n(pred_t, total_samples);

  int offset = 0;
  for (int i = 0; i < ensemble->model_count; ++i) {
    memcpy(&combined_x[offset * n_features], ensemble->train_x[i],
           ensemble->sample_counts[i] * n_features * sizeof(feature_t));
    memcpy(&combined_y[offset], ensemble->train_y[i], ensemble->sample_counts[i] * sizeof(pred_t));
    offset += ensemble->sample_counts[i];
  }
  safe_call(XGDMatrixCreateFromMat(combined_x, total_samples, n_features, -2, &ensemble->global_data));
  safe_call(XGDMatrixSetFloatInfo(ensemble->global_data, "label", combined_y, total_samples));

  DMatrixHandle eval_dmats[1] = {ensemble->global_data};
  const char *eval_names[1] = {"global"};

  safe_call(XGBoosterCreate(eval_dmats, 1, &ensemble->global_tree));
  safe_call(XGBoosterSetParam(ensemble->global_tree, "booster", "gbtree"));
  safe_call(XGBoosterSetParam(ensemble->global_tree, "verbosity", "1"));
  safe_call(XGBoosterSetParam(ensemble->global_tree, "nthread", "1"));
  // Set max depth to 6 for now
#if OBJECTIVE == REG
  safe_call(XGBoosterSetParam(ensemble->global_tree, "objective", "reg:squarederror"));
#endif

  for (int i = 0; i < N_TRAIN_ITER; ++i) {
    safe_call(XGBoosterUpdateOneIter(ensemble->global_tree, i, ensemble->global_data));
  }

  INFO("Global model trained on combined data from all models.\n");

  my_free(sizeof(feature_t) * total_samples * n_features, combined_x);
  my_free(sizeof(pred_t) * total_samples, combined_y);
  safe_call(XGDMatrixFree(ensemble->global_data));

  clear_global_leaf_distribution(ensemble);

  // Now classify each model's data using the trained global tree and populate leaf distribution
  for (int model_idx = 0; model_idx < ensemble->model_count; ++model_idx) {
    // Create DMatrix from the model's training data
    DMatrixHandle model_data;
    safe_call(XGDMatrixCreateFromMat(ensemble->train_x[model_idx], ensemble->sample_counts[model_idx], n_features, -2,
                                     &model_data));
    safe_call(
        XGDMatrixSetFloatInfo(model_data, "label", ensemble->train_y[model_idx], ensemble->sample_counts[model_idx]));

    char const config[] =
        "{\"type\": 6, \"training\": false, \"iteration_begin\": 0, \"iteration_end\": 0, \"strict_shape\": true}";

    // Variables for prediction results
    bst_ulong const *out_shape = NULL;
    bst_ulong out_dim = 0;
    float const *out_result = NULL;

    // Make prediction with the global tree, outputting leaf indices
    safe_call(
        XGBoosterPredictFromDMatrix(ensemble->global_tree, model_data, config, &out_shape, &out_dim, &out_result));

    // Process the leaf indices for each data point
    for (bst_ulong i = 0; i < out_shape[0]; ++i) {    // Loop over each sample
      for (bst_ulong j = 0; j < out_shape[1]; ++j) {  // Loop over each tree
        int tree_idx = j;
        int leaf_idx = (int)out_result[i * out_shape[1] + j];
        // printf("Model %d, Tree %d, Leaf %d\n", model_idx, tree_idx, leaf_idx);
        // Update leaf_distribution matrix
        ensemble->leaf_distribution[model_idx][tree_idx][leaf_idx]++;
      }
    }

    // // Clean up model_data DMatrix
    safe_call(XGDMatrixFree(model_data));
  }
}

// Add a model to the ensemble from a saved file, along with training data
void ENSEMBLE_add_model_from_file(ENSEMBLE *ensemble, const char *file_path, feature_t *train_x, pred_t *train_y,
                                  int n_samples, int n_features) {
  // If the ensemble is full, remove the lowest-ranked model
  if (ensemble->model_count >= MAX_LEARNERS) {
    ENSEMBLE_remove_lowest_ranked_model(ensemble);
  }

  // Load model from file
  BoosterHandle model;
  safe_call(XGBoosterCreate(NULL, 0, &model));
  if (XGBoosterLoadModel(model, file_path) == 0) {
    // // Make deep copies of train_x and train_y to store them in the ensemble
    feature_t *train_x_copy = my_malloc_n(feature_t, n_samples * n_features);
    memset(train_x_copy, 0, sizeof(feature_t) * n_samples * n_features);
    memcpy(train_x_copy, train_x, n_samples * n_features * sizeof(feature_t));

    pred_t *train_y_copy = my_malloc_n(pred_t, n_samples);
    memset(train_y_copy, 0, sizeof(feature_t) * n_samples);
    memcpy(train_y_copy, train_y, n_samples * sizeof(pred_t));

    // Add the model, training data copies, and sample count to the ensemble
    int index = ensemble->model_count;
    ensemble->models[index] = model;
    ensemble->train_x[index] = train_x_copy;
    ensemble->train_y[index] = train_y_copy;
    ensemble->sample_counts[index] = n_samples;
    ensemble->model_count++;
    printf("Model loaded from %s and added to the ensemble.\n", file_path);
  } else {
    fprintf(stderr, "Error: Could not load model from %s\n", file_path);
    safe_call(XGBoosterFree(model));
  }
  ENSEMBLE_train_global_tree(ensemble, n_features);
}

// Placeholder for covariate ranking method
void evaluate_covariate_ranking(ENSEMBLE *ensemble, DMatrixHandle inference_data) {
  if (ensemble->model_count == 0) {
    fprintf(stderr, "Error: No models in the ensemble.\n");
    return;
  }

  // Initialize scores array for each model
  int model_scores[MAX_LEARNERS] = {0};

  // Configuration for leaf prediction
  char const config[] =
      "{\"type\": 6, \"training\": false, \"iteration_begin\": 0, \"iteration_end\": 0, \"strict_shape\": true}";

  // Variables for prediction results
  bst_ulong const *out_shape = NULL;
  bst_ulong out_dim = 0;
  float const *out_result = NULL;

  // Run prediction to get leaf indices for each sample in inference_data
  safe_call(
      XGBoosterPredictFromDMatrix(ensemble->global_tree, inference_data, config, &out_shape, &out_dim, &out_result));

  // Check output shape to ensure compatibility
  bst_ulong n_samples = out_shape[0];
  bst_ulong n_trees = out_shape[1];

  // Ensure that output dimensions match expectations
  if (n_trees != NUM_TREES) {
    fprintf(stderr, "Warning: Number of trees in prediction (%lu) does not match expected NUM_TREES (%d).\n", n_trees,
            NUM_TREES);
  }

  // Process leaf indices for each sample and update model scores
  for (bst_ulong sample_idx = 0; sample_idx < n_samples; ++sample_idx) {
    for (bst_ulong tree_idx = 0; tree_idx < n_trees; ++tree_idx) {
      int leaf_idx = (int)out_result[sample_idx * n_trees + tree_idx];

      // Add to model scores based on `leaf_distribution`
      for (int model_idx = 0; model_idx < ensemble->model_count; ++model_idx) {
        model_scores[model_idx] += ensemble->leaf_distribution[model_idx][tree_idx][leaf_idx];
      }
    }
  }

  // Update the covariate ranking based on scores
  for (int i = 0; i < ensemble->model_count; ++i) {
    ensemble->covariate_rankings[i].index = i;
    ensemble->covariate_rankings[i].validation_loss = (float)model_scores[i];  // Directly use scores as ranking metric
  }

  // Sort the covariate rankings based on scores
  for (int i = 0; i < ensemble->model_count - 1; ++i) {
    for (int j = i + 1; j < ensemble->model_count; ++j) {
      if (ensemble->covariate_rankings[i].validation_loss < ensemble->covariate_rankings[j].validation_loss) {
        ranking_t temp = ensemble->covariate_rankings[i];
        ensemble->covariate_rankings[i] = ensemble->covariate_rankings[j];
        ensemble->covariate_rankings[j] = temp;
      }
    }
  }
}

// Get the best-performing model using Borda count
BoosterHandle ENSEMBLE_get_best_model(ENSEMBLE *ensemble) {
  if (ensemble->model_count == 0) {
    return NULL;  // No models available
  }

  // Borda count: assign points based on ranking positions
  int scores[MAX_LEARNERS] = {0};
  for (int i = 0; i < ensemble->model_count; ++i) {
    // printf("Score place %d, Concept Model: %d (%f), Covariate Model: %d (%f)\n", i,
    // ensemble->concept_rankings[i].index,
    //        ensemble->concept_rankings[i].validation_loss, ensemble->covariate_rankings[i].index,
    //        ensemble->covariate_rankings[i].validation_loss);
    scores[ensemble->concept_rankings[i].index] += ensemble->model_count - i;
    scores[ensemble->covariate_rankings[i].index] += ensemble->model_count - i;
  }
  // Find the model with the highest Borda count score
  int best_model_index = 0;
  for (int i = 1; i < ensemble->model_count; ++i) {
    // printf("Model %d: Final Score: %d\n", i, scores[i]);
    if (scores[i] > scores[best_model_index]) {
      best_model_index = i;
    }
  }

  // printf("Best model by Borda count: %d\n", best_model_index);
  return ensemble->models[best_model_index];
}

static void train_xgboost(cache_t *cache) {
  GLCache_params_t *params = (GLCache_params_t *)cache->eviction_params;
  learner_t *learner = &params->learner;

  // if (learner->n_train != 0) {
  //   safe_call(XGBoosterFree(learner->booster));
  //   safe_call(XGDMatrixFree(learner->train_dm));
  //   safe_call(XGDMatrixFree(learner->valid_dm));
  // }
  safe_call(XGBoosterCreate(NULL, 0, &learner->booster));

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
  printf("EVAL RESULT: %s\n", eval_result);
  printf(
      "%.2lf hour, cache size %.2lf MB, vtime %ld, train/valid %d/%d samples, "
      "%d trees, "
      "rank intvl %.4lf\n",
      (double)params->curr_rtime / 3600.0, (double)cache->cache_size / 1024.0 / 1024.0, (long)params->curr_vtime,
      (int)learner->n_train_samples, (int)learner->n_valid_samples, learner->n_trees, params->rank_intvl);
  // if global tree exist, try to eval
  if (global_ensemble.global_tree) {
    DMatrixHandle global_dmats[1] = {learner->valid_dm};
    const char *global_eval_names[1] = {"valid"};
    safe_call(XGBoosterEvalOneIter(global_ensemble.global_tree, 0, global_dmats, global_eval_names, 1, &eval_result));
    printf("Global model eval: %s\n", eval_result);
  }

  if (cache->should_dump || cache->is_matchmaker) {
    {
      static __thread char s[128];
      snprintf(s, 128, "dump/model_%d.bin", learner->n_train);
      safe_call(XGBoosterSaveModel(learner->booster, s));
      printf("dump model %s\n", s);
      // for (int m = 0; m < 100; m++) {
      //   printf("BEFORE: %f: ", learner->train_y[m]);
      //   for (int n = 0; n < learner->n_feature; n++) {
      //     printf("%6f,", learner->train_x[learner->n_feature * m + n]);
      //   }
      //   printf("\n");
      // }
      ENSEMBLE_add_model_from_file(&global_ensemble, s, learner->train_x, learner->train_y, learner->n_train_samples,
                                   learner->n_feature);
      ENSEMBLE_evaluate_concept_ranking(&global_ensemble, eval_dmats, eval_names);
    }
  }
}

void proc_rank_best_model(cache_t *cache) {
  if (!cache->is_matchmaker) {
    return;
  }
  // rank covariate
  GLCache_params_t *params = (GLCache_params_t *)cache->eviction_params;
  learner_t *learner = &params->learner;

  evaluate_covariate_ranking(&global_ensemble, learner->inf_dm);

  BoosterHandle best_model = ENSEMBLE_get_best_model(&global_ensemble);
  if (best_model != NULL) {
    learner->booster = best_model;
  }
}

bool have_loaded = false;
bool is_first_time = true;

// Flow:
// 1. Upon training, will create new model. If matchmaker, will also create global model.
// 2. On inference, will select most relevant model and use it by calling proc_rank_best_model.
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
    bst_ulong num_of_features = 0;

    safe_call(XGBoosterGetNumFeature(learner->booster, &num_of_features));

  } else {
    printf("Training XGBOOST\n");
    train_xgboost(cache);
  }

  uint64_t end_time = gettime_usec();

  // INFO("training time %.4lf sec\n", (end_time - start_time) / 1000000.0);
  // BoosterHandle best_model = ENSEMBLE_get_best_model(&global_ensemble);
  // if (best_model != NULL && (cache->should_dump || cache->is_matchmaker)) {
  //   printf("Setting best model\n");
  //   learner->booster = best_model;
  // }
  params->learner.n_train += 1;
  params->learner.last_train_rtime = params->curr_rtime;
  params->learner.n_train_samples = 0;
  params->learner.n_valid_samples = 0;
}