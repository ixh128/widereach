#include <float.h>
#include <stddef.h>

#include "widereach.h"
#include "helper.h"

void copy_hyperplane(size_t dimension, double *dest, double *src) {
    for (size_t i = 0; i <= dimension; i++) {
        dest[i] = src[i];
    }
}


double *random_hyperplane(size_t dimension) {
    double *w = CALLOC(dimension + 1, double);
    random_unit_vector(dimension, w);
    double *origin = random_point(dimension);
    w[dimension] = 0.;
    for (size_t i = 0; i < dimension; i++) {
        w[dimension] -= w[i] * origin[i];
    }
    free(origin);
    return w;
}


double *best_random_hyperplane(int initial, env_t *env) {
    params_t *params = env->params;
    int rnd_trials = initial ? params->rnd_trials : params->rnd_trials_cont;
    if (!rnd_trials) {
        return NULL;
    }
    
    samples_t *samples = env->samples;
    size_t dimension = samples->dimension;
    double best_value = -DBL_MAX;
    double *best_hyperplane = CALLOC(dimension + 1, double);
    for (int k = 0; k < rnd_trials; k++) {
        double *hyperplane = random_hyperplane(dimension);
        double value = hyperplane_to_solution(hyperplane, NULL, env);
        /*for (size_t t = 0; t <= dimension; t++) {
          glp_printf("%g ", hyperplane[t]);
        }
        glp_printf(" -> %g\n", value);*/
        if (value > best_value) {
            best_value = value;
            copy_hyperplane(dimension, best_hyperplane, hyperplane);
            // glp_printf("%i\t%g\n", k, best_value);
        }
        free(hyperplane);
    }
    return best_hyperplane;
}

double *random_hyperplane_unbiased(size_t dimension) {
    double *w = CALLOC(dimension + 1, double);
    random_unit_vector(dimension, w);
    w[dimension] = 0;
    return w;
}

double *best_random_hyperplane_unbiased(int initial, env_t *env) {
  params_t *params = env->params;
  int rnd_trials = initial ? params->rnd_trials : params->rnd_trials_cont;
  if (!rnd_trials) {
    return NULL;
  }

  printf("dimension = %ld\n", env->samples->dimension);

  samples_t *samples = env->samples;
  size_t dimension = samples->dimension;
  double best_value = -DBL_MAX;
  double *best_hyperplane = CALLOC(dimension, double);
  for (int k = 0; k < rnd_trials; k++) {
    //double *hyperplane = random_hyperplane_unbiased(dimension);
    double *hyperplane = CALLOC(dimension, double);
    random_unit_vector(dimension, hyperplane);
    double value = hyperplane_to_solution(hyperplane, NULL, env);
     if (value > best_value) {
      best_value = value;
      for(int i = 0; i < dimension; i++) {
	best_hyperplane[i] = hyperplane[i];
      }
     }
    free(hyperplane);
  }
  return best_hyperplane;
}
