#include <float.h>
#include <stddef.h>

#include "widereach.h"
#include "helper.h"
#include <gsl/gsl_linalg.h>

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
  //printf("dimension = %ld\n", env->samples->dimension);

  samples_t *samples = env->samples;
  size_t dimension = samples->dimension;
  double best_value = -DBL_MAX;
  double *best_hyperplane = CALLOC(dimension+1, double);
  for (int k = 0; k < rnd_trials; k++) {
    //double *hyperplane = random_hyperplane_unbiased(dimension);
    double *hyperplane = CALLOC(dimension+1, double);
    random_unit_vector(dimension, hyperplane);
    hyperplane[dimension] = 0;
    double value = hyperplane_to_solution(hyperplane, NULL, env);
     if (value > best_value) {
      best_value = value;
      for(int i = 0; i < dimension; i++) {
	best_hyperplane[i] = hyperplane[i];
      }
     }
    free(hyperplane);
  }
  //printf("best random hyperplane obj = %g\n", best_value);
  best_hyperplane[dimension] = 0;
  return best_hyperplane;
}

int *random_indices(int n, int N) {
  //generates n unique random numbers from 0 to N-1
  if(n > N) return NULL;
  int *pts = CALLOC(n, int);
  for(int i = 0; i < n; i++) {
    int repeat, newpt;
    do {
      repeat = 0;
      newpt = rand() % N;
      for(int j = 0; j < i; j++) {
	if(pts[j] == newpt) {
	  repeat = 1;
	  break;
	}
      }
    } while(repeat);
    pts[i] = newpt;
  }
  return pts;
}

gsl_vector *best_random_hyperplane_projection(int initial, env_t *env) {
  //tries N random projections and returns the best resulting hyperplane
  params_t *params = env->params;
  int rnd_trials = initial ? params->rnd_trials : params->rnd_trials_cont;

  //int rnd_trials = 1e6;

  samples_t *samples = env->samples;
  vsamples_t *vs = samples_to_vec(samples);
  size_t d = samples->dimension;
  double best_obj = -1e101;
  gsl_vector *best_w = gsl_vector_alloc(d);
  int n = samples_total(env->samples);
  for(int k = 0; k < rnd_trials; k++) {
    int *pts = random_indices(d-1, n);
    gsl_matrix *A = gsl_matrix_alloc(d, d-1);
    for(int i = 0; i < d-1; i++) {
      int idx = pts[i];
      int class = idx >= vs->count[0] ? 1 : 0;
      if(class == 1)
	idx -= vs->count[0];
      gsl_vector *s = vs->samples[class][idx];
      gsl_matrix_set_col(A, i, s);
    }
    gsl_matrix *T = gsl_matrix_alloc(d-1, d-1);
    gsl_linalg_QR_decomp_r(A, T);
    //final column of Q is an ONB for the null space of A^T, assuming A had full rank
    gsl_matrix *Q = gsl_matrix_alloc(d, d);
    gsl_matrix *R = gsl_matrix_alloc(d-1, d-1);
    gsl_linalg_QR_unpack_r(A, T, Q, R);
    
    gsl_vector *v = gsl_vector_alloc(d);
    gsl_matrix_get_col(v, Q, d-1);

    gsl_matrix_free(A);
    gsl_matrix_free(T);
    gsl_matrix_free(Q);
    gsl_matrix_free(R);

    class_res res = classify(env, v);
    if(res.obj > best_obj) {
      printf("best obj = %g\n", best_obj);
      best_obj = res.obj;
      gsl_vector_memcpy(best_w, v);
    }
    gsl_vector_free(v);
    free_class_res(&res);
    free(pts);    
  }
  vsamples_free(vs);
  printf("best hyperplane has obj %g\n", best_obj);
  return best_w;
}

gsl_vector *approx_within_cone(env_t *env, hplane_data *w) {
  //finds a good hyperplane in the cone containing the feasible hyperplane w
  //assumes closed version
  
}
