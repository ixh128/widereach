#include "general.h"
#include "helper.h"

#include <math.h>
#include <assert.h>

sample_locator_t random_sample(samples_t *samples) {
  size_t n = samples_total(samples);
  int i = rand() % n;
  int P = positives(samples);
  if(i < P) {
    return (sample_locator_t) {.class = 1, .index = i};
  } else {
    return (sample_locator_t) {.class = 0, .index = i - P};
  }
}

sample_locator_t *choose_random_samples(samples_t *samples, size_t num) {
  //generate num random sample locators
  sample_locator_t *inds = CALLOC(num, sample_locator_t);

  for(int i = 0; i < num; i++) {
    inds[i] = random_sample(samples);
    int good = 1;
    for(int j = 0; j < i; j++) {
      if(inds[j].class == inds[i].class && inds[j].index == inds[i].index) {
	good = 0;
	break;
      }
    }
    if(!good) i--;
  }
  return inds;
}

gsl_matrix *random_sample_matrix(env_t *env) {
  vsamples_t *vs = env->vsamples;
  size_t d = vs->dimension;
  gsl_matrix *A = gsl_matrix_calloc(d, d-1);

  sample_locator_t *locs = choose_random_samples(env->samples, d-1);

  for(int i = 0; i < d-1; i++) {
    sample_locator_t loc = locs[i];
    gsl_vector *s = vs->samples[loc.class][loc.index];
    gsl_matrix_set_col(A, i, s);
  }

  free(locs);
  return A;
}

hplane_data *random_bdy_hplane(env_t *env) {
  gsl_matrix *A = random_sample_matrix(env);
  gsl_vector *x = obtain_null_vec(A);
  gsl_matrix_free(A);
  return make_hplane_data(x, classify(env, x));
}

hplane_list *get_cone_vectors(env_t *env, int N) {
  //generate N hyperplanes in the interiors of randomly chosen cones
  hplane_list *B = create_list();
  for(int i = 0; i < N; i++) {
    list_add(B, random_bdy_hplane(env));
  }

  hplane_list *A = create_list();
  gr_displace(env, B, A);

  return A;
}

double apx_cone_ball_rad(env_t *env, gsl_vector *v) {
  //finds the max radius of a unit ball centered at v which is contained entirely in the interior of v's cone
  //equal to the min distance from vd to any hyperplane orthogonal to a sample
  //doesn't assume samples have unit norm (may cause numerical issues if norms are small)

  vsamples_t *vs = env->vsamples;
  double min = 1e101;

  for(int class = 0; class < vs->class_cnt; class++) {
    for(int i = 0; i < vs->count[class]; i++) {
      gsl_vector *s = vs->samples[class][i];

      double dist = fabs(dot(v, s))/sqrt(dot(s,s));

      if(dist < min) {
	min = dist;
      }
    }
  }

  return min;
}

double eps_sample(env_t *env, int N) {
  //sample N random cones and get a value of epsilon such that no optimal cone will be lost

  env->vsamples = samples_to_vec(env->samples);
  env->params->greer_params.skip_rec = 1;

  double min = 1e101;
  hplane_list *A = get_cone_vectors(env, N);
  lnode *curr = A->head;
  while(curr) {
    double r = apx_cone_ball_rad(env, curr->data->v);
    if(r < min)
      min = r;
    curr = curr->next;
  }

  free(list_free(A));

  return min;
}
