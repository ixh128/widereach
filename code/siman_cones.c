#include "widereach.h"
#include "helper.h"

#include <gsl/gsl_rng.h>
#include <gsl/gsl_siman.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_blas.h>
#include <math.h>

#define N_TRIES 200 /* how many points do we try before stepping */
#define ITERS_FIXED_T 2000 /* how many iterations for each T? */
#define STEP_SIZE 10000.0 /* max step size in random walk */
#define K 1.0 /* Boltzmann constant */
#define T_INITIAL 1.0 /* initial temperature */
#define MU_T 1.003 /* damping factor for temperature */
#define T_MIN 0.1 // NOTE: changed

extern env_t *env; //this probably could be included in the current state xp
vsamples_t *vs;

void add_new_proj_ip(gsl_matrix *c, const gsl_vector *x) {
  //compute C' = C - uuT/uTu
  double denom;
  gsl_blas_ddot(x, x, &denom);
  gsl_blas_dger(-1/denom, x, x, c);
}

gsl_vector *get_proj_basis_idxs(env_t *env, int *idxs) {
  size_t d = env->samples->dimension;
  gsl_vector **V = CALLOC(d-1, gsl_vector *);
  for(int i = 0; i < d-1; i++) {
    sample_locator_t *loc = locator(idxs[i]+d+2, env->samples);
    V[i] = vs->samples[loc->class][loc->index];
    free(loc);
  }
  return get_proj_basis(env, V);
}

double cones_energy(void *xp) {
  //returns -1 * obj of best hyperplane on projection defined by points in xp
  //this will be minimized
  int *idxs = xp;
  //printf("state = %d %d\n", idxs[0], idxs[1]);
  int d = env->samples->dimension;
  /*gsl_matrix *c = gsl_matrix_calloc(d, d);
  for(size_t i = 0; i < d; i++)
    gsl_matrix_set(c, i, i, 1);
    for(int i = 0; i < d-1; i++) {
    sample_locator_t *loc = locator(idxs[i]+d+2, env->samples);
    //printf("adding u = \n");
    gsl_vector *u = vs->samples[loc->class][loc->index];
    for(int k = 0; k < u->size; k++)
      printf("%g%s", gsl_vector_get(u, k), k == u->size-1 ? "\n" : " ");
    gsl_vector *cu = apply_proj(c, u);
    add_new_proj_ip(c, cu);
    gsl_vector_free(cu);
    free(loc);
  }*/
  
  /*print_matrix(stdout, c);
  if(idxs[0] == 2 && idxs[1] == 1)
  exit(0);*/

  /*gsl_vector *u1 = gsl_vector_calloc(d);
  gsl_vector_set_all(u1, 1);
  gsl_vector *v1 = apply_proj(c, u1);
  gsl_vector *v2 = gsl_vector_calloc(d);
  gsl_vector_axpby(-1, v1, 0, v2); 
  gsl_vector_free(u1);
  double obj1 = compute_obj(env, vs, v1, c).obj;
  double obj2 = compute_obj(env, vs, v2, c).obj;
  //printf("obj1 = %g, obj2 = %g\n", obj1, obj2);
  gsl_matrix_free(c);*/
  
  gsl_vector *v1 = get_proj_basis_idxs(env, idxs);
  gsl_vector *v2 = gsl_vector_calloc(d);
  gsl_vector_axpby(-1, v1, 0, v2); 
  double obj1 = compute_obj(env, vs, v1, NULL).obj;
  double obj2 = compute_obj(env, vs, v2, NULL).obj;
 
  if(obj1 > obj2) {
    gsl_vector_free(v2);
    //printf("obj = %g\n", obj1);
    return -obj1;
  } else {
    gsl_vector_free(v1);
    //printf("obj = %g\n", obj2);
    return -obj2;
  }
}

void siman_cones_simple_step(const gsl_rng *r, void *xp, double step_size) {
  //replace a random projected-upon sample with a randomly-chosen different one
  int d = env->samples->dimension;
  int rnd_i = gsl_rng_uniform_int(r, env->samples->dimension);
  int *idxs = xp;
  idxs[rnd_i] = -1;
  int newidx;
  int taken;
  do {
    newidx = gsl_rng_uniform_int(r, samples_total(env->samples));
    taken = 0;
    for(int j = 0; j < d-1; j++) {
      if(idxs[j] == newidx) {
	taken = 1;
	break;
      }
    }
  } while(taken == 1);
  idxs[rnd_i] = newidx;
}

void siman_cones_step(const gsl_rng *r, void *xp, double step_size) {
  //choose a random sample from the current state vector
  //exchange it with the furthest sample within a distance step_size
  //this may be bugged - haven't gotten any good results with it yet
  int d = env->samples->dimension;
  int *idxs = xp;
  int changed = 0;
  while(!changed) {
    int rnd_i = gsl_rng_uniform_int(r, d);
    sample_locator_t *rnd_loc = locator(idxs[rnd_i]+d+2, env->samples);
    gsl_vector *rnd_s = vs->samples[rnd_loc->class][rnd_loc->index];
    free(rnd_loc);
    double highest_dist = -1;
    int best_i = -1;
    for(int i = 0; i < samples_total(env->samples); i++) {
      sample_locator_t *loc = locator(i+d+2, env->samples);
      gsl_vector *s = vs->samples[loc->class][loc->index];
      free(loc);
      double dot; //probably should actually compute the angle/dist instead of dot
      //since dot is the cosine of the angle, which isn't monotone
      gsl_blas_ddot(rnd_s, s, &dot);
      if(dot < step_size && dot > highest_dist) {
	int taken = 0;
	for(int j = 0; j < d - 1; j++) {
	  if(idxs[j] == i) {
	    taken = 1;
	    break;
	  }
	}
	if(taken) continue;
	highest_dist = dot;
	best_i = i;
      }
    }
    if(best_i == -1)
      continue;
    changed = 1;
    //printf("exchanging %d for %d\n", idxs[rnd_i], best_i);
    idxs[rnd_i] = best_i;
    /*printf("now idxs = {");
    for(int i = 0; i < d - 1; i++)
    printf("%d%s", idxs[i], i == d-2 ? "}\n" : ", ");*/
    return;
  }
}

double siman_cones_dist(void *xp, void *yp) {
  int *idxs1 = xp, *idxs2 = yp;
  //return the number of components which differ between the two
  //TODO: could try calculating the total distance between the differing points
  /*double dist = 0;
  for(int i = 0; i < env->samples->dimension-1; i++) {
    if(idxs1[i] == idxs2[i])
      dist++;
  }
  return dist;*/
  gsl_vector *v1 = get_proj_basis_idxs(env, idxs1);
  gsl_vector *v2 = get_proj_basis_idxs(env, idxs2);
  double dot;
  gsl_blas_ddot(v1, v2, &dot);
  double theta = acos(dot / (gsl_blas_dnrm2(v1)*gsl_blas_dnrm2(v2)));
  gsl_vector_free(v1);
  gsl_vector_free(v2);
  return theta;
}

void print_siman_cones(void *xp) {
  //double *h = xp;
  /*int n = env->samples->dimension + 1;
  for(int i = 0; i < n; i++)
    printf("%g ", h[i]);*/
  //printf("%g ", h[0]); //placeholder - full plane takes too much space
  //  printf("n/a");
}

double *single_siman_cones_run(unsigned int *seed, int iter_lim, env_t *env_p, int *init) {
  env = env_p;
  vs = samples_to_vec(env->samples);
  gsl_siman_params_t params = {N_TRIES, ITERS_FIXED_T, STEP_SIZE, K, T_INITIAL, MU_T, T_MIN};
  srand48(*seed);
  int d = env->samples->dimension;
  if(!init) {
    //todo: find a better initializer - use best_random_hyperplane_proj

    //this assumes that there are at least d-1 samples, which is fine
    init = CALLOC(d-1, int);
    for(int i = 0; i < d-1; i++)
      init[i] = i;
  }
  gsl_rng *r = gsl_rng_alloc(gsl_rng_taus);
  gsl_rng_set(r, rand());

  gsl_siman_solve(r, init, cones_energy, siman_cones_step, siman_cones_dist, print_siman_cones, NULL, NULL, NULL, (env->samples->dimension+1)*sizeof(double), params);

  /*double *random_solution = blank_solution(env->samples);
  double random_objective_value = hyperplane_to_solution(h0, random_solution, env);
  printf("obj = %g\n", random_objective_value);
  return random_solution;*/
  printf("obj = %g\n", -cones_energy(init));
  vsamples_free(vs);
  exit(0);
  return NULL;
}
