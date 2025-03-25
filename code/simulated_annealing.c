#include "helper.h"
#include "widereach.h"

#include <gsl/gsl_blas.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_siman.h>
#include <math.h>

#define N_TRIES 200        /* how many points do we try before stepping */
#define ITERS_FIXED_T 2000 /* how many iterations for each T? */
#define STEP_SIZE 10.0     /* max step size in random walk */
#define K 1.0              /* Boltzmann constant */
#define T_INITIAL 1.0      /* initial temperature */
#define MU_T 1.003         /* damping factor for temperature */
#define T_MIN 0.1          // NOTE: changed

gsl_siman_params_t params = {N_TRIES,   ITERS_FIXED_T, STEP_SIZE, K,
                             T_INITIAL, MU_T,          T_MIN};
env_t *env; // this probably could be included in the current state xp

double energy(void *xp) {
  // returns -1 * obj of hyperplane specified in xp
  // this will be minimized
  double *h = xp;
  double obj = hyperplane_to_solution(h, NULL, env);
  return -obj;
}

/*void siman_step(const gsl_rng *r, void *xp, double step_size) {
  int n = env->samples->dimension + 1; //add 1 to include c in the space
  double *h = (double *) xp;
  double *u = CALLOC(n, double);
  gsl_ran_dir_nd(r, n, u); //random unit vector
  for(int i = 0; i < n; i++) {
    double newhi = h[i] + step_size * u[i];
    h[i] = newhi;
  }
  free(u);
}*/

double arr_dot(double *a, double *b, size_t d) {
  double res = 0;
  for (int i = 0; i < d; i++)
    res += a[i] * b[i];
  return res;
}

void siman_step(const gsl_rng *r, void *xp, double step_size) {
  int d = env->samples->dimension; // assuming unbiased problem
  double *w = xp;
  double *u = (double *)malloc(d * sizeof(double));
  gsl_ran_dir_nd(r, d, u); // draw random unit vector u

  // project u onto tangent space at w:
  double wu = arr_dot(w, u, d);
  for (size_t i = 0; i < d; i++) {
    u[i] -= wu * w[i]; // remove component in direction w
  }

  // normalize tangent vector
  double u_norm = sqrt(arr_dot(u, u, d));
  if (u_norm > 0) {
    for (size_t i = 0; i < d; i++) {
      u[i] = (step_size / u_norm) * u[i];
      w[i] += u[i];
    }
  }

  // renormalize w
  double w_norm = sqrt(arr_dot(w, w, d));
  for (size_t i = 0; i < d; i++) {
    w[i] /= w_norm;
  }

  free(u);
}

/*double hplane_dist(void *xp, void *yp) {
  double *h1 = xp, *h2 = yp;
  // return the distance between the two corresponding vectors in (d+1)-space
  int n = env->samples->dimension + 1;
  gsl_vector x1 = gsl_vector_view_array(h1, n).vector;
  gsl_vector x2 = gsl_vector_view_array(h2, n).vector;
  gsl_vector_sub(&x1, &x2); // x1 -= x2
  double dist = gsl_blas_dnrm2(&x1);
  return dist;
}*/

double hplane_dist(void *xp, void *yp) {
  // returns the angle between the two hyperplanes, thought of as points on the
  // unit sphere
  double *w1 = xp, *w2 = yp;
  return acos(arr_dot(w1, w2, env->samples->dimension));
}

void print_hplane(void *xp) {
  // double *h = xp;
  /*int n = env->samples->dimension + 1;
  for(int i = 0; i < n; i++)
    printf("%g ", h[i]);*/
  // printf("%g ", h[0]); //placeholder - full plane takes too much space
  printf("n/a");
}

double *single_siman_run(unsigned int *seed, int iter_lim, env_t *env_p,
                         double *h0) {
  env = env_p;
  srand48(*seed);
  if (!h0) {
    // initialize to all zeros
    // this could be best random hyperplane instead
    // h0 = CALLOC(env->samples->dimension+1, double);
    h0 = best_random_hyperplane_unbiased(1, env);
  }
  gsl_rng *r = gsl_rng_alloc(gsl_rng_taus);
  gsl_rng_set(r, rand());

  printf("starting solve\n ");
  gsl_siman_solve(r, h0, energy, siman_step, hplane_dist, print_hplane, NULL,
                  NULL, NULL, env->samples->dimension * sizeof(double), params);
  printf("done\n");

  double *random_solution = blank_solution(env->samples);
  double random_objective_value =
      hyperplane_to_solution(h0, random_solution, env);
  printf("obj = %g\n", random_objective_value);
  return random_solution;
}
