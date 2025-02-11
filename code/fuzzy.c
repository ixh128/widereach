#include "widereach.h"
#include "helper.h"

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_roots.h>
#include <math.h>
#include <memory.h>


double sample_dist(fuzzy_info_t *fuzzy_info, gsl_rng *rng) {
  double p = fuzzy_info->param;
  size_t d = fuzzy_info->dimension;
  size_t d0 = fuzzy_info->dist_dims;
  //TOOD: should we use d or d0 for chi-squared?
  switch(fuzzy_info->dist) {
  case CHI_SQ:
    return p*p*gsl_ran_chisq(rng, d0);
  case CHI:
    //we can only sample a standard chi-squared variable, so we have to take its square root and scale it by sigma
    return p*sqrt(gsl_ran_chisq(rng, d0));
  case EXP:
    return gsl_ran_exponential(rng, p);
  case STUDENT_T:
    return gsl_ran_tdist(rng, p);
  }
}

int need_to_truncate(size_t dimension, double *s, int norm) {
  //ensure s is in hypercube and mag is <= 1. if not, returns 1
  double mag = 0;
  for(int i = 0; i < dimension; i++) {
    if(s[i] > 1 || s[i] < 0) return 1;
    if(norm == 2) {
      mag += s[i]*s[i]; //this will technically get mag^2, but we are checking if it is <= 1, so it doesn't matter
    } else if(norm == 1) {
      mag += fabs(s[i]);
    } else {
      printf("ERR: invalid norm\n");
    }
  }
  return !(mag <= 1);
}

void unit_vec(double *x, size_t d, int norm) {
  //generates a unit vector x in R^d, wrt the given norm (1 or 2)
  //currently, any L1 unit vector will be automatically in the first quadrant - that's okay for now, because we will take its absolute value anyways
  if(norm == 2) {
    random_unit_vector(d, x);
  } else if(norm == 1) {
    double *s = random_simplex_point(1, d);
    double norm = 0;
    for(int i = 0; i < d; i++)
      norm += fabs(s[i]);
    for(int i = 0; i < d; i++)
      x[i] = s[i]/norm;
    free(s);
  } else {
    printf("ERR: invalid norm\n");
  }
}

double **random_fuzzy_points(size_t count, fuzzy_info_t *fuzzy_info, gsl_rng *rng) {
  /* this is the old function to generate the cluster benchmark */
  double **samples = CALLOC(count, double *);
  size_t count_cluster = count * (1 - fuzzy_info->noise_ratio);
  size_t mirror_count = count_cluster / fuzzy_info->cluster_cnt;
  size_t dimension = fuzzy_info->dimension;
  size_t dist_dims = fuzzy_info->dist_dims;
  int norm = fuzzy_info->norm;
  for (size_t j = 0; j < count_cluster; j++) {
    double *s = CALLOC(dimension, double);
    //repeatedly sample until we get something that it is the hypercube
    //if the distribution is too heavy-tailed, this could be a little slow
    do {
      unit_vec(s, dist_dims, norm);
      double mag = sample_dist(fuzzy_info, rng);
      for(int i = 0; i < dist_dims; i++)
	s[i] = fabs(s[i]) * mag;
    } while(need_to_truncate(dist_dims, s, norm));
    for(int i = dist_dims; i < dimension; i++) {
      s[i] = drand48();
    }
    samples[j] = s;
    if (j >= mirror_count) {
      mirror_sample(dimension, samples[j]);
    }
  }
  for (size_t j = count_cluster; j < count; j++) {
    samples[j] = random_point(dimension);
  }
  return samples;
}

void set_sample_class_fuzzy(
		samples_t *samples, 
		size_t class, 
		int label, 
		size_t count,
		fuzzy_info_t *fuzzy_info) {
  samples->label[class] = label;
  samples->count[class] = count;
  gsl_rng *rng = gsl_rng_alloc(gsl_rng_taus);
  samples->samples[class] = random_fuzzy_points(count, fuzzy_info, rng);
  gsl_rng_free(rng);
}

samples_t *random_fuzzy_samples(fuzzy_info_t *fuzzy_info) {
  samples_t *samples = CALLOC(1, samples_t);
  samples->dimension = fuzzy_info->dimension;
  samples->class_cnt = 2;
  samples->label = CALLOC(2, int);
  samples->count = CALLOC(2, size_t);
  samples->samples = CALLOC(2, double **);
  size_t *positives = &(fuzzy_info->positives);
  size_t count = fuzzy_info->count;
  if (*positives > count) {
    *positives = count;
  }
  set_sample_class(samples, 0, -1, count - *positives);
  set_sample_class_fuzzy(samples, 1,  1, *positives, fuzzy_info);
  return samples;
}

double cdf_chi(double x, double nu) {
  //gsl doesn't have the chi dist (only chi-sq), so we do it ourselves
  //we will use its chi-squared cdf function, which takes nu = # deg of freedom as an argument
  //that should equal our dimension
  //TODO: fix that
  return gsl_cdf_chisq_P(x*x, nu);
}

double (*getCdf(cluster_dist dist))(double, double) {
  //returns the CDF for the given distribution
  switch(dist) {
  case CHI:
    return cdf_chi;
  case EXP:
    return gsl_cdf_exponential_P;
  case STUDENT_T:
    return gsl_cdf_tdist_P;
  default:
    printf("invalid dist in getCdf\n");
    return NULL;
  }
}

typedef struct dist_params {
  cluster_dist dist;
  double local_ratio;
  double global_ratio;
  double max_rad;
  double dims;
  double r;
  int norm;
} dist_params;

double dist_to_solve(double sigma, void *params) {
  dist_params p = *(dist_params *) params;
  double r = p.r;
  double f = p.local_ratio;
  double d = p.dims;
  double (*cdf)(double, double) = getCdf(p.dist);

  double ball_vol = 0;
  //if norm is incorrect, ball_vol will stay at 0
  if(p.norm == 1) {
    ball_vol = pow(r, d)/gsl_sf_fact(d);
  } else if(p.norm == 2) {
    ball_vol = pow(M_PI, (double) d/2) * pow(r, d) / (pow(2, d)*gsl_sf_gamma(1+(double) d/2));
  }

  return cdf(r, sigma) - (f/(p.global_ratio * ball_vol))*cdf(p.max_rad, sigma);
}

double compute_param(size_t dimension, double global_ratio, double local_ratio, double rad, cluster_dist dist) {
  //computes the parameter of the distribution needed to get the given local ratio within the truncated ball of
  //global ratio = N/P
  //local ratio = desired N/P within the cluster (a.k.a. f)
  //max rad = see def of fuzzy_info struct (probably should be 1)
  
  const gsl_root_fsolver_type *T = gsl_root_fsolver_brent;
  gsl_root_fsolver *s = gsl_root_fsolver_alloc (T);

  gsl_function f;
  dist_params p = {.dist = dist, .local_ratio = local_ratio, .global_ratio = global_ratio, .max_rad = 1, .dims = dimension, .r = rad};
  f.function = dist_to_solve;
  f.params = &p;
  double x_lo = 0;
  double x_hi = 1000;  //todo: come up with better bounds
  gsl_root_fsolver_set(s, &f, x_lo, x_hi);
  printf("computing param using %s method\n", gsl_root_fsolver_name(s));
  printf ("%5s [%9s, %9s] %9s %9s\n",
	  "iter", "lower", "upper", "root", "err(est)");
  int iter = 0;
  int status;
  double root;
  do {
      iter++;
      status = gsl_root_fsolver_iterate (s);
      root = gsl_root_fsolver_root (s);
      x_lo = gsl_root_fsolver_x_lower (s);
      x_hi = gsl_root_fsolver_x_upper (s);
      status = gsl_root_test_interval (x_lo, x_hi,
				       0, 0.001);
      if (status == GSL_SUCCESS)
	printf ("Converged:\n");
      printf ("%5d [%.7f , %.7f ] %.7f %+.7f \n",
	      iter, x_lo, x_hi,
	      root, x_hi - x_lo);
    }
  while (status == GSL_CONTINUE && iter < 100);

  if(status != GSL_SUCCESS) {
    printf("failed to converge\n");
  }
  gsl_root_fsolver_free(s);
  
  return root;
}

double *l1_cut_hplane(fuzzy_info_t *fuzzy_info) {
  double *w = CALLOC(fuzzy_info->dimension + 1, double);
  for(int i = 0; i < fuzzy_info->dist_dims; i++) {
    w[i] = -1;
  }
  w[fuzzy_info->dimension] = -fuzzy_info->rad;
  return w;
}
