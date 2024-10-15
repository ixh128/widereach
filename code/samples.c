#include "widereach.h"
#include "helper.h"
#include <memory.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <math.h>


/* -------------------- Samples ------------------------------------------ */

int is_binary(const samples_t *samples) {
	return 	 2 == samples->class_cnt &&
		-1 == samples->label[0] &&
		 1 == samples->label[1];
}


size_t samples_total(const samples_t *samples) {
	size_t *count = samples->count;
	int cnt = 0;
	for (size_t class = 0; class < samples->class_cnt; class++) {
		cnt += count[class];
	}
	return cnt;
}


size_t positives(const samples_t *samples) {
	return samples->count[1];
}

size_t negatives(const samples_t *samples) {
	return samples->count[0];
}

samples_t *delete_samples(samples_t *samples) {
	free(samples->label);
	size_t class_cnt = samples->class_cnt;
	for (size_t i = 0; i < class_cnt; i++) {
		size_t cnt = samples->count[i];
		for (size_t j = 0; j < cnt; j++) {
			free(samples->samples[i][j]);
		}
		free(samples->samples[i]);
	}
	free(samples->samples);
	free(samples->count);
	return samples;
}

double primary_value(int label) {
    return label > 0 ? 1. : 0.;
}

double **random_points(size_t count, size_t dimension) {
	double **samples = CALLOC(count, double *);
	for (size_t j = 0; j < count; j++) {
		samples[j] = random_point(dimension);
	}
	return samples;
}


void set_sample_class(
		samples_t *samples, 
		size_t class, 
		int label, 
		size_t count) {
	samples->label[class] = label;
	samples->count[class] = count;
	samples->samples[class] = random_points(count, samples->dimension);
}


samples_t *random_samples(
		size_t count, size_t positives, size_t dimension) {
	samples_t *samples = CALLOC(1, samples_t);
	samples->dimension = dimension;
	samples->class_cnt = 2;
	samples->label = CALLOC(2, int);
	samples->count = CALLOC(2, size_t);
	samples->samples = CALLOC(2, double **);
	if (positives > count) {
		positives = count;
	}
	set_sample_class(samples, 0, -1, count - positives);
	set_sample_class(samples, 1,  1, positives);
	return samples;
}


void print_sample(sample_locator_t loc, samples_t *samples) {
	size_t class = loc.class;
	glp_printf("%i, ", samples->label[class]);

	double *sample = samples->samples[class][loc.index];
	size_t dimension = samples->dimension;
	for (size_t j = 0; j < dimension; j++) {
		glp_printf("%g, ", sample[j]);
	}
	glp_printf("\n");

}

void print_samples(samples_t *samples) {
	size_t class_cnt = samples->class_cnt;
	size_t *counts = samples->count;
	for (size_t class = 0; class < class_cnt; class ++) {
		size_t count = counts[class];
		for (size_t i = 0; i < count; i++) {
			sample_locator_t loc = { class, i };
			print_sample(loc, samples);
		}
	}
}


double distance(
        sample_locator_t *loc, 
        samples_t *samples, 
        double *hyperplane, 
		double precision) {
	size_t class = loc->class;
	double *sample = samples->samples[class][loc->index];
    size_t dimension = samples->dimension;
	double product = - hyperplane[dimension] 
                     - samples->label[class] * precision;
	for (size_t i = 0; i < dimension; i++) {
		product += sample[i] * hyperplane[i];
	}
    return product;
}

double sample_violation(
        sample_locator_t *loc, 
        samples_t *samples, 
        double *hyperplane, 
		double precision) {
    double dist = distance(loc, samples, hyperplane, precision);
    if (loc->class > 0) {
        dist = -dist;
    }
    return dist;
}


int side(
		sample_locator_t *loc, 
		samples_t *samples, 
		double *hyperplane, 
		double precision) {
    double dist = distance(loc, samples, hyperplane, precision);
    return 0. == dist ? loc->class : dist > 0.;
    //return dist >= 0;
}

int side_cnt(
		int class, 
		samples_t *samples, 
		double *hyperplane,
		double precision) {
    sample_locator_t loc;
    loc.class = (int) class;
    int positive_cnt = 0;
    size_t count = samples->count[class];
    for (size_t i = 0; i < count; i++) {
        loc.index = i;
        positive_cnt += side(&loc, samples, hyperplane, precision);
    }
    return positive_cnt;
}

int reduce(
    samples_t *samples,
    void *initial,
    int (*accumulator)(samples_t *, sample_locator_t, void *, void *),
    void *aux) {
  void *result = initial;
  size_t classes[] = { 1, 0 };
  for (int class_index = 0; class_index < 2; class_index++) {
    size_t class = classes[class_index];
    int cnt = samples->count[class];
    for (size_t idx = 0; idx < cnt; idx++) {
      sample_locator_t locator = { class, idx };
      int state = accumulator(samples, locator, result, aux);
      if (state != 0) {
        return state;
      }
	}
  }

  return 0;
}


void add_bias(samples_t *samples) {
  for(int class = 0; class < 2; class++) {
    for(size_t i = 0; i < samples->count[class]; i++) {
      double *s = samples->samples[class][i];
      double *new_s = realloc(s, (samples->dimension+1)*sizeof(double));
      if(!new_s) {
	printf("add_bias: realloc failed\n");
	exit(EXIT_FAILURE);
      }
      s = new_s;
      s[samples->dimension] = 1;
      samples->samples[class][i] = s;
    }
  }
  samples->dimension++;
}

void normalize_samples(samples_t *samples) {
  //scales all samples to have unit norm
  //this is valid if the problem is unbiased
  size_t d = samples->dimension;
  for(int class = 0; class < 2; class++) {
    for(size_t i = 0; i < samples->count[class]; i++) {
      double *s = samples->samples[class][i];
      double norm = 0;
      for(int j = 0; j < d; j++)
	norm += s[j]*s[j];
      norm = sqrt(norm);
      for(int j = 0; j < d; j++)
	s[j] /= norm;
    }
  }
}

gsl_vector **uniform_sphere_points(size_t n, size_t d) {
  gsl_vector **pts = CALLOC(n, gsl_vector *);
  gsl_rng *r = gsl_rng_alloc(gsl_rng_taus);
  gsl_rng_set(r, rand());
  for(int i = 0; i < n; i++) {
    pts[i] = gsl_vector_alloc(d);
    double *v = CALLOC(d, double);
    gsl_ran_dir_nd(r, d, v);
    gsl_vector_view view = gsl_vector_view_array(v, d);
    gsl_vector_memcpy(pts[i], &view.vector);
    free(v);
  }
  gsl_rng_free(r);
  return pts;
}

round_pts_t random_round_points(env_t *env, vsamples_t *vs, size_t n) {
  size_t d = env->samples->dimension;
  if(n > samples_total(env->samples)) {
    printf("ERR: too many round pts\n");
    return (round_pts_t) {0};
  }
  int *taken_pts = CALLOC(samples_total(env->samples), int); //=1 if taken, 0 otherwise
  gsl_rng *r = gsl_rng_alloc(gsl_rng_taus);
  gsl_rng_set(r, rand());
  
  int idx;
  for(int i = 0; i < n; i++) {
    do {
      idx = gsl_rng_uniform_int(r, samples_total(env->samples));
    } while (taken_pts[idx]);
    taken_pts[idx] = 1;
  }

  gsl_vector **pts = CALLOC(n, gsl_vector *);
  int k = 0;
  for(int i = 0; i < samples_total(env->samples); i++) {
    if(!taken_pts[i]) continue;
    sample_locator_t *loc = locator(i+d+2, env->samples);
    //pts[k++] = vs->samples[loc->class][loc->index];
    pts[k] = gsl_vector_calloc(d);
    gsl_vector_memcpy(pts[k++], vs->samples[loc->class][loc->index]);
    free(loc);
  }
  
  gsl_rng_free(r);
  free(taken_pts);
  return (round_pts_t) {n, pts};
}

vsamples_t *create_rounded_vs(vsamples_t *vs, round_pts_t *round_pts) {
  size_t n = round_pts->n;
  size_t d = vs->dimension;
  vsamples_t *rvs = malloc(sizeof(vsamples_t));
  rvs->dimension = d;
  rvs->class_cnt = vs->class_cnt;
  rvs->count = CALLOC(vs->class_cnt, size_t);
  memcpy(rvs->count, vs->count, vs->class_cnt*sizeof(size_t));
  rvs->label = CALLOC(vs->class_cnt, int);
  memcpy(rvs->label, vs->label, vs->class_cnt*sizeof(int));
  rvs->samples = CALLOC(vs->class_cnt, gsl_vector **);

  round_pts->npos = CALLOC(n, int);
  round_pts->nneg = CALLOC(n, int);
  
  for(int class = 0; class < vs->class_cnt; class++) {
    rvs->samples[class] = CALLOC(rvs->count[class], gsl_vector *);
    for(int i = 0; i < vs->count[class]; i++) {
      rvs->samples[class][i] = gsl_vector_calloc(d);
      double min_angle = 1e101;
      int closest_idx;
      for(int j = 0; j < n; j++) {
	//assuming normalized data, angle = acos(dot product)
	double cos_ang, angle;
	gsl_blas_ddot(vs->samples[class][i], round_pts->pts[j], &cos_ang);
	angle = acos(cos_ang);
	if(angle < min_angle) {
	  min_angle = angle;
	  closest_idx = j; 
	}
      }
      gsl_vector_memcpy(rvs->samples[class][i], round_pts->pts[closest_idx]);
      if(vs->label[class] == 1)
	round_pts->npos[closest_idx]++;
      else
	round_pts->nneg[closest_idx]++;
    }
  }
  return rvs;
}

rounded_pts_t random_rounded_points(env_t *env, vsamples_t *vs, size_t n) {
  round_pts_t round_pts = random_round_points(env, vs, n);
  return (rounded_pts_t) {round_pts, create_rounded_vs(vs, &round_pts)};
}
